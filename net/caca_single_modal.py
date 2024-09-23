import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# Loss
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# MLP class for projector
class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# MLP class for predictor
class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)  # using BN may cause unstable
        )

    def forward(self, x):
        x = self.net(x)
        return x


class CACA(nn.Module):
    """ Referring to the code of BYOL """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, projection_hidden_size=1024,
                 projection_size=128, prediction_hidden_size=1024, moving_average_decay=0.99, use_momentum=True,
                 in_channels=3, hidden_channels=16, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        projection_size: the feature dimension after encoder and projection head
        projection_hidden_size: the feature dimension in the middle of MLP
        moving_average_decay: (default: 0.99)
        use_momentum: True: BYOL, False: Simsiam
        """
        super().__init__()
        self.pretrain = pretrain
        self.use_momentum = use_momentum

        str_encoder = base_encoder
        base_encoder = import_class(base_encoder)
        if str_encoder == 'net.st_gcn.Model':
            self.online_backbone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                    feature_dim=feature_dim, dropout=dropout, graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting,
                                    **kwargs)
        elif str_encoder == 'net.unik.Model':
            self.online_backbone = base_encoder(feature_dim=feature_dim, num_joints=25,
                                                num_person=2, tau=1, num_heads=3,in_channels=in_channels)
        elif str_encoder == 'net.ctrgcn.Model':
            self.online_backbone = base_encoder(feature_dim=feature_dim, num_point=25, num_person=2,
                                                graph='graph.ntu_rgb_d.Graph',
                                                graph_args={'labeling_mode': 'spatial'},
                                                in_channels=in_channels, drop_out=0, adaptive=True)
        elif str_encoder == 'net.gru.GRU_model':
            self.online_backbone = base_encoder(input_size=150, hidden_size=feature_dim // 2, num_layers=3)
        else:
            pass

        if self.pretrain:
            self.online_projector = projection_MLP(feature_dim, projection_hidden_size, projection_size)

            if self.use_momentum:
                self.target_backbone = copy.deepcopy(self.online_backbone)
                set_requires_grad(self.target_backbone, False)

                self.target_projector = copy.deepcopy(self.online_projector)
                set_requires_grad(self.target_projector, False)

            else:
                self.target_backbone = self.online_backbone
                self.target_projector = self.online_projector

            self.target_ema_updater = EMA(moving_average_decay)
            self.online_predictor = prediction_MLP(projection_size, prediction_hidden_size, projection_size)
        else:
            self.online_projector = nn.Linear(feature_dim, num_class)

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        for current_params, ma_params in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

        for current_params, ma_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(self, seq_1, seq_2=None):
        """
        Input:
            seq_1: a batch of sequences
            seq_2: a batch of sequences
        """

        if not self.pretrain:
            return self.online_projector(self.online_backbone(seq_1))

        # Compute the feature
        online_proj_1 = self.online_projector(self.online_backbone(seq_1))
        online_proj_2 = self.online_projector(self.online_backbone(seq_2))

        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with torch.no_grad():
            target_proj_1 = self.target_projector(self.target_backbone(seq_1))
            target_proj_2 = self.target_projector(self.target_backbone(seq_2))
            target_proj_1.detach_()
            target_proj_2.detach_()

        loss_min = loss_fn(online_pred_1, online_pred_2)
        loss_max1 = loss_fn(online_pred_1, target_proj_1.detach())
        loss_max2 = loss_fn(online_pred_2, target_proj_2.detach())
        loss = loss_min - 0.5 * (loss_max1 + loss_max2)

        return loss.mean()

