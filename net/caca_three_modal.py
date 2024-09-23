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


class CACA_3views(nn.Module):
    """ Referring to the code of BYOL """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, projection_hidden_size=1024,
                 projection_size=128, prediction_hidden_size=1024, moving_average_decay=0.99, use_momentum=True,
                 in_channels=3, hidden_channels=16, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, topk=2, lamb=10, **kwargs):
        """
        projection_size: the feature dimension after encoder and projection head
        projection_hidden_size: the feature dimension in the middle of MLP
        moving_average_decay: (default: 0.99)
        use_momentum: True: BYOL, False: Simsiam
        """
        super().__init__()
        self.pretrain = pretrain
        self.use_momentum = use_momentum
        self.lamb = lamb
        self.topk = topk
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        if graph_args['layout'] == 'uav-human':
            self.Bone = [(10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)]
            self.Bone = [(i + 1, j + 1) for (i, j) in self.Bone]

        base_encoder = import_class(base_encoder)
        self.online_backbone_joint = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                feature_dim=feature_dim, dropout=dropout, graph_args=graph_args,
                                edge_importance_weighting=edge_importance_weighting,
                                **kwargs)
        self.online_backbone_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                      feature_dim=feature_dim, dropout=dropout, graph_args=graph_args,
                                      edge_importance_weighting=edge_importance_weighting,
                                      **kwargs)
        self.online_backbone_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                      feature_dim=feature_dim, dropout=dropout, graph_args=graph_args,
                                      edge_importance_weighting=edge_importance_weighting,
                                      **kwargs)

        if self.pretrain:
            self.online_projector_joint = projection_MLP(feature_dim, projection_hidden_size, projection_size)
            self.online_projector_motion = projection_MLP(feature_dim, projection_hidden_size, projection_size)
            self.online_projector_bone = projection_MLP(feature_dim, projection_hidden_size, projection_size)

            if self.use_momentum:
                self.target_backbone_joint = copy.deepcopy(self.online_backbone_joint)
                set_requires_grad(self.target_backbone_joint, False)
                self.target_backbone_motion = copy.deepcopy(self.online_backbone_motion)
                set_requires_grad(self.target_backbone_motion, False)
                self.target_backbone_bone = copy.deepcopy(self.online_backbone_bone)
                set_requires_grad(self.target_backbone_bone, False)

                self.target_projector_joint = copy.deepcopy(self.online_projector_joint)
                set_requires_grad(self.target_projector_joint, False)
                self.target_projector_motion = copy.deepcopy(self.online_projector_motion)
                set_requires_grad(self.target_projector_motion, False)
                self.target_projector_bone = copy.deepcopy(self.online_projector_bone)
                set_requires_grad(self.target_projector_bone, False)

            else:
                self.target_backbone_joint = self.online_backbone_joint
                self.target_backbone_motion = self.online_backbone_motion
                self.target_backbone_bone = self.online_backbone_bone

                self.target_projector_joint = self.online_projector_joint
                self.target_projector_motion = self.online_projector_motion
                self.target_projector_bone = self.online_projector_bone

            self.target_ema_updater = EMA(moving_average_decay)

            self.online_predictor_joint = prediction_MLP(projection_size, prediction_hidden_size, projection_size)
            self.online_predictor_motion = prediction_MLP(projection_size, prediction_hidden_size, projection_size)
            self.online_predictor_bone = prediction_MLP(projection_size, prediction_hidden_size, projection_size)
        else:
            self.online_projector_joint = nn.Linear(feature_dim, num_class)
            self.online_projector_motion = nn.Linear(feature_dim, num_class)
            self.online_projector_bone = nn.Linear(feature_dim, num_class)

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        for current_params, ma_params in zip(self.online_backbone_joint.parameters(), self.target_backbone_joint.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)
        for current_params, ma_params in zip(self.online_backbone_motion.parameters(), self.target_backbone_motion.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)
        for current_params, ma_params in zip(self.online_backbone_bone.parameters(), self.target_backbone_bone.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

        for current_params, ma_params in zip(self.online_projector_joint.parameters(), self.target_projector_joint.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)
        for current_params, ma_params in zip(self.online_projector_motion.parameters(), self.target_projector_motion.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)
        for current_params, ma_params in zip(self.online_projector_bone.parameters(), self.target_projector_bone.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(self, seq_1, seq_2=None, stream='all', cross=False):
        """
        Input:
            seq_1: a batch of sequences
            seq_2: a batch of sequences
        """

        seq_1_motion = torch.zeros_like(seq_1)
        seq_1_motion[:, :, :-1, :, :] = seq_1[:, :, 1:, :, :] - seq_1[:, :, :-1, :, :]

        seq_1_bone = torch.zeros_like(seq_1)
        for v1, v2 in self.Bone:
            seq_1_bone[:, :, :, v1 - 1, :] = seq_1[:, :, :, v1 - 1, :] - seq_1[:, :, :, v2 - 1, :]

        if not self.pretrain:
            if stream == 'joint':
                return self.online_projector_joint(self.online_backbone_joint(seq_1))
            elif stream == 'motion':
                return self.online_projector_motion(self.online_backbone_motion(seq_1_motion))
            elif stream == 'bone':
                return self.online_projector_bone(self.online_backbone_bone(seq_1_bone))
            elif stream == 'all':
                return self.online_projector_joint(self.online_backbone_joint(seq_1))\
                       + self.online_projector_motion(self.online_backbone_motion(seq_1_motion))\
                       + self.online_projector_bone(self.online_backbone_bone(seq_1_bone))
            else:
                raise ValueError

        if cross:
            return self.cross_training(seq_1, seq_2)

        seq_2_motion = torch.zeros_like(seq_2)
        seq_2_motion[:, :, :-1, :, :] = seq_2[:, :, 1:, :, :] - seq_2[:, :, :-1, :, :]

        seq_2_bone = torch.zeros_like(seq_2)
        for v1, v2 in self.Bone:
            seq_2_bone[:, :, :, v1 - 1, :] = seq_2[:, :, :, v1 - 1, :] - seq_2[:, :, :, v2 - 1, :]

        # Compute the joint feature
        online_proj_j1 = self.online_projector_joint(self.online_backbone_joint(seq_1))
        online_proj_j2 = self.online_projector_joint(self.online_backbone_joint(seq_2))

        online_pred_j1 = self.online_predictor_joint(online_proj_j1)
        online_pred_j2 = self.online_predictor_joint(online_proj_j2)

        # Compute the motion feature
        online_proj_m1 = self.online_projector_motion(self.online_backbone_motion(seq_1_motion))
        online_proj_m2 = self.online_projector_motion(self.online_backbone_motion(seq_2_motion))

        online_pred_m1 = self.online_predictor_motion(online_proj_m1)
        online_pred_m2 = self.online_predictor_motion(online_proj_m2)

        # Compute the bone feature
        online_proj_b1 = self.online_projector_bone(self.online_backbone_bone(seq_1_bone))
        online_proj_b2 = self.online_projector_bone(self.online_backbone_bone(seq_2_bone))

        online_pred_b1 = self.online_predictor_bone(online_proj_b1)
        online_pred_b2 = self.online_predictor_bone(online_proj_b2)

        with torch.no_grad():
            # joint
            target_proj_j1 = self.target_projector_joint(self.target_backbone_joint(seq_1))
            target_proj_j2 = self.target_projector_joint(self.target_backbone_joint(seq_2))
            target_proj_j1.detach_()
            target_proj_j2.detach_()

            # motion
            target_proj_m1 = self.target_projector_motion(self.target_backbone_motion(seq_1_motion))
            target_proj_m2 = self.target_projector_motion(self.target_backbone_motion(seq_2_motion))
            target_proj_m1.detach_()
            target_proj_m2.detach_()

            # bone
            target_proj_b1 = self.target_projector_bone(self.target_backbone_bone(seq_1_bone))
            target_proj_b2 = self.target_projector_bone(self.target_backbone_bone(seq_2_bone))
            target_proj_b1.detach_()
            target_proj_b2.detach_()

        loss_joint_min = loss_fn(online_pred_j1, online_pred_j2)
        loss_joint_max1 = loss_fn(online_pred_j1, target_proj_j1.detach())
        loss_joint_max2 = loss_fn(online_pred_j2, target_proj_j2.detach())
        loss_joint = loss_joint_min - 0.5 * (loss_joint_max1 + loss_joint_max2)

        loss_motion_min = loss_fn(online_pred_m1, online_pred_m2)
        loss_motion_max1 = loss_fn(online_pred_m1, target_proj_m1.detach())
        loss_motion_max2 = loss_fn(online_pred_m2, target_proj_m2.detach())
        loss_motion = loss_motion_min - 0.5 * (loss_motion_max1 + loss_motion_max2)

        loss_bone_min = loss_fn(online_pred_b1, online_pred_b2)
        loss_bone_max1 = loss_fn(online_pred_b1, target_proj_b1.detach())
        loss_bone_max2 = loss_fn(online_pred_b2, target_proj_b2.detach())
        loss_bone = loss_bone_min - 0.5 * (loss_bone_max1 + loss_bone_max2)

        return loss_joint.mean(), loss_motion.mean(), loss_bone.mean()

    def cross_training(self, seq_1, seq_2):

        seq_1_motion = torch.zeros_like(seq_1)
        seq_1_motion[:, :, :-1, :, :] = seq_1[:, :, 1:, :, :] - seq_1[:, :, :-1, :, :]

        seq_1_bone = torch.zeros_like(seq_1)
        for v1, v2 in self.Bone:
            seq_1_bone[:, :, :, v1 - 1, :] = seq_1[:, :, :, v1 - 1, :] - seq_1[:, :, :, v2 - 1, :]

        seq_2_motion = torch.zeros_like(seq_2)
        seq_2_motion[:, :, :-1, :, :] = seq_2[:, :, 1:, :, :] - seq_2[:, :, :-1, :, :]

        seq_2_bone = torch.zeros_like(seq_2)
        for v1, v2 in self.Bone:
            seq_2_bone[:, :, :, v1 - 1, :] = seq_2[:, :, :, v1 - 1, :] - seq_2[:, :, :, v2 - 1, :]

        # Compute the joint feature
        online_proj_j1 = self.online_projector_joint(self.online_backbone_joint(seq_1))
        online_proj_j2 = self.online_projector_joint(self.online_backbone_joint(seq_2))

        online_pred_j1 = self.online_predictor_joint(online_proj_j1)
        online_pred_j2 = self.online_predictor_joint(online_proj_j2)

        # Compute the motion feature
        online_proj_m1 = self.online_projector_motion(self.online_backbone_motion(seq_1_motion))
        online_proj_m2 = self.online_projector_motion(self.online_backbone_motion(seq_2_motion))

        online_pred_m1 = self.online_predictor_motion(online_proj_m1)
        online_pred_m2 = self.online_predictor_motion(online_proj_m2)

        # Compute the bone feature
        online_proj_b1 = self.online_projector_bone(self.online_backbone_bone(seq_1_bone))
        online_proj_b2 = self.online_projector_bone(self.online_backbone_bone(seq_2_bone))

        online_pred_b1 = self.online_predictor_bone(online_proj_b1)
        online_pred_b2 = self.online_predictor_bone(online_proj_b2)

        with torch.no_grad():
            # joint
            target_proj_j1 = self.target_projector_joint(self.target_backbone_joint(seq_1))
            target_proj_j2 = self.target_projector_joint(self.target_backbone_joint(seq_2))
            target_proj_j1.detach_()
            target_proj_j2.detach_()

            # motion
            target_proj_m1 = self.target_projector_motion(self.target_backbone_motion(seq_1_motion))
            target_proj_m2 = self.target_projector_motion(self.target_backbone_motion(seq_2_motion))
            target_proj_m1.detach_()
            target_proj_m2.detach_()

            # bone
            target_proj_b1 = self.target_projector_bone(self.target_backbone_bone(seq_1_bone))
            target_proj_b2 = self.target_projector_bone(self.target_backbone_bone(seq_2_bone))
            target_proj_b1.detach_()
            target_proj_b2.detach_()

        # The distribute
        sim_j1_proj = F.cosine_similarity(online_pred_j1.unsqueeze(1), target_proj_j2.unsqueeze(0), dim=2)
        sim_j2_proj = F.cosine_similarity(online_pred_j2.unsqueeze(1), target_proj_j1.unsqueeze(0), dim=2)

        sim_m1_proj = F.cosine_similarity(online_pred_m1.unsqueeze(1), target_proj_m2.unsqueeze(0), dim=2)
        sim_m2_proj = F.cosine_similarity(online_pred_m2.unsqueeze(1), target_proj_m1.unsqueeze(0), dim=2)

        sim_b1_proj = F.cosine_similarity(online_pred_b1.unsqueeze(1), target_proj_b2.unsqueeze(0), dim=2)
        sim_b2_proj = F.cosine_similarity(online_pred_b2.unsqueeze(1), target_proj_b1.unsqueeze(0), dim=2)

        # Label
        top_k = self.topk
        label_j1 = torch.eye(sim_j1_proj.shape[0]).to(sim_j1_proj.device)
        label_j2 = torch.eye(sim_j2_proj.shape[0]).to(sim_j2_proj.device)

        label_m1 = torch.eye(sim_m1_proj.shape[0]).to(sim_m1_proj.device)
        label_m2 = torch.eye(sim_m2_proj.shape[0]).to(sim_m2_proj.device)

        label_b1 = torch.eye(sim_b1_proj.shape[0]).to(sim_b1_proj.device)
        label_b2 = torch.eye(sim_b2_proj.shape[0]).to(sim_b2_proj.device)

        _, topk_ix_j1 = torch.topk(sim_j1_proj, top_k, dim=1)
        _, topk_ix_j2 = torch.topk(sim_j2_proj, top_k, dim=1)

        _, topk_ix_m1 = torch.topk(sim_m1_proj, top_k, dim=1)
        _, topk_ix_m2 = torch.topk(sim_m2_proj, top_k, dim=1)

        _, topk_ix_b1 = torch.topk(sim_b1_proj, top_k, dim=1)
        _, topk_ix_b2 = torch.topk(sim_b2_proj, top_k, dim=1)

        label_j1.scatter_(1, topk_ix_j1, 1)
        label_j2.scatter_(1, topk_ix_j2, 1)

        label_m1.scatter_(1, topk_ix_m1, 1)
        label_m2.scatter_(1, topk_ix_m2, 1)

        label_b1.scatter_(1, topk_ix_b1, 1)
        label_b2.scatter_(1, topk_ix_b2, 1)

        label_3views = label_j1 * label_j2 + label_m1 * label_m2 + label_b1 * label_b2
        label_3views = torch.where(label_3views >= 2, torch.ones(1).to(label_3views.device),
                                   torch.zeros(1).to(label_3views.device))
        label_3views.detach_()

        # Logits
        log_j1 = torch.softmax(sim_j1_proj, dim=1)
        log_j2 = torch.softmax(sim_j2_proj, dim=1)

        log_m1 = torch.softmax(sim_m1_proj, dim=1)
        log_m2 = torch.softmax(sim_m2_proj, dim=1)

        log_b1 = torch.softmax(sim_b1_proj, dim=1)
        log_b2 = torch.softmax(sim_b2_proj, dim=1)

        ddm_loss_j1 = -torch.mean(torch.sum(torch.log(log_j1) * label_3views, dim=1))  # DDM loss
        ddm_loss_j2 = -torch.mean(torch.sum(torch.log(log_j2) * label_3views, dim=1))  # DDM loss

        ddm_loss_m1 = -torch.mean(torch.sum(torch.log(log_m1) * label_3views, dim=1))  # DDM loss
        ddm_loss_m2 = -torch.mean(torch.sum(torch.log(log_m2) * label_3views, dim=1))  # DDM loss

        ddm_loss_b1 = -torch.mean(torch.sum(torch.log(log_b1) * label_3views, dim=1))  # DDM loss
        ddm_loss_b2 = -torch.mean(torch.sum(torch.log(log_b2) * label_3views, dim=1))  # DDM loss

        loss_joint_min = loss_fn(online_pred_j1, online_pred_j2)
        loss_joint_max1 = loss_fn(online_pred_j1, target_proj_j1.detach())
        loss_joint_max2 = loss_fn(online_pred_j2, target_proj_j2.detach())
        loss_joint = loss_joint_min - 0.5 * (loss_joint_max1 + loss_joint_max2) + self.lamb * (ddm_loss_j1 + ddm_loss_j2)

        loss_motion_min = loss_fn(online_pred_m1, online_pred_m2)
        loss_motion_max1 = loss_fn(online_pred_m1, target_proj_m1.detach())
        loss_motion_max2 = loss_fn(online_pred_m2, target_proj_m2.detach())
        loss_motion = loss_motion_min - 0.5 * (loss_motion_max1 + loss_motion_max2) + self.lamb * (ddm_loss_m1 + ddm_loss_m2)

        loss_bone_min = loss_fn(online_pred_b1, online_pred_b2)
        loss_bone_max1 = loss_fn(online_pred_b1, target_proj_b1.detach())
        loss_bone_max2 = loss_fn(online_pred_b2, target_proj_b2.detach())
        loss_bone = loss_bone_min - 0.5 * (loss_bone_max1 + loss_bone_max2) + self.lamb * (ddm_loss_b1 + ddm_loss_b2)

        loss = loss_joint + loss_motion + loss_bone

        return loss.mean()