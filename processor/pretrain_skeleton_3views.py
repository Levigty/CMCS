import sys
import argparse
import yaml
import math
import random
import numpy as np
import pickle

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor
from .knn_monitor import knn_monitor


class Skeleton_3views_Processor(PT_Processor):
    """
        Processor for 3s-Skeleton Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_motion_value = []
        loss_bone_value = []

        class KNN_Model(nn.Module):
            def __init__(self, knn_model):
                super().__init__()
                self.knn_model = knn_model
                self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

            def forward(self, seq_1, stream=self.arg.stream):
                seq_1_motion = torch.zeros_like(seq_1)
                seq_1_motion[:, :, :-1, :, :] = seq_1[:, :, 1:, :, :] - seq_1[:, :, :-1, :, :]

                seq_1_bone = torch.zeros_like(seq_1)
                for v1, v2 in self.Bone:
                    seq_1_bone[:, :, :, v1 - 1, :] = seq_1[:, :, :, v1 - 1, :] - seq_1[:, :, :, v2 - 1, :]
                    
                if stream == 'joint':
                    return self.knn_model.online_projector_joint(self.knn_model.online_backbone_joint(seq_1))
                elif stream == 'motion':
                    return self.knn_model.online_projector_motion(self.knn_model.online_backbone_motion(seq_1_motion))
                elif stream == 'bone':
                    return self.knn_model.online_projector_bone(self.knn_model.online_backbone_bone(seq_1_bone))
                elif stream == 'all':
                    return (self.knn_model.online_projector_joint(self.knn_model.online_backbone_joint(seq_1)) +
                            self.knn_model.online_projector_motion(self.knn_model.online_backbone_motion(seq_1_motion)) +
                            self.knn_model.online_projector_bone(self.knn_model.online_backbone_bone(seq_1_bone))) / 3.
                else:
                    raise ValueError

        self.net = KNN_Model(self.model)

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            if epoch <= self.arg.cross_epoch:
                loss, loss_motion, loss_bone = self.model(data1, data2, stream=self.arg.stream, cross=False)

                # backward
                self.optimizer.zero_grad()
                if loss != 0:
                    loss.backward()
                if loss_motion != 0:
                    loss_motion.backward()
                if loss_bone != 0:
                    loss_bone.backward()
                self.optimizer.step()
                self.model.update_moving_average()

                # statistics
                if loss != 0:
                    self.iter_info['loss'] = loss.data.item()
                else:
                    self.iter_info['loss'] = 0
                if loss_motion != 0:
                    self.iter_info['loss_motion'] = loss_motion.data.item()
                else:
                    self.iter_info['loss_motion'] = 0
                if loss_bone != 0:
                    self.iter_info['loss_bone'] = loss_bone.data.item()
                else:
                    self.iter_info['loss_bone'] = 0
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])

                self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                self.show_iter_info()
                self.meta_info['iter'] += 1
                self.train_log_writer(epoch)
            else:
                loss = self.model(data1, data2, stream=self.arg.stream, cross=True)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.model.update_moving_average()

                # statistics
                self.iter_info['loss'] = loss.data.item()
                loss_value.append(self.iter_info['loss'])

                self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                self.show_iter_info()
                self.meta_info['iter'] += 1
                self.train_log_writer(epoch)

        if self.arg.knn_monitor and epoch % self.arg.knn_interval == 0:
            fea_tsne, gt_tsne, best_k, best_accuracy = knn_monitor(self.net, self.data_loader['mem_train'], self.data_loader['mem_test'], self.arg.model_args['num_class'], hide_progress=False)
            self.epoch_info['best_k'] = best_k
            self.epoch_info['best_knn_acc'] = best_accuracy
            self.train_writer.add_scalar('knn_acc', self.epoch_info['best_knn_acc'], epoch)
            self.epoch_info['train_mean_loss'] = np.mean(loss_value)
            self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
            self.show_epoch_info()
            del self.epoch_info['best_k']
            del self.epoch_info['best_knn_acc']
            save_name = self.arg.work_dir + '/' + str(epoch) + '_tsne_data.pkl'
            with open(save_name, 'wb') as f:
                pickle.dump((fea_tsne, gt_tsne), f)
        else:
            self.epoch_info['train_mean_loss']= np.mean(loss_value)
            self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
            self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--cross_epoch', type=int, default=1000, help='use cross')
        
        parser.add_argument('--knn_monitor', type=str2bool, default=True, help='use knn_monitor or not')
        parser.add_argument('--knn_interval', type=int, default=1, help='the knn_interval')

        return parser
