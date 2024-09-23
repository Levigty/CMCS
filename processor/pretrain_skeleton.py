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


class Skeleton_Processor(PT_Processor):
    """
        Processor for SkeletonBL Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if self.arg.stream == 'joint':
                pass
            elif self.arg.stream == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
            elif self.arg.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]

                data1 = bone1
                data2 = bone2
            else:
                raise ValueError

            # forward
            loss = self.model(data1, data2)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.update_moving_average()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        if self.arg.knn_monitor and epoch % self.arg.knn_interval == 0:
            fea_tsne, gt_tsne, best_k, best_accuracy = knn_monitor(self.model.online_backbone, self.data_loader['mem_train'],
                                   self.data_loader['mem_test'], self.arg.model_args['num_class'], hide_progress=False)
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
            self.epoch_info['train_mean_loss'] = np.mean(loss_value)
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
        
        parser.add_argument('--knn_monitor', type=str2bool, default=True, help='use knn_monitor or not')
        parser.add_argument('--knn_interval', type=int, default=1, help='the knn_interval')

        return parser
