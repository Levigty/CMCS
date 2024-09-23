#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np
import pickle
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor
from .knn_monitor import knn_monitor


class CrosSCLR_3views_Processor(PT_Processor):
    """
        Processor for 3view-CrosSCLR Pretraining.
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

            def forward(self, seq_1, stream=self.arg.view):
                seq_1_motion = torch.zeros_like(seq_1)
                seq_1_motion[:, :, :-1, :, :] = seq_1[:, :, 1:, :, :] - seq_1[:, :, :-1, :, :]

                seq_1_bone = torch.zeros_like(seq_1)
                for v1, v2 in self.Bone:
                    seq_1_bone[:, :, :, v1 - 1, :] = seq_1[:, :, :, v1 - 1, :] - seq_1[:, :, :, v2 - 1, :]

                if stream == 'joint':
                    return self.knn_model.encoder_q(seq_1)
                elif stream == 'motion':
                    return self.knn_model.encoder_q_motion(seq_1_motion)
                elif stream == 'bone':
                    return self.knn_model.encoder_q_bone(seq_1_bone)
                elif stream == 'all':
                    return (self.knn_model.encoder_q(seq_1) + self.knn_model.encoder_q_motion(seq_1_motion) + self.knn_model.encoder_q_bone(seq_1_bone)) / 3.
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
                output, output_motion, output_bone, target = self.model(data1, data2)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
                loss_motion = self.loss(output_motion, target)
                loss_bone = self.loss(output_bone, target)

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss = loss + loss_motion + loss_bone
            else:
                output_jm, output_jb, output_mj, output_mb, output_bj, output_bm, mask_jm, mask_jb, mask_mj, mask_mb, mask_bj, mask_bm = self.model(data1, data2, cross=True, topk=self.arg.topk, context=self.arg.context)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output_jm.size(0))
                else:
                    self.model.update_ptr(output_jm.size(0))
                loss_jm = - (F.log_softmax(output_jm, dim=1) * mask_jm).sum(1) / mask_jm.sum(1)
                loss_jb = - (F.log_softmax(output_jb, dim=1) * mask_jb).sum(1) / mask_jb.sum(1)
                loss_mj = - (F.log_softmax(output_mj, dim=1) * mask_mj).sum(1) / mask_mj.sum(1)
                loss_mb = - (F.log_softmax(output_mb, dim=1) * mask_mb).sum(1) / mask_mb.sum(1)
                loss_bj = - (F.log_softmax(output_bj, dim=1) * mask_bj).sum(1) / mask_bj.sum(1)
                loss_bm = - (F.log_softmax(output_bm, dim=1) * mask_bm).sum(1) / mask_bm.sum(1)
                loss = (loss_jm + loss_jb) / 2.
                loss_motion = (loss_mj + loss_mb) / 2.
                loss_bone = (loss_bj + loss_bm) / 2.
                loss = loss.mean()
                loss_motion = loss_motion.mean()
                loss_bone = loss_bone.mean()

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss = loss + loss_motion + loss_bone

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)
            self.train_writer.add_scalar('batch_loss_motion', self.iter_info['loss_motion'], self.global_step)
            self.train_writer.add_scalar('batch_loss_bone', self.iter_info['loss_bone'], self.global_step)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_motion']= np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone']= np.mean(loss_bone_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)
        self.show_epoch_info()
        if self.arg.knn_monitor and epoch % self.arg.knn_interval == 0:
            fea_tsne, gt_tsne, best_k, best_accuracy = knn_monitor(self.net, self.data_loader['mem_train'],
                                                                   self.data_loader['mem_test'],
                                                                   self.arg.model_args['num_class'],
                                                                   hide_progress=False)
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

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-view training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-view training')

        parser.add_argument('--knn_monitor', type=str2bool, default=True, help='use knn_monitor or not')
        parser.add_argument('--knn_interval', type=int, default=1, help='the knn_interval')
        # endregion yapf: enable

        return parser
