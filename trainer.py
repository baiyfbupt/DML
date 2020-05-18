# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid.dygraph.base import to_variable
from paddleslim.common import AvgrageMeter, get_logger
logger = get_logger(__name__, level=logging.INFO)

class Trainer(object):
    def __init__(self, models, parallel_models, optimizers, dataloaders, epochs, log_freq):
        self.models = models
        self.model_num = len(self.models)
        self.use_data_parallel = False
        self.parallel_models = parallel_models
        if self.parallel_models is not None:
            self.use_data_parallel = True
        self.optimizers = optimizers
        assert len(self.optimizers) == self.model_num
        self.train_loader = dataloaders[0]
        self.valid_loader = dataloaders[1]
        self.epochs = epochs
        self.log_freq = log_freq
        self.start_epoch = 0

        self.best_valid_accs = [0.] * self.model_num

    def train(self):

        for epoch in range(self.start_epoch, self.epochs):

            # train 1 epoch on trainset
            train_losses, train_accs = self.train_one_epoch(epoch)
            # valid 1 epoch on validset
            valid_losses, valid_accs = self.valid_one_epoch(epoch)

            for i in range(self.model_num):
                is_best = valid_accs[i].avg> self.best_valid_accs[i]
                msg1 = "model_{:d}: lr: {:.6f}, train loss: {:.3f}, train acc: {:.3f}% "
                msg2 = ", val loss: {:.3f}, val acc: {:.3f}%"
                if is_best:
                    msg2 += " [*]"
                msg = msg1 + msg2
                logger.info(msg.format(i+1, self.optimizers[i].current_step_lr(), train_losses[i].avg[0], train_accs[i].avg[0], valid_losses[i].avg[0], valid_accs[i].avg[0]))
                self.best_valid_accs[i] = max(valid_accs[i].avg[0], self.best_valid_accs[i])


    def train_one_epoch(self, epoch):
        losses = []
        accs = []

        for i in range(self.model_num):
            if self.use_data_parallel:
                self.parallel_models[i].train()
            else:
                self.models[i].train()
            losses.append(AvgrageMeter())
            accs.append(AvgrageMeter())

        for step_indx, (images, labels) in enumerate(self.train_loader):
            images, labels = to_variable(images), to_variable(labels)
            batch_size = images.shape[0]

            logits=[]
            if self.use_data_parallel:
                for model in self.parallel_models:
                    logits.append(model(images))
            else:
                for model in self.models:
                    logits.append(model(images))


            log_msg = 'Train Epoch {}, Step {}'.format(epoch, step_indx)
            for i in range(self.model_num):
                gt_loss = self.models[i].loss(logits[i], labels)
                kl_loss = 0
                for j in range(self.model_num):
                    if i!=j:
                        x = F.log_softmax(logits[i], axis=1)
                        y = fluid.layers.softmax(logits[j], axis=1)
                        kl_loss += fluid.layers.kldiv_loss(x, y, reduction='batchmean')

                loss = gt_loss
                if (self.model_num > 1):
                    loss += kl_loss / (self.model_num - 1)

                prec = fluid.layers.accuracy(input=logits[i], label=labels, k=1)
                losses[i].update(loss.numpy(), batch_size)
                accs[i].update(prec.numpy()*100, batch_size)

                if self.use_data_parallel:
                    loss = self.parallel_models[i].scale_loss(loss)
                    loss.backward()
                    self.parallel_models[i].apply_collective_grads()
                else:
                    loss.backward()
                self.optimizers[i].minimize(loss)
                if self.use_data_parallel:
                    self.parallel_models[i].clear_gradients()
                else:
                    self.models[i].clear_gradients()

                log_msg += ', model{}_loss: {:.3f}'.format(i+1, losses[i].avg[0])

            if step_indx % self.log_freq==0:
                logger.info(log_msg)
        return losses, accs



    def valid_one_epoch(self, epoch):
        losses = []
        accs = []
        for i in range(self.model_num):
            if self.use_data_parallel:
                self.parallel_models[i].eval()
            else:
                self.models[i].eval()
            losses.append(AvgrageMeter())
            accs.append(AvgrageMeter())

        for _, (images, labels) in enumerate(self.valid_loader):
            images, labels = to_variable(images), to_variable(labels)
            batch_size = images.shape[0]

            logits=[]
            if self.use_data_parallel:
                for model in self.parallel_models:
                    logits.append(model(images))
            else:
                for model in self.models:
                    logits.append(model(images))
            for i in range(self.model_num):
                gt_loss = self.models[i].loss(logits[i], labels)
                kl_loss = 0
                for j in range(self.model_num):
                    if i!=j:
                        x = F.log_softmax(logits[i], axis=1)
                        y = fluid.layers.softmax(logits[j], axis=1)
                        kl_loss += fluid.layers.kldiv_loss(x, y, reduction='batchmean')

                loss = gt_loss
                if (self.model_num > 1):
                    loss += kl_loss / (self.model_num - 1)

                prec = fluid.layers.accuracy(input=logits[i], label=labels, k=1)
                losses[i].update(loss.numpy(), batch_size)
                accs[i].update(prec.numpy()*100, batch_size)
        return losses, accs
