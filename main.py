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

import argparse
import functools
import paddle.fluid as fluid

import reader
from trainer import Trainer
from model import ResNet
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('log_freq',          int,   10,              "Log frequency.")
add_arg('data',              str,   'dataset',       "The dir of dataset.")
add_arg('batch_size',        int,   64,              "Minibatch size.")
add_arg('learning_rate',     float, 0.025,           "The start learning rate.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   50,              "Epoch number.")
add_arg('class_num',         int,   10,              "Class number of dataset.")
add_arg('trainset_num',      int,   50000,           "Images number of trainset.")
add_arg('model_save_dir',    str,   'saved_models',  "The path to save model.")
add_arg('use_data_parallel', bool,  False,           "The flag indicating whether to use data parallel mode to train the model.")
# yapf: enable

def main(args):
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    train_reader = reader.train_valid(
            batch_size=args.batch_size,
            is_train = True,
            is_shuffle=True,
            args=args)

    valid_reader = reader.train_valid(
            batch_size=args.batch_size,
            is_train = False,
            is_shuffle=False,
            args=args)

    with fluid.dygraph.guard(place):
        train_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(valid_reader, places=place)

        models = [ResNet(152), ResNet(50)]
        step_per_epoch = int(args.trainset_num / args.batch_size)
        lr_a = fluid.dygraph.CosineDecay(0.1, step_per_epoch, 100)
        lr_b = fluid.dygraph.CosineDecay(0.1, step_per_epoch, 100)
        opt_a = fluid.optimizer.MomentumOptimizer(lr_a, 0.9, parameter_list=models[0].parameters(), use_nesterov=True, regularization=fluid.regularizer.L2DecayRegularizer(4e-5))
        opt_b = fluid.optimizer.MomentumOptimizer(lr_b, 0.9, parameter_list=models[1].parameters(), use_nesterov=True, regularization=fluid.regularizer.L2DecayRegularizer(4e-5))
        optimizers = [opt_a, opt_b]
        dataloaders = [train_loader, valid_loader]
        trainer = Trainer(models, optimizers, dataloaders, 200, args.log_freq)
        trainer.train()




if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    main(args)
