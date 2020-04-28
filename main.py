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
from resnet import ResNet
from mobilenet import MobileNetV1
from utility import add_arguments, print_arguments, count_parameters_in_MB

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('log_freq',          int,   10,              "Log frequency.")
add_arg('data',              str,   'dataset',       "The dir of dataset.")
add_arg('batch_size',        int,   256,            "Minibatch size.")
add_arg('init_lr',           float, 0.1,             "The start learning rate.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   200,             "Epoch number.")
add_arg('class_num',         int,   100,             "Class number of dataset.")
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
            is_shuffle=True)

    valid_reader = reader.train_valid(
            batch_size=args.batch_size,
            is_train = False,
            is_shuffle=False)

    with fluid.dygraph.guard(place):
        train_loader = fluid.io.DataLoader.from_generator(
            capacity=2048,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=2048,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=True)
        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(valid_reader, places=place)
        dataloaders = [train_loader, valid_loader]

        models = [ResNet()]#, ResNet()]
        print(count_parameters_in_MB(models[0].parameters()))

        step = int(args.trainset_num / args.batch_size)
        epochs = [60, 120, 180]
        bd = [step * e for e in epochs]
        lr = [args.init_lr * (0.1**i) for i in range(len(bd) + 1)]

        lr_a = fluid.dygraph.PiecewiseDecay(bd, lr, 0)
        #lr_b = fluid.dygraph.PiecewiseDecay(bd, lr, 0)
        opt_a = fluid.optimizer.MomentumOptimizer(lr_a, 0.9, parameter_list=models[0].parameters(), use_nesterov=True, regularization=fluid.regularizer.L2DecayRegularizer(5e-4))
        #opt_b = fluid.optimizer.MomentumOptimizer(lr_b, 0.9, parameter_list=models[1].parameters(), use_nesterov=True, regularization=fluid.regularizer.L2DecayRegularizer(5e-4))
        optimizers = [opt_a]#, opt_b]
        trainer = Trainer(models, optimizers, dataloaders, args.epochs, args.log_freq)
        trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    main(args)
