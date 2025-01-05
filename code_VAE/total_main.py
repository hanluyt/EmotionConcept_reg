#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2023/12/20 22:16
# @Author: hanluyt

import torch
import pandas as pd
import os
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
import dist_train
from vanilla_vae import VectorVAE
import torch.backends.cudnn as cudnn
import random
import time
import datetime
from dataset import IMAGEN_contrast
from data_util import configure_optimizers, training_one_epoch, save_model
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='vae.yaml')
    parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    args = get_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dist_train.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = config['exp_params']['manual_seed']
    set_seed(seed)

    cudnn.benchmark = True

    train_files = os.listdir(config['dataset_params']['directory'])
    dataset_train = IMAGEN_contrast(**config['dataset_params'], cv_name=train_files)

    print('TrainSet:', len(dataset_train))


    if args.distributed:
        num_tasks = dist_train.get_world_size()  # 4
        global_rank = dist_train.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks,
                                                            rank=global_rank, shuffle=True)
        print("Length of Sampler_train = %s" % len(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        print("Length of Sampler_train = %s" % len(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    sampler=sampler_train,
                                                    **config['dataloader_params'])


    # Model
    print('-----------------------------------------')
    print(f"==> Creating model, Train on IMAGEN_VAE")
    model = VectorVAE(**config['model_params'])
    model.to(device)
    model_without_ddp = model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of params in the VAE model:", n_params / 1e6, "M")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # training params
    print(config['exp_params'])

    # Configure
    optimizer_integrate = configure_optimizers(model_without_ddp, config['exp_params'])
    if len(optimizer_integrate) == 2:
        optimizer, scheduler_cosine = optimizer_integrate
    else:
        optimizer = optimizer_integrate
        scheduler_cosine = None

    total_epoch = config['exp_params']['global_epoch']
    print(f"Start training for {total_epoch} epochs")
    total_start_time = time.time()

    model.zero_grad()
    best_loss = 1.
    total_train_loss = []

    for epoch in range(1, total_epoch + 1):
        epoch_start_time = time.time()

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_loss, _, _ = training_one_epoch(args, model, device, epoch, data_loader_train,
                                              optimizer, lr_scheduler=scheduler_cosine, **config['exp_params'])
        total_train_loss.append(train_loss)
        model.train()

        if train_loss < best_loss:
            save_model(model, cv_step='_occ', **config['logging_params'])
            best_loss = train_loss
        total_time = time.time() - epoch_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Best loss: %f, time for this epoch: %s" % (best_loss, total_time_str))

    # save_model(model, save_dir="experiment/VAE/result/", name='lastmodel.pth')
    print("###################################")
    print("End training!!!")
    # print("Best loss: \t%f" % best_loss)

    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


