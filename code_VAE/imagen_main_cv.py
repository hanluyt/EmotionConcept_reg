#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2023/9/3 14:45
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
from vanilla_vae import VectorVAE, ConvVAE
import torch.backends.cudnn as cudnn
import random
import time
import datetime
from dataset import IMAGEN_contrast, split_dataset
from data_util import configure_optimizers, training_one_epoch, validation_one_epoch, save_model
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
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

    # load dataset
    cv_num = 5
    data_split = split_dataset(**config['dataset_params'], random_state=seed, num_part=cv_num)

    json_name = os.path.join(config['logging_params']['save_dir'],config['logging_params']['idname'])
    with open(json_name, 'w') as f:
        json.dump(data_split, f)

    for i in range(cv_num):
        print(f"################### {i + 1} fold ###################")

        valid_files = data_split[i]
        train_files = list(set(sum(data_split, [])) - set(valid_files))

        dataset_train = IMAGEN_contrast(**config['dataset_params'], cv_name=train_files)
        dataset_val = IMAGEN_contrast(**config['dataset_params'], cv_name=valid_files)

        print('TrainSet:', len(dataset_train))
        print('ValSet:', len(dataset_val))

        if args.distributed:
            num_tasks = dist_train.get_world_size()  # 4
            global_rank = dist_train.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks,
                                                                rank=global_rank, shuffle=True)
            print("Length of Sampler_train = %s" % len(sampler_train))
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            print("Length of Sampler_train = %s" % len(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                        sampler=sampler_train,
                                                        **config['dataloader_params'])

        data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                      sampler=sampler_val,
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
        total_train_loss, total_val_loss = [], []

        for epoch in range(1, total_epoch + 1):
            epoch_start_time = time.time()

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_loss, _, _ = training_one_epoch(args, model, device, epoch, data_loader_train,
                                                  optimizer, lr_scheduler=scheduler_cosine, **config['exp_params'])

            total_train_loss.append(train_loss)

            val_loss, _, _ = validation_one_epoch(model, device, epoch, data_loader_val, **config['exp_params'])
            total_val_loss.append(val_loss)

            if best_loss > val_loss:
                save_model(model, cv_step=i+1, **config['logging_params'])
                best_loss = val_loss

            model.train()

            total_time = time.time() - epoch_start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Best loss: %f, time for this epoch: %s" % (best_loss, total_time_str))

        print("###################################")
        print("Best loss: \t%f" % best_loss)
        print(f"End Training {i + 1} fold!")

        total_time = time.time() - total_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x = range(1, total_epoch + 1)
        ax[0].plot(x, total_train_loss)
        ax[1].plot(x, total_val_loss)
        ax[0].set_title('train loss')
        ax[1].set_title('val loss')
        ax[1].set_xlabel("Best loss: %f, Training time: %s" % (best_loss, total_time_str))
        plt.suptitle(config['exp_params'])

        lr = config['exp_params']['LR']
        kld = config['exp_params']['kld_weight']
        plt.savefig(f'/experiment/VAE/model/loss_{lr}_{kld}_{i+1}.png', dpi=300)
        # metric_loss = [[train_loss, val_loss] for train_loss, val_loss in zip(total_train_loss, total_val_loss)]
        # metric_loss = pd.DataFrame(metric_loss, columns=['train_loss', 'val_loss'])
        # metric_loss.to_csv(os.path.join(config['logging_params']['save_dir'], 'metric.csv'), index=False)

    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("##########################################")
    print('Total Training time {}'.format(total_time_str))









