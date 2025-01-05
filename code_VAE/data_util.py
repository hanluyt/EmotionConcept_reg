#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2023/9/4 20:53
# @Author: hanluyt

import torch
from torch import optim
import math
import sys
import os
import torchvision.utils as vutils
import torch.distributed as dist
from dist_train import get_rank
import random

class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """In the first total_iters iterations,
        the learning rate increases linearly from 0 to base_lr"""
        return [base_lr * self.last_epoch / self.total_iters for base_lr in self.base_lrs]


def configure_optimizers(model: torch.nn.Module, params: dict):
    optimizer = optim.Adam(model.parameters(),
                           lr=params['LR'],
                           weight_decay=params['weight_decay'],
                           betas=(0.500, 0.999))

    if params['scheduler'] is not None:
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
        #                                              gamma=params['scheduler_gamma'])
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['cosine_Tmax'])
        # scheduler_warmup =  WarmUpLR(optimizer, total_iters=params['warmup_iter'])
        return optimizer, scheduler_cosine
    else:
        return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def save_model(model, cv_step, **kwargs):
    path = kwargs['modelname'] + str(cv_step) +'.pth'
    checkpoint_path = os.path.join(kwargs['save_dir'], path)
    to_save = {
        'model': model.state_dict()
    }

    if get_rank() == 0:
        torch.save(to_save, checkpoint_path)
    print("Saved model checkpoint to [DIR: %s]" % (kwargs['save_dir']))


def training_one_epoch(args, model, device, step_epoch, data_loader, optimizer,
                       lr_scheduler=None, **kwargs):
    global_epoch = kwargs['global_epoch']
    model.train()

    train_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    KLD_losses = AverageMeter()

    for step, data in enumerate(data_loader):
        if len(data) == 2:
            real_img, labels = data
            real_img = real_img.to(device)
            labels = labels.to(device)
            results = model(real_img, labels=labels)
        else:
            real_img = data
            real_img = real_img.to(device)
            results = model(real_img)

        train_loss = model.loss_function(*results, M_N=kwargs['kld_weight'])

        if not math.isfinite(train_loss['loss'].item()):
            print("Loss is {}, stopping training.".format(train_loss['loss'].item()))
            sys.exit(1)

        train_losses.update(train_loss['loss'].item())
        reconstruction_losses.update(train_loss['Reconstruction_Loss'].item())
        KLD_losses.update(train_loss['KLD'].item())

        # print(train_loss['loss'].item())

        optimizer.zero_grad()
        train_loss['loss'].backward()
        optimizer.step()

        if lr_scheduler is not None:
            # scheduler_cosine, scheduler_warmup = lr_scheduler
            # if step_epoch < kwargs['warmup_iter']:
            #     scheduler_warmup.step()
            # else:
            #     scheduler_cosine.step()

            lr_scheduler.step()

        torch.cuda.synchronize()

        # print(optimizer.param_groups[0]['lr'])

    if args.distributed:
        t_train_loss = torch.tensor([train_losses.count, train_losses.sum], dtype=torch.float64, device="cuda")
        t_recon = torch.tensor([reconstruction_losses.count, reconstruction_losses.sum], dtype=torch.float64, device="cuda")
        t_kld = torch.tensor([KLD_losses.count, KLD_losses.sum], dtype=torch.float64, device="cuda")
        dist.barrier() # synchronizes all processes
        dist.all_reduce(t_train_loss, op=torch.distributed.ReduceOp.SUM, )
        dist.all_reduce(t_recon, op=torch.distributed.ReduceOp.SUM, )
        dist.all_reduce(t_kld, op=torch.distributed.ReduceOp.SUM, )
        t_train_loss = t_train_loss.tolist()
        t_recon = t_recon.tolist()
        t_kld = t_kld.tolist()
        t_train_loss_count, t_train_loss_sum = int(t_train_loss[0]), t_train_loss[1]
        t_recon_count, t_recon_sum = int(t_recon[0]), t_recon[1]
        t_kld_count, t_kld_sum = int(t_kld[0]), t_kld[1]

        loss_avg = t_train_loss_sum / t_train_loss_count
        recon_avg = t_recon_sum / t_recon_count
        KLD_avg = t_kld_sum / t_kld_count
    else:
        loss_avg = train_losses.avg
        recon_avg = reconstruction_losses.avg
        KLD_avg = KLD_losses.avg
    print('Train: {%d / %d epochs} ----- Train loss: %2.5f, Reconstruction_Loss: %2.5f, KLD: %2.5f' % (
    step_epoch, global_epoch, loss_avg, recon_avg, KLD_avg))
    return loss_avg, recon_avg,  KLD_avg


def validation_one_epoch(model, device, step_epoch, data_loader, **kwargs):
    global_epoch = kwargs['global_epoch']
    model.eval()

    val_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    KLD_losses = AverageMeter()

    for step, data in enumerate(data_loader):
        if len(data) == 2:
            real_img, labels = data
            real_img = real_img.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                results = model(real_img, labels=labels)
        else:
            real_img = data
            real_img = real_img.to(device)
            with torch.no_grad():
                results = model(real_img)


        val_loss = model.loss_function(*results, M_N=kwargs['kld_weight'])

        if not math.isfinite(val_loss['loss'].item()):
            print("Loss is {}, stopping training.".format(val_loss['loss'].item()))
            sys.exit(1)

        val_losses.update(val_loss['loss'].item())
        reconstruction_losses.update(val_loss['Reconstruction_Loss'].item())
        KLD_losses.update(val_loss['KLD'].item())

    loss_avg = val_losses.avg
    recon_avg = reconstruction_losses.avg
    KLD_avg = KLD_losses.avg

    print('Validation: {%d / %d epoch} ----- Val loss: %2.5f, Reconstruction_Loss: %2.5f, KLD: %2.5f' % (
    step_epoch, global_epoch, loss_avg, recon_avg, KLD_avg))
    return loss_avg, recon_avg, KLD_avg


def sample_images(model, device, data_loader, idx=0, **kwargs):
    """Get sample reconstruction image"""
    for i, (test_input, test_label) in enumerate(data_loader):
        if i == idx:
            test_input = test_input.to(device)
            test_label = test_label.to(device)
            break

    recons = model.generate(test_input, labels=test_label)
    vutils.save_image(recons.data,
                      os.path.join(kwargs['save_dir'],
                                   "Reconstructions",
                                   f"recons_{idx}.png"),
                      normalize=True,
                      nrow=16)

    vutils.save_image(test_input.data,
                      os.path.join(kwargs['save_dir'],
                                   "Reconstructions",
                                   f"input_{idx}.png"),
                      normalize=True,
                      nrow=16)

    samples = model.sample(128, current_device=device, label=test_label)
    vutils.save_image(samples.cpu().data,
                      os.path.join(kwargs['save_dir'],
                                   "Samples",
                                   f"sample_{idx}.png"),
                      normalize=True,
                      nrow=16)







