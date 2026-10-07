"""Trains a plain (non-adaptive) backbone from arch/models_baseline.py.

This is the first stage of the pipeline: a backbone is trained here with no
compression/gating at all, and its checkpoint is later used both as the
reference accuracy and as the required warm-start source for
adaptive_training.py.
"""

from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchsummary import summary

import os
import shutil
import argparse
import time
import logging
import math
from arch import models_baseline as models
from configs.config_utils import load_config, BASE_DEFAULTS, build_optimizer, build_criterion, build_metric, set_seed
from data import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    """Reads --config, merges it onto BASE_DEFAULTS, and validates
    `arch`/`cmd`. Returns the resulting argparse.Namespace."""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training')
    parser.add_argument('--config', type=str, required=True,
                        help='path to a JSON config file with all run parameters')
    cli_args = parser.parse_args()

    args = load_config(cli_args.config, BASE_DEFAULTS)
    if args.arch not in model_names:
        raise ValueError(f"Unknown arch {args.arch!r} in config. Choices: {', '.join(model_names)}")
    if args.cmd not in ('train', 'test'):
        raise ValueError(f"config 'cmd' must be 'train' or 'test', got {args.cmd!r}")
    return args


def main():
    """Script entry point: parses the config, sets up logging, and
    dispatches to run_training or test_model based on `args.cmd`."""
    args = parse_args()
    set_seed(args.seed)
    save_path = args.save_path = os.path.join(args.save_folder, args.save_name or args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("======================================== Input Parameters: =============================================")
    for arg, value in vars(args).items():
        print(f"{arg:>15}: {value}")
    print("========================================================================================================\n\n")

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def input_shape_for(dataset):
    """Returns the (channels, height, width) of one sample for `dataset`
    (matched by name substring, as in data.py). Raises ValueError for an
    unknown dataset."""
    if 'mnist' in dataset:
        return (1, 28, 28)
    if any(name in dataset for name in ('cifar', 'cif10', 'cif100', 'svhn')):
        return (3, 32, 32)
    raise ValueError(f"No input shape known for dataset {dataset!r}")


def run_training(args):
    """Entry point for `cmd: "train"`: builds the plain backbone, optionally
    resumes from `args.resume`, and trains it for `args.epochs`, validating
    and checkpointing after every epoch.

    Input: the parsed config (see configs/config_utils.py's BASE_DEFAULTS
    for every field used).

    Output: none - trains in place and writes checkpoints to
    `args.save_path`."""
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = model.cuda()

    best_prec1 = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            new_state_dict = {
                k: v for k, v in checkpoint['state_dict'].items()
                if not k.startswith('classifier.')
            }
            model.load_state_dict(new_state_dict, strict=False)
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, 0
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    summary(model, input_size=input_shape_for(args.dataset))

    criterion = build_criterion(args.criterion, **args.criterion_kwargs).cuda()
    metric_fn = build_metric(args.metric)
    metric_label = args.metric

    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    print(f"------------------------------ Optimizer is {args.optimizer} ----------------------------")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    iters_per_epoch = len(train_loader)
    total_iters = args.epochs * iters_per_epoch
    warmup_iters = 5 * iters_per_epoch
    end = time.time()
    for i in range(args.start_epoch, args.epochs):
        model.train()
        for batch_idx, (input, target) in enumerate(train_loader):
            current_iter = i * len(train_loader) + batch_idx
            current_lr = adjust_learning_rate_v2(
                optimizer,
                current_iter,
                warmup_iters,
                total_iters,
                args.lr,
                i,
                iters_per_epoch
            )
            # measuring data loading time
            data_time.update(time.time() - end)

            target = target.squeeze().long().cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure metric and record loss
            prec1 = metric_fn(output.data, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            input = input.cuda()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if batch_idx % args.print_freq == 0:
                logging.info("Epoch: [{0}/{1}]\t"
                             "Batch: {2}\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "{metric_label} {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                i,
                                args.epochs,
                                batch_idx,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                metric_label=metric_label,
                                top1=top1)
                )

        # evaluate and checkpoint after every epoch
        if True:
            with torch.no_grad():
                prec1 = validate(args, test_loader, model, criterion, metric_fn)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path,
                                        'checkpoint_{:05d}.pth.tar'.format(
                                            i))
            save_checkpoint({
                'epoch': i,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
                is_best, filename=checkpoint_path)
            shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                        'checkpoint_latest'
                                                        '.pth.tar'))


def validate(args, test_loader, model, criterion, metric_fn):
    """Runs `model` in eval mode over `test_loader`, logging loss/metric.

    Input: the model and test loader, and the criterion/metric to
    evaluate with.

    Output: the average metric over the test set."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    metric_label = args.metric

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure metric and record loss
        prec1 = metric_fn(output.data, target)
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                '{metric_label}: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, metric_label=metric_label, top1=top1
                )
            )

    logging.info(' * {metric_label} {top1.avg:.3f}'.format(metric_label=metric_label, top1=top1))
    return top1.avg


def test_model(args):
    """Entry point for `cmd: "test"`: loads a checkpoint from `args.resume`
    and evaluates it.

    Input: the parsed config.

    Output: none - prints the evaluation results."""
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = model.cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = build_criterion(args.criterion, **args.criterion_kwargs).cuda()
    metric_fn = build_metric(args.metric)

    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, metric_fn)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves `state` to `filename`, and also to `model_best.pth.tar` in
    the same directory if `is_best`. No return value."""
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate_v2(optimizer, current_iter, warmup_iters, total_iters, initial_lr, epoch, iters_per_epoch):
    """Sets `optimizer`'s LR for `current_iter`: linear warmup up to
    `warmup_iters`, then cosine decay to 0 by `total_iters`. Returns the
    LR it set."""
    if current_iter < warmup_iters:
        # Linear Warmup
        lr = initial_lr * current_iter / warmup_iters
    else:
        # Cosine Decay
        c_iter = current_iter - warmup_iters
        t_iter = total_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * c_iter / t_iter))
        lr = initial_lr * cosine_decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if current_iter % iters_per_epoch == 0:
        logging.info('Epoch [{}] Learning rate: {:.6f}'.format(epoch, lr))

    return lr


if __name__ == '__main__':
    main()
