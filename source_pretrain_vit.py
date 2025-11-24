from __future__ import print_function, absolute_import
import sys
import argparse
import random

import numpy as np
import os.path as osp

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from timm.data.random_erasing import RandomErasing


from SAGE_reID import datasets
from SAGE_reID import models
from SAGE_reID.trainers import PreTrainer, PreTrainer_multi
from SAGE_reID.evaluators import Evaluator
from SAGE_reID.utils.data import IterLoader
from SAGE_reID.utils.data import transforms as T
from SAGE_reID.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGalleryDistributedSampler
from SAGE_reID.utils.data.preprocessor import Preprocessor
from SAGE_reID.utils.logging import Logger
from SAGE_reID.utils.lr_scheduler import CosineLRScheduler
from SAGE_reID.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from SAGE_reID.utils.model_utils import print_params
from SAGE_reID.utils.misc import (
    get_rank, 
    init_distributed_mode, 
    get_sha
)


start_epoch = best_mAP = 0


def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=200):
    root = osp.join(data_dir)

    dataset = datasets.create(name, root)
    
    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])

    train_set = dataset.train
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             # T.AugMix(),
             T.ToTensor(),
             RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
             normalizer
         ])


    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, 
                                    transform=train_transformer),
                        batch_size=batch_size, num_workers=workers, sampler=sampler,
                        shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader

def main():
    args = parser.parse_args()

    main_worker(args)


def main_worker(args):
    init_distributed_mode(args)
    global start_epoch, best_mAP
    
    if args.seed is not None:
        seed = args.seed  + get_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source, num_classes, train_loader_source, test_loader_source = \
        get_data(args.dataset_source, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters)
    
    dataset_target, _, train_loader_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, 0, iters)
    
    device = torch.device(args.device)
    
    model = models.create(args.arch,num_classes=[num_classes],lintype=args.lintype)
    model.cuda()

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))
    
    # Evaluator
    evaluator = Evaluator(model)
    # args.evaluate=True
    if args.evaluate:
        print("Test on source domain:")
        evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True, rerank=args.rerank)
        print("Test on target domain:")
        evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)
        return

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    print_params(model)

    # Evaluator
    evaluator = Evaluator(model)

    # create optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.SGD(params)

    # create scheduler
    num_epochs = args.epochs
    lr_min = 0.002 * args.lr
    warmup_lr_init = 0.01 * args.lr
    warmup_t = args.warmup_step
    noise_range = None
    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    trainer = PreTrainer(model, num_classes, margin=args.margin) 

      # Start training
    for epoch in range(start_epoch, args.epochs):

        train_loader_source.new_epoch()
        
        trainer.train(epoch, train_loader_source, optimizer,
                    train_iters=len(train_loader_source), print_freq=args.print_freq)
        lr_scheduler.step(epoch=epoch)
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):

            _, mAP = evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  source mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    print("Test on target domain:")
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")

    # datasets
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                    help="each minibatch consist of "
                            "(batch_size // num_instances) identities, and "
                            "each identity has num_instances instances, "
                            "default: 0 (NOT USE)")
        
    # model
    parser.add_argument('-lin', '--lintype', type=str, default='gate',
                choices=["gate","lora", "full"])
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    choices=models.names())
    parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
    
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    
    
    # training configs
    parser.add_argument('--resume', type=str, default="", metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs'))
    main()
