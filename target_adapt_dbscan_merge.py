from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import os
import numpy as np
import sys
import collections
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from timm.data.random_erasing import RandomErasing

import loralib as lora

from SAGE_reID import datasets
from SAGE_reID import models
from SAGE_reID.trainers import ClusterBaseTrainer
from SAGE_reID.evaluators import Evaluator, extract_features
from SAGE_reID.utils.data import IterLoader
from SAGE_reID.utils.data import transforms as T
from SAGE_reID.utils.data.sampler import RandomMultipleGallerySampler
from SAGE_reID.utils.data.preprocessor import Preprocessor
from SAGE_reID.utils.logging import Logger
from SAGE_reID.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from SAGE_reID.utils.rerank import compute_jaccard_dist
from SAGE_reID.utils.model_utils import (
    print_params, state_dict_cleaner, 
    compute_cluster_entropy_and_weights, 
    mark_only_gate_vector_as_trainable
)
from SAGE_reID.utils.misc import (
    get_rank, 
    init_distributed_mode, 
    get_sha
)



start_epoch = best_mAP = 0

def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):
    
    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
             normalizer,
         ])

    train_set = sorted(dataset.train) if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=False),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):

    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader



def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)



def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    # 1 epochs 10% of steps 
    iters = ((iters * 10) // 100) + 1 # 10% of total steps per epoch
    #dataset_source = get_data(args.dataset_source, args.data_dir)
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    tar_cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)
    #sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers, testset=dataset_source.train)

    # Create model
    classes = len(dataset_target.train)
    model = models.create(args.arch, num_classes=[classes], lintype=args.lintype)
    mark_only_gate_vector_as_trainable(model)
    print_params(model)
    model.cuda()
    model = nn.DataParallel(model)

    # LoRA loading  
    print("lora loading")

    sources = [args.dataset_source1, args.dataset_source2, args.dataset_source3]

    
    checkpoints_list = []
    for source in sources:
        if source:  # Check if the source is provided
            checkpoints_list.append(os.path.join('logs/adapt', f'{source}TO{args.dataset_target}', 'vit_base_patch16_224-baseline'))
    print(checkpoints_list)

    filtered_state_dict = defaultdict(list)
    for exp_idx, ch in enumerate(checkpoints_list):
        checkpoint = load_checkpoint(osp.join(ch, 'model_best_lora.pth.tar'))
        for layer_idx, (k, v) in enumerate(checkpoint['state_dict'].items()):
            if 'blocks' in k and ('mlp' in k or 'attn' in k):
                if 'lora_A' in k or 'lora_B' in k:
                    filtered_state_dict[k].append(v)
    
    for k in filtered_state_dict.keys():
        filtered_state_dict[k] = torch.stack(filtered_state_dict[k], dim=0)
    model.load_state_dict(filtered_state_dict, strict=False)

    print("source loading")
    base_list = []
    for source in sources:
        if source:  # Check if the source is provided
            base_list.append(os.path.join('logs/pretrain', f'{source}', 'vitbase/model_best.pth.tar'))
    state_combined = dict()
    state_dicts = [load_checkpoint(base)['state_dict'] for base in base_list]
    for k in state_dicts[0].keys():
        if not ("classifier" in k or "head" in k):
            state_combined[k] = sum(state_dict[k] for state_dict in state_dicts) / len(state_dicts)
    copy_state_dict(state_combined, model)
    
    print("bias loading")
    checkpoint_comb = dict()
    checkpoints = [load_checkpoint(osp.join(ch, 'model_best_lora.pth.tar'))['state_dict'] for ch in checkpoints_list]
    for k in checkpoints[0].keys():
        if not ("classifier" in k or "head" in k):
            checkpoint_comb[k] = sum(checkpoint[k] for checkpoint in checkpoints) / len(checkpoints)
    # bias all loading 
    filtered_bias = {k: v for k, v in checkpoint_comb.items() if 'bias' in k}     
    model.load_state_dict(filtered_bias, strict=False)

    
    # Evaluator
    evaluator = Evaluator(model)
    
    for epoch in range(args.epochs):
        
        dict_f, _ = extract_features(model, tar_cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f.values()))

  
        rerank_dist = compute_jaccard_dist(cf, use_gpu=args.rr_gpu).numpy()

        if (epoch==0):
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            rho = 1.6e-3
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps for cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

  
            
        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        args.num_clusters = num_ids
        print('\n Clustered into {} classes \n'.format(args.num_clusters))

        # generate new dataset and calculate cluster centers
        new_dataset = []
        cluster_centers = collections.defaultdict(list)
        for i, ((fname, _, cid), label) in enumerate(zip(dataset_target.train, labels)):
            if label==-1: continue
            new_dataset.append((fname,label,cid))
            cluster_centers[label].append(cf[i])
  
        
        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)
        model.module.classifier.weight.data[:args.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())

        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters, trainset=new_dataset)

                
        # addition of cacl
        cluster_cams = [cid for (_, _, cid), label in zip(sorted(dataset_target.train), labels) if label != -1]
        cluster_labels = [label for label in labels if label != -1]
        cluster_weight_dict = compute_cluster_entropy_and_weights(cluster_labels, cluster_cams)

        # Optimizer
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
         
        optimizer = torch.optim.SGD(params)

        # Trainer
        trainer = ClusterBaseTrainer(model, num_cluster=args.num_clusters)

        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_target, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader_target),cluster_weight_dict=cluster_weight_dict)


        def save_model(model, is_best, best_mAP):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP) 
            best_mAP = max(mAP, best_mAP)
            save_model(model, is_best, best_mAP)

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))


    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster Baseline Training")
    # data
    parser.add_argument('-ds1', '--dataset-source1', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-ds2', '--dataset-source2', type=str, default='dukemtmc',
                    choices=datasets.names())
    parser.add_argument('-ds3', '--dataset-source3', type=str, default='dukemtmc',
                    choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-lin', '--lintype', type=str, default='gate',
                    choices=["gate","full"])
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # loras 
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--lora_alpha', type=int, default=2)
    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--lambda-value', type=float, default=0)
    parser.add_argument('--rr-gpu', action='store_true', 
                        help="use GPU for accelerating clustering")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
