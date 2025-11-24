from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import (
    SoftTripletLoss_vallia, 
    CrossEntropyLabelSmooth,
    CrossEntropyLabelSmoothCD, 
    SoftTripletLoss, 
    SoftEntropy, 
    TripletLoss,
    TripletLossCD,
    SoftTripletLossCD
)
from .utils.meters import AverageMeter
from torch.cuda import amp

class PreTrainer_multi(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer_multi, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss_vallia(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()
        losses_ce_3 = AverageMeter()
        losses_tr_3 = AverageMeter()
        precisions_3 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out,_,_,s_cls_out_3,s_features_3 = self.model(s_inputs,training=True)
            # target samples: only forward
            self.model(t_inputs,training=True)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out[0], targets)
            loss_ce_3, loss_tr_3, prec1_3 = self._forward(s_features_3, s_cls_out_3[0], targets)
            loss = loss_ce + loss_tr + loss_ce_3 + loss_tr_3

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)
            losses_ce_3.update(loss_ce_3.item())
            losses_tr_3.update(loss_tr_3.item())
            precisions_3.update(prec1_3)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()


            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      'Loss_ce_3 {:.3f} ({:.3f})\t'
                      'Loss_tr_3 {:.3f} ({:.3f})\t'
                      'Prec_3 {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg,
                              losses_ce_3.val, losses_ce_3.avg,
                              losses_tr_3.val, losses_tr_3.avg,
                              precisions_3.val, precisions_3.avg))

    def _parse_data(self, inputs):
        imgs, _, pids,_, _ = inputs#, pids, index
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets


    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss_vallia(margin=margin).cuda()
    
        # self.criterion_ce = torch.nn.CrossEntropyLoss()
        # self.criterion_triple = TripletLoss()
    
    def train(self, epoch, data_loader_source, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        
        scaler = amp.GradScaler()
        
        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()
            
            optimizer.zero_grad()
            source_inputs = data_loader_source.next()
            data_time.update(time.time() - end)

            with amp.autocast(enabled=True):
                s_inputs, targets = self._parse_data(source_inputs)
                
                s_features, s_cls_out = self.model(s_inputs,training=True)
                # backward main #
                loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
                loss = loss_ce + loss_tr
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            
            optimizer.zero_grad()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()


            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids,_, _ = inputs#, pids, index
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets


    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class ClusterBaseTrainer(object):
    def __init__(self, model, num_cluster=500):
        super(ClusterBaseTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster
  
        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_cd = CrossEntropyLabelSmoothCD(num_cluster).cuda()
        #self.criterion_ce = CrossEntropyLabelSmoothW(num_cluster, weight=self.cluster_weights).cuda()
        
        
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_cd = SoftTripletLossCD(margin=0.0).cuda()

    
    def train(self, epoch, data_loader_target, optimizer, print_freq=1, train_iters=200, cluster_weight_dict=None):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_total = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        scaler = amp.GradScaler()

        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)
            
        
            # forward
            f_out_t, p_out_t = self.model(inputs,training=True)
            #f_out_t, p_out_t = self.model(inputs)
            p_out_t = p_out_t[:,:self.num_cluster]
            
            optimizer.zero_grad()
            
            with amp.autocast():  # Forward pass in mixed precision
                #if epoch == 0:
                loss_ce = self.criterion_ce(p_out_t, targets)
                loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
                #else:
                #    loss_ce = self.criterion_ce_cd(p_out_t, targets, cluster_weight_dict)
                #    loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
                loss = loss_ce + loss_tri

            #Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()

            prec, = accuracy(p_out_t.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_total.update(loss.item())
            precisions.update(prec[0])




            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_total {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_total.val, losses_total.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

 
    def _parse_data(self, inputs):
        imgs, _, pids,_, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets
    