from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


import numpy as np
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4, choice_c=0):

        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances
        self.choice_c=choice_c

        for index, items in enumerate(data_source):# items: (_, pid, ..., pid2, cam)
            self.index_pid[index] = items[self.choice_c+1]
            self.pid_cam[items[self.choice_c+1]].append(items[-1])
            self.pid_index[items[self.choice_c+1]].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            i_pid, i_cam = self.data_source[i][self.choice_c+1],self.data_source[i][-1]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)

import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomMultipleGalleryDistributedSampler(Sampler):
    """
    DDP-aware version of RandomMultipleGallerySampler.

    Each rank gets a disjoint subset of PIDs, and for each PID we sample
    `num_instances` images using the same camera/track logic as the original.
    """

    def __init__(self, data_source, num_instances=4, choice_c=0,
                 num_replicas=None, rank=None, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.num_instances = num_instances
        self.choice_c = choice_c
        self.shuffle = shuffle
        self.drop_last = drop_last

        # ---- DDP world info ----
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank

        # ---- same bookkeeping as original sampler ----
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, items in enumerate(data_source):  # items: (_, pid, ..., pid2, cam)
            pid = items[self.choice_c + 1]
            cam = items[-1]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_pids = len(self.pids)

        # how many PIDs per rank
        self.num_pids_per_rank = int(math.ceil(self.num_pids * 1.0 / self.num_replicas))
        self.total_pids = self.num_pids_per_rank * self.num_replicas

        # samples per rank (PIDs * instances)
        self.num_samples = self.num_pids_per_rank * self.num_instances

        # epoch for deterministic shuffling
        self.epoch = 0

    def __len__(self):
        # Dataloader will see this per-rank
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        # ---- 1) choose PID order globally ----
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            pid_indices = torch.randperm(self.num_pids, generator=g).tolist()
        else:
            pid_indices = list(range(self.num_pids))

        # pad to be divisible by num_replicas
        if len(pid_indices) < self.total_pids:
            pid_indices = (pid_indices * math.ceil(self.total_pids / len(pid_indices)))[:self.total_pids]
        else:
            pid_indices = pid_indices[:self.total_pids]

        # ---- 2) shard PIDs across ranks ----
        offset = self.num_pids_per_rank * self.rank
        pid_indices_rank = pid_indices[offset: offset + self.num_pids_per_rank]

        # ---- 3) apply your original per-PID sampling logic ----
        ret = []

        for kid in pid_indices_rank:
            pid = self.pids[kid]

            # choose a random index for this PID
            i = random.choice(self.pid_index[pid])
            i_pid, i_cam = self.data_source[i][self.choice_c + 1], self.data_source[i][-1]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index_list = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)

                for kk in cam_indexes:
                    ret.append(index_list[kk])
            else:
                select_indexes = No_index(index_list, i)
                if not select_indexes:
                    continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index_list[kk])

        # ---- 4) make sure length is exactly num_samples ----
        # your original sampler also has a mismatch risk; DDP is pickier
        if len(ret) < self.num_samples:
            # pad by repeating from existing indices
            if len(ret) == 0:
                # degenerate case: dataset super weird
                # just randomly sample any index from dataset
                extra = np.random.choice(len(self.data_source), size=self.num_samples, replace=True).tolist()
                ret = extra
            else:
                need = self.num_samples - len(ret)
                extra = np.random.choice(ret, size=need, replace=True).tolist()
                ret.extend(extra)
        elif len(ret) > self.num_samples:
            ret = ret[:self.num_samples]

        assert len(ret) == self.num_samples

        return iter(ret)
