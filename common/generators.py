from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from functools import reduce


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, subj):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self.subject = subj
        self._actions = reduce(lambda x, y: x + y, actions)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action

    def __len__(self):
        return len(self._actions)


class CamPoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, subj):
        assert poses_3d is not None

        self.num_cameras = 4
        self.num_sequences = len(poses_3d) // self.num_cameras + 1

        bounds = [0] + np.cumsum([len(x) for x in poses_3d]).tolist()
        inds = [list(range(bounds[i], bounds[i+1])) for i in range(len(bounds[:-1]))]
        lens = []
        for i, pose in enumerate(inds[::4]):
            idx = list(zip(*inds[i*4:(i*4)+4]))
            lens.append(idx)


        self.indices = lens
        self.flat_indices = [x for y in lens for x in y]

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)
        self.subject = subj
        self._subject = reduce(lambda x, y: x + y, subj)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]
        out_subj = self._subject[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action, out_subj

    def __len__(self):
        return len(self._actions)


class BatchSampler(Sampler):
    """
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    """
    def __init__(self, dataset):
        super(BatchSampler, self).__init__(dataset)

        self.dataset = dataset
        self.flat_indices = dataset.flat_indices

    def __iter__(self):
        for inds in self.flat_indices:
            yield inds

    def __len__(self):
        return len(self.flat_indices)