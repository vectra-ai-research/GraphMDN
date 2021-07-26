"""
This code is based on (and an extension of) the publicly-available implementation of SemGCN.
https://github.com/garyzhao/SemGCN
"""

from __future__ import print_function, absolute_import, division

import os
import sys
import time
import numpy as np
import os.path as path
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.arguments import parse_args, args_to_file
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.graph_utils import adj_mx_from_skeleton
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator, CamPoseGenerator, BatchSampler
from common import loss
from common import mdn_loss
from models.sem_gcn_mdn import SemGCN_MDN_Graph

def main(args):
    print('==> Using settings {}'.format(args))

    # Load data in a way that is semi-robust to path differences
    print('==> Loading dataset...')
    directory_for_this_file = os.path.dirname(__file__)
    data_folder_name = 'data'
    data_root = os.path.join(directory_for_this_file, data_folder_name)
    dataset_path = path.join(data_root, 'data_3d_' + args.dataset + '.npz')

    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join(data_root, 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = list(map(lambda x: dataset.define_actions(x)[0], action_filter))
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample

    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device("cuda")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")

    # Define Metrics
    metrics = ['best_p1', 'best_p2', 'mean_p1', 'mean_p2', 'max_p1', 'max_p2']

    # Create model
    print("==> Creating model...")

    p_dropout = (None if args.dropout == 0.0 else args.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = SemGCN_MDN_Graph(
        adj,
        args.hid_dim,
        num_gaussians=args.num_gaussians,
        num_layers=args.num_layers,
        p_dropout=p_dropout,
        tanh_out=args.tanh_out,
        pose_level_pi=args.pose_level_pi,
        uniform_sigma=args.uniform_sigma,
        multivariate=args.multivariate,
        nodes_group=dataset.skeleton().joints_group() if args.non_local else None,
    ).to(device)

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr, weight_decay=args.l2_norm)

    # Optionally resume from a checkpoint
    if args.model_file:
        ckpt_path = args.model_file

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
            ckpt_dir_path = path.dirname(ckpt_path)

            if args.resume:
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        print('evaluation file required to run eval')
        sys.exit(1)

    if args.evaluate:
        ckpt_type = (ckpt_path.split('_')[-1]).split('.')[0]
        eval_logger = Logger(os.path.join(ckpt_dir_path, 'eval_{}.txt'.format(ckpt_type)))
        eval_logger.set_names(['action'] + metrics)

        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        total_loss = {}

        for i, action in enumerate(action_filter):
            print(action)

            poses_valid, poses_valid_2d, actions_valid, subj_valid = fetch(subjects_test, dataset, keypoints, [action], stride)
            valid_ds = CamPoseGenerator(poses_valid, poses_valid_2d, actions_valid, subj_valid)
            valid_sampler = BatchSampler(valid_ds)
            valid_loader = DataLoader(valid_ds, batch_sampler=valid_sampler, num_workers=0)

            eval_loss = evaluate_all(valid_loader, model_pos, device)
            total_loss[action] = {k: v.avg for k, v in eval_loss.items()}

            eval_logger.append([action, eval_loss['best_p1'].avg, eval_loss['best_p2'].avg,
                                eval_loss['mean_p1'].avg, eval_loss['mean_p2'].avg,
                                eval_loss['max_p1'].avg, eval_loss['max_p2'].avg])

        metric_loss = {s: np.mean([v[s] for v in total_loss.values()]) for s in metrics}

        eval_logger.append(['Avg'] + [np.mean([v[s] for v in total_loss.values()]) for s in metrics])

        print('Best Kernel Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(metric_loss['best_p1']))
        print('Best Kernel Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(metric_loss['best_p2']))
        print('Mean of Distribution Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(metric_loss['mean_p1']))
        print('Mean of Distribution Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(metric_loss['mean_p2']))
        print('MaxK of Distribution Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(metric_loss['max_p1']))
        print('MaxK of Distribution Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(metric_loss['max_p2']))
        sys.exit(0)


def evaluate_all(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = defaultdict(AverageMeter)

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()
    start = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, subj) in enumerate(data_loader):

        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d)
        #zero hip joint
        mu = outputs_3d[0] 
        mu[:, :1] = 0*mu[:, :1]
        outputs_3d = (mu, outputs_3d[1], outputs_3d[2])
        """
        Below, we get all the predictions made by the GraphMDN model:
            * Best: Prediction using the hypothesis that agrees best with the target
            * Mean: Weighted average of the hypotheses according to their mixing coefficients
            * Max: Prediction from the kernel with the highest mixing coefficient
            * Aligned: Prediction that best matches a multi-view alignment of kernels
        """
        preds = mdn_loss.get_all_preds(outputs_3d, targets_3d, aligned=False)#True, subj=subj)
        for k,v in preds.items():

            # MPJPE (Protocol 1)
            losses[k + '_p1'].update(loss.mpjpe(preds[k], targets_3d).item() * 1000.0, num_poses)
            
            # P-MPJPE (Protocol 2)
            if k=='best':
                best_error = mdn_loss.best_p_mpjpe(mu.detach().cpu().numpy(), targets_3d.numpy()).item()
                losses[k + '_p2'].update(best_error * 1000.0, num_poses)
            else:
                losses[k + '_p2'].update(loss.p_mpjpe(preds[k].numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) | Total: {ttl:} | ETA: {eta:} ' \
                     '| High: ({h1: .2f}, {h2: .2f}), Avg: ({a1: .2f}, {a2: .2f}), ' \
                     'Best: ({b1: .2f}, {b2: .2f})' \
            .format(batch=i + 1, size=len(data_loader),
                    ttl=bar.elapsed_td, eta=bar.eta_td, h1=losses['max_p1'].avg, h2=losses['max_p2'].avg,
                    a1=losses['mean_p1'].avg, a2=losses['mean_p2'].avg,
                    b1=losses['best_p1'].avg, b2=losses['best_p2'].avg)

        bar.next()

    bar.finish()
    return losses


if __name__ == '__main__':
    main(parse_args())
