"""
This code is based on (and an extension of) the publicly-available implementation of SemGCN.
https://github.com/garyzhao/SemGCN
"""

from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import numpy as np
import os.path as path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    print("==> Using settings {}".format(args))

    # Load data in a way that is semi-robust to path differences
    print("==> Loading dataset...")
    directory_for_this_file = os.path.dirname(__file__)
    data_folder_name = "data"
    data_root = os.path.join(directory_for_this_file, data_folder_name)
    dataset_path = path.join(data_root, "data_3d_" + args.dataset + ".npz")

    if args.dataset == "h36m":
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS

        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError("Invalid dataset")

    print("==> Preparing data...")
    dataset = read_3d_data(dataset)

    print("==> Loading 2D detections...")
    keypoints = create_2d_data(path.join(data_root, "data_2d_" + args.dataset + "_" + args.keypoints + ".npz"), dataset)

    action_filter = None if args.actions == "*" else args.actions.split(",")
    if action_filter is not None:
        action_filter = list(map(lambda x: dataset.define_actions(x)[0], action_filter))
        print("==> Selected actions: {}".format(action_filter))

    stride = args.downsample

    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")

    # Define Metrics
    metrics = ["best_p1", "best_p2", "mean_p1", "mean_p2", "max_p1", "max_p2"]

    # Create model
    print("==> Creating model...")

    p_dropout = None if args.dropout == 0.0 else args.dropout
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

    if args.pose_level_pi:
        criterion = mdn_loss.mdn_loss_fn_pose_distributions
    else:
        criterion = mdn_loss.mdn_loss_fn
    
    # Optionally resume from a checkpoint
    if args.resume:
        ckpt_path = args.resume

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            start_epoch = ckpt["epoch"]
            error_best = ckpt["error"]
            glob_step = ckpt["step"]
            lr_now = ckpt["lr"]
            model_pos.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
            ckpt_dir_path = path.dirname(ckpt_path)

            if args.resume:
                logger = Logger(path.join(ckpt_dir_path, "log.txt"), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr

        # Encode various useful parameters in the checkpoint filename
        ckpt_dir_path = path.join(
            args.checkpoint,
            "ds_{}_numK{}_drop{}_nhid_{}_tanh{}_pPi_{}_mvar_{}_{}".format(
                args.keypoints,
                args.num_gaussians,
                args.dropout,
                args.hid_dim,
                args.tanh_out,
                args.pose_level_pi,
                args.multivariate,
                datetime.datetime.now().isoformat(":", "seconds"),
            ),
        )

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print("==> Making checkpoint dir: {}".format(ckpt_dir_path))

        args_to_file(args, os.path.join(ckpt_dir_path, "params.txt"))

        logger = Logger(os.path.join(ckpt_dir_path, "log.txt"))
        logger.set_names(["epoch", "lr", "loss_train"] + metrics)
    
    # Setup the training data
    poses_train, poses_train_2d, actions_train, subj_train = fetch(
        subjects_train, dataset, keypoints, action_filter, stride
    )
    train_loader = DataLoader(
        PoseGenerator(poses_train, poses_train_2d, actions_train, subj_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Set up the validation data
    poses_valid, poses_valid_2d, actions_valid, subj_valid = fetch(
        subjects_test, dataset, keypoints, action_filter, stride
    )
    valid_loader = DataLoader(
        PoseGenerator(poses_valid, poses_valid_2d, actions_valid, subj_valid),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    d_scheduler = None
    if args.onecycle_lr:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0 ,max_lr=args.lr, 
                    step_size_up=int((args.epochs-start_epoch)*len(train_loader)*0.5),
                    #step_size_down=int((args.epochs-start_epoch)*len(train_loader)*0.7),
                    cycle_momentum=False)
    else:
        scheduler = None

    for epoch in range(start_epoch, args.epochs):
        print("\nEpoch: %d | LR: %.8f" % (epoch + 1, lr_now))
        #eval_loss = evaluate_all(valid_loader, model_pos, device, criterion)
        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(
            train_loader,
            model_pos,
            criterion,
            optimizer,
            device,
            args.lr,
            lr_now,
            glob_step,
            args.lr_decay,
            args.lr_gamma,
            max_norm=args.max_norm,
            onecycle_scheduler = scheduler,
        )

        # Evaluate
        eval_loss = evaluate_all(valid_loader, model_pos, device, criterion)

        # Update log file
        logger.append(
            [
                epoch + 1,
                lr_now,
                epoch_loss,
                eval_loss["best_p1"].avg,
                eval_loss["best_p2"].avg,
                eval_loss["mean_p1"].avg,
                eval_loss["mean_p2"].avg,
                eval_loss["max_p1"].avg,
                eval_loss["max_p2"].avg,
            ]
        )

        # Save checkpoint
        if error_best is None or error_best > eval_loss["best_p1"].avg:
            error_best = eval_loss["best_p1"].avg
            save_ckpt(
                {
                    "epoch": epoch + 1,
                    "lr": lr_now,
                    "step": glob_step,
                    "state_dict": model_pos.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "error": eval_loss["best_p1"].avg,
                },
                ckpt_dir_path,
                suffix="best",
            )

    save_ckpt(
        {
            "epoch": epoch + 1,
            "lr": lr_now,
            "step": glob_step,
            "state_dict": model_pos.state_dict(),
            "optimizer": optimizer.state_dict(),
            "error": eval_loss["best_p1"].avg,
        },
        ckpt_dir_path,
        suffix="final",
    )
    logger.close()
    logger.plot(["loss_train", "best_p1"])
    savefig(path.join(ckpt_dir_path, "log.eps"))

    return


def train(
    data_loader,
    model_pos,
    criterion,
    optimizer,
    device,
    lr_init,
    lr_now,
    step,
    decay,
    gamma,
    max_norm=True,
    onecycle_scheduler=None,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()
    start = time.time()

    dataset_size = len(data_loader)

    bar = Bar("Train", max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, subj) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if not onecycle_scheduler:
            if step % decay == 0 or step == 1:
                lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)

        # Predict 3D poses from inputs
        outputs_3d = model_pos(inputs_2d)

        # Calculate loss and backprop
        loss_3d_pos = criterion(outputs_3d, targets_3d)

        optimizer.zero_grad()
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        if onecycle_scheduler:
            onecycle_scheduler.step()
            
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = (
            "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} "
            "| Loss: {loss: .3f} ".format(
                batch=i + 1,
                size=len(data_loader),
                data=data_time.val,
                bt=batch_time.avg,
                ttl=bar.elapsed_td,
                eta=bar.eta_td,
                loss=epoch_loss_3d_pos.avg
            )
        )
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


def evaluate_all(data_loader, model_pos, device, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = defaultdict(AverageMeter)

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()
    start = time.time()

    bar = Bar("Eval ", max=len(data_loader))
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
        """
        loss_3d_pos = criterion(outputs_3d, targets_3d.to(device))
        losses['loss'].update(loss_3d_pos, num_poses)
        preds = mdn_loss.get_all_preds(outputs_3d, targets_3d, aligned=False, subj=subj)

        # Cycle through the types of predictions (see above) and calculate error for Protocol 1 & 2.
        for k, v in preds.items():
            # MPJPE (Protocol 1)
            losses[k + "_p1"].update(
                loss.mpjpe(preds[k], targets_3d).item() * 1000.0, num_poses
            )
            # P-MPJPE (Protocol 2)
            losses[k + "_p2"].update(
                loss.p_mpjpe(preds[k].numpy(), targets_3d.numpy()).item() * 1000.0,
                num_poses,
            )

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = (
            "({batch}/{size}) | Total: {ttl:} | ETA: {eta:} |"
            " High: ({h1: .3f}, {h2: .3f}), "
            "Avg: ({a1: .3f}, {a2: .3f}), Best: ({b1: .3f}, {b2: .3f}) | "
            "Loss: {l: .3f}".format(
                batch=i + 1,
                size=len(data_loader),
                ttl=bar.elapsed_td,
                eta=bar.eta_td,
                h1=losses["max_p1"].avg,
                h2=losses["max_p2"].avg,
                a1=losses["mean_p1"].avg,
                a2=losses["mean_p2"].avg,
                b1=losses["best_p1"].avg,
                b2=losses["best_p2"].avg,
                l=losses["loss"].avg,
            )
            )

        bar.next()
        #if i%100 == 0:
        #    for k, avg_meter in losses.items():
        #        print('{}:{:.3f}'.format(k, avg_meter.avg))

    bar.finish()
    #for k, avg_meter in losses.items():
    #    print('{}:{:.3f}'.format(k, avg_meter.avg))
    return losses


if __name__ == "__main__":
    main(parse_args())
