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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.arguments import parse_args, args_to_file
from common.log import Logger
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.graph_utils import adj_mx_from_skeleton
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from models.sem_gcn import SemGCN


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
    keypoints = create_2d_data(
        path.join(data_root, "data_2d_" + args.dataset + "_" + args.keypoints + ".npz"), dataset)

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

    # Create model
    print("==> Creating model...")

    p_dropout = None if args.dropout == 0.0 else args.dropout
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = SemGCN(adj, args.hid_dim, num_layers=args.num_layers, p_dropout=p_dropout,
                       nodes_group=dataset.skeleton().joints_group() if args.non_local else None).to(
        device)

    print("==> Total parameters: {:.2f}M".format(
        sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr, weight_decay=args.l2_norm)

    criterion = nn.MSELoss(reduction='mean').to(device)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

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
            "gcn_ds_{}_drop{}_nhid_{}_{}".format(
                args.keypoints,
                args.dropout,
                args.hid_dim,
                datetime.datetime.now().isoformat(":", "seconds"),
            ),
        )

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print("==> Making checkpoint dir: {}".format(ckpt_dir_path))

        args_to_file(args, os.path.join(ckpt_dir_path, "params.txt"))

        logger = Logger(os.path.join(ckpt_dir_path, "log.txt"))
        logger.set_names(["epoch", "lr", "loss_train", "eval error p1", "eval error p2"])

    if args.evaluate:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            print(action)
            poses_valid, poses_valid_2d, actions_valid, subj_valid = fetch(subjects_test, dataset,
                                                                           keypoints, [action],
                                                                           stride)
            valid_loader = DataLoader(
                PoseGenerator(poses_valid, poses_valid_2d, actions_valid, subj_valid),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            errors_p1[i], errors_p2[i] = evaluate(valid_loader, model_pos, device)

        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(
            np.mean(errors_p1).item()))
        print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(
            np.mean(errors_p2).item()))
        exit(0)

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
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.lr,
                                                      step_size_up=int(
                                                          (args.epochs - start_epoch) * len(
                                                              train_loader) * 0.5),
                                                      # step_size_down=int((args.epochs-start_epoch)*len(train_loader)*0.7),
                                                      cycle_momentum=False)
    else:
        scheduler = None

    for epoch in range(start_epoch, args.epochs):
        print("\nEpoch: %d | LR: %.8f" % (epoch + 1, lr_now))
        # error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device)
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
            onecycle_scheduler=scheduler,
        )

        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device)

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step,
                       'state_dict': model_pos.state_dict(), 'optimizer': optimizer.state_dict(),
                       'error': error_eval_p1}, ckpt_dir_path, suffix='best')

    save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step,
               'state_dict': model_pos.state_dict(), 'optimizer': optimizer.state_dict(),
               'error': error_eval_p1}, ckpt_dir_path, suffix='final')
    logger.close()
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


def evaluate(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()
    start = time.time()
    for i, (targets_3d, inputs_2d, _, subj) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d).cpu()
        outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(
            p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:.3f}' \
          '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
          .format(batch=i + 1, size=len(data_loader), data=data_time.val, bt=batch_time.avg,
                  ttl=time.time() - start, e1=epoch_loss_3d_pos.avg,
                  e2=epoch_loss_3d_pos_procrustes.avg))

    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == "__main__":
    main(parse_args())