import argparse
import os
import json
import copy


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('--actions', default='*', type=str, metavar='LIST', help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--predict', default='', type=str, metavar='FILENAME', help='checkpoint to predict for one of each action (file name)')
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')

    # Model arguments
    parser.add_argument('--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('--eval_batch_size', default=1024, type=int, metavar='N', help='evaluation batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=30, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--non_local', type=bool, default=True, help='if use non-local layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')

    # Experimental
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--num_gaussians', default=5, type=int, help='amnt gaussians in mdn')
    parser.add_argument('--l2_norm', default=0, type=float, help='amount of l2 weight normalization applied')
    parser.add_argument('--no_tanh', dest='tanh_out', action='store_false', help='dont use tanh activation on mu outputs')
    parser.set_defaults(tanh_out=True)
    parser.add_argument('--pose_level_pi', dest='pose_level_pi', action='store_true', 
                        help='constrain mixture coefficients pi of each node to be the same by taking mean over nodes')
    parser.set_defaults(pose_level_pi=False)
    parser.add_argument('--uniform_sigma', dest='uniform_sigma', action='store_true', 
                        help='make each joint have the same variance by averaging sigma outputs in each kernel')
    parser.set_defaults(uniform_sigma=False)
    parser.add_argument('--multivariate', default=False, type=bool,
                        help='use multivariate distribution (unique sigma per x,y,z coordinate)')
    parser.add_argument('--onecycle_lr', dest='onecycle_lr', action='store_true', help='whether to use onecycle lr schedule')
    parser.set_defaults(onecycle_lr=False)

    args = parser.parse_args()

    # Check invalid configuration - these can't be simultaneously active.
    if bool(args.predict) + bool(args.resume) + bool(args.evaluate) > 1:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    model_file = args.predict or args.resume or args.evaluate

    # If the model file is specified, overload the argparse arguments with parameters from the model.
    if model_file:
        tmp_pred, tmp_resume, tmp_eval = args.predict, args.resume, args.evaluate
        tmp_actions = args.actions

        dirname = os.path.dirname(model_file)
        with open(os.path.join(dirname, 'params.txt'), 'r') as f:
            json_dict = json.load(f)

        args = argparse.Namespace(**json_dict)
        args.predict, args.resume, args.evaluate, args.model_file = tmp_pred, tmp_resume, tmp_eval, model_file
        args.actions = tmp_actions

    return args


def args_to_file(args, filename="./args_for_run.txt"):
    """ A helper method for publishing arguments to disk. """

    argparse_dict = copy.deepcopy(vars(args))
    argparse_dict.pop('evaluate')
    argparse_dict.pop('resume')
    argparse_dict.pop('predict')
    argparse_dict.pop('checkpoint')

    with open(filename, 'w') as f:
        f.write(json.dumps(argparse_dict, indent=4))
