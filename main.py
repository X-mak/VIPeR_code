#!/usr/bin/env python3
# augment 31         scaled_rotation = torch.block_diag(kn.geometry.angle_to_rotation_matrix(angle)[0] @ torch.diag(scale), torch.ones(1))
# utils/geometry 150          points = kn.geometry.conversions.normalize_pixel_coordinates(points, w + 1, h + 1).unsqueeze(0).expand(B, -1, -1, -1)
#                 45          return kn.geometry.camera.pinhole.PinholeCamera(intrinsics, extrinsics, height, width)
# kornia 0.5.8
import re
from pathlib import Path
import configargparse
import os
from main_single import run
import torch
import random
import numpy as np

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # set CPU seed
    torch.cuda.manual_seed(seed)  # set GPU seed
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def main(args, extra_args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    set_random_seed(42)
    passed_envs, last_dataset = [], None
    args.datasets = args.dataset.split(',')

    # default environment orders
    out_dir = Path(args.out_dir)
    save_path = create_dir(out_dir / 'train') / 'model.pth'
    if not args.cross:
        args.datasets = [args.datasets[0]]
    for dataset in args.datasets:
        args.envs, args.epochs = None, None
        if not args.skip_train:
            if dataset == "tartanair":
                args.envs = ["carwelding", "neighborhood", "office2", "abandonedfactory_night", "westerndesert"]
                args.epochs = [1] if args.epochs is None else args.epochs
            if dataset == "nordland":
                args.envs = ["spring", "summer", "fall", "winter"]
                args.epochs = [1] if args.epochs is None else args.epochs
            if dataset == "robotcar":
                args.envs = ["sun", "night", "overcast"]
                args.epochs = [1] if args.epochs is None else args.epochs
        else:
            args.envs = []
            args.epochs = [1] if args.epochs is None else args.epochs
            for ds in args.datasets:
                if ds == "tartanair":
                    args.envs.extend(["carwelding", "neighborhood", "office2", "abandonedfactory_night", "westerndesert"])
                elif ds == "nordland":
                    args.envs.extend(["spring", "summer", "fall", "winter"])
                elif ds == "robotcar":
                    args.envs.extend(["sun", "night", "overcast"])
            save_path = create_dir(out_dir / 'train') / 'model.pth'
        args.epochs = [1] if args.epochs is None else args.epochs

        n_env = len(args.envs)
        if len(args.epochs) == 1:
            args.epochs = args.epochs * n_env
        else:
            assert args.method != 'joint' and len(args.epochs) == n_env

        # out_dir = Path(args.out_dir)
        if dataset == 'tartanair':
            regex = ["carwelding", "neighborhood", "office2", "abandonedfactory_night", "westerndesert"]
        elif dataset == 'nordland':
            regex = ["spring", "summer", "fall", "winter"]
        elif dataset == 'robotcar':
            regex = ["sun", "night", "overcast"]

        all_env_regex = '(' + '|'.join(regex) + ')'
        # save_path = create_dir(out_dir / 'train') / 'model.pth'
        for i, (epoch, env) in enumerate(zip(args.epochs, args.envs)):
            if not args.skip_train:
                train_args = ['--task', 'train-joint' if args.method == 'joint' else 'train-seq']
                train_args += ['--dataset', dataset]
                train_args += ['--include', all_env_regex if args.method == 'joint' else env]
                train_args += ['--epoch', str(epoch)]
                train_args += ['--memory-path', args.memory_path]
                train_args += ['--all-data', args.dataset]
                train_args += ['--backbone', args.backbone]
                train_args += ['--method', args.method]
                if args.method == 'viper' or args.backbone == 'dinov2':
                    train_args += ['--aggregator', 'vlad']

                # load model saved from previous runx
                if i > 0 or last_dataset is not None:
                    train_args += ['--load', str(save_path)]
                train_args += ['--save', str(save_path)]
                if args.cross:
                    train_args += ['--cross']
                # weights loading for lifelong methods
                if args.method not in ['finetune', 'joint']:
                    train_args += ['--ll-method', args.method]
                    train_args += ['--ll-weight-dir', str(create_dir(out_dir / 'll-weights'))]
                    if i > 0 or last_dataset is not None:
                        # train_args += ['--ll-weight-load'] + args.envs[:i]
                        train_args += ['--ll-weight-load'] + passed_envs
                if args.method == 'viper':
                    train_args += ['--memory']
                run(train_args + extra_args)

            if args.method == 'joint':
                save_path = save_path.parent / (save_path.name + f'.epoch{epoch - 1}')
            else:
                save_path = save_path.parent / (save_path.name + (f".{env}.{epoch - 1}" if epoch > 1 else f".{env}"))

            if not args.skip_eval:
                eval_args = ['--task', 'eval', '--dataset', dataset, '--include', all_env_regex, '--load', str(save_path), '--backbone', args.backbone, '--method', args.method]
                if args.method == 'viper' or args.backbone == 'dinov2':
                    eval_args += ['--aggregator', 'vlad']
                run(eval_args + extra_args)
            if len(passed_envs) != 0 and passed_envs[-1] != env:
                passed_envs.append(env)
            elif len(passed_envs) == 0:
                passed_envs.append(env)
        last_dataset = dataset



def create_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    return directory


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    # meta
    parser.add_argument('--out-dir', type=str, default="./run/")
    parser.add_argument('--memory-path', type=str, default="./run/memory.pth")
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    # launch
    # parser.add_argument("--dataset", type=str, default='tartanair',
    #                     choices=['tartanair', 'nordland', 'robotcar'], help="Dataset to use")
    parser.add_argument("--dataset", type=str, default='tartanair', help="Dataset to use")
    parser.add_argument('--envs', type=str, nargs='+')
    parser.add_argument('--epochs', type=int, nargs='+')
    parser.add_argument('--method', type=str, required=True,
                        choices=['finetune', 'viper', 'airloop'])
    parser.add_argument('--backbone', type=str, required=False, default='vgg',
                        choices=['vgg', 'dinov2'])
    parser.add_argument('--cross', action='store_true')

    parserd_args, unknown_args = parser.parse_known_args()

    main(parserd_args, unknown_args)
