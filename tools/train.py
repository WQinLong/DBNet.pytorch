# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function

import argparse
import os

import anyconfig
import numpy as np
import random
import time
import yaml
from utils import setup_logger


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--config_file', default='config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')

    args = parser.parse_args()
    return args


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0, 0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, -3:] * w).sum(1)


def print_mutation(results, yaml_file, evolve_file, meta_value):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    # a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(meta_value) % tuple(meta_value)  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\nEvolved fitness: %s\n' % (b, c))

    with open(evolve_file, 'a') as f:  # append result
        f.write(b + c + '\n')
    x = np.unique(np.loadtxt(evolve_file, ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt(evolve_file, x, '%10.3g')  # save sort by fitness

    # # Save yaml
    # for i, k in enumerate(hyp.keys()):
    #     hyp[k] = float(x[0, i + 7])
    # with open(yaml_file, 'w') as f:
    #     results = tuple(x[0, :7])
    #     c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    #     f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
    #     yaml.dump(hyp, f, sort_keys=False)


def main(config):
    import torch
    from models import build_model, build_loss
    from data_loader import get_dataloader
    from trainer import Trainer
    from post_processing import get_post_processing
    from utils import get_metric
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(), rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank

    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train']['dataset']['args'][
                                                         'img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])

    if config['local_rank'] == 0:
        save_dir = os.path.join(config['trainer']['output_dir'], config['name'] + '_' + model.name)
        logger = setup_logger(os.path.join(save_dir, 'train.log'))

    if 'evolve' in config.keys() and config['evolve']['flag'] and not config['distributed']:
        meta = {'optimizer.args.lr': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lr_scheduler.args.warmup_epoch': (1, 0, 5),
                'loss.alpha': (1, 0.5, 3),
                'loss.beta': (2, 5, 20),
                'loss.ohem_ratio': (1, 1, 5),
                'post_processing.args.box_thresh': (0.3, 0.4, 1.0),
                'dataset.train.dataset.args.pre_processes.[1].args.min_crop_side_ratio': (1, 0.1, 0.9),
                'dataset.train.dataset.args.pre_processes.[2].args.thresh_max': (0.3, 0.4, 1.0),
                }  # image mixup (probability)
        config['notest'] = True
        config['nosave'] = True
        saved_path = os.path.join(config['trainer']['output_dir'], config['name'] + '_' + model.name)
        if not os.path.exists(os.path.join(saved_path, 'evolve')):
            os.makedirs(os.path.join(saved_path, 'evolve'))
        yaml_file = os.path.join(saved_path, 'evolve', 'hyp_evolved.yaml')
        evolve_file = os.path.join(saved_path, 'evolve', 'evolve.txt')
        for _ in range(300):
            if os.path.exists(evolve_file):
                parent = 'single'
                x = np.loadtxt(evolve_file, ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min()
                if len(x) == 1:
                    x = x[0]
                elif parent == 'single':
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                #     hyp[k] = float(x[i + 7] * v[i])  # mutate
                for i, k in enumerate(meta.keys()):
                    config_keys = k.split('.')
                    str_config = 'config'
                    for config_key in config_keys:
                        if config_key.startswith('[') and config_key.endswith(']'):
                            str_config = str_config + config_key
                        else:
                            str_config = str_config + '[\'' + config_key + '\']'
                    exec(str_config + '=x[i]*v[i]')

            meta_value = []
            for k, v in meta.items():
                config_keys = k.split('.')
                str_config = 'config'
                for config_key in config_keys:
                    if config_key.startswith('[') and config_key.endswith(']'):
                        str_config = str_config + config_key
                    else:
                        str_config = str_config + '[\'' + config_key + '\']'
                # str_config = 'config[\'' + '\'][\''.join(k.split('.')) + '\']'
                exec('print(' + str_config + ')')
                exec(str_config + '=max(' + str_config + ', v[1])')
                exec(str_config + ' = min(' + str_config + ', v[2])')
                exec(str_config + ' = round(' + str_config + ', 5)')
                exec('meta_value.append(' + str_config + ')')
                # hyp[k] = max(hyp[k], v[1])  # lower limit
                # hyp[k] = min(hyp[k], v[2])  # upper limit
                # hyp[k] = round(hyp[k], 5)  # significant digits

            train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
            assert train_loader is not None
            if 'validate' in config['dataset']:
                validate_loader = get_dataloader(config['dataset']['validate'], False)
            else:
                validate_loader = None

            criterion = build_loss(config['loss']).cuda()

            post_p = get_post_processing(config['post_processing'])
            metric = get_metric(config['metric'])

            trainer = Trainer(config=config,
                              model=model,
                              criterion=criterion,
                              train_loader=train_loader,
                              post_process=post_p,
                              metric_cls=metric,
                              validate_loader=validate_loader,
                              logger=(logger if config['local_rank'] == 0 else None))
            results = trainer.train()

            print_mutation(results, yaml_file, evolve_file, meta_value)

    else:
        train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
        assert train_loader is not None
        if 'validate' in config['dataset']:
            validate_loader = get_dataloader(config['dataset']['validate'], False)
        else:
            validate_loader = None

        criterion = build_loss(config['loss']).cuda()

        post_p = get_post_processing(config['post_processing'])
        metric = get_metric(config['metric'])

        trainer = Trainer(config=config,
                          model=model,
                          criterion=criterion,
                          train_loader=train_loader,
                          post_process=post_p,
                          metric_cls=metric,
                          validate_loader=validate_loader,
                          logger=(logger if config['local_rank'] == 0 else None))
        trainer.train()


if __name__ == '__main__':
    import sys
    import pathlib
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    print(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    print(str(__dir__.parent.parent))
    # project = 'DBNet.pytorch'  # 工作项目根目录
    # sys.path.append(os.getcwd().split(project)[0] + project)

    from utils import parse_config

    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    main(config)
