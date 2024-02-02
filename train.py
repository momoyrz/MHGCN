import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from timm import create_model, model_entrypoint

from datasets.read_data import get_data
from utils.classification_metrics import all_metrics
from utils.engine import train_one_epoch
from utils.logger import create_logger
from utils.loss import FocalLoss
from utils.lr_factory import cosine_scheduler
from utils.optimizer_factory import create_optimizer


def get_args_parser():
    parser = argparse.ArgumentParser('Set arguments for training and evaluation', add_help=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu_id', default='1', type=str)

    parser.add_argument('--features_path', default='', type=str)
    parser.add_argument('--csv_dir', default='./datasets/all_data.csv', type=str)
    parser.add_argument('--attributes_H_path', default='', type=str)
    parser.add_argument('--using_attributes', default='True', type=str)

    parser.add_argument('--model', default='dhgnn', type=str)
    parser.add_argument('--nhid', default=256, type=int)
    parser.add_argument('--nb_class', default=7, type=int)
    parser.add_argument('--seed', default=37, type=int)
    parser.add_argument('--torch_seed', default=21, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--residual', default='False', type=str)
    parser.add_argument('--cosine_threshold', default=0.6, type=float)
    parser.add_argument('--cosine_k', default=0, type=int)
    parser.add_argument('--kmeans_k', default=250, type=int)
    parser.add_argument('--gaussian_gamma', default=2, type=float)
    parser.add_argument('--Euclidean_threshold', default=0, type=float)
    parser.add_argument('--gaussian_threshold', default=0, type=float)
    parser.add_argument('--gaussian_k', default=0, type=int)
    parser.add_argument('--scaler', default=1, type=int)

    parser.add_argument('--opt', default='AdamW', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=(0.9, 0.999), type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--base_lr', default=0.01, type=float)
    parser.add_argument('--warmup_epochs', default=20, type=int)
    parser.add_argument('--final_lr', default=0.001, type=float)
    parser.add_argument('--start_warmup_value', default=0, type=float)

    parser.add_argument('--loss', default='focal', type=str)
    parser.add_argument('--gamma', default=2.0, type=float)
    parser.add_argument('--alpha', default=0.25, type=float)


    return parser


def main(args):
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    np.random.seed(args.seed)

    args.residual = args.residual.lower() in ['true', '1', 'yes']
    args.using_attributes = args.using_attributes.lower() in ['true', '1', 'yes']
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_dir = ('train_output/other/' + datetime.now().strftime('%Y%m%d') + '/' + args.model + '_' + args.prefix +
                    str(args.torch_seed) + '_' + str(args.seed) + '_' + str(args.num_layers) + '_' + str(
                args.residual) + '_' + str(args.cosine_threshold) + '_' + str(args.cosine_k) + '_' + str(
                args.gaussian_gamma) + '_' + str(args.gaussian_threshold) + '_' + str(args.gaussian_k) + '_' + str(
                args.scaler) + '_' + str(args.base_lr) + '_' + str(args.final_lr) + '_' + args.attributes_H_path.split('/')[-1].split('.')[0])
    # 如果'train_output/other/' + args.model + '_' + args.prefix这个文件夹不存在，则创建
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_logger(args.log_dir, '')

    logger.info('Creating data')
    features, labels, idx_train, idx_val, idx_test, H = get_data(args)

    if args.using_attributes:
        num_hyperedge = H.shape[1]
    else:
        num_hyperedge = 0

    if (args.cosine_threshold > 0 or args.cosine_k > 0) and (args.gaussian_threshold > 0 or args.gaussian_k > 0):
        raise Exception('Only one of cosine_threshold, cosine_k, gaussian_threshold, gaussian_k can be greater than 0')
    if args.cosine_threshold > 0 or args.cosine_k > 0:
        num_hyperedge += features.shape[0]
    elif args.gaussian_threshold > 0 or args.gaussian_k > 0:
        num_hyperedge += features.shape[0]
    elif args.Euclidean_threshold > 0:
        num_hyperedge += features.shape[0]

    if args.kmeans_k > 0:
        num_hyperedge += args.kmeans_k


    logger.info('Creating model')
    create_fn = model_entrypoint(args.model)
    model = create_fn(in_ch=features.shape[1],
                      n_class=args.nb_class,
                      n_hid=args.nhid,
                      num_layers=args.num_layers,
                      dropout=args.dropout,
                      residual=args.residual,
                      cosine_threshold=args.cosine_threshold,
                      Euclidean_threshold=args.Euclidean_threshold,
                      cosine_k=args.cosine_k,
                      gaussian_gamma=args.gaussian_gamma,
                      gaussian_threshold=args.gaussian_threshold,
                      gaussian_k=args.gaussian_k,
                      num_node=features.shape[0],
                      num_hyperedge=num_hyperedge,
                      scaler=args.scaler,
                      kmeans_k=args.kmeans_k)
    logger.info('Model created')

    optimizer = create_optimizer(args, model)
    lr_schedule_values = cosine_scheduler(args.base_lr, args.final_lr, args.epochs,
                                          args.warmup_epochs, args.start_warmup_value)
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accurency = 0.0

    for epoch in range(args.epochs):
        output = train_one_epoch(model, criterion, optimizer, features, H, labels, idx_train, idx_val, idx_test,
                                 args.device, epoch, lr_schedule_values, logger)
        y_score = torch.softmax(output, dim=1)
        acc, pre, sen, f1, spec, kappa, my_auc, qwk = all_metrics(labels[idx_val].cpu().detach().numpy(),
                                                                  y_score[idx_val].cpu().detach().numpy())
        if acc > max_accurency:
            max_accurency = acc
            acc, pre, sen, f1, spec, kappa, my_auc, qwk = all_metrics(labels[idx_test].cpu().detach().numpy(),
                                                                      y_score[idx_test].cpu().detach().numpy())
            best_state = {'Test acc': acc, 'Test pre': pre, 'Test sen': sen, 'Test f1': f1, 'Test spec': spec,
                          'Test kappa': kappa, 'Test auc': my_auc, 'Test qwk': qwk}

    logger.info({**{f'{k}': v for k, v in best_state.items()}})
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('All_models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
