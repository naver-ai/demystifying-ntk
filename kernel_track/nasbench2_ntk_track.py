import pickle
import torch
import torch.nn.functional as F
import argparse
import logging
import random
import json
import math
from scipy import stats
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net
from train import train
from track_utils import *


def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def parse_arguments():
    parser = argparse.ArgumentParser(description='NTK Tracking Experiments')
    parser.add_argument('--api_loc', default='../bench_data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--arch_loc', default='../bench_data/nb201_cf10_arch.json',
                        type=str, help='path to binned architectures')
    parser.add_argument('--outdir', default='./nb201_ntk_tracking',
                        type=str, help='output directory')
    parser.add_argument('--save_string', default='exp_1',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization')
    parser.add_argument('--tr_batch_size', default=1024, type=int)
    parser.add_argument('--val_batch_size', default=256, type=int)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=0, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--data_seed', type=int, default=42, help='random/numpy manual seed')
    parser.add_argument('--model_seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--train_epoch', type=int, default=10, help='start index')
    parser.add_argument('--track_freq', type=int, default=1, help='ntk tracking frequency')
    parser.add_argument('--sampling', type=int, default=1, help='number of samples to compute ntk')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=3000, help='end index')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_arguments()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    save_path = f'{args.outdir}/{args.save_string}'
    os.makedirs(save_path, exist_ok=True)
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", args)

    logger = SummaryWriter(save_path)

    train_loader, val_loader = get_cifar_dataloaders(args.tr_batch_size, args.val_batch_size, args.dataset,
                                                     args.num_data_workers)

    ntk_norms = []
    param_norms = []
    corrs = []

    ntk_diff_init = []
    ntk_rdiff_init = []
    corrs_diff_init = []
    sim_diff_init = []

    ntk_diffs = []
    ntk_rdiffs = []
    corrs_diff = []

    sims = []
    epochs = []

    from nas_201_api import NASBench201API as API
    api = API(args.api_loc)

    models_bin = json.load(open(args.arch_loc))
    np.random.seed(args.model_seed)
    rand_indices = np.random.randint(0, len(models_bin), size=1)

    #loop over nasbench2 archs
    for iter, (idx) in enumerate(rand_indices):
        arch_str = api.query_meta_info_by_index(int(models_bin[idx][0])).arch_str

        res = {'i': iter, 'arch': arch_str}

        random.seed(args.model_seed)
        np.random.seed(args.model_seed)
        torch.manual_seed(args.model_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args))
        net.to(args.device)

        init_net(net, args.init_w_type, args.init_b_type)

        arch_str2 = nasbench2.get_arch_str_from_model(net)
        if arch_str != arch_str2:
            print(arch_str)
            print(arch_str2)
            raise ValueError

        ntk_matrix_init = batch_wise_ntk(net, train_loader, device=torch.device("cuda:" + str(args.gpu)), samplesize=args.sampling)
        ntk_norm = np.linalg.norm(ntk_matrix_init.flatten())
        print(f'The total norm of the NTK sample is {ntk_norm:.2f}')
        param_norm = np.sqrt(np.sum([p.pow(2).sum().detach().cpu().numpy() for p in net.parameters()]))
        print(f'The L2 norm of the parameter vector is {param_norm:.2f}')
        matrix_corr, corr_coeff, corr_tom = corr(ntk_matrix_init, ntk_matrix_init)
        print(f'The Correlation coefficient of the NTK sample from init to now is {corr_coeff:.2f}')
        print(f'The Similarity coefficient of the NTK sample from init to now is {corr_tom:.2f}')
        ntk_matrix_prev = ntk_matrix_init.copy()
        sims.append(corr_tom)
        param_norms.append(param_norm)
        ntk_norms.append(ntk_norm)
        corrs.append(corr_coeff)
        corrs_diff.append(corr_coeff)
        epochs.append(0)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        optimizer = torch.optim.SGD(
            net.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        for epoch_idx in range(args.train_epoch):
            net.train()
            train_accu, train_error = train(train_loader, net, criterion, optimizer)
            logging.info(f'{epoch_idx} / {args.train_epoch} | train_accu = {train_accu} | train_error = {train_error}')
            logger.add_scalar("accuracy", train_accu, epoch_idx)
            logger.add_scalar("train_error", train_error, epoch_idx)

            if int(epoch_idx + 1) % args.track_freq == 0:

                ## Calculate NTK ##
                ntk_matrix = batch_wise_ntk(net, train_loader, device=torch.device("cuda:" + str(args.gpu)), samplesize=args.sampling)

                ## Calculate diff in NTK from init to now ##
                ntk_matrix_diff_init = np.abs(ntk_matrix_init - ntk_matrix)
                ntk_matrix_diff_norm_init = np.linalg.norm(ntk_matrix_diff_init.flatten())
                print(f'The total norm of the NTK sample diff from init to now is {ntk_matrix_diff_norm_init:.2f}')
                ntk_matrix_rdiff_init = np.abs(ntk_matrix_init - ntk_matrix) / (np.abs(ntk_matrix_init) + 1e-4)
                ntk_matrix_rdiff_norm_init = np.linalg.norm(ntk_matrix_rdiff_init.flatten())
                print(f'The total norm of the NTK sample relative diff from init to now is {ntk_matrix_rdiff_norm_init:.2f}')

                ## Calculate correlation coefficients ##
                _, corr_coeff_init, corr_tom_init = corr(ntk_matrix_init, ntk_matrix)
                print(f'The Correlation coefficient of the NTK sample from init to now is {corr_coeff_init:.2f}')

                ntk_matrix_prev = ntk_matrix.copy()

                logger.add_scalar("ntk_rdiffs_init", ntk_matrix_rdiff_norm_init, epoch_idx)
                logger.add_scalar("corrs_init", corr_coeff_init, epoch_idx)
