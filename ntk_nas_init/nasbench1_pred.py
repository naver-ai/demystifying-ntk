# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pickle
import torch
import torch.nn.functional as F
import argparse
import json
import random
import nasspace
import logging
from scipy import stats
import numpy as np
from thop import profile

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='NAS-Bench-101')
    # parser.add_argument('--api_loc', default='data/nasbench_only108.tfrecord',
    #                     type=str, help='path to API')
    parser.add_argument('--json_loc', default='../bench_data/all_graphs.json',
                        type=str, help='path to JSON database')
    parser.add_argument('--GPU', default='1', type=str)
    parser.add_argument('--api_loc', default='../bench_data/nasbench_only108.tfrecord',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./results_nb101',
                        type=str, help='output directory')
    parser.add_argument('--save_string', default='debugging',
                        type=str, help='output directory')
    parser.add_argument('--nasspace', default='nasbench101',
                        type=str, help='output directory')
    parser.add_argument('--outfname', default='test',
                        type=str, help='output filename')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--init_channels', default=128, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=0, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=1000, help='end index')
    parser.add_argument('--write_freq', type=int, default=100, help='frequency of write to file')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--choice', default='others_all_bn', type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args

def get_op_names(v):
    o = []
    for op in v:
        if op == -1:
            o.append('input')
        elif op == -2:
            o.append('output')
        elif op == 0:
            o.append('conv3x3-bn-relu')
        elif op == 1:
            o.append('conv1x1-bn-relu')
        elif op == 2:
            o.append('maxpool3x3')
    return o
    
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

    searchspace = nasspace.get_search_space(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    models = json.load(open(args.json_loc))

    print(f'Running models {args.start} to {args.end} out of {len(models.keys())}')

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)


    #loop over nasbench1 archs (k=hash, v=[adj_matrix, ops])
    idx = 0
    # cached_res = []
    knas_metrics = []
    tenas_metrics = []
    fnorm_metrics = []
    avg_metrics = []
    diff_metrics = []
    last_metrics = []
    accs = []
    for k,v in models.items():

        if idx < args.start:
            idx += 1
            continue
        if idx >= args.end:
            break 
        logging.info(f'idx = {idx}')
        idx += 1

        res = {}
        res['hash']=k

        # model
        spec = nasbench1_spec._ToModelSpec(v[0], get_op_names(v[1]))
        net = nasbench1.Network(spec, stem_out=args.init_channels, num_stacks=3, num_mods=3, num_classes=get_num_classes(args))
        net.to(args.device)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        measures = predictive.find_measures(net, 
                                        train_loader, 
                                        (args.dataload, args.dataload_info, get_num_classes(args)),
                                        args.device,
                                        loss_fn=F.cross_entropy,
                                        measure_names=[args.choice])

        acc = searchspace.get_final_accuracy(k, acc_type='ori-test', trainval=True)


        res['logmeasures']= measures
        res['accuracy'] = acc

        knas_metrics.append(measures.get(args.choice)[0].item())
        tenas_metrics.append(measures.get(args.choice)[1].item())
        fnorm_metrics.append(measures.get(args.choice)[2].item())
        accs.append(acc)

        knas_kt, _ = stats.kendalltau(accs, knas_metrics)
        knas_roh , _ = stats.spearmanr(accs, knas_metrics)
        res['knas_kendall'] = knas_kt
        res['knas_spearman'] = knas_roh

        tenas_kt, _ = stats.kendalltau(accs, tenas_metrics)
        tenas_roh , _ = stats.spearmanr(accs, tenas_metrics)
        res['tenas_kendall'] = tenas_kt
        res['tenas_spearman'] = tenas_roh

        fnorm_kt, _ = stats.kendalltau(accs, fnorm_metrics)
        fnorm_roh , _ = stats.spearmanr(accs, fnorm_metrics)
        res['fnorm_kendall'] = fnorm_kt
        res['fnorm_spearman'] = fnorm_roh

        logging.info(res)

    knasscore_filename = os.path.join(save_path, 'knas_scores.npy')
    tenasscore_filename =os.path.join(save_path, 'tenas_scores.npy')
    fnormscore_filename =os.path.join(save_path, 'fnorm_scores.npy')
    acc_filename = os.path.join(save_path, 'accs.npy')

    np.save(knasscore_filename, knas_metrics)
    np.save(tenasscore_filename, tenas_metrics)
    np.save(fnormscore_filename, fnorm_metrics)
    np.save(acc_filename, accs)
    