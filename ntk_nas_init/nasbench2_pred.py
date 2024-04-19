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
import logging
import random
import math
from scipy import stats

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='NAS-Bench-201')
    parser.add_argument('--api_loc', default='../bench_data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./results_nb201',
                        type=str, help='output directory')
    parser.add_argument('--save_string', default='debugging',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=0, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--data_seed', type=int, default=42, help='random/numpy manual seed')
    parser.add_argument('--model_seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=3000, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true', help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    parser.add_argument('--choice', default='others_all_bn', type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
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

    if args.noacc:
        api = pickle.load(open(args.api_loc,'rb'))
    else:
        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)
    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)

    knas_metrics = []
    tenas_metrics = []
    fnorm_metrics = []
    test_accs = []

    
    args.end = len(api) if args.end == 0 else args.end

    #loop over nasbench2 archs
    for i, arch_str in enumerate(api):

        if i < args.start:
            continue
        if i >= args.end:
            break 

        res = {'i':i, 'arch':arch_str}

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

        random.seed(args.data_seed)
        np.random.seed(args.data_seed)
        torch.manual_seed(args.data_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        measures = predictive.find_measures(net,
                                        train_loader,
                                        (args.dataload, args.dataload_info, get_num_classes(args)),
                                        args.device,
                                        loss_fn=F.cross_entropy,
                                        measure_names=[args.choice])

        res['logmeasures']= measures

        if not args.noacc:
            info = api.get_more_info(i, 'cifar10-valid' if args.dataset=='cifar10' else args.dataset, iepoch=None, hp='200', is_random=False)

            testacc  = info['test-accuracy']

            res['testacc']=testacc

        knas_metrics.append(measures.get(args.choice)[0].item())
        tenas_metrics.append(measures.get(args.choice)[1].item())
        fnorm_metrics.append(measures.get(args.choice)[2].item())

        test_accs.append(testacc)

        if len(knas_metrics) == 1001:
            break
        else:
            knas_test_kt, _ = stats.kendalltau(test_accs, knas_metrics)
            knas_test_roh , _ = stats.spearmanr(test_accs, knas_metrics)

            res['knas_test_kendall'] = knas_test_kt
            res['knas_test_spearman'] = knas_test_roh

            tenas_test_kt, _ = stats.kendalltau(test_accs, tenas_metrics)
            tenas_test_roh , _ = stats.spearmanr(test_accs, tenas_metrics)

            res['tenas_test_kendall'] = tenas_test_kt
            res['tenas_test_spearman'] = tenas_test_roh

            fnorm_test_kt, _ = stats.kendalltau(test_accs, fnorm_metrics)
            fnorm_test_roh , _ = stats.spearmanr(test_accs, fnorm_metrics)

            res['fnorm_test_kendall'] = fnorm_test_kt
            res['fnorm_test_spearman'] = fnorm_test_roh

            logging.info(res)

    knasscore_filename = os.path.join(save_path, 'knas_scores.npy')
    tenasscore_filename =os.path.join(save_path, 'tenas_scores.npy')
    fnormscore_filename = os.path.join(save_path, 'fnorm_scores.npy')
    testacc_filename = os.path.join(save_path, 'test_accs.npy')

    np.save(knasscore_filename, knas_metrics)
    np.save(tenasscore_filename, tenas_metrics)
    np.save(fnormscore_filename, fnorm_metrics)
    np.save(testacc_filename, test_accs)
