from nasbench import api as nasbench101api
from nas_101_api.model_spec import ModelSpec
import random
import json
import torch
from models.cell_searchs.genotypes import Structure
from copy import deepcopy

class Nasbench101:
    def __init__(self, dataset, apiloc, args):
        self.dataset = dataset
        self.api = nasbench101api.NASBench(apiloc)
        self.args = args

    def get_accuracy(self, unique_hash, acc_type, trainval=True):
        spec = self.get_spec(unique_hash)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = 0.
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
        return maxacc
    def get_final_accuracy(self, uid, acc_type, trainval):
        return self.get_accuracy(uid, acc_type, trainval)
    def get_spec(self, unique_hash):
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        spec = ModelSpec(matrix, operations)
        return spec


class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod(x), x
                

def return_feature_layer(network, prefix=''):
    #for attr_str in dir(network):
    #    target_attr = getattr(network, attr_str)
    #    if isinstance(target_attr, torch.nn.Linear):
    #        setattr(network, attr_str, ReturnFeatureLayer(target_attr))
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')



def get_search_space(args):
    # if args.nasspace == 'nasbench201':
    #     return Nasbench201(args.dataset, args.api_loc)
    if args.nasspace == 'nasbench101':
        return Nasbench101(args.dataset, args.api_loc, args)
