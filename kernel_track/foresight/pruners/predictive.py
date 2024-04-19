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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .p_utils import *
from . import measures

import types
import copy


def find_measures(net_orig,                  # neural network
                  dataloader,                # a data loader (typically for training data)
                  dataload_info,             # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                  device,                    # GPU/CPU device used
                  loss_fn,   # loss function to use within the zero-cost metrics
                  measure_names,        # an array of measure names to compute, if left blank, all measures are computed by default
                  measures_arr=None):        # [not used] if the measures are already computed but need to be summarized, pass them here
    
    #Given a neural net
    #and some information about the input data (dataloader)
    #and loss function (loss_fn)
    #this function returns an array of zero-cost proxy metrics.
    #given 1 minibatch of data

    done, ds = False, 1
    dataload, num_imgs_or_batches, num_classes = dataload_info

    if dataload == 'random':
        inputs, targets = get_some_data(dataloader, num_batches=num_imgs_or_batches, device=device)
    elif dataload == 'grasp':
        inputs, targets = get_some_data_grasp(dataloader, num_classes, samples_per_class=num_imgs_or_batches, device=device)
    else:
        raise NotImplementedError(f'dataload {dataload} is not supported')

    measure_values = {}
    for measure_name in measure_names:
        if measure_name not in measure_values:
            val = measures.calc_measure(measure_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds)
            measure_values[measure_name] = val


    return measure_values
