"""
demystifying-ntk
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

import torch
import numpy as np


def label_sim_matrix(input, target):
    sim_matrix = np.zeros(len(target), len(target))

    for i in range(len(target)):
        for j in range(len(target)):
            if i == j:
                sim_matrix[i,j] = 1
            else:
                sim_matrix[i,j] = -1

    return sim_matrix
