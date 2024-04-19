"""
demystifying-ntk
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

import torch
import numpy as np

def corr(n1, n2):
    """Pearson Corr."""
    n1_mean = np.mean(n1)
    n2_mean = np.mean(n2)
    matrix_corr = (n1 - n1_mean) * (n2 - n2_mean) / \
        np.std(n1) / np.std(n2)
    corr_coeff = np.mean(matrix_corr)
    corr_tom = np.sum(n1 * n2) / np.sqrt(np.sum(n1 * n1) * np.sum(n2 * n2))
    return matrix_corr, corr_coeff, corr_tom

def kernel_dist(n1, n2):
    numerator = np.trace(n1 * np.transpose(n2))
    denominator = np.sqrt(np.trace(n1 * np.transpose(n1)))*np.sqrt(np.trace(n2*np.transpose(n2)))
    dist = 1- (numerator /denominator)
    return dist


def batch_wise_ntk(net, dataloader, device=torch.device('cpu'), samplesize=10):
    r"""Evaluate NTK on a batch sample level.

    1) Draw a batch of images from the batch
    2) Compute gradients w.r.t to all logits for all images
    3) compute n_logits² matrix by pairwise multiplication of all grads and summing over parameters
    4) Tesselate batch_size² matrix with n_logits²-sized submatrices

    1) Choose 10 images
    2) For each image pair, compute \nabla_theta F(x, theta) and \nabla_theta F(y, theta), both in R^{p x N_logits}
       then take the product of these quantities to get an N_logitsxN_logits matrix.
       This matrix will be 10x10 since you have 10 logits.
    """
    net.eval()
    net.to(device)

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device=device), targets.to(device=device)
        # grad_outputs should be a sequence of length matching output containing the “vector” in
        # Jacobian-vector product, usually the pre-computed gradients w.r.t. each of the outputs.
        # If an output doesn’t require_grad, then the gradient can be None).

        # Sort sample
        targets, indices = torch.sort(targets)
        inputs = inputs[indices, :]
        # Compute gradients per image sample per logit

        image_grads = []
        length_last = 0
        for n in range(samplesize):
            print(n)
            output = net(inputs[n:n + 1, :, :, :]).squeeze()
            logit_dim = output.shape[0]
            D_ft = []
            for l in range(logit_dim):
                net.zero_grad()
                D_ft.append(torch.autograd.grad(output[l], net.parameters(), allow_unused=True, retain_graph=True))
            image_grads.append(D_ft)
        for p in image_grads[-1][-1]:
            if p is not None:
                length_last += p.numel()

        print(f'Gradients computed in dim (samples x logits x params) :'
              f' ({samplesize} x {logit_dim} x {length_last})')
        
        ntk_sample = []

        for ni in range(samplesize):                # image 1
            ntk_row = []
            for nj in range(samplesize):            # image 2
                ntk_entry = np.empty((logit_dim, logit_dim))
                for i in range(logit_dim):          # iterate over logits
                    for j in range(logit_dim):
                        prod = 0
                        for p1, p2 in zip(image_grads[ni][i], image_grads[nj][j]):
                            if p1 is not None and p2 is not None:
                                outer = (p1 * p2).sum().cpu().numpy()
                            if np.isfinite(outer):
                                prod += outer
                        ntk_entry[i, j] = prod

                ntk_row.append(ntk_entry)

            ntk_sample.append(ntk_row)

        # Retile to matrix
        ntk_matrix = np.block(ntk_sample)
        return ntk_matrix

def save_output(out_dir, name, **kwargs):
    """Save keys to .csv files. Function from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_evo_{args.net}_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    if not args.dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)
        print('\nResults saved to ' + fname + '.')
    else:
        print(f'Would save results to {fname}.')
