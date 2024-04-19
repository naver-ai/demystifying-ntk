"""
demystifying-ntk
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

import train_utils
import logging
import argparse
import torch.nn as nn

from torch.autograd import Variable


def train(train_queue, model, criterion, optimizer):
  objs = train_utils.AvgrageMeter()
  top1 = train_utils.AvgrageMeter()
  #top5 = train_utils.AvgrageMeter()
  model.train()

  iterations = len(train_queue)
  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda()

    optimizer.zero_grad()
    logits, _ = model(input)
    loss = criterion(logits, target)

    loss.backward()

    optimizer.step()

    prec1 = train_utils.accuracy(logits, target)
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1, n)
    #top5.update(prec5.data.item(), n)

    #if step % args.report_freq == 0 or step == iterations-1:
    #  logging.info('train (%03d/%d) %e %f %f', step, iterations, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg
