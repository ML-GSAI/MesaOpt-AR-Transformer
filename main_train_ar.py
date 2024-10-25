import argparse
import os
import shutil
import json

import numpy as np
from tqdm import tqdm

import torch

from data.generate_data import generate_ar_process

from models import create_model

from utils.tools import *
from utils import Logger


parser = argparse.ArgumentParser(description='Simulation')

# path
parser.add_argument('--data_root', default='./datasets/ar', type=str, help='data root.')
parser.add_argument('--log_dir', type=str, default='./log/ar')

# data
parser.add_argument('--dim', default=2, type=int, metavar='N', help='dim of element')
parser.add_argument('--T', default=100, type=int, metavar='N', help='length of seq')
parser.add_argument('--m', default=10000, type=int, help='train size')
parser.add_argument('--isotropic', default=0, type=int, help='if use isotropic gaussian')
parser.add_argument('--scale', default=1., type=float, help='scale')

# model training
parser.add_argument('--model', default='lsa', type=str, choices=['lsa'], help='model.')
parser.add_argument('--a', default=1., type=float, help='WKQ')
parser.add_argument('--b', default=1., type=float, help='WPV')

parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=0, type=float, help='weight decay')
parser.add_argument('--epochs', default=200, type=int, help='iteration')
parser.add_argument('--eval_freq', default=1, type=int, help='eval frequence')
parser.add_argument('--save', action='store_true', help='if save model')
parser.add_argument('--clip', action='store_true', help='if clip non-diagnol gradient')

parser.add_argument('--seed', default=1, type=int, help='random seed')

def get_embedding(batch):
    device = batch.device
    embed_list = []
    dim = batch.shape[1]
    t = batch.shape[2]
    for i in range(batch.shape[0]):
        embed = torch.zeros(size=(2*dim, t), dtype=torch.cfloat)
        embed[:dim, :] = batch[i]
        embed[dim:, 1:] = batch[i][:,:t-1]
        embed_list.append(embed)
    return torch.stack(embed_list, dim=0).to(device)

def eval_log(model, output, labels_test, logger, epoch, args):

    logger.log(f'=============Epoch {epoch}===============')

    loss_eval = 0.5 * ((output - labels_test).conj_physical() * (output - labels_test)).sum().real / labels_test.shape[0]
    logger.log('MSE-\tEval:')
    logger.log(loss_eval.item())

    with torch.no_grad():
        WKQ = model.WKQ.real[args.dim:, :args.dim]
        WPV = model.WPV.real[:, :args.dim]
        W = WKQ * WPV
        ab = torch.diag(W).mean().item()
    
    logger.log('ab:')
    logger.log(ab)
    args.log_ab.append(ab)

    if args.isotropic == 1 or args.isotropic == 0:
        ratio = (output[:, :,-1] / (labels_test[:, :,-1] + 1e-5)).mean().real.item()
        logger.log('Ratio:')
        logger.log(ratio)
        args.log_gap.append(ratio)

    if args.isotropic == 2:
        args.log_gap.append(loss_eval.item())

    if args.isotropic == 0 and epoch == args.epochs:
        with torch.no_grad():
            args.WKQ = model.WKQ.real.detach().cpu().numpy()
            args.WPV = model.WPV.real.detach().cpu().numpy()
            logger.log('WPV:')
            logger.log(args.WPV)
            logger.log('WKQ:')
            logger.log(args.WKQ)
            

def main():
    args = parser.parse_args()
    log_dir = os.path.join(args.log_dir, f'dim{args.dim}_T{args.T}_iso{args.isotropic}')

    if args.model == 'lsa':
        args.desc = f'{args.model}_a{args.a}_b{args.b}_scale{args.scale}_clip{args.clip}_lr{args.lr}_seed{args.seed}'
    else:
        raise ValueError('Invalid model name {}!'.format(args.model))
    
    log_dir = os.path.join(log_dir, args.desc)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    # logger
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    logger = Logger(os.path.join(log_dir, 'log-train.log'))

    # random seed (generate and train)
    seed_torch(args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log('Using device: {}'.format(args.device))

    train_set, test_set = generate_ar_process(args.dim, args.T, m=args.m, isotropic=args.isotropic, scale=args.scale)
    train_set, test_set = train_set.to(args.device), test_set.to(args.device)
    embed_train = get_embedding(train_set)
    embed_test = get_embedding(test_set)
    labels_test = embed_test[:, :args.dim, 2:]

    args.log_ab = []
    args.log_gap = []

    model = create_model(args).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer.zero_grad()

    epoch = 0
    output = model.forward_train(embed_test)
    
    eval_log(model, output, labels_test, logger, epoch, args)


    for epoch in (range(args.epochs)):
        optimizer.zero_grad()

        output = model.forward_train(embed_train)
        labels = embed_train[:, :args.dim, 2:]

        loss = 0.5 * ((output - labels).conj_physical() * (output - labels)).sum().real / embed_train.shape[0]
        loss = loss.sum()
        loss.backward()

        optimizer.step()

        if args.isotropic == 0 and args.clip:
            logger.log('clip the gradient!!!')
            with torch.no_grad():
                for i in range(args.dim):
                    for j in range(args.dim):
                        if j != i:
                            model.WPV[i,j].real = 0.
                            model.WPV[i,j].imag = 0.
                            model.WKQ[i+args.dim,j].real = 0.
                            model.WKQ[i+args.dim,j].imag = 0.

        if epoch == args.epochs - 1:

            output = model.forward_train(embed_test)

            eval_log(model, output, labels_test, logger, epoch + 1, args)

            logger.log(args.log_ab)
            logger.log(args.log_gap)

            ab_path = os.path.join(log_dir, 'ab.npy')
            np.save(ab_path, np.array(args.log_ab))

            gap_path = os.path.join(log_dir, 'gap.npy')
            np.save(gap_path, np.array(args.log_gap))

            if args.isotropic == 0:
                WPV_path = os.path.join(log_dir, 'WPV.npy')
                np.save(WPV_path, np.array(args.WPV))
                WKQ_path = os.path.join(log_dir, 'WKQ.npy')
                np.save(WKQ_path, np.array(args.WKQ))
          
            if args.save:
                ckpt_dir = log_dir
                ckpt_path = os.path.join(ckpt_dir, 'model.pth')
                if os.path.exists(ckpt_path):
                    shutil.rmtree(ckpt_path)
                torch.save(model.state_dict(), ckpt_path)

        elif (epoch + 1) % args.eval_freq == 0:

            output = model.forward_train(embed_test)

            eval_log(model, output, labels_test, logger, epoch + 1, args)

            model.train()

if __name__ == '__main__':
    main()