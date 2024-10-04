from scipy.sparse import coo_matrix
from model import ELiCiT
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
import numpy as np
import random

import tqdm
import time
import sys
import argparse
import shutil, os

class Evaluator(nn.Module):
    def __init__(self, filename, num_feats, batch_size=524288, qlevel=4, gpus=[0, 1, 2, 3]):
        super().__init__()
        self.filename = filename
        self.num_feats = num_feats
        self.qlevel = qlevel
        self.load(filename)
        self.model = ELiCiT(dims=self.dims, num_feats=num_feats, qlevel=qlevel)
        self.model = nn.DataParallel(self.model, device_ids=gpus)
        self.batch_size = batch_size
        
    def load(self, filename):
        target = np.load(filename)
        self.register_buffer('target', torch.from_numpy(target).contiguous().view(-1))
        self.square_sum = (target ** 2).sum()
        self.dims = target.shape
        self.order = len(self.dims)
        
    def handle_model(self, optimizer=None):
        device = self.model.module.feats.device
        eps = 1e-15
        if optimizer is not None:
            optimizer.zero_grad()
        offset_dim_full = [1 for _ in range(self.order + 1)]
        for i in range(self.order - 1, -1, -1):
            offset_dim_full[i] = offset_dim_full[i+1] * self.dims[i]
        total_loss = 0.
        # minibatch update (random minibatching)
        full_target_indices = torch.randperm(offset_dim_full[0])
        batch_size = (offset_dim_full[0] // ((offset_dim_full[0] + self.batch_size - 1) // self.batch_size)) + 1
        len_loader = (offset_dim_full[0] + batch_size - 1) // batch_size
        for i in range(0, offset_dim_full[0], batch_size):
            if optimizer is not None:
                optimizer.zero_grad()
            bsize = min(batch_size, offset_dim_full[0] - i)
            target_indices = full_target_indices[i + torch.arange(bsize)]
            idxs = [(target_indices % offset_dim_full[j]) // offset_dim_full[j+1] for j in range(self.order)]
            vals = self.model(idxs)

            partial_loss = ((vals - self.target[target_indices]) ** 2).sum()
            
            if optimizer is not None:
                partial_loss.backward()
                optimizer.step()
                
            total_loss += partial_loss.item()
        return total_loss
    
    def fitness(self, loss):
        return 1. - ((loss / self.square_sum) ** 0.5)
    
    def compute_fitness(self):
        return self.fitness(self.handle_model(optimizer=None))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-path", default='./uber.npy', type=str, help="original tensor to measure the fitness")
    parser.add_argument("--compressed-path", default='./example_uber_40.elicit', type=str, help="compressed tensor to measure the fitness")
    parser.add_argument("--gpus", action="store", nargs='+', default=[0,1], type=int, help="GPU ids for running the evaluation process")
    
    args = parser.parse_args()
    
    # read basic information (order, shape, num_features)
    raw_compressed_file = np.load(args.compressed_path).item()
    order = np.frombuffer(raw_compressed_file[0:2], dtype='<u2').item()
    shape = np.frombuffer(raw_compressed_file[2:2*(order+1)], dtype='<u2').tolist()
    nfeats = np.frombuffer(raw_compressed_file[2*(order+1):2*(order+2)], dtype='<u2').item()
    print('order:', order, '| shape:', tuple(shape), '| num_features:', nfeats)
    
    # load evaluator
    evaluator = Evaluator(args.original_path, nfeats, gpus=args.gpus).to(args.gpus[0]).double()
    with torch.no_grad():
        # restore candidates
        pointer = 2 * (order + 2)
        candidate_size = np.prod(evaluator.model.module.candidates.shape)
        candidates = torch.from_numpy(np.frombuffer(raw_compressed_file[pointer:pointer + 8 * candidate_size], dtype=np.float64)).view(evaluator.model.module.candidates.shape)
        pointer += 8 * candidate_size
        evaluator.model.module.candidates.copy_(candidates)
        
        # restore values
        value_size = np.prod(evaluator.model.module.inner.values.shape)
        values = torch.from_numpy(np.frombuffer(raw_compressed_file[pointer:pointer + 8 * value_size], dtype=np.float64)).view(evaluator.model.module.inner.values.shape)
        pointer += 8 * value_size
        evaluator.model.module.inner.values.copy_(values)
            
        # restore scale and bias
        evaluator.model.module.scale.fill_(np.frombuffer(raw_compressed_file[pointer:pointer+8], dtype=np.float64).item())
        evaluator.model.module.bias.fill_(np.frombuffer(raw_compressed_file[pointer+8:pointer+16], dtype=np.float64).item())
        pointer += 16

        # restore feats
        raw_feats = torch.from_numpy(np.frombuffer(raw_compressed_file[pointer:], dtype='<u1')).long()
        feats = torch.stack((raw_feats // 16, raw_feats % 16), dim=-1).view(-1)
        real_feats = candidates[evaluator.model.module.which_axis.data.cpu()].view(-1, 16)[torch.arange(feats.shape[0]), feats].view(evaluator.model.module.feats.shape)
        evaluator.model.module.feats.copy_(real_feats)

    # result
    print('final fitness:', evaluator.compute_fitness())