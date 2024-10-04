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

class Trainer(nn.Module):
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
    
    def train_iteration(self, optimizer, max_iter=500):
        device = self.model.module.feats.device
        start_time = time.time()
        best_score = -1e10
        patience = 0
        for epoch_cnt in range(max_iter):
            _ = self.handle_model(optimizer=optimizer)
            curr_fitness = self.fitness(self.handle_model(optimizer=None))
            if best_score + 0.0001 < curr_fitness:
                best_score = curr_fitness
                torch.save(self.model.module.state_dict(), 'temp.pkt')
                patience = 0
            else:
                patience += 1
               
            elapsed = (time.time() - start_time)

            print(f'epoch {epoch_cnt} | score {curr_fitness:.6f} best {best_score:.6f} patience {patience} time {elapsed:.3f}')
            
            if patience == 10:
                break
        
        return best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default='./uber.npy', type=str, help="target tensor for compression")
    parser.add_argument("--output-path", default='./output.elicit', type=str, help="output path of the compressed data")
    parser.add_argument("--num-features", default=10, type=int, help="number of features")
    parser.add_argument("--lr1", type=float, default=1e-3, help="learning rate of the values of the reference states")
    parser.add_argument("--lr2", type=float, default=1e-2, help="learning rate of the rest of the parameters (including the features and the candidates)")
    parser.add_argument("--gpus", action="store", nargs='+', default=[0], type=int, help="GPU ids for running the compression process")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.num_features = (args.num_features // 2) * 2
    
    # train
    trainer = Trainer(args.input_path, args.num_features, gpus=args.gpus).to(args.gpus[0]).double()
    optimizer = torch.optim.Adam([{'params': trainer.model.module.inner.parameters(), 'lr': args.lr1},
                                  {'params': [trainer.model.module.feats, trainer.model.module.candidates, trainer.model.module.scale, trainer.model.module.bias], 'lr': args.lr2}], lr=args.lr1)
    
    print('final fitness:', trainer.train_iteration(optimizer))

    # save the compressed model
    best_params = torch.load('temp.pkt', map_location='cpu')
    os.remove('temp.pkt')
    dims = trainer.dims
    order = trainer.order
    nfeats = args.num_features
    
    header = np.array((order, *dims, nfeats), dtype='<u2').tobytes()
    except_feats = torch.cat((best_params['candidates'].view(-1), best_params['inner.values'].view(-1), best_params['scale'].view(-1), best_params['bias'].view(-1)), dim=-1).numpy().tobytes()
    feats = ((best_params['candidates'][best_params['which_axis']] - best_params['feats'].unsqueeze(-1)).abs().argmin(dim=-1).view(-1, 2) * torch.LongTensor([16, 1])).sum(-1).byte().numpy().tobytes()
    np.save(args.output_path, header + except_feats + feats)
    os.rename(args.output_path + '.npy', args.output_path)