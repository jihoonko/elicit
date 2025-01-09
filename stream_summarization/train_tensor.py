from scipy.sparse import coo_matrix
from model_tensor import ELiCiT
import torch
torch.set_num_threads(4)
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
import optuna
from itertools import chain
from torch_scatter import scatter
import pickle
    
def load_npz(path):
    file = np.load(path)
    support_y = file['support_y']
    support_x = file['support_x']
    query_x = file['query_x']
    query_y = file['query_y']
    # if support_y.min() <= 0.0001 or query_y.min() <= 0.0001:
    #     print('find illegal frequency: zero!')
    #     exit()
    return support_x, support_y, query_x, query_y
    
def load_all_npz_in_dir(dir_path):
    file_name_list = os.listdir(dir_path)
    file_path_list = [os.path.join(dir_path, '0.npz')] #  for file_name in file_name_list]
    support_x_list = []
    support_y_list = []
    query_x_list = []
    query_y_list = []
    for file_path in file_path_list:
        support_x, support_y, query_x, query_y = load_npz(file_path)
        support_x_list.append(support_x)
        support_y_list.append(support_y)
        query_x_list.append(query_x)
        query_y_list.append(query_y)
        
    return (support_x_list[0][..., :16].astype('int64') * (2 ** (15 - np.arange(16)))).sum(-1), (support_x_list[0][..., 16:].astype('int64') * (2 ** (15 - np.arange(16)))).sum(-1), support_y_list[0], (query_x_list[0][..., :16].astype('int64') * (2 ** (15 - np.arange(16)))).sum(-1), (query_x_list[0][..., 16:].astype('int64') * (2 ** (15 - np.arange(16)))).sum(-1), query_y_list[0]

class Trainer(nn.Module):
    def __init__(self, filename, num_feats, batch_size=524288, qlevel=4, gpus=[0, 1, 2, 3]):
        super().__init__()
        self.filename = filename
        self.num_feats = num_feats
        self.qlevel = qlevel
        self.load(filename)
        self.model = ELiCiT(dims=(65536, 65536, self.train_vals.shape[0]), num_feats=num_feats, qlevel=qlevel)
        # self.model = nn.DataParallel(self.model, device_ids=gpus)
        self.batch_size = batch_size
        self.record = False
        
    def load(self, filename):
        train_srcs, train_dsts, train_vals, test_srcs, test_dsts, test_vals = load_all_npz_in_dir('./' + args.input_path)
        # target = np.load(filename)
        self.register_buffer('train_srcs', torch.from_numpy(train_srcs))
        self.register_buffer('train_dsts', torch.from_numpy(train_dsts))
        self.register_buffer('train_vals', torch.from_numpy(train_vals).contiguous().view(-1))
        self.register_buffer('test_srcs', torch.from_numpy(test_srcs))
        self.register_buffer('test_dsts', torch.from_numpy(test_dsts))
        self.register_buffer('test_vals', torch.from_numpy(test_vals).contiguous().view(-1))
        
    def eval_model(self):
        device = self.model.which_axis.device
        eps = 1e-15
        total_aae, total_are = 0., 0.
        n_samples = self.test_vals.shape[0]
        
        batch_size = (n_samples // ((n_samples + self.batch_size - 1) // self.batch_size)) + 1
        len_loader = (n_samples + batch_size - 1) // batch_size

        self.model.eval()
        vals = self.model.predict([self.test_srcs, self.test_dsts, torch.arange(self.test_srcs.shape[0]).to(device)], n_neg=0)
        partial_aae = (vals - self.test_vals).abs().sum()
        partial_are = ((vals - self.test_vals) / self.test_vals).abs().sum()
        total_aae += partial_aae.item()
        total_are += partial_are.item()

        details_aae = ' '.join(map(str, scatter((vals - self.test_vals).abs(), torch.log2(self.test_vals).long(), dim=0, reduce='mean').tolist()))
        details_are = ' '.join(map(str, scatter((vals - self.test_vals).abs() / self.test_vals, torch.log2(self.test_vals).long(), dim=0, reduce='mean').tolist()))
        
        total_aae /= n_samples
        total_are /= n_samples

        return total_aae, total_are, 0., details_aae, details_are
        
    def handle_model(self, optimizer=None, save=False):
        device = self.model.which_axis.device
        eps = 1e-15
        optimizer.zero_grad()
        total_aae, total_are = 0., 0.

        samples = torch.LongTensor(list(chain.from_iterable(self.bins)))
        binidx = torch.cat([(i * torch.ones(len(self.bins[i]), dtype=torch.long)) for i in range(len(self.bins))], dim=-1).to(device)
        train_ws = torch.ones_like(binidx) # torch.cat([((self.sz[i] / (len(self.bins[i]) + 1e-6)) * torch.ones(len(self.bins[i]), dtype=torch.float)) for i in range(len(self.bins))], dim=-1).to(device)
        n_samples = samples.shape[0]
        
        self.model.train()
        target_indices = samples

        target_srcs = self.train_srcs[target_indices]
        target_dsts = self.train_dsts[target_indices]
        target_objs = self.train_obj[target_indices]
        n_neg = (65536 // (self.model.order * n_samples))
        outs = self.model([target_srcs, target_dsts, target_indices.to(device)], n_neg=n_neg)
        vals, neg_vals = outs['vals'], outs['neg_vals']
        
        loss = 0.
        loss = ((vals - target_objs).abs() + ((vals - target_objs).abs() / target_objs)).sum()
        loss = loss + neg_vals.clamp(min=args.w1).mean(-1).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.)
        optimizer.step()

        preds = vals
        gt = self.train_obj[target_indices]
        partial_aae = (preds - gt).abs().sum()
        partial_are = ((preds - gt) / gt).abs().sum()
        total_aae += partial_aae.item()
        total_are += partial_are.item()

        total_aae /= n_samples
        total_are /= n_samples

        return vals.mean(), neg_vals.mean()
        
    
    def train_iteration(self, optimizer, cache_size=0, neg_cache_size=0):
        device = self.model.which_axis.device
        self.train_obj = torch.zeros_like(self.train_vals)
        self.train_weight = torch.zeros_like(self.train_vals)
        self.pen_weights = {name: [torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)] for name, p in self.model.named_parameters()}
        self.filter = torch.zeros(65536, dtype=torch.bool).to(device)
        start_time = time.time()
        best_score = -1e10
        patience = 0
        
        prev_value_sum = 0.
        max_val = 0.
        self.bins = []
        self.sz = []
        # self.neg_cache = [torch.zeros(0, dtype=torch.long).to(device), torch.zeros(0, dtype=torch.long).to(device)]
        self.total_sz = 0
        print(self.train_vals.shape, self.test_vals.shape, self.train_vals.sum(), self.test_vals.sum())
        hi_cnt = 0
        
        global_mean = 0.
        zsum = 0.
        ok_cnt = 0
        for epoch_cnt in range(0, self.train_vals.shape[0], 1):
            self.model.eval()
            
            with torch.no_grad():
                cache_hit = -1
                self.train_obj[epoch_cnt] = (self.train_vals[epoch_cnt].item())
                self.train_weight[epoch_cnt] = 1.
                
                max_val = max(self.train_obj[epoch_cnt].item(), max_val)
                bin_idx = int(np.log2(self.train_obj[epoch_cnt].item()))
                while len(self.bins) <= bin_idx:
                    self.bins.append([])
                    self.sz.append(0)

                while self.total_sz >= cache_size:
                    proportion = (-1, -1)
                    for i in range(len(self.bins)):
                        proportion = max(proportion, ((len(self.bins[i]) - 1) / ((self.sz[i] + 1e-6) ** 0.5), i))
                    target_bin = proportion[1]
                    target_sample = torch.randperm(len(self.bins[target_bin]))[0].item()
                    # eliminated.append(self.bins[target_bin][target_sample])
                    if target_sample == len(self.bins[target_bin])-1:
                        self.bins[target_bin] = self.bins[target_bin][:target_sample]
                    else:
                        self.bins[target_bin] = self.bins[target_bin][:target_sample] + self.bins[target_bin][target_sample+1:]
                    self.total_sz -= 1
                    
                self.sz[bin_idx] += 1
                self.bins[bin_idx].append(epoch_cnt)
                self.total_sz += 1

            self.model.train()
            if epoch_cnt % 1 == 0:
                train_aae, train_are = self.handle_model(optimizer=optimizer)
                
            self.record = False
            if epoch_cnt % 100 == 0:

                samples = torch.LongTensor(list(chain.from_iterable(self.bins)))
                test_aae, test_are, test_zero, details_aae, details_are = self.eval_model()
                args.fp.write(f'epoch {epoch_cnt} | AAE = {test_aae} ARE = {test_are} TIME = {time.time() - start_time}\n')
                
                prev_value_sum = 0.
                args.fp.flush()

        test_aae, test_are, test_zero, details_aae, details_are = self.eval_model()
        args.fp.write(f'LAST | AAE = {test_aae} ARE = {test_are} TIME = {time.time() - start_time}\n')
        prev_value_sum = 0.
        args.fp.flush()
        return test_aae

def objective():
    # train
    args.num_features = 16
    learning_rate = 1e-3
    args.w1 = 0.5
    trainer = Trainer(args.input_path, args.num_features, gpus=args.gpus).to(args.gpus[0])

    args.fp = open(f'{args.input_path}.log', 'w')
    optimizer = torch.optim.SGD([
        {'params': trainer.model.feat_net.parameters(), 'lr': learning_rate, 'weight_decay': args.reg},
        {'params': trainer.model.inner.parameters(), 'lr': learning_rate, 'weight_decay': 0.0}],
        lr=learning_rate)

    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(total_params)
    cache_size = ((128 * 1024) - (total_params * 4)) // 8
    ret = trainer.train_iteration(optimizer, cache_size=cache_size)
    args.fp.close()
    torch.save(trainer.model.state_dict(), 'compressed_output')
    return ret
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default='example', type=str, help="target tensor for compression")
    parser.add_argument("--reg", type=float, default=0.01, help="regularization coefficient")
    parser.add_argument("--gpus", action="store", nargs='+', default=[0], type=int, help="GPU ids for running the compression process")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    objective()
    