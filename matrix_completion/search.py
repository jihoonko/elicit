from torch import nn
import torch
from utils import AlgoBase
import optuna
import sys
import time
import random
import numpy as np
import copy
import time
from torch_scatter import scatter
import torch.nn.functional as F
import argparse

class ELiCiTInner(nn.Module):
    def __init__(self, num_feats, dims):
        super().__init__()
        self.num_feats = num_feats
        num_dims = len(dims)
        self.num_dims = num_dims
        self.num_values = (1 << num_dims)
        # the corresponding values of the reference states
        self.values = nn.Parameter(torch.empty(1, self.num_values, num_feats))
        self.scale = nn.Parameter(torch.empty(1))
        with torch.no_grad():
            scale = 1. / (self.num_feats ** 0.5)
            nn.init.normal_(self.values, 0, scale)
            nn.init.normal_(self.scale, 0, scale)
            
    def forward(self, probs):
        return probs
        
class ELiCiT(nn.Module):
    def __init__(self, dims, num_feats, qlevel=4):
        super().__init__()
        self.num_feats = num_feats
        self.dims = dims
        self.qlevel = qlevel
        self.order = len(self.dims)
        self.inner = ELiCiTInner(num_feats=num_feats, dims = dims)
        
        # (implicit) features of indices
        self.feats = nn.Parameter(torch.empty(sum(self.dims), self.num_feats))
        self.ifeats = nn.Parameter(torch.empty(sum(self.dims), self.num_feats))
        
        # (implicit) candidates
        self.keys = nn.Parameter(torch.empty(self.order, self.num_feats, (1 << self.qlevel)))
        self.ikeys = nn.Parameter(torch.empty(self.order, self.num_feats, (1 << self.qlevel)))
        
        self.register_buffer('which_axis', torch.cat([i * torch.ones(dim, dtype=torch.long) for i, dim in enumerate(self.dims)], dim=-1))
        
        nn.init.trunc_normal_(self.feats) 
        nn.init.trunc_normal_(self.ifeats)
        nn.init.trunc_normal_(self.keys) 
        nn.init.trunc_normal_(self.ikeys)
        
    def prepare_inputs(self):
        device = self.feats.data.device
        target_keys = torch.sigmoid(self.keys[self.which_axis])
        target_feats = torch.sigmoid(self.feats)
        # find the target candidate
        assigned = torch.argmin((target_feats.unsqueeze(-1) - target_keys).abs(), dim=-1).unsqueeze(-1)
        # straight-through technique
        probs = -(target_feats.detach()) + (torch.gather(target_keys, -1, assigned).squeeze(-1) + target_feats)
        return torch.stack((probs, 1. - probs), dim=0)

    def register_idxs(self, idxs):
        # for considering implicit feedback
        self.iidxs = idxs
        self.cnts_first = ((torch.bincount(self.iidxs[0], minlength=self.dims[0]) + 1e-12) ** 0.5)
        self.cnts_second = ((torch.bincount(self.iidxs[-1], minlength=self.dims[-1]) + 1e-12) ** 0.5)
        
    def prepare_iinputs(self):
        device = self.feats.data.device
        target_keys = torch.sigmoid(self.ikeys[self.which_axis])
        target_feats = torch.sigmoid(self.ifeats)
        # find the target candidate
        assigned = torch.argmin((target_feats.unsqueeze(-1) - target_keys).abs(), dim=-1).unsqueeze(-1) # sum_dims x num_feats x 1 - constant < (1 << qlevel)
        # straight-through technique
        probs = -(target_feats.detach()) + (torch.gather(target_keys, -1, assigned).squeeze(-1) + target_feats)
        # compute the final implicit feature
        iprobs_first = scatter(probs[self.dims[0]:][self.iidxs[-1]] - 0.5, self.iidxs[0], dim=0, dim_size=self.dims[0], reduce='sum') / (self.cnts_first.unsqueeze(-1))
        iprobs_second = scatter(probs[:self.dims[0]][self.iidxs[0]] - 0.5, self.iidxs[-1], dim=0, dim_size=self.dims[-1], reduce='sum') / (self.cnts_second.unsqueeze(-1))
        iprobs = torch.cat((iprobs_first, iprobs_second), dim=0)
        return torch.stack((iprobs, -iprobs), dim=0)
        
    def forward(self, idxs):
        device = self.feats[0].device
        eps = 1e-15
        # compute weights of the reference states
        feats = torch.split(self.prepare_inputs() + self.prepare_iinputs(), self.dims, dim=1) # 2 x dim x feat
        views = (torch.eye(self.order, dtype=torch.long) + 1).tolist()
        sub_probs = [feats[i][:, idx_dim].view(*views[i], -1, self.num_feats) for i, idx_dim in enumerate(idxs)]
        probs = sub_probs[0]
        for i in range(1, self.order): probs = probs * sub_probs[i]
        
        vals = (self.inner.values.unsqueeze(-1) * probs.view(1 << self.order, -1, self.num_feats).permute(0, 2, 1)).sum(1)
        vals = vals.view(1, self.num_feats, -1).sum(1)
        preds = (self.inner.scale.unsqueeze(-1) * vals).sum(0) # hdim x batch_size
        return preds

class Net(nn.Module):
    def __init__(self, dims, num_feats, global_mean=0.):
        super().__init__()
        self.inner = ELiCiT(dims, num_feats)
        self._mean = global_mean
    
    def forward(self, row_indices, col_indices):
        return self.inner((row_indices, col_indices)) + self._mean
        
    def register_indices(self, row_indices, col_indices):
        self.inner.register_idxs((row_indices, col_indices))
        
class CompletionModel(AlgoBase):
    def __init__(self, dataset_name, is_test, **kwargs):
        super().__init__(dataset_name, is_test)
        self.budget = kwargs.get('budget', 100)
        self.device = kwargs.get('device', 0)
        self.batch_size = kwargs.get('batch_size', (1 << 20))
        self.lr_and_regs = kwargs.get('lr_and_regs', None)
        
    def compute_loss(self, preds, gt, metric='sse'):
        if metric == 'sse':
            return ((preds - gt) ** 2).sum()
        elif metric == 'rmse':
            return ((((preds - gt) ** 2).mean()) ** 0.5)
                
    def fit(self, trainset, valset):
        self.dims = (self.n_users, self.n_items)
        print(self.dims)
        train_users, train_items, train_ratings = map(np.array, zip(*trainset))
        val_users, val_items, val_ratings = map(np.array, zip(*valset))
        num_feats = int(((self.dims[0] + self.dims[1]) * self.budget - 1) / (68 + ((self.dims[0] + self.dims[1]) / 4.)))
        
        wd1, wd2, wd3, lr, lr2 = self.lr_and_regs
        self.model = Net(dims=self.dims, num_feats = num_feats, global_mean = train_ratings.mean()).to(self.device)
        self.model.register_indices(torch.LongTensor(train_users).to(self.device), torch.LongTensor(train_items).to(self.device))
        self.optimizer = torch.optim.Adam([{'params': list(self.model.inner.inner.parameters()), 'lr': lr, 'weight_decay': wd1},
                                           {'params': [self.model.inner.feats, self.model.inner.ifeats], 'lr': lr2, 'weight_decay': 0},
                                           {'params': [self.model.inner.keys, self.model.inner.ikeys], 'lr': lr2, 'weight_decay': 0}], lr=lr) # features and candidates: compute l2 regularization later
        
        patience, best_score = 0, 1e10
        best_params = copy.deepcopy(self.model.state_dict())
        for epoch in range(5000):
            self.model.train()
            self.model.zero_grad()
            for _idx in range(0, len(trainset), self.batch_size):
                _until = min(len(trainset), _idx + self.batch_size) 
                loss = self.compute_loss(self.model(train_users[_idx:_until], train_items[_idx:_until]), torch.DoubleTensor(train_ratings[_idx:_until]).to(self.device), metric='sse')
                loss = loss + 0.5 * wd2 * ( ((F.sigmoid(self.model.inner.feats) - 0.5) ** 2).sum() + ((F.sigmoid(self.model.inner.ifeats) - 0.5) ** 2).sum() ) # l2 regularization for the features
                loss = loss + 0.5 * wd3 * ( ((F.sigmoid(self.model.inner.keys) - 0.5) ** 2).sum() + ((F.sigmoid(self.model.inner.ikeys) - 0.5) ** 2).sum() ) # l2 regularization for the candidates
                loss.backward()
            self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                curr_score = self.compute_loss(self.model(val_users, val_items), torch.DoubleTensor(val_ratings).to(self.device), metric='rmse').item()
            if best_score > curr_score:
                best_score = curr_score
                patience = 0
                best_params = copy.deepcopy(self.model.state_dict())
            else:
                patience += 1
            elapsed = (time.time() - self.start_time)
            if patience == 100:
                break
        with torch.no_grad():
            self.model.load_state_dict(best_params)
        return self

    def estimate_batch(self, _us, _is):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.LongTensor(_us).to(self.device), torch.LongTensor(_is).to(self.device))
        return preds

def objective(trial):
    wd1 = trial.suggest_float('lamb1', 1e-4, 1e4, log=True) # regularization coefficient of the values of the reference states and scale
    wd2 = trial.suggest_float('lamb2', 1e-4, 1e4, log=True) # regularization coefficient of the (implicit) features
    wd3 = trial.suggest_float('lamb3', 1e-4, 1e4, log=True) # regularization coefficient of the (implicit) candidates ('keys' and 'ikeys' in this code)
    lr1 = trial.suggest_float('lr1', 1e-4, 1e-0, log=True) # learning rate of the values of the reference states and scale
    lr2 = trial.suggest_float('lr2', 1e-4, 1e-0, log=True) # learning rate of the (implicit) features and the candidates
    algo = CompletionModel(dataset_name=args.dataset_name, is_test=0, budget=args.budget, lr_and_regs = (wd1, wd2, wd3, lr1, lr2), device=args.gpu)
    result = algo.run()
    return result['rmse']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default='douban', type=str, help="target matrix for completion")
    parser.add_argument("--budget", default=32., type=float, help="target budget / (num_rows + num_cols)")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id for searching the hyperparameters")
    args = parser.parse_args()
    
    study = optuna.create_study(study_name='search', sampler=optuna.samplers.TPESampler(multivariate=True))
    study.optimize(objective, n_trials=200)