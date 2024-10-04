import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class ELiCiTInner(nn.Module):
    def __init__(self, num_feats, dims, eps=1e-12):
        super().__init__()
        self.num_feats = num_feats
        num_dims = len(dims)
        self.num_dims = num_dims
        self.num_values = (1 << num_dims)
        # the corresponding values of the reference states
        self.values = nn.Parameter(torch.empty(1, self.num_values, num_feats))
        with torch.no_grad():
            scale = .01 / (self.num_feats ** 0.5)
            nn.init.trunc_normal_(self.values, 0, scale)
            
    def forward(self, probs):
        return probs
        
class ELiCiT(nn.Module):
    def __init__(self, dims, num_feats, qlevel=4):
        super().__init__()
        self.num_feats = num_feats
        self.dims = dims
        self.qlevel = qlevel
        self.order = len(self.dims)
        # for handling the corresponding values of the reference states
        self.inner = ELiCiTInner(num_feats=num_feats, dims=dims)
        # features of the indices
        self.feats = nn.Parameter(torch.rand(sum(self.dims), self.num_feats))
        # candidate values
        if qlevel != -1:
            self.candidates = nn.Parameter(torch.linspace(1. / (1 << (self.qlevel + 1)), 1. - 1. / (1 << (self.qlevel + 1)), steps=(1 << self.qlevel)).view(1, 1, (1 << self.qlevel)).repeat(self.order, self.num_feats, 1))
        else:
            self.candidates = nn.Parameter(torch.zeros(1))

        # scale and bias to stablize the training
        self.scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer('which_axis', torch.cat([i * torch.ones(dim, dtype=torch.long) for i, dim in enumerate(self.dims)], dim=-1))
        
    def prepare_inputs(self):
        device = self.feats.data.device
        if self.qlevel == -1:
            probs = self.feats
        else:
            target_keys = self.candidates[self.which_axis]
            target_feats = self.feats
            # find the target candidate
            assigned = torch.argmin((target_feats.unsqueeze(-1) - target_keys).abs(), dim=-1).unsqueeze(-1)
            # straight-through technique
            probs = -(target_feats.detach()) + (torch.gather(target_keys, -1, assigned).squeeze(-1) + target_feats)
        return torch.stack((probs, 1. - probs), dim=0)

    def forward(self, idxs):
        device = self.feats[0].device
        eps = 1e-15
        feats = torch.split(self.prepare_inputs(), self.dims, dim=1)
        # compute weights of the reference states
        views = (torch.eye(self.order, dtype=torch.long) + 1).tolist()
        sub_probs = [feats[i][:, idx_dim].view(*views[i], -1, self.num_feats) for i, idx_dim in enumerate(idxs)]
        probs = sub_probs[0]
        for i in range(1, self.order): probs = probs * sub_probs[i]
        # compute weighted sum
        vals = (self.inner.values.unsqueeze(-1) * probs.view(1 << self.order, -1, self.num_feats).permute(0, 2, 1)).sum(1)
        vals = vals.view(1, 2, self.num_feats // 2, -1).sum(-2)
        # use advanced reduce function
        preds = (vals[:, 0] * F.tanh(vals[:, 1])).sum(0) * torch.exp(self.scale) + self.bias
        return preds
