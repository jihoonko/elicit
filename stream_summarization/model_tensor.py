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
        self.values = nn.Parameter(torch.empty(self.num_values, num_feats))
        # self.bias = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            scale = .01 / (self.num_feats ** 0.5)
            nn.init.trunc_normal_(self.values, 0., scale) # -4.60517018599 / self.num_feats
            
    def forward(self, probs):
        return probs

import torch
import hashlib
import math

def deterministic_normal_samples(input_data, num_samples=16, order=0):
    num_inputs = input_data.shape[0]
    samples = torch.empty((num_inputs, num_samples), dtype=torch.float64)
    for idx in range(num_samples):
        input_strs = [f"{order}_{x.item()}_{idx + 1}" for x in input_data]
        u_values = []
        for s in input_strs:
            hash_bytes = hashlib.sha256(s.encode('utf-8')).digest()
            integer_value = int.from_bytes(hash_bytes, 'big')
            u = integer_value / ((1 << (len(hash_bytes) * 8)) - 1)
            u = min(max(u, 1e-10), 1 - 1e-10)
            u_values.append(u)
        u_tensor = torch.tensor(u_values, dtype=torch.float64)
        z = torch.erfinv(2 * u_tensor - 1) * math.sqrt(2)
        samples[:, idx] = z
    return samples
    
class ELiCiT(nn.Module):
    def __init__(self, dims, num_feats, qlevel=4):
        super().__init__()
        self.num_feats = num_feats
        self.dims = dims
        self.qlevel = qlevel
        self.order = len(self.dims)
        self.cumul = [0]
        for i in range(self.order):
            self.cumul.append(self.cumul[-1] + self.dims[i])
        self.inner = ELiCiTInner(num_feats=self.num_feats, dims=dims)
        samples = torch.cat([deterministic_normal_samples(torch.arange(self.dims[i]), num_samples=self.num_feats, order=i+1).float() for i in range(self.order)], dim=0)
        
        self.register_buffer('init_feat', samples)
        print(samples.shape)
        # self.init_feat = nn.Embedding(131072, self.num_feats)
        self.feat_net = nn.Sequential(
            nn.Linear(self.num_feats * self.order, self.num_feats * self.order),
            nn.LayerNorm(self.num_feats * self.order),
            nn.GELU(),
            nn.Linear(self.num_feats * self.order, self.num_feats * self.order),
            nn.LayerNorm(self.num_feats * self.order),
            nn.GELU(),
            nn.Linear(self.num_feats * self.order, self.num_feats * self.order),
            nn.LayerNorm(self.num_feats * self.order),
            nn.GELU(),
            nn.Linear(self.num_feats * self.order, self.num_feats * self.order)
        )
        # scale and bias to stablize the training
        self.register_buffer('which_axis', torch.cat([i * torch.ones(dim, dtype=torch.long) for i, dim in enumerate(self.dims)], dim=-1))

    def predict(self, idxs, n_neg=0):
        outs = self.forward(idxs, n_neg=0)
        return outs['vals'].clamp(min=0.)
        
    def forward(self, idxs, n_neg=0):
        device = self.which_axis.device
        eps = 1e-15
        if (n_neg > 0):
            deltas = [torch.cat([torch.zeros(1, device=device).long()] + [(torch.randint(self.dims[i]-1, (n_neg,), device=device) + 1) if (i != j) else (torch.zeros(n_neg, device=device).long()) for j in range(self.order)], dim=-1) for i in range(self.order)]
            idxs = [(idxs[i].unsqueeze(-1) + deltas[i] + self.dims[i]).view(-1) % self.dims[i] for i in range(self.order)]

        prob_one = torch.cat([self.init_feat[idxs[i] + self.cumul[i]] for i in range(self.order)], dim=-1)
        prob = self.feat_net(prob_one)
        raw_sub_probs = torch.split(torch.stack((0.5 + prob, 0.5 - prob), dim=0), self.num_feats, dim=-1) # 2 x num_samples x self.num_feats
        views = (torch.eye(self.order, dtype=torch.long) + 1).tolist()
        
        sub_probs = [raw_sub_probs[i].view(*views[i], -1, self.num_feats) for i in range(self.order)]
        probs = sub_probs[0]
        for i in range(1, self.order):
            probs = probs * sub_probs[i]
            
        vals = (self.inner.values.unsqueeze(-1) * probs.view(1 << self.order, -1, self.num_feats).permute(0, 2, 1)).sum(0)
        vals = vals.view(self.num_feats, -1).sum(0).view(-1, n_neg * self.order + 1)
        
        vals_real = vals[:, 0]
        neg_vals_real = vals[:, 1:]
        
        return {'vals': vals_real, 'neg_vals': neg_vals_real}
        