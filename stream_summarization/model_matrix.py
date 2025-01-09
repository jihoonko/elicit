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

def deterministic_normal_samples(input_data, num_samples=16):
    num_inputs = input_data.shape[0]
    samples = torch.empty((num_inputs, num_samples), dtype=torch.float64)
    for idx in range(num_samples):
        input_strs = [f"{(x // 65536).item() + 1}_{(x % 65536).item()}_{idx + 1}" for x in input_data]
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
        # for handling the corresponding values of the reference states
        self.inner = ELiCiTInner(num_feats=self.num_feats, dims=dims)
        input_data = torch.arange(0, 2**17, dtype=torch.int64) 
        samples = deterministic_normal_samples(input_data, num_samples=self.num_feats).float()
        self.register_buffer('init_feat', samples)
        # self.init_feat = nn.Embedding(131072, self.num_feats)
        self.feat_net = nn.Sequential(
            nn.Linear(self.num_feats * 2, self.num_feats * 2),
            nn.LayerNorm(self.num_feats * 2),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(self.num_feats * 2, self.num_feats * 2),
            nn.LayerNorm(self.num_feats * 2),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(self.num_feats * 2, self.num_feats * 2),
            nn.LayerNorm(self.num_feats * 2),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(self.num_feats * 2, self.num_feats * 2)
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
            delta0 = torch.cat((torch.zeros(1, device=device).long(), torch.randperm(65535, device=device)[:n_neg] + 1, torch.zeros(n_neg, device=device).long()), dim=-1)
            delta1 = torch.cat((torch.zeros(n_neg + 1, device=device).long(), torch.randperm(65535, device=device)[:n_neg] + 1), dim=-1)
            idxs[0] = (idxs[0].unsqueeze(-1) + delta0 + 65536).view(-1) % 65536
            idxs[1] = (idxs[1].unsqueeze(-1) + delta1 + 65536).view(-1) % 65536
            
        prob_one = torch.cat((self.init_feat[idxs[0]], self.init_feat[idxs[1]+65536]), dim=-1)
        prob = self.feat_net(prob_one)
        prob1, prob2 = torch.split(prob, self.num_feats, dim=-1)
        probs = torch.stack((0.5 + prob1, 0.5 - prob1), dim=0).unsqueeze(1) * torch.stack((0.5 + prob2, 0.5 - prob2), dim=0).unsqueeze(0)
        vals = (self.inner.values.unsqueeze(-1) * probs.view(1 << self.order, -1, self.num_feats).permute(0, 2, 1)).sum(0)

        wprob = vals.permute(1, 0).reshape(-1, n_neg + n_neg + 1, self.num_feats)
        vals = vals.view(self.num_feats, -1).sum(0).view(-1, n_neg + n_neg + 1)
        # (vals[0] * vals[1])
        
        vals_real = vals[:, 0]
        feats = wprob[:, 0]
        neg_vals_real = vals[:, 1:]
        neg_feats = wprob[:, 1:]
        
        return {'vals': vals_real,
                'feats': feats,
                'neg_vals': neg_vals_real,
                'neg_feats': neg_feats}
        