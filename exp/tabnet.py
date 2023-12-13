import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input, device=torch.device("cuda")):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(
            start=1, end=number_of_logits + 1, device=device, step=1, dtype=input.dtype
        ).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        self.output = torch.max(torch.zeros_like(input), input - taus)
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output

    def backward(self, grad_output):
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
        return self.grad_input


def initialize_non_glu(module, inp_dim, out_dim):
    gain = np.sqrt((inp_dim + out_dim) / np.sqrt(4 * inp_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain)


class GBN(nn.Module):
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        chunk = torch.chunk(x, max(1, x.size(0) // self.vbs), 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        return x[:, : self.od] * torch.sigmoid(x[:, self.od :])


class FeatureTransformer(nn.Module):
    def __init__(
        self, inp_dim, out_dim, shared, n_ind, vbs=128, device=torch.device("cuda")
    ):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = torch.sqrt(torch.tensor([0.5], device=device))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class AttentionTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, relax, vbs=128, device=torch.device("cuda")):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
        self.bn = GBN(out_dim, vbs=vbs)
        #         self.smax = Sparsemax()
        self.r = torch.tensor([relax], device=device)

    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = torch.sigmoid(a * priors)
        priors = priors * (self.r - mask)
        return mask


class DecisionStep(nn.Module):
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, loss


class TabNet(nn.Module):
    def __init__(
        self,
        inp_dim,
        final_out_dim=1,
        n_d=256,
        n_a=256,
        n_shared=3,
        n_ind=4,
        n_steps=4,
        relax=1.2,
        vbs=512,
        cat_map=None,
    ):
        super().__init__()
        self.cat_map = cat_map
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
        else:
            self.shared = None
        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind)
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            self.steps.append(
                DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs)
            )
        self.fc = nn.Linear(n_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d

    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d :]
        loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:, : self.n_d])
            x_a = x_te[:, self.n_d :]
            loss += l
        return self.fc(out)

class TabNetEmb(nn.Module):
    def __init__(
        self,
        inp_dim,
        final_out_dim=1,
        n_d=256,
        n_a=256,
        n_shared=3,
        n_ind=4,
        n_steps=4,
        relax=1.2,
        vbs=512,
        cat_map=None,
        embs_dim = 8
    ):
        super().__init__()
        self.cat_map = cat_map
        self.emb_layers = nn.ModuleList()
        self.cats = []
        inp_dim = 0
        for cat in cat_map:
            i, j = cat_map[cat]
            if isinstance(embs_dim, dict):
                self.emb_layers.append(nn.Linear(j - i + 1, embs_dim.get(cat, embs_dim["_default_"]), bias=False))
                inp_dim += embs_dim.get(cat, embs_dim["_default_"])
            else:
                self.emb_layers.append(nn.Linear(j - i + 1, embs_dim, bias=False))
                inp_dim += embs_dim
            self.cats.append((i, j))

        for num in range(self.cats[0][0]):
            self.emb_layers.append(nn.Linear(1, 1, bias=False))
            self.cats.append((num, num))
        inp_dim += self.cats[0][0]

        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
        else:
            self.shared = None
        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind)
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            self.steps.append(
                DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs)
            )
        self.fc = nn.Linear(n_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d

    def forward(self, x):

        enc = []
        for k, emb in enumerate(self.emb_layers):
            i, j = self.cats[k]
            enc.append(emb(x[:, i:j+1]))
        x = torch.cat(enc, dim=1)
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d :]
        loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:, : self.n_d])
            x_a = x_te[:, self.n_d :]
            loss += l
        return self.fc(out)
