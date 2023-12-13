import sys

import numpy as np

sys.path.append("..")
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.tabnet import TabNet, TabNetEmb
from exp.utils import *


class FCNN(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.b2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.b3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 32)
        self.b4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(self.bn0(x)))
        x = self.b1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.b2(x)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.b3(x)
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.b4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


class FCNN_small(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, 4)
        self.b1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 4)
        self.b2 = nn.BatchNorm1d(4)
        self.fc3 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(self.bn0(x)))
        x = self.b1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.b2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FCNN_large(nn.Module):
    def __init__(self, inp_dim, width=256):
        super().__init__()
        self.b0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, width * 32)
        self.b1 = nn.BatchNorm1d(width * 32)
        self.fc2 = nn.Linear(width * 32, width * 4)
        self.b2 = nn.BatchNorm1d(width * 4)
        self.fc3 = nn.Linear(width * 4, width * 4)
        self.b3 = nn.BatchNorm1d(width * 4)
        self.fc5 = nn.Linear(width * 4, width * 4)
        self.b5 = nn.BatchNorm1d(width * 4)
        self.fc6 = nn.Linear(width * 4, width)
        self.b6 = nn.BatchNorm1d(width)
        self.fc7 = nn.Linear(width, 1)
        self.dropout = nn.Dropout(p=0.8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.b0(x)
        x = self.relu(self.fc1(x))
        x = self.b1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.b2(x)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.b3(x)
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.b5(x)
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        x = self.b6(x)
        x = self.dropout(x)
        x = self.fc7(x)
        return x


class FCNN2(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.b4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.b1(x)
        x = self.relu(self.fc2(x))
        x = self.b4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


class Perc(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        #self.b0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, 1)
        # self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.b0(x)
        x = self.fc1(x)  # self.relu(self.fc1(x))
        return x


class TabNet_ieeecis(TabNetEmb):
    def __init__(self, inp_dim, cat_map=None):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=32,
            n_a=32,
            n_shared=2,
            n_ind=2,
            n_steps=4,
            relax=1.2,
            vbs=512,
            embs_dim={
                "ProductCD": 4,
                "card3": 32,
                "addr2": 32,
                "P_emaildomain": 32,
                "R_emaildomain" : 32,
                "card_type": 8,
                "DeviceType" : 4,
                "_default_": 8
            },
            cat_map=cat_map,
        )

class TabNet_syn(TabNetEmb):
    def __init__(self, inp_dim, cat_map=None):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=32,
            n_a=32,
            n_shared=2,
            n_ind=2,
            n_steps=3,
            relax=1.2,
            vbs=512,
            embs_dim=4,
            cat_map=cat_map,
        )

class TabNet_homecredit(TabNetEmb):
    def __init__(self, inp_dim, cat_map=None):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=16,
            n_a=16,
            n_shared=3,
            n_ind=2,
            n_steps=4,
            relax=1.2,
            vbs=512,
            cat_map=cat_map,
        )

class TabNet_credit_sim(TabNetEmb):
    def __init__(self, inp_dim, cat_map=None):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=64,
            n_a=64,
            n_shared=2,
            n_ind=2,
            n_steps=4,
            relax=1.2,
            vbs=512,
            embs_dim={
                'Merchant City':64,
                "_default_": 8
            },
            cat_map=cat_map,
        )

class TabNet_bank_fraud(TabNetEmb):
    def __init__(self, inp_dim, cat_map=None):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=16,
            n_a=16,
            n_shared=2,
            n_ind=1,
            n_steps=3,
            relax=1.2,
            vbs=512,
            embs_dim={
                'employment_status':4,
                'foreign_request':2, 
                'device_os':2,
                'source':2,
                'email_is_free':2,
                'keep_alive_session':2,
                'phone_home_valid':2,
                'phone_mobile_valid':2,
                "_default_": 8
            },
            cat_map=cat_map,
        )
class TabNet_lending_club(TabNetEmb):
    def __init__(self, inp_dim, cat_map=None):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=128,
            n_a=128,
            n_shared=2,
            n_ind=2,
            n_steps=4,
            relax=1.2,
            vbs=512,
            embs_dim={
                "pub_rec_bankruptcies": 8,
                "annual_inc": 32,
                "dti": 32,
                "revol_util": 32,
                "revol_bal" : 32,
                "zip_code": 32,
                "_default_": 8
            },
            cat_map=cat_map,
        )
