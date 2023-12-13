import pickle
import sys
import time
import typing
import warnings
import random
import re

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import notebook as tqdm

sys.path.append("..")
# -

from src.utils.data import one_hot_encode
from src.utils.data import diff
from src.utils.hash import fast_hash

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import ast

CONST_MAX_COST = 10000.0

def get_w_vec(df, weights, one_hot=True, sep="_", num_list=[]):
    global CONST_MAX_COST
    col_names = df.columns.values
    w = torch.zeros(len(col_names))
    for i, c in enumerate(col_names):
        if sep in c and one_hot == True:
            if c in num_list:
                name = c
            else:
                name = sep.join((c.split(sep)[:-1]))
        else:
            name = c
        w[i] = weights.get(name, CONST_MAX_COST)
    return w

def get_sym_matrix(d, cost, w_max=CONST_MAX_COST):
    global CONST_MAX_COST
    cm = np.ones((d, d)) * cost - np.eye(d) * cost
    return cm

def get_num_matrix(d, inc_cost=0, dec_cost=0, w_max=CONST_MAX_COST):
    global CONST_MAX_COST
    cm = np.ones((d, d)) - np.eye(d)
    for i in range(d):
        for j in range(0, i):
            cm[i, j] *= dec_cost * (i - j)
        for j in range(i, d):
            cm[i, j] *= inc_cost * (j - i)
    return cm

def get_shop_matrix(d, prices, w_max=CONST_MAX_COST):
    cm = np.zeros((d, d))
    if d != len(prices.items()):
        print("WARNING: Some features do not have costs. Cost matrix might be incomplete")
    for i, (name, cost) in enumerate(prices.items()):
        cm[:, i] = cost
        cm[i, i] = 0.0
    return cm


def get_w_rep(X, costs, cat_map, sep="_", num_list=[], w_max=0.0):
    global CONST_MAX_COST
    w_rep = np.ones_like(X) * CONST_MAX_COST
    for i, num in enumerate(num_list):
        w_rep[:, i] = CONST_MAX_COST

    for name, cost in costs.items():
        if name not in num_list:
            i, j = cat_map[name]
            if w_max != 0.0:
                cost[cost >= w_max] = CONST_MAX_COST
            w_rep[:, i:j+1] = np.matmul(X[:, i:j+1], cost)
        else:
            pass
    return w_rep

def get_cat_map(df, cat_var, bin_var=[], sep="_"):
    col_names = df.columns.values
    c_map = {}
    for i, c in enumerate(col_names):
        if sep in c:
            if c in bin_var:
                c_map[c] = [i, i]
                continue
            name = sep.join((c.split(sep)[:-1]))
            #print(name, c)
            if name in cat_var:
                if name in c_map:
                    c_map[name][1] = i
                else:
                    c_map[name] = [i, i]
    return c_map

class BankFraud(Dataset):

    def __init__(
        self,
        balanced=True,
        seed=42,
        cat_map=False,
        same_cost=False,
        q_num = 10,
        mode = "train",
        w_max=0.0
    ):
        global CONST_MAX_COST

        self.mode = mode
        self.same_cost = same_cost
        self.max_eps = CONST_MAX_COST

        df = pd.read_csv("../data/bank_fraud/reduced_balanced.csv")

        numerical_columns = [
            "proposed_credit_limit",
            "month",
        ]
        quantized_columns = [
            'name_email_similarity', 
            'zip_count_4w',
        ]
        categorical_columns = [
            'income',
            'customer_age',
            'employment_status',
            'foreign_request',
            'device_os',
            'source',
            'email_is_free',
            'keep_alive_session',
            'phone_home_valid',
            'phone_mobile_valid',
            "device_distinct_emails_8w",
        ]


        df[categorical_columns] = df[categorical_columns].astype('category')

        phone_prices = {
            "0": 0.1,
            "1": 10,
        }

        email_prices = {
            "0": 0.1,
            "1": 10,
        }
        costs = {
            "income": get_sym_matrix(9, 10),
            "zip_count_4w": get_sym_matrix(q_num, 1),
            "foreign_request": get_sym_matrix(2, 0.1),
            "keep_alive_session": get_sym_matrix(2, 0.1),
            "source": get_sym_matrix(2, 0.1),
            'device_os': get_sym_matrix(5, 0.1),
            'email_is_free': get_shop_matrix(len(email_prices), email_prices),
            "phone_home_valid": get_shop_matrix(len(phone_prices), phone_prices),
            "phone_mobile_valid": get_shop_matrix(len(phone_prices), phone_prices),
            "device_distinct_emails_8w": get_num_matrix(4, inc_cost=0.1, dec_cost=CONST_MAX_COST),       
        }
        self.costs = costs
        qcols = pd.DataFrame()
        for qc in quantized_columns:
            qcol = pd.cut(df[qc], q_num, duplicates="drop").astype('category')
            qcols[qc] = qcol
        
        df = df[categorical_columns + numerical_columns + ["fraud_bool"]]
        df = df.join(qcols)

        
        self.cost_orig_df = df["proposed_credit_limit"]
        self.gain_col = "proposed_credit_limit"
        df = one_hot_encode(
                df, binary_vars=["fraud_bool"], standardize=False, prefix_sep="_"
        )

        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_\78-]+', '', x))
        y = df["fraud_bool"]
        X = df.drop(columns=["fraud_bool"], axis=0)
        self.orig_df = df

        X_train = X[X['month']<6]
        X_test = X[X['month']>=6]
        y_train = y[X['month']<6]
        y_test = y[X['month']>=6]

        X_train.drop('month', axis=1, inplace=True)
        X_test.drop('month', axis=1, inplace=True)
        self.cat_map = get_cat_map(X_train, categorical_columns + quantized_columns, sep='_')

        del X
        del y
        del df

        # 
        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)
            self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)
        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)
            self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)

        else:
            raise ValueError

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx]])
        return (inpt, oupt, cost, self.w_rep[idx])

    
class CreditCard(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=False,
        discrete_one_hot=False,
        same_cost=False,
        cat_map=False,
        ballet=False,
        w_max=0.0
    ):
        self.max_eps = 1000.0
        self.mode = mode
        self.same_cost = same_cost
        df = pd.read_csv(f"{folder_path}/credit_sim/credit_card_transactions-balanced-v7.csv", index_col=0)
        self.gain_col = "Amount"
        numerical_columns = [
            "Amount"
        ]
        categorical_columns = [
            "Use Chip",
            "Merchant City",
            "Errors",
            "card_brand",
            "card_type",
            "Fraud",
        ]
        weights = {
            "Use Chip": 20,
            "card_brand": 20,
            "card_type": 20,
            "Month": 0.1,
            "Day": 0.1,
            "Hour": 0.1,
            "Minutes": 0.1,
            "MCC_City": 100,
            "Errors": 10,
        }
        df = df[numerical_columns + categorical_columns]
        df = one_hot_encode(
            df,
            cat_cols=categorical_columns,
            num_cols=numerical_columns,
            binary_vars=["Fraud"],
            standardize=False,
            prefix_sep="_",
        )
        scaling_factor = 0.1 # 1 eps == 10$
        
        card_brand_price = {
            "Amex": 25,
            "Discover": 25,
            "Mastercard": 20,
            "Visa": 20,
        }
        
        costs = {
            "Merchant City": np.load("../data/credit_sim/costs_cities.npy") * scaling_factor,
            "card_type": get_sym_matrix(3, 20) * scaling_factor,     
            "card_brand": get_shop_matrix(len(card_brand_price), card_brand_price, w_max=w_max) * scaling_factor,

        }
        self.cat_map = get_cat_map(df, categorical_columns, sep='_')

        df["Fraud"] = np.where(
            df["Fraud"].str.contains("Y"), "1", "0"
        )
        y = df["Fraud"]
        X = df.drop(columns=["Fraud"])
        #print(list(df.columns))
        self.orig_df = df
        self.cost_orig_df = df["Amount"]
        self.gain_col = "Amount"
        self.costs = costs


        X_train, X_test, y_train, y_test = train_test_split( # ALREADY BALANCED DATA
                X, y, test_size=3000, random_state=seed
            )

        del X
        del y
        del df

        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train.to_numpy(dtype=np.float32)
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)
            self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test.to_numpy(dtype=np.float32)
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)
            self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)

        else:
            raise ValueError

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx]])
        return (inpt, oupt, cost, self.w_rep[idx])
    

class IEEECISDataset(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=False,
        discrete_one_hot=False,
        same_cost=False,
        cat_map=False,
        ballet=False,
        w_max=0.0
    ):
        global CONST_MAX_COST
        self.max_eps = CONST_MAX_COST
        self.mode = mode
        self.same_cost = same_cost
        train_identity = pd.read_csv(f"{folder_path}/ieeecis/train_identity.csv")
        train_transaction = pd.read_csv(f"{folder_path}/ieeecis/train_transaction.csv")

        df = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
        cost_orig = df["TransactionAmt"]
        self.gain_col = "TransactionAmt"
        self.cost_orig_df = cost_orig
        if discrete_one_hot:
            num_cols = [
                "TransactionAmt",
                "card1",
                "card2",
                "card5",
                "addr1",
            ]
            
            cat_cols = [
                "ProductCD",
                "card4",
                "card3",
                "card6",
                "addr2",
                "P_emaildomain",
                "R_emaildomain",
                "DeviceType",
                "isFraud",
            ]

            email_price = {
                "aim.com" : 0.14,
                "anonymous.com": CONST_MAX_COST,
                "aol.com" : 0.14,
                "att.net": 0.5,
                "bellsouth.net": 0.5,
                "cableone.net": CONST_MAX_COST,
                "centurylink.net": 30,
                "cfl.rr.com": CONST_MAX_COST,
                "charter.net": 0.5,
                "comcast.net": 0.5,
                "cox.net": 50,
                "earthlink.net": 50,
                "embarqmail.com": CONST_MAX_COST,
                "frontier.com" : 0.5,
                "frontiernet.net": 0.5,
                "gmail": 0.6,
                "gmail.com": 0.6,
                "gmx.de": 0.5,
                "hotmail.co.uk": 0.12,
                "hotmail.com": 0.12,
                "hotmail.de": 0.12,
                "hotmail.es": 0.12,
                "hotmail.fr": 0.12,
                "icloud.com" : 6,
                "juno.com" : 10,
                "live.com" : CONST_MAX_COST,
                "live.com.mx" : CONST_MAX_COST,
                "live.fr" : CONST_MAX_COST,
                "mac.com" : CONST_MAX_COST,
                "mail.com" : 0.5,
                "me.com" : CONST_MAX_COST,
                "msn.com" : CONST_MAX_COST,
                "netzero.com" : 20,
                "netzero.net" : 20,
                "optonline.net" : 10,
                "outlook.com" : 0.15,
                "outlook.es": 0.5,
                "prodigy.net.mx": CONST_MAX_COST,
                "protonmail.com": 0.5,
                "ptd.net": 15,
                "q.com": 10,
                "roadrunner.com": CONST_MAX_COST,
                "rocketmail.com": CONST_MAX_COST,
                "sbcglobal.ne": CONST_MAX_COST,
                "sc.rr": CONST_MAX_COST,
                "servicios-ta.com": CONST_MAX_COST,
                "suddenlink.net": 30,
                "twc.com": CONST_MAX_COST,
                "verizon.net": CONST_MAX_COST,
                "web.de": 0.5,
                "windstream.net": 50,
                "yahoo.co.jp" : 0.24,
                "yahoo.co.uk" : 0.24,
                "yahoo.com": 0.24,
                "yahoo.com.mx": 0.24,
                "yahoo.de": 0.24,
                "yahoo.es": 0.24,
                "yahoo.fr": 0.24,
                "ymail.com": 0.24,
                "nan": CONST_MAX_COST
                ,
            }

            card_type_price = {
                "american express-charge card": CONST_MAX_COST,
                "american express-credit": 25,
                "american express-debit":25,
                "discover-credit":25,
                "discover-debit":25,
                "mastercard-credit":20,
                "mastercard-debit":20,
                "mastercard-debit or credit":20,
                "visa-credit":20,
                "visa-debit":20,
                "<NA>": CONST_MAX_COST,
            }

            weights = {
                # Do not need it 

                "card_type": 20,
                "P_emaildomain": 0.2,
                "DeviceType": 0.1,
            }
            costs = {
                "card_type": get_shop_matrix(len(card_type_price), card_type_price, w_max=w_max),
                "P_emaildomain": get_shop_matrix(len(email_price), email_price, w_max=w_max),
                "DeviceType": get_sym_matrix(3, 0.1, w_max=w_max)
            }
            self.costs = costs
            one_hot_vars = ["ProductCD",
            "card_type",
            "P_emaildomain",
            "R_emaildomain",
            "addr2", "card3", "DeviceType"]
            df = df[num_cols + cat_cols]
            df[cat_cols] = df[cat_cols].astype("category")

            df = df[~df[num_cols].isna().any(axis=1)]

            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            df["card_type"] = (
                df["card4"].astype("string") + "-" + df["card6"].astype("string")
            ).astype("category")
            df = df.drop(columns=["card4", "card6"])
            df = one_hot_encode(
                df, binary_vars=["isFraud"], standardize=False, prefix_sep="_"
            )
            if cat_map:
                self.cat_map = get_cat_map(df, one_hot_vars, sep='_')
            else:
                self.cat_map = None
        else:
            df = (
                df.select_dtypes(exclude=["object"])
                .dropna(axis=1, how="any")
                .drop(columns=["TransactionID", "TransactionDT"])
            )

        X = df.drop(columns="isFraud")
        y = df["isFraud"]
        self.orig_df = df
        w_vector = get_w_vec(X, weights, one_hot=True, sep="_").unsqueeze(0)
        self.w = w_vector
        if balanced:
            X_resampled, self.y_resampled = RandomUnderSampler(
                random_state=seed
            ).fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, self.y_resampled, test_size=3000, random_state=seed
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=3000, random_state=seed
            )

        del X
        del y
        del df

        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)
            self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=num_cols, w_max=w_max)
            print(self.w_rep[0])

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)
            self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=num_cols, w_max=w_max)

        else:
            raise ValueError

        self.mean = 1

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx] / self.mean])
        return (inpt, oupt, cost, self.w_rep[idx])


def _transform_source_identity(X_k, sources_count=7):
    """
    Helper to transform the source_identity field.
    """
    X_k = X_k.apply(lambda x: x.replace(";", ","))
    X_k = X_k.apply(ast.literal_eval)

    N, K = X_k.shape[0], sources_count * 2
    X_k_transformed = np.zeros((N, K), dtype="intc")

    # Set (1, 0) if the source is present for the user and (0, 1) if absent.
    for i in range(N):
        for j in range(sources_count):
            if j in X_k[i]:
                X_k_transformed[i, j * 2] = 1
            else:
                X_k_transformed[i, j * 2 + 1] = 1

    return X_k_transformed



class Synthetic(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=13,
        same_cost=False,
        cat_map=False, # TODO remove
        noise='0',
        mat_mode=True,
        w_max=0.0
    ):
        self.mode = mode
        self.same_cost = same_cost
        if noise == '0':
            df = pd.read_csv(folder_path + "/syn.csv")
        else:
            df = pd.read_csv(folder_path + "/syn_" + noise + ".csv")

        if mat_mode:
            costs = np.load(folder_path + "/syn_" + noise + "_costs.npy", allow_pickle=True)[()]
            self.costs = costs

        categorical_columns = ["cat" + str(_) for _ in range(50)]
        numerical_columns = ["num" + str(_) for _ in range(50)]

        y = df["target"]
        df["gain"] = 1
        cost_orig = df["gain"]
        self.gain_col = "gain"
        self.cost_orig_df = cost_orig
        self.orig_df = df

        random.seed(seed)

        weights = {f: random.choice([0.1, 1.0, 10.0, 100.0]) for f in df.columns}


        for categorical_column in categorical_columns:
            df[categorical_column].fillna("NULL", inplace=True)
            df[categorical_column] = df[categorical_column].astype('category',copy=False)

        df = one_hot_encode(
            df,
            cat_cols=categorical_columns,
            num_cols=numerical_columns,
            binary_vars=["target"],
            standardize=False,
            prefix_sep="_",
        )

        X = df.drop(columns=["target"])


        self.max_eps = 100000.0
        self.cat_map = get_cat_map(df, categorical_columns, sep='_')
        if mat_mode:
            pass
        else:
            w_vector = get_w_vec(df.drop(columns=["target"]), weights, sep="_", one_hot=True).unsqueeze(0)
            self.w = w_vector
            print(self.w)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=10000, random_state=seed
        )
        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)
            if mat_mode:
                self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)
            if mat_mode:
                self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)

        self.mat_mode = mat_mode

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx] / 1])
        if self.mat_mode:
            w_rep = torch.Tensor(self.w_rep[idx])

        return (inpt, oupt, cost, w_rep)

class HomeCreditDataset(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=True,
        discrete_one_hot=False,
        same_cost=False,
        drop_features=None,
        cat_map=False,
        ballet=False
    ):
        self.mode = mode
        self.same_cost = same_cost
        
        self.max_eps = 30000.0
        all_weights = {
            "NAME_CONTRACT_TYPE": 0.1,
            "FLAG_OWN_CAR": 100,
            "FLAG_OWN_REALTY": 100,
            "AMT_INCOME_TOTAL": 1,
            "NAME_TYPE_SUITE": 0.1,
            "NAME_INCOME_TYPE": 100,
            "has_children": 1000,
            "house_variables_sum_isnull": 100,
            "cluster_days_employed": 100,
            "NAME_EDUCATION_TYPE": 1000,
            "NAME_FAMILY_STATUS": 1000,
            "NAME_HOUSING_TYPE": 100,
            "REGION_RATING_CLIENT": 100,
            "REG_REGION_NOT_LIVE_REGION": 100,
            "REG_REGION_NOT_WORK_REGION": 100,
            "LIVE_REGION_NOT_WORK_REGION": 100,
            "REG_CITY_NOT_LIVE_CITY": 100,
            "REG_CITY_NOT_WORK_CITY": 100,
            "LIVE_CITY_NOT_WORK_CITY": 100,
            "FLAG_MOBIL": 10,
            "FLAG_EMP_PHONE": 10,
            "FLAG_WORK_PHONE": 10,
            "FLAG_CONT_MOBILE": 10,
            "FLAG_PHONE": 10,
            "FLAG_EMAIL": 0.1,
            "WEEKDAY_APPR_PROCESS_START": 0.1,
            "HOUR_APPR_PROCESS_START": 0.1,
            "OCCUPATION_TYPE": 100,
            "ORGANIZATION_TYPE": 100,
            "EXT_SOURCE_1":10000,
            "EXT_SOURCE_2":10000,
            "EXT_SOURCE_3":10000,
        }

        cons_weights = {
            "NAME_CONTRACT_TYPE": -1,
            "FLAG_OWN_CAR": -1,
            "FLAG_OWN_REALTY": -1,
            "AMT_INCOME_TOTAL": -1,
            "NAME_TYPE_SUITE":  -1,
            "NAME_INCOME_TYPE": -1,
            "NAME_EDUCATION_TYPE": -1,
            "NAME_FAMILY_STATUS": -1,
            "NAME_HOUSING_TYPE": -1,
            "FLAG_MOBIL": -1,
            "cluster_days_employed": -1,
            "FLAG_EMP_PHONE": -1,
            "FLAG_WORK_PHONE": -1,
            "FLAG_CONT_MOBILE": -1,
            "FLAG_PHONE": -1,
            "FLAG_EMAIL": -1,
            "WEEKDAY_APPR_PROCESS_START": -1,
            "HOUR_APPR_PROCESS_START": -1,
            "OCCUPATION_TYPE": -1,
            "ORGANIZATION_TYPE": -1,

        }

        weights = all_weights

        # For robust baseline
        #for f, w in weights.items():
        #    weights[f] = -1
        #print(weights)

        application_train_df = pd.read_csv(
            folder_path + "/home-credit-default-risk/application_train.csv"
        ).sample(frac=1, random_state=seed)
        application_test_df = pd.read_csv(
            folder_path + "/home-credit-default-risk/application_test.csv"
        )
        previous_application_df = pd.read_csv(
            folder_path + "/home-credit-default-risk/previous_application.csv"
        )

        application_train_df["CSV_SOURCE"] = "application_train.csv"
        application_test_df["CSV_SOURCE"] = "application_test.csv"
        df = pd.concat([application_train_df, application_test_df])

        # MANAGE previous_applications.csv
        temp_previous_df = previous_application_df.groupby(
            "SK_ID_CURR", as_index=False
        ).agg({"NAME_CONTRACT_STATUS": lambda x: ",".join(set(",".join(x).split(",")))})
        temp_previous_df["has_only_approved"] = np.where(
            temp_previous_df["NAME_CONTRACT_STATUS"] == "Approved", "1", "0"
        )
        temp_previous_df["has_been_rejected"] = np.where(
            temp_previous_df["NAME_CONTRACT_STATUS"].str.contains("Refused"), "1", "0"
        )

        # JOIN DATA
        df = pd.merge(df, temp_previous_df, on="SK_ID_CURR", how="left")

        # CREATE CUSTOM COLUMNS
        #################################################### total_amt_req_credit_bureau
        df["total_amt_req_credit_bureau"] = (
            df["AMT_REQ_CREDIT_BUREAU_YEAR"] * 1
            + df["AMT_REQ_CREDIT_BUREAU_QRT"] * 2
            + df["AMT_REQ_CREDIT_BUREAU_MON"] * 8
            + df["AMT_REQ_CREDIT_BUREAU_WEEK"] * 16
            + df["AMT_REQ_CREDIT_BUREAU_DAY"] * 32
            + df["AMT_REQ_CREDIT_BUREAU_HOUR"] * 64
        )
        df["total_amt_req_credit_bureau_isnull"] = np.where(
            df["total_amt_req_credit_bureau"].isnull(), "1", "0"
        )
        df["total_amt_req_credit_bureau"].fillna(0, inplace=True)

        #######################################################################  has_job
        #df["has_job"] = np.where(
        #    df["NAME_INCOME_TYPE"].isin(["Pensioner", "Student", "Unemployed"]),
        #    "1",
        #    "0",
        #)

        #######################################################################  has_children
        df["has_children"] = np.where(df["CNT_CHILDREN"] > 0, "1", "0")

        ####################################################### clusterise_days_employed
        def clusterise_days_employed(x):
            days = x["DAYS_EMPLOYED"]
            if days > 0:
                return "not available"
            else:
                days = abs(days)
                if days < 30:
                    return "less 1 month"
                elif days < 180:
                    return "less 6 months"
                elif days < 365:
                    return "less 1 year"
                elif days < 1095:
                    return "less 3 years"
                elif days < 1825:
                    return "less 5 years"
                elif days < 3600:
                    return "less 10 years"
                elif days < 7200:
                    return "less 20 years"
                elif days >= 7200:
                    return "more 20 years"
                else:
                    return "not available"

        df["cluster_days_employed"] = df.apply(clusterise_days_employed, axis=1)

        #######################################################################  custom_ext_source_3
        def clusterise_ext_source(x):
            if str(x) == "nan":
                return "not available"
            else:
                if x < 0.1:
                    return "less 0.1"
                elif x < 0.2:
                    return "less 0.2"
                elif x < 0.3:
                    return "less 0.3"
                elif x < 0.4:
                    return "less 0.4"
                elif x < 0.5:
                    return "less 0.5"
                elif x < 0.6:
                    return "less 0.6"
                elif x < 0.7:
                    return "less 0.7"
                elif x < 0.8:
                    return "less 0.8"
                elif x < 0.9:
                    return "less 0.9"
                elif x <= 1:
                    return "less 1"

        df["clusterise_ext_source_1"] = df["EXT_SOURCE_1"].apply(
            lambda x: clusterise_ext_source(x)
        )
        df["clusterise_ext_source_2"] = df["EXT_SOURCE_2"].apply(
            lambda x: clusterise_ext_source(x)
        )
        df["clusterise_ext_source_3"] = df["EXT_SOURCE_3"].apply(
            lambda x: clusterise_ext_source(x)
        )

        #######################################################################  house_variables_sum
        house_vars = [
            "APARTMENTS_AVG",
            "APARTMENTS_MEDI",
            "APARTMENTS_MODE",
            "BASEMENTAREA_AVG",
            "BASEMENTAREA_MEDI",
            "BASEMENTAREA_MODE",
            "COMMONAREA_AVG",
            "COMMONAREA_MEDI",
            "COMMONAREA_MODE",
            "ELEVATORS_AVG",
            "ELEVATORS_MEDI",
            "ELEVATORS_MODE",
            "EMERGENCYSTATE_MODE",
            "ENTRANCES_AVG",
            "ENTRANCES_MEDI",
            "ENTRANCES_MODE",
            "FLOORSMAX_AVG",
            "FLOORSMAX_MEDI",
            "FLOORSMAX_MODE",
            "FLOORSMIN_AVG",
            "FLOORSMIN_MEDI",
            "FLOORSMIN_MODE",
            "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE",
            "LANDAREA_AVG",
            "LANDAREA_MEDI",
            "LANDAREA_MODE",
            "LIVINGAPARTMENTS_AVG",
            "LIVINGAPARTMENTS_MEDI",
            "LIVINGAPARTMENTS_MODE",
            "LIVINGAREA_AVG",
            "LIVINGAREA_MEDI",
            "LIVINGAREA_MODE",
            "NONLIVINGAPARTMENTS_AVG",
            "NONLIVINGAPARTMENTS_MEDI",
            "NONLIVINGAPARTMENTS_MODE",
            "NONLIVINGAREA_AVG",
            "NONLIVINGAREA_MEDI",
            "NONLIVINGAREA_MODE",
            "TOTALAREA_MODE",
            "WALLSMATERIAL_MODE",
            "YEARS_BEGINEXPLUATATION_AVG",
            "YEARS_BEGINEXPLUATATION_MEDI",
            "YEARS_BEGINEXPLUATATION_MODE",
            "YEARS_BUILD_AVG",
            "YEARS_BUILD_MEDI",
            "YEARS_BUILD_MODE",
        ]
        df["house_variables_sum"] = df[house_vars].sum(axis=1)
        df["house_variables_sum_isnull"] = np.where(
            df["house_variables_sum"].isnull(), "1", "0"
        )
        df["house_variables_sum"].fillna(
            value=df["house_variables_sum"].median(), inplace=True
        )

        # SELECT COLUMNS
        numerical_columns = [
            "AMT_ANNUITY",
            "AMT_CREDIT",
            #"AMT_GOODS_PRICE",
            "AMT_INCOME_TOTAL",
            #"REGION_POPULATION_RELATIVE",
            #"DAYS_BIRTH",
            #"DAYS_ID_PUBLISH",
            #"DAYS_REGISTRATION",
            #CNT_CHILDREN",
            #"CNT_FAM_MEMBERS",
            #"DAYS_EMPLOYED",
            #"DAYS_LAST_PHONE_CHANGE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            #"total_amt_req_credit_bureau",
            #"house_variables_sum",
        ]
        categorical_columns = [
            "CODE_GENDER",
            "CSV_SOURCE",
            "NAME_EDUCATION_TYPE",
            #"CNT_CHILDREN",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
            "NAME_CONTRACT_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "NAME_INCOME_TYPE",
            "NAME_TYPE_SUITE",
            "WEEKDAY_APPR_PROCESS_START",
            "HOUR_APPR_PROCESS_START",
            "REGION_RATING_CLIENT",
            #"has_only_approved",
            #"has_been_rejected",
            #"has_job",
            "cluster_days_employed",
            #"clusterise_ext_source_1",
            #"clusterise_ext_source_2",
            #"clusterise_ext_source_3",
            #"total_amt_req_credit_bureau_isnull",
            "house_variables_sum_isnull",
        ]

        binary_columns = [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_EMAIL",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "has_children",
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
        ]

        one_hot_vars = [
            "NAME_CONTRACT_TYPE",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            #"CNT_CHILDREN",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "cluster_days_employed",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "WEEKDAY_APPR_PROCESS_START",
            "HOUR_APPR_PROCESS_START",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE"
        ]

        target_column = ["TARGET"]
        df = df[numerical_columns + categorical_columns + target_column + binary_columns]
        # MANAGE MISSING VALUES
        for numerical_column in numerical_columns:
            if df[numerical_column].isnull().values.any():
                df[numerical_column + "_isnull"] = np.where(
                    df[numerical_column].isnull(), "1", "0"
                )
            df[numerical_column].fillna(
                value=df[numerical_column].median(), inplace=True
            )

        for categorical_column in categorical_columns:
            df[categorical_column].fillna("NULL", inplace=True)
            df[categorical_column] = df[categorical_column].astype('category',copy=False)

        df["FLAG_OWN_CAR"] = np.where(
            df["FLAG_OWN_CAR"].str.contains("Y"), "1", "0"
        )
        df["FLAG_OWN_REALTY"] = np.where(
            df["FLAG_OWN_CAR"].str.contains("Y"), "1", "0"
        )
        # STANDARDISE
        # min_max_scaler = preprocessing.MinMaxScaler()
        # df[numerical_columns] = pd.DataFrame(min_max_scaler.fit_transform(df[numerical_columns]))

        # CONVERT CATEGORICAL COLUMNS INTO TYPE "category"
        categorical_columns.remove("CSV_SOURCE")

        cost_orig = df["AMT_CREDIT"]
        self.gain_col = "AMT_CREDIT"
        self.orig_df = df

        #df["EXT_SOURCE_1"] = 10 * np.log10(df["EXT_SOURCE_1"])
        df["EXT_SOURCE_1"] = np.log2(1 - df["EXT_SOURCE_1"])
        #df["EXT_SOURCE_2"] = 10 * np.log10(df["EXT_SOURCE_2"])
        #df["EXT_SOURCE_3"] = 10 * np.log10(df["EXT_SOURCE_3"])
        df["EXT_SOURCE_2"] = np.log2(1 - df["EXT_SOURCE_2"])
        df["EXT_SOURCE_3"] = np.log2(1 - df["EXT_SOURCE_3"])

        df = one_hot_encode(
            df,
            cat_cols=categorical_columns,
            num_cols=numerical_columns,
            binary_vars=["TARGET", "CSV_SOURCE"],
            standardize=False,
            prefix_sep="_",
        )
        w_vector = get_w_vec(
            df.drop(columns=["TARGET", "CSV_SOURCE"]), weights, sep="_", num_list=(numerical_columns + binary_columns)
        ).unsqueeze(0)
        self.w = w_vector
        #print(w_vector)

        # SPLIT DATA INTO TRAINING vs TRAIN
        df = df[df["CSV_SOURCE"] == "application_train.csv"]
        y = df["TARGET"]
        self.cost_orig_df = cost_orig

        # REMOVE NOT USEFUL COLUMNS
        if cat_map:
            self.cat_map = get_cat_map(df, categorical_columns, bin_var=binary_columns, sep='_')
            print(self.cat_map)
        else:
            self.cat_map = None

        X = df.drop(columns=["CSV_SOURCE", "TARGET"], axis=0)
        #print(list(X.columns))
        if balanced:
            X_resampled, self.y_resampled = RandomUnderSampler(
                random_state=seed
            ).fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, self.y_resampled, test_size=3000, random_state=seed
            )

        del X
        del y
        del df

        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)

        elif mode == "test":
            self.X_test = X_test
            #print(self.X_test.iloc[0])
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx] / 1])
        # (oupt.shape, oupt)
        return (inpt, oupt, cost)


shape_dict = {"ieeecis": 147, "twitter_bot": 19, "home_credit": 183, "syn": 100, "credit_app": 70} # TODO: check if I need it at all 