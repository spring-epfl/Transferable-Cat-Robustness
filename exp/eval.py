import argparse
import os
import pickle
import sys
import time
import typing
from pathlib import Path

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
from sklearn.metrics import classification_report
from sklearn.cluster import AgglomerativeClustering
import lightgbm as lgb
from scipy import stats

sys.path.append("..")
sys.path.append(".")

from src.utils.data import one_hot_encode
from src.utils.data import diff
from src.utils.hash import fast_hash
from src.utils.counter import ExpansionCounter, CounterLimitExceededError
from src.transformations import TransformationGenerator
from src.transformations import CategoricalFeature, NumFeature
from src.search import a_star_search as generalized_a_star_search

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import autonotebook as tqdm

from utils import *
from train import default_model_dict as model_dict

from loaders import shape_dict
from exp.framework import ExperimentSuite
from exp.utils import TorchWrapper, EmbWrapper, LastEmbWrapper, WangWrapper
from exp import settings
from exp.settings import get_dataset
from sklearn.ensemble import GradientBoostingClassifier



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--results_dir", default="../out", type=str)
    parser.add_argument(
        "--dataset",
        default="ieeecis",
        choices=["ieeecis", "syn", "bank_fraud", "credit_sim"],
        type=str,
    )
    parser.add_argument("--attack", default="greedy", type=str)
    parser.add_argument("--embs", default="", type=str)
    parser.add_argument("--cost_bound", default=None, type=float)
    parser.add_argument("--tr", default=0.0, type=float)
    parser.add_argument("--model_path", default="../models/default.pt", type=str)
    parser.add_argument("--wang_path", default="../models/", type=str)
    parser.add_argument(
        "--utility_type",
        default="success_rate",
        choices=["maximum", "satisficing", "cost-restrictred", "average-attack-cost", "success_rate"],
        type=str,
    )
    parser.add_argument(
        "--satisfaction-value", default=-1, type=float, help="Value for satisfaction"
    )
    parser.add_argument(
        "--max-cost-value", default=-1, type=float, help="Max-cost value"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test-lim", default=10000, type=int)
    parser.add_argument("--noise", default="0", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--last-layer-embs", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--merge-type",
        default="linf",
        choices=["linf", "l2"],
        type=str,
    )
    return parser.parse_args()

def cluster_vec(vec, t):
    idx = np.argsort(vec)
    svec = vec.copy()
    mns = []
    for i, a in enumerate(idx):
        if mns == []:
            mns.append(vec[a])
            continue
        elif vec[a] - mns[-1] < t:
            svec[a] = mns[-1]
            continue
        else:
            mns.append(vec[a])
    return svec
    

def prone_embs(embs, t=0.01, merge_type="linf"):
    for emb in embs:
        w = emb.weight.detach().cpu().numpy()
        if w.shape[0] == 1:
            continue # do not treat numeric yet
        if merge_type == "linf":
            #print(w.shape)
            for i in range(w.shape[0]):
                w[i] = cluster_vec(w[i], t)
                #print(emb.weight[i], w[i])
        elif merge_type == "l2":
            w = emb.weight.detach().cpu().numpy()
            clustering = AgglomerativeClustering(n_clusters=None, affinity='l2', linkage="complete", distance_threshold=t+0.00001)
            out = clustering.fit(w.T)
            k = max(out.labels_)
            print(k)
            for i in range(k+1):
                av = w.T[out.labels_ == i]
                avm = np.mean(av, axis=0)
                #print(avm)
                avm = avm / (np.linalg.norm(avm) + 0.0001) # to avoid division by zero
                w.T[out.labels_ == i] = avm
                #print(i, av)
            #print(w.T, out.labels_)
        emb.weight = torch.nn.Parameter(torch.Tensor(w).to(emb.weight.device))

def get_embs_dists(embs, merge_type="linf"):
    dists = []
    for emb in embs:
        w = emb.weight.detach().cpu().numpy()
        if w.shape[0] == 1:
            continue # do not treat numeric yet
        if merge_type == "linf":
            for i in range(w.shape[1]):
                for j in range(w.shape[1]):
                    for k in range(w.shape[0]):
                        if i != j:
                            d = np.abs(w[k,i] - w[k,j])
                            dists.append(d)
        elif merge_type == "l2":
            for i in range(w.shape[1]):
                for j in range(w.shape[1]):
                    if i != j:
                        d = np.linalg.norm(w[:,i] - w[:,j], ord=2)
                        dists.append(d)
    return np.array(dists)

            
def dump_costs_embs(costs, embs, n=50):
    for i in range(n):
        name = "cat" + str(i)
        print("Cost matrix for ", name)
        print(costs[name])
        print("Embeddings for ", name)
        print(embs[i].weight.T)
    
    for i in range(n):
        name = "num" + str(i)
        print("Cost matrix for ", name)
        print(costs[name])
        print("Embeddings for ", name)
        print(embs[n+i].weight.T)

def get_utility(results, cost_orig, X_test, y, mode="maximum", cost=-1, t_value=-1):
    total_ut = 0
    divider = 0
    for i, r in results.iterrows():
        #print(i, r.cost, cost_orig.iloc[int(r.orig_index)], r.orig_index)
        #if y.iloc[int(r.orig_index)] == 0: No need for it, only y = target is evaluated here
        #    continue
        if r.cost is None:
            if mode == "success_rate":
                divider += 1
            continue
        else:
            if mode == "maximum":
                total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "satisficing":
                if cost_orig.iloc[int(r.orig_index)] - r.cost > t_value:
                    total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "cost-restrictred":
                if r.cost < cost:
                    total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "average-attack-cost":
                if r.cost > 0.001:
                    total_ut += r.cost
                    divider += 1
            elif mode == "success_rate":
                if r.cost > 0.001:
                    total_ut += 1
                divider += 1
    
    if mode == "average-attack-cost" or mode == "success_rate":
        return total_ut / (divider + 0.001)
    else:
        return total_ut / len(X_test)


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # print("Cuda Device Available")
        # print("Name of the Cuda Device: ", torch.cuda.get_device_name())
        # print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
    else:
        device = torch.device("cpu")

    eval_settings = settings.setup_dataset_eval(args.dataset, args.data_dir, seed=0, noise = args.noise)
    eval_settings.working_datasets.X_test = eval_settings.working_datasets.X_test[:args.test_lim]
    eval_settings.working_datasets.y_test = eval_settings.working_datasets.y_test[:args.test_lim]
    experiment_path = settings.get_experiment_path(
        args.results_dir, args.model_path, args.attack, args.cost_bound, embs=args.embs, tr=args.tr
    )
    print(experiment_path)
    inp_dim = eval_settings.working_datasets.X_test.shape[1]
    if os.path.isfile(experiment_path) and not args.force:
        print(f"{experiment_path} already exists. Skipping attack...")
    else:
        if args.model_path == "lgbm":
                torch.manual_seed(args.seed) # Reset seed to get the same dataset
                np.random.seed(args.seed)

                data_train = get_dataset(
                    args.dataset, args.data_dir, mode="train", cat_map=True, noise=args.noise
                )
                data_test = get_dataset(
                    args.dataset, args.data_dir, mode="test", cat_map=True, noise=args.noise
                )
                #clf = lgb.LGBMClassifier(max_depth=4, n_estimators=100, min_split_gain=0.1)
                if args.dataset == 'ieeecis':
                    clf = lgb.LGBMClassifier(max_depth=-1, n_estimators=400)
                elif args.dataset == 'credit_sim':
                    clf = lgb.LGBMClassifier(max_depth=-1, n_estimators=400)
                else:
                    clf = lgb.LGBMClassifier()
                if args.embs != "":
                    if not args.last_layer_embs:
                        emb_model = model_dict[args.dataset](inp_dim=inp_dim, cat_map=data_train.cat_map).to(device)
                        emb_model.load_state_dict(torch.load(args.embs))
                        np.set_printoptions(precision=4)
                        #dump_costs_embs(data_test.costs, emb_model.emb_layers)
                        if args.dataset == 'ieeecis':
                            embs2prone = [emb_model.emb_layers[3], emb_model.emb_layers[5], emb_model.emb_layers[6]]
                        elif args.dataset == 'bank_fraud':
                            embs2prone = [emb_model.emb_layers[0], 
                                          emb_model.emb_layers[3],
                                          emb_model.emb_layers[4],
                                          emb_model.emb_layers[5],
                                          emb_model.emb_layers[6],
                                          emb_model.emb_layers[7],
                                          emb_model.emb_layers[8],
                                          emb_model.emb_layers[9],
                                          emb_model.emb_layers[10],
                                          emb_model.emb_layers[12],
                            ]
                        elif args.dataset == 'credit_sim':
                            embs2prone = [emb_model.emb_layers[1], 
                                          emb_model.emb_layers[3],
                                          emb_model.emb_layers[4],
                            ]
                        else:
                            embs2prone = emb_model.emb_layers

                        dsts = get_embs_dists(embs2prone, merge_type=args.merge_type)
                        tr = stats.scoreatpercentile(dsts, args.tr * 100)
                        print("Threshold: " + str(tr))
                        prone_embs(embs2prone, t=tr, merge_type=args.merge_type)
                        clf = EmbWrapper(clf, emb_model.emb_layers, emb_model.cats, device)
                    else:
                        emb_model = model_dict[args.dataset](inp_dim=inp_dim, cat_map=data_train.cat_map).to(device)
                        emb_model.load_state_dict(torch.load(args.embs))
                        clf = LastEmbWrapper(clf, emb_model, emb_model.cats, device)

                clf.fit(data_train.X_train.to_numpy(dtype=np.float32), data_train.y_train)
                y_pred=clf.predict(data_test.X_test)
                y_pred_train=clf.predict(data_train.X_train)
                print("Train:")
                print(classification_report(data_train.y_train, y_pred_train))
                print("Test:")
                print(classification_report(data_test.y_test, y_pred))

        elif args.model_path == "rf":
                torch.manual_seed(args.seed) # Reset seed to get the same dataset
                np.random.seed(args.seed)
                
                data_train = get_dataset(
                    args.dataset, args.data_dir, mode="train", cat_map=True, noise=args.noise
                )
                data_test = get_dataset(
                    args.dataset, args.data_dir, mode="test", cat_map=True, noise=args.noise
                )
                #clf = lgb.LGBMClassifier(max_depth=4, n_estimators=100, min_split_gain=0.1)
                clf = RandomForestClassifier()
                if args.embs != "":
                    if not args.last_layer_embs:
                        emb_model = model_dict[args.dataset](inp_dim=inp_dim, cat_map=data_train.cat_map).to(device)
                        emb_model.load_state_dict(torch.load(args.embs))
                        np.set_printoptions(precision=4)
                    #dump_costs_embs(data_test.costs, emb_model.emb_layers)
                        if args.dataset == 'ieeecis':
                            embs2prone = [emb_model.emb_layers[3], emb_model.emb_layers[5], emb_model.emb_layers[6]]
                        elif args.dataset == 'bank_fraud':
                            embs2prone = [emb_model.emb_layers[0], 
                                          emb_model.emb_layers[3],
                                          emb_model.emb_layers[4],
                                          emb_model.emb_layers[5],
                                          emb_model.emb_layers[6],
                                          emb_model.emb_layers[7],
                                          emb_model.emb_layers[8],
                                          emb_model.emb_layers[9],
                                          emb_model.emb_layers[10],
                                          emb_model.emb_layers[12],
                            ]
                        elif args.dataset == 'credit_sim':
                            embs2prone = [emb_model.emb_layers[1], 
                                          emb_model.emb_layers[3],
                                          emb_model.emb_layers[4],
                            ]
                        else:
                            embs2prone = emb_model.emb_layers
                        #start = time.time()
                        dsts = get_embs_dists(embs2prone, merge_type=args.merge_type)
                        tr = stats.scoreatpercentile(dsts, args.tr * 100)
                        print("Threshold: " + str(tr))
                        prone_embs(embs2prone, t=tr, merge_type=args.merge_type)
                        clf = EmbWrapper(clf, emb_model.emb_layers, emb_model.cats, device)
                    else:
                        emb_model = model_dict[args.dataset](inp_dim=inp_dim, cat_map=data_train.cat_map).to(device)
                        emb_model.load_state_dict(torch.load(args.embs))
                        clf = LastEmbWrapper(clf, emb_model, emb_model.cats, device)                        
                clf.fit(data_train.X_train.to_numpy(dtype=np.float32), data_train.y_train)
                y_pred=clf.predict(data_test.X_test)
                #end = time.time()
                #print("Time: ", end - start)
                print(classification_report(data_test.y_test, y_pred))

        elif args.model_path == "ds":
                torch.manual_seed(args.seed) # Reset seed to get the same dataset
                np.random.seed(args.seed)
                
                data_train = get_dataset(
                    args.dataset, args.data_dir, mode="train", cat_map=True, noise=args.noise
                )
                data_test = get_dataset(
                    args.dataset, args.data_dir, mode="test", cat_map=True, noise=args.noise
                )
                #clf = lgb.LGBMClassifier(max_depth=4, n_estimators=100, min_split_gain=0.1)
                if args.dataset == 'credit_sim':
                    clf = GradientBoostingClassifier(n_estimators=80, learning_rate=1.0, max_depth=1, random_state=0)
                elif args.dataset == 'bank_fraud':
                    clf = GradientBoostingClassifier(n_estimators=40, learning_rate=0.4, max_depth=1, random_state=0)
                else:
                    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
                if args.embs != "":
                    if not args.last_layer_embs:
                        emb_model = model_dict[args.dataset](inp_dim=inp_dim, cat_map=data_train.cat_map).to(device)
                        emb_model.load_state_dict(torch.load(args.embs))
                        np.set_printoptions(precision=4)
                    #dump_costs_embs(data_test.costs, emb_model.emb_layers)
                        if args.dataset == 'ieeecis':
                            embs2prone = [emb_model.emb_layers[3], emb_model.emb_layers[5], emb_model.emb_layers[6]]
                        elif args.dataset == 'bank_fraud':
                            embs2prone = [emb_model.emb_layers[0], 
                                          emb_model.emb_layers[3],
                                          emb_model.emb_layers[4],
                                          emb_model.emb_layers[5],
                                          emb_model.emb_layers[6],
                                          emb_model.emb_layers[7],
                                          emb_model.emb_layers[8],
                                          emb_model.emb_layers[9],
                                          emb_model.emb_layers[10],
                                          emb_model.emb_layers[12],
                            ]
                        elif args.dataset == 'credit_sim':
                            embs2prone = [emb_model.emb_layers[1], 
                                          emb_model.emb_layers[3],
                                          emb_model.emb_layers[4],
                            ]
                        else:
                            embs2prone = emb_model.emb_layers
                        
                        #start = time.time()
                        dsts = get_embs_dists(embs2prone, merge_type=args.merge_type)
                        tr = stats.scoreatpercentile(dsts, args.tr * 100)
                        print("Threshold: " + str(tr))
                        if args.tr != 0.0:
                            prone_embs(embs2prone, t=tr, merge_type=args.merge_type)
                        clf = EmbWrapper(clf, emb_model.emb_layers, emb_model.cats, device)
                    else:
                        emb_model = model_dict[args.dataset](inp_dim=inp_dim, cat_map=data_train.cat_map).to(device)
                        emb_model.load_state_dict(torch.load(args.embs))
                        clf = LastEmbWrapper(clf, emb_model, emb_model.cats, device)                        
                clf.fit(data_train.X_train.to_numpy(dtype=np.float32), data_train.y_train)
                y_pred=clf.predict(data_test.X_test)
                #end = time.time()
                #print("Time: ", end - start)
                print(classification_report(data_test.y_test, y_pred))
        else:
            net = model_dict[args.dataset](inp_dim=inp_dim, cat_map=eval_settings.working_datasets.cat_map).to(device)
            net.load_state_dict(torch.load(args.model_path))
            net.eval()
            clf = TorchWrapper(net, device)

        if args.attack == "Ballet":
            eval_settings.working_datasets.orig_df["isFraud"] = \
                eval_settings.working_datasets.orig_df["isFraud"].astype('float')
            cr = eval_settings.working_datasets.orig_df.corr()
            corr_vec = cr["isFraud"]
            corr_vec.drop(index='isFraud')
            corr_vec = corr_vec.to_numpy(dtype=np.float32)
            for i, w in enumerate(eval_settings.working_datasets.w):
                if w < 10000:
                    eval_settings.working_datasets.w[0,i] = np.abs(corr_vec[i]) / (np.norm(corr_vec[i]) ** 2)

        exp_suite = ExperimentSuite(
            clf,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.y_test,
            target_class=eval_settings.target_class,
            cost_bound=args.cost_bound,
            spec=eval_settings.spec,
            gain_col=eval_settings.gain_col,
            dataset=eval_settings.working_datasets.dataset,
            iter_lim=1000
        )
        preds = clf.predict(eval_settings.working_datasets.X_test)
        print(preds, eval_settings.working_datasets.y_test)
        print("Acc: ", sum(preds == eval_settings.working_datasets.y_test) / len(eval_settings.working_datasets.X_test))

        attack_config = {a.name: a for a in eval_settings.experiments}[args.attack]

        results = exp_suite.run(attack_config)
        #results['x']
        results.to_pickle(experiment_path)

    result = pd.read_pickle(experiment_path)
    ut = get_utility(
            result,
            eval_settings.working_datasets.orig_cost,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.orig_y,
            mode=args.utility_type,
            cost=args.max_cost_value,
            t_value=args.satisfaction_value,
        )
    #print(ut)
    if (args.utility_type == "success_rate"):
        print("Attack Success rate: ", ut)

if __name__ == "__main__":
    main()
