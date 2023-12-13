import sys
import time

import numpy as np
from numpy.core.numeric import zeros_like
from torch._C import unify_type_list

sys.path.append("..")
sys.path.append(".")

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp import loaders
from exp.tabnet import TabNet
from exp.settings import get_dataset
from exp.utils import *
from exp.models import FCNN, TabNet_ieeecis, TabNet_homecredit, TabNet_credit_sim, TabNet_syn, TabNet_lending_club, TabNet_bank_fraud, Perc, FCNN_small
import torch.autograd.profiler as profiler

import argparse

default_model_dict = {
    "home_credit": TabNet_homecredit,
    "credit_sim": TabNet_credit_sim,
    "lending_club": TabNet_lending_club,
    "bank_fraud": TabNet_bank_fraud,
    "ieeecis": TabNet_ieeecis,
    "twitter_bot": TabNet,
    "syn": TabNet_syn,
}
model_dict = {
    "tabnet_homecredit": TabNet_homecredit,
    "tabnet_ieeecis": TabNet_ieeecis,
    "tabnet": TabNet,
    "perc": Perc,
    "fcnn": FCNN,
}

def fast_cdist(x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    # But be clever and do it with a single matmul call
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    # Zero out negative values
    res.clamp_min_(1e-30).sqrt_()
    return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument(
        "--dataset",
        default="ieeecis",
        choices=["ieeecis", "syn", "credit_sim", "bank_fraud"],
        type=str,
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--target-class", default=-1, type=int)
    parser.add_argument("--iter_lim", default=20, type=int)
    parser.add_argument(
        "--lr_schedule", default="piecewise", choices=["cyclic", "piecewise"]
    )
    parser.add_argument(
        "--lr_max", default=0.1, type=float, help="0.05 in Table 1, 0.2 in Figure 2"
    )
    parser.add_argument("--attack", default="none", type=str, choices=["none", "pgd"])
    parser.add_argument("--eps", default=1.0, type=float)
    parser.add_argument(
        "--attack_iters", default=5, type=int, help="n_iter of pgd for evaluation"
    )
    parser.add_argument("--alpha", default=100, type=float)
    parser.add_argument(
        "--normreg", default=0.0, type=float
    )  # Can be used for numerical stability
    parser.add_argument("--grad-reg", default=0.00, type=float)
    parser.add_argument("--lamb", default=1.00, type=float)
    parser.add_argument("--noise", default='0', type=str)
    parser.add_argument("--distance", default="l-mix", type=str)
    parser.add_argument("--model", default="default", type=str)
    parser.add_argument("--model_path", default="../models/default.pt", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument(
        "--n_train", default=-1, type=int, help="Number of training points."
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="weight decay aka l2 regularization",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--eps-sched", action="store_true")
    parser.add_argument("--emb-only", action="store_true")
    parser.add_argument("--emb-fine", action="store_true")
    parser.add_argument("--emb-adv-iters", default=1, type=int)
    parser.add_argument("--keep-one-hot", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--unit-ball", action="store_true")  # for unweighted l1
    parser.add_argument("--same-cost", action="store_true")
    parser.add_argument("--utility-max", action="store_true")
    parser.add_argument("--robust-baseline", action="store_true")
    parser.add_argument(
        "--utility-type",
        default="constant",
        choices=["constant", "additive", "multiplicative"],
        type=str,
    )
    parser.add_argument("--mixed-loss", default=False, action="store_true")
    return parser.parse_args()

def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()

def norm_embs(embs):
    for emb in embs:
        w = emb.weight
        if len(w.shape) == 1:
            continue
        for i in range(w.shape[1]):
            with torch.no_grad():
                emb.weight[:,i] = w[:, i] / torch.norm(w[:, i])

def train(
    model,
    x,
    y,
    optimizer,
    criterion,
    eps=0.0,
    pgd_steps=0,
    pgd_alpha=0.5,
    costs=None,
    w_rep=None,
    mixed_loss=False,
    utility_max=False,
    _lambda=0,
    cat_map=None,
    utility_type="constant",
    eps_part=1,
    eps_max=0,
    opte=None,
    dist="l1",
    iter_lim=10,
    emb_only=False,
    target_class=-1
):
    model.zero_grad()
    x[torch.where(w_rep == -1)] = 0
    if target_class != -1:
        costs[torch.where(y == target_class)] = 0 
    ut_loss = 0
    if pgd_steps != 0:
        model.eval()
        if utility_max:
            eps=20.4
            delta = attack_pgd_training(
                model,
                x,
                y,
                eps,
                pgd_alpha,
                pgd_steps,
                dist=dist,
                gains=costs,
                w_rep=w_rep,
                utility_type="constant",
                eps_part=eps_part,
                utility_max=utility_max,
                eps_max=eps_max
            )
        else:
            delta = attack_pgd_training(
                model,
                x,
                y,
                eps,
                pgd_alpha,
                pgd_steps,
                dist=dist,
                gains=costs,
                w_rep=w_rep,
                utility_type=utility_type,
                eps_part=eps_part,
                eps_max=eps_max,
                cat_map=cat_map,
                iter_lim=iter_lim
            )
        model.train()

        optimizer.zero_grad()
        if emb_only:
            opte.zero_grad()
            norm_embs(model.emb_layers)
            for p in model.parameters():
                p.requires_grad = False
            for emb in model.emb_layers:
                emb.weight.requires_grad = True
        
        if mixed_loss:
            if utility_max:
                x_adv = torch.autograd.Variable(x.data + delta, requires_grad=False)
                adv_output = model(x_adv)
                cl_output = model(x)
                adv_loss = util_loss(adv_output, delta, y, costs, w_vec)
                cl_loss = criterion(cl_output, y)
                loss = adv_loss * _lambda + cl_loss * (1 - _lambda)
                ut_loss = adv_loss.item()
            else:
                b_size = x.size()[0]
                x_adv = torch.autograd.Variable(x.data + delta, requires_grad=False)

                adv_output = model(x_adv)
                adv_loss = criterion(adv_output, y)
                if emb_only:
                    for p in model.parameters():
                        p.requires_grad = True
                cl_output = model(x)
                cl_loss = criterion(cl_output, y)

                if cl_loss > adv_loss and not utility_max:
                    loss = cl_loss
                else:
                    loss = adv_loss * _lambda + cl_loss * (1 - _lambda)
                ut_loss = adv_loss.item()
        else:
            model.eval()
            with torch.no_grad():
                cl_output = model(x)
                cl_loss = criterion(cl_output, y)
                x_adv = x + delta
                adv_output = model(x_adv)
                loss = criterion(adv_output, y) 
                ad_l_vec = torch.nn.functional.binary_cross_entropy_with_logits(adv_output.detach(), y.detach(), reduction='none')
                cl_l_vec = torch.nn.functional.binary_cross_entropy_with_logits(cl_output.detach(), y.detach(), reduction='none')
                loss_mask = ad_l_vec - cl_l_vec
                loss_mask = loss_mask.repeat(1, x.shape[1])
            
                delta.data[torch.where(loss_mask < 0)] = 0

            model.train()
            if loss < cl_loss:
                print("trigger", loss, cl_loss)
                adv_output = model(x + delta)
                loss = criterion(adv_output, y)
            else:
                adv_output = model(x + delta)
                loss = criterion(adv_output, y)

    else:
        model.train()
        delta = torch.randn(x.shape).cuda() * 0.05
        output = model(x + delta)
        adv_output = output
        loss = criterion(output, y)

    optimizer.zero_grad()

    if emb_only and _lambda != 1.0:

        loss.backward()
        opte.step()

        for p in model.parameters():
            p.requires_grad = not p.requires_grad
        if _lambda != 0:
            cl_output = model(x)
            cl_loss = criterion(cl_output, y) * _lambda
            cl_loss.backward()
    else:
        loss.backward()

    if _lambda != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) # To avoid catastrofic overfitting
        optimizer.step()
    model.eval()
    cl_output = model(x)
    
    if "foreign_request" in cat_map.keys():
        f0 = cat_map["foreign_request"]
        f1 = cat_map["zip_count_4w"]
        print(x[0, f0[0]:f0[1]+1], delta[0, f0[0]:f0[1]+1], torch.max(delta[0, f1[0]:f1[1]+1]), y[0], torch.sigmoid(cl_output).detach()[0],torch.sigmoid(adv_output).detach()[0])
    if "Merchant City" in cat_map.keys():
        f1 = cat_map["Merchant City"]
        print(torch.max(delta[0, f1[0]:f1[1]+1]), y[0], torch.sigmoid(cl_output).detach()[0],torch.sigmoid(adv_output).detach()[0])


    return (
        loss.item(),
        torch.round(torch.sigmoid(adv_output)).detach(),
        ut_loss * delta.shape[0],
        torch.round(torch.sigmoid(cl_output)).detach(),
    )


def train_model(
    epochs,
    train_loader,
    test_loader,
    dataset,
    model_name,
    device=torch.device("cpu"),
    eps=0.0,
    pgd_steps=0,
    unit_ball=False,
    utility_max=False,
    mixed_loss=False,
    eps_sched=False,
    _lambda=0,
    utility_type="constant",
    model_path=None,
    dist="l1",
    iter_lim=10,
    emb_only=False,
    args=None
):
    start_eps = eps
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    inp_dim = train_loader.dataset.X_train.shape[1]
    print(inp_dim)
    if pgd_steps != 0:
        pgd_alpha = args.alpha #2 * (1 / pgd_steps)
    else:
        pgd_alpha = 0
    torch.manual_seed(2)
    if model_name != "default":
        net = model_dict[model_name](inp_dim=inp_dim, cat_map=train_loader.dataset.cat_map).to(device)
    else:
        net = default_model_dict[dataset](inp_dim=inp_dim, cat_map=train_loader.dataset.cat_map).to(device)

    if args.checkpoint != "":
        net.load_state_dict(torch.load(args.checkpoint))
    torch.save(net.state_dict(), model_path + "_tmp_" + str(0))
    if emb_only:
        cl_params = list(net.shared.parameters()) + list(net.first_step.parameters()) + list(net.steps.parameters()) + list(net.fc.parameters()) + list(net.bn.parameters())
        emb_params = net.emb_layers.parameters()
        optm = Adam(cl_params, lr=0.001, weight_decay=5e-4)
        opte = Adam(emb_params, lr=0.01, weight_decay=0)
        for i, cat in enumerate(train_loader.dataset.cat_map.keys()):
            with torch.no_grad():
                #print(emb)
                emb = net.emb_layers[i]
                if cat in train_loader.dataset.costs.keys():
                    for emb_p in emb.parameters():
                        emb_p.data = torch.ones_like(emb_p.data)
                        print(cat, emb_p)
    else:
        optm = Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
        opte = None

    for epoch in range(epochs):
        
        epoch_loss = 0
        correct = 0
        clean_correct = 0
        ut_avg = 0
        eps_part = 1
        if eps_sched:
            if utility_max:
                eps_part = 1 * (epoch / epochs)
            else:
                eps_part = 1 * (2 * epoch / epochs)
            if eps_part > 1:
                eps_part = 1
        total = 0
        
        pgd_steps_eval = args.attack_iters

        if args.emb_only and args.emb_fine:
            counter = 0
            c_state = True
        elif args.emb_only and epoch % (args.emb_adv_iters + 1) != 0:
            c_state = False
        elif args.emb_only:
            c_state = True

        for bidx, (x_train, y_train, c, w_rep) in enumerate(train_loader):
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                w_rep = w_rep.to(device)
                c = c.to(device)
                total += len(x_train)

                if args.emb_only and c_state: # clean
                    _lambda = 1.0
                    pgd_steps = 0
                elif args.emb_only:
                    _lambda = 0.0
                    pgd_steps = args.attack_iters
                loss, predictions, ut_loss, clean_pred = train(
                    net,
                    x_train,
                    y_train,
                    optm,
                    criterion,
                    eps=eps,
                    pgd_steps=pgd_steps,
                    pgd_alpha=pgd_alpha,
                    costs=c,
                    w_rep=w_rep,
                    utility_max=utility_max,
                    mixed_loss=mixed_loss,
                    _lambda=_lambda,
                    cat_map=train_loader.dataset.cat_map,
                    utility_type=utility_type,
                    eps_part=eps_part,
                    eps_max=train_loader.dataset.max_eps,
                    dist=dist,
                    iter_lim=iter_lim,
                    opte=opte,
                    emb_only=emb_only,
                    target_class=args.target_class
                )
                for idx, i in enumerate(predictions):
                    i = torch.round(i)
                    if i == y_train[idx]:
                        correct += 1
                for idx, i in enumerate(clean_pred):
                    i = torch.round(i)
                    if i == y_train[idx]:
                        clean_correct += 1
                ut_avg += ut_loss
                acc = correct / total
                clean_acc = clean_correct / total
                epoch_loss += loss * len(x_train)

                if args.emb_only and args.emb_fine:
                    counter += 1
                    if counter == args.emb_adv_iters + 1:
                        counter = 0
                    if counter != 0:
                        c_state = False
                    else:
                        c_state = True

        epoch_loss = epoch_loss / len(train_loader.dataset)
        test_acc, test_loss = dataset_eval(net, test_loader, score="acc", device=device)
        test_rob_acc, _ = dataset_eval_rob(
            net,
            test_loader,
            eps=eps,
            pgd_steps=pgd_steps_eval,
            pgd_alpha=pgd_alpha,
            score="acc",
            w_rep=None,
            device=device,
            cat_map=train_loader.dataset.cat_map,
            utility_type=utility_type,
            utility_max=utility_max,
            eps_max=train_loader.dataset.max_eps,
            dist=dist,
            iter_lim=iter_lim,
            target_class=args.target_class
        )

        print(
            "Epoch {} Train Accuracy : {} (clean {}), Test Accuracy : {}, Test Robust Accuracy: {}".format(
                epoch + 1, acc, clean_acc, test_acc, test_rob_acc
            )
        )
        print(
            "Epoch {} Train Loss : {}, Test Loss : {}".format(
                (epoch + 1), epoch_loss, test_loss
            )
        )
        print(
            "Epoch {} Train Utility : {}".format(
                (epoch + 1), ut_avg / len(train_loader.dataset)
            )
        )

        if args.dataset == "ieeecis":
            embs0 = net.emb_layers[5].weight
            with torch.no_grad():
                d = torch.cdist(embs0.T, embs0.T)
            #print(embs0)
            #print(d)
            embs1 = net.emb_layers[3].weight
            with torch.no_grad():
                d = torch.cdist(embs1.T, embs1.T)
            #print(d)


            costs1 = train_loader.dataset.costs["P_emaildomain"]
            log_costs1 = costs1.copy()
            log_costs1[np.where(log_costs1 == 0)] = 0.01
            log_costs1 = np.log10(log_costs1)
        elif args.dataset == "bank_fraud":
            embs0 = net.emb_layers[3].weight
            with torch.no_grad():
                d = torch.cdist(embs0.T, embs0.T)
            #print(embs0)
            #rint(d)
            embs1 = net.emb_layers[3].weight
            with torch.no_grad():
                d = torch.cdist(embs1.T, embs1.T)


            costs1 = train_loader.dataset.costs["device_os"]
            log_costs1 = costs1.copy()
            log_costs1[np.where(log_costs1 == 0)] = 0.01
            log_costs1 = np.log10(log_costs1)

        torch.save(net.state_dict(), model_path + "_tmp_" + str(epoch + 1))


    return net


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Cuda Device Available")
        print("Name of the Cuda Device: ", torch.cuda.get_device_name())
        print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
    else:
        device = torch.device("cpu")
        print("No Cuda Device Detected")

    folder_path = args.data_dir

    data_train = get_dataset(
        args.dataset, args.data_dir, mode="train", same_cost=args.same_cost, cat_map=True, noise=args.noise, w_max=0
    )
    data_test = get_dataset(
        args.dataset, args.data_dir, mode="test", same_cost=args.same_cost, cat_map=True, noise=args.noise, w_max=0
    )
    train_loader = DataLoader(
        dataset=data_train, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        dataset=data_test, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    print(len(train_loader.dataset), len(test_loader.dataset))

    config = dict(
        epochs = args.epochs,
        dataset = args.dataset,
        model_name = args.model,
        eps=args.eps,
        pgd_steps=args.attack_iters,
        dist=args.distance,
        iter_lim=args.iter_lim,
        eps_sched=args.eps_sched,
        utility_type=args.utility_type,
        mixed_loss=args.mixed_loss
    )

    start_time = time.time()
    model = train_model(
        args.epochs,
        train_loader,
        test_loader,
        args.dataset,
        args.model,
        device=device,
        eps=args.eps,
        pgd_steps=args.attack_iters,
        unit_ball=args.unit_ball,
        utility_max=args.utility_max,
        mixed_loss=args.mixed_loss,
        _lambda=args.lamb,
        eps_sched=args.eps_sched,
        utility_type=args.utility_type,
        model_path=args.model_path,
        dist=args.distance,
        iter_lim=args.iter_lim,
        emb_only=args.emb_only,
        args=args
    )
    #print("--- Training time per epoch %s seconds ---" % ((time.time() - start_time) / args.epochs))
    torch.save(model.state_dict(), args.model_path)
    print("Saved: ", args.model_path)


if __name__ == "__main__":
    main()
