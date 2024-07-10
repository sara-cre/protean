#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, test_inference_new_het, test_inference
from models import CNNMnist, CNNFemnist, CustomCNN
from utils import get_dataset, average_weights, average_weights_, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem


def Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list):
    
    
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_accs = [], [], []
        print(f'\n | Global Training Round : {round + 1} |\n')


        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc = local_model.update_weights(args, idx, model=copy.deepcopy(global_model), global_round=round)
            

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_accs.append(copy.deepcopy(acc))

        global_weights = average_weights_(local_weights)
        # update global weights

        
        global_model_ = copy.deepcopy(global_model)
        global_model_.load_state_dict(global_weights, strict=True)
        global_model = global_model_
        #local_model_list[idx] = local_model



        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        global_protos = []
        acc, loss = test_inference(args, global_model, test_dataset, global_protos)
        print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
    
    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.dataset == 'ciciot':
        args.num_features = 46
        args.num_classes = 8
    elif args.dataset == 'xiiotid':
        args.num_features = 74
        args.num_classes = 10
    elif args.dataset == '5gnidd':
        args.num_features = 34
        args.num_classes = 7
    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'xiiotid' or args.dataset == 'ciciot' or args.dataset == '5gnidd':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)


    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            global_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            global_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i<10:
                    args.stride = [1,4]
                else:
                    args.stride = [2,2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]

            global_model.load_state_dict(initial_weight)
        elif args.dataset == 'ciciot':
            args.num_features = 46
            args.num_classes = 8
            global_model = CustomCNN(args=args)
        elif args.dataset == 'xiiotid':
            global_model = CustomCNN(args=args)
        elif args.dataset == '5gnidd':
            global_model = CustomCNN(args=args)
        global_model.to(args.device)
        global_model.train()


    Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)