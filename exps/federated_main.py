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
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, test_inference_new_het, test_inference, test_inference_new_het_by_attack
from models import CNNMnist, CNNFemnist, CustomCNN
from utils import get_dataset, average_weights, average_weights_, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from plot import plot_fl_accuracies, plot_fedproto_accuracies
import time
import time

def Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list):
    timestamp = time.time()
    filename = f'../save/accuracies_FL{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u{timestamp}.txt'

    # Open the file using the created file name
    accuracies_file = open(filename, 'w')
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    accuracies = []
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
        #print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        accuracies.append(acc)
    accuracies_file.write(str(accuracies))
    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)
    accuracies_file.close()
    plot_fl_accuracies(filename)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def aggregate_mapping_layers(local_weights):
    aggregated_weights = copy.deepcopy(local_weights[0])
    for key in aggregated_weights.keys():
        for i in range(1, len(local_weights)):
            aggregated_weights[key] += local_weights[i][key]
        aggregated_weights[key] = torch.div(aggregated_weights[key], len(local_weights))
    return aggregated_weights
def verify_mapping_layers_among_clients(models):
    reference_weights = models[0].state_dict()
    for idx, model in enumerate(models[1:], start=1):
        model_weights = model.state_dict()
        for key in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'dense1.weight', 'dense1.bias']:
            if not torch.equal(reference_weights[key], model_weights[key]):
                print(f'Discrepancy found between model 1 and model {idx + 1} for layer {key}')
                return False
    print('All mapping layers are identical across clients.')
    return True

def aggregate_weights(local_weights):
    aggregated_weights = copy.deepcopy(local_weights[0])
    for key in aggregated_weights.keys():
        for i in range(1, len(local_weights)):
            aggregated_weights[key] += local_weights[i][key]
        aggregated_weights[key] = torch.div(aggregated_weights[key], len(local_weights))
    return aggregated_weights

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, aggregated = 'none'):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')
    timestamp = time.time()
    filename_wo = f'../save/accuracies_FedProto_wo_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u{timestamp}.txt'
    filename_w = f'../save/accuracies_FedProto_w_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u{time}.txt'
    accuracies_file_wo = open (filename_wo, 'w') #open(f'../save/accuracies_FedProto_wo_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u.txt', 'w')
    accuracies_file_w = open(filename_w, 'w') #open(f'../save/accuracies_FedProto_w_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u.txt', 'w')
    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    global_model = copy.deepcopy(local_model_list[0])

    for round in tqdm(range(args.rounds)):
        local_mapping_weights, local_weights, local_losses, local_protos = [], [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)


            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']
            mapping_layers_weights = {k: v for k, v in w.items() if 'conv1' in k or 'conv2' in k or 'dense1' in k}
            local_mapping_weights.append(copy.deepcopy(mapping_layers_weights))


        """# Aggregate mapping layers
        global_mapping_weights = aggregate_weights(local_weights) #aggregate_mapping_layers(local_mapping_weights)

        # Update local models' mapping layers with aggregated weights
        for idx in idxs_users:
            local_model = local_model_list[idx]
            model_state_dict = local_model.state_dict()
            model_state_dict.update(global_mapping_weights)
            local_model.load_state_dict(model_state_dict, strict=False)
            local_model_list[idx] = local_model"""
        if aggregated == 'none':
            local_weights_list = local_weights

            for idx in idxs_users:
                local_model = copy.deepcopy(local_model_list[idx])
                local_model.load_state_dict(local_weights_list[idx], strict=True)
                local_model_list[idx] =  local_model
        elif aggregated == 'mapping_layers':
            print('Aggregating mapping layers')
            # Aggregate mapping layers
            global_mapping_weights = aggregate_mapping_layers(local_mapping_weights)

            # Update local models' mapping layers with aggregated weights
            for idx in idxs_users:
                local_model = local_model_list[idx]
                model_state_dict = local_model.state_dict()
                model_state_dict.update(global_mapping_weights)
                local_model.load_state_dict(model_state_dict, strict=False)
                local_model_list[idx] = local_model
        elif aggregated == 'all_layers':

            global_weights = average_weights_(local_weights)
            # update global weights

            
            global_model_ = copy.deepcopy(global_model)
            global_model_.load_state_dict(global_weights, strict=True)
            global_model = global_model_
            # update global weights
            local_weights_list = local_weights

            for idx in idxs_users:
                local_model = copy.deepcopy(global_model)
                local_model.load_state_dict(local_weights_list[idx], strict=True)
                local_model_list[idx] = global_model #local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    # Verify that all clients have the same mapping layers
    if not verify_mapping_layers_among_clients(local_model_list):
        print('Mismatch found in mapping layers among clients.')
    else:
        print('All clients have identical mapping layers.')
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
    accuracies_file_wo.write(str(acc_list_l))
    accuracies_file_w.write(str(acc_list_g))
    accuracies_file_wo.write('\n')
    accuracies_file_w.write('\n')
    for label in range(args.num_classes):
        print("--------------------------------------------------------------------------")
        print(f'For class {label}')
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_by_attack(args, local_model_list, test_dataset, user_groups_lt, global_protos, label)
        print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
        print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
        print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
        accuracies_file_wo.write(str(acc_list_l))
        accuracies_file_w.write(str(acc_list_g))
        accuracies_file_wo.write('\n')
        accuracies_file_w.write('\n')
    """for idx in idxs_users:
        acc, loss = test_inference(args, local_model_list[idx], test_dataset, global_protos)
        print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        acc = test_inference_new_het(args, local_model_list, test_dataset,global_protos)
        print('For all users, mean of test acc is {:.5f}'.format(acc))"""

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)
    
    accuracies_file_wo.close()
    accuracies_file_w.close()
    plot_fedproto_accuracies(filename_wo)
    plot_fedproto_accuracies(filename_w)


def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    
    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

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

    # Build models
    local_model_list = []
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

            local_model = CNNMnist(args=args)

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
            local_model = CNNFemnist(args=args)

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

            local_model.load_state_dict(initial_weight)
        elif args.dataset == 'ciciot':
            args.num_features = 46
            args.num_classes = 8
            local_model = CustomCNN(args=args)
            global_model = CustomCNN(args=args)
        elif args.dataset == 'xiiotid':
            local_model = CustomCNN(args=args)
            global_model = CustomCNN(args=args)
        elif args.dataset == '5gnidd':
            local_model = CustomCNN(args=args)
            global_model = CustomCNN(args=args)
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)
        global_model.to(args.device)
        global_model.train()

    unique_labels = set(range(args.num_classes))
    # Save classes distribution between clients
    classes_distribution = []
    for idx, user in user_groups.items():
        user_classes = {}
        for data_idx in user:
            label = train_dataset[int(data_idx)][1].item()  # Get the label for the data index
            if label not in user_classes:
                user_classes[label] = 0
            user_classes[label] += 1
        classes_distribution.append((idx, user_classes))

    # Print classes_distribution for debugging
    print(classes_distribution)

    # Save classes distribution to a file
    #file_path = '../save/classes_distribution_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u.txt'
    #with open(file_path, 'w') as f:
    with open(f'../save/classes_distribution_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u_{time.time()}.txt', 'w') as f:    
        for idx, user_classes in classes_distribution:
            f.write(f"User {idx}:\n")
            for label, count in user_classes.items():
                f.write(f"  Class {label}: {count} instances\n")
            f.write("\n")
    """if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)"""
    Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)
    aggregated_layers = ['none', 'mapping_layers', 'all_layers']
    aggregated = aggregated_layers[2]
    FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,aggregated)