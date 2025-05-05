#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#https://github.com/mnsalim/IoT-Related-Dataset-and-Resources
#iotid20
import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
from itertools import pairwise
import math


lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from update import DatasetSplit

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, test_inference_new_het, test_inference, test_inference_new_het_by_attack, test_inference_new_het_lt_new, test_inference_new_het_lt_new_op, test_inference_metrics, test_inference_by_attack_server
from models import CNNMnist, CNNFemnist, CustomCNN, EdgeCustomCNN
from utils import get_dataset, average_weights, average_weights_, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem, proto_anomaly_detection
from plot import plot_fl_accuracies, plot_fedproto_accuracies, plot_metrics
import time
import time
from models import Proj, Embedder, DenseModel
from update import test_inference_all_classes, test_inference_metrics_proto, test_inference_metrics_proto_new, test_inference_by_attack_server_proto, test_inference_by_attack_server_proto_new

from plot import plot_accuracy_comparison, plot_accuracy_comparison_global
from inference import reconstruct_input,  evaluate_reconstruction, compute_baseline_mse #sample_original_data,
from poisoning import class_wise_outlier_detection, evaluate_outlier_detection, determine_attacker_outlier_status
from poisoning import intra_client_analysis, get_min_prototype_distances, inter_client_analysis, get_min_prototype_distances_simple, inter_client_analysis_max_distance, inter_client_analysis_isolation_forest

def split_server_and_client_params(client_mode, layers_to_client=[], adapter_hidden_dim=-1, dropout=0.):
    assert client_mode in ['none', 'representation', 'out_layer', 'interpolate']

    def is_on_client(name):
        if client_mode == 'none':
            return False
        elif client_mode == 'representation':
            return 'conv' in name or 'reshape_layer' in name
        elif client_mode == 'out_layer':
            return 'dense2' in name
        elif client_mode == 'interpolate':
            return True

    def is_on_server(name):
        return not is_on_client(name)

    return is_on_client, is_on_server


def split_server_and_client_params_old(client_mode, layers_to_client=[], adapter_hidden_dim=-1, dropout=0.):

    assert client_mode in ['none', 'representation', 'out_layer', 'interpolate'] #['none', 'res_layer', 'inp_layer', 'out_layer', 'adapter', 'interpolate', 'finetune'] 
    is_on_server = None
    if client_mode == 'none':
        def is_on_client(name):
            return False
    elif client_mode == 'representation':
        def is_on_client(name):
            return 'conv' in name
    elif client_mode == 'out_layer':
        def is_on_client(name):
            return 'fc' in name
    elif client_mode == 'interpolate':
        is_on_client = lambda _: True
        is_on_server = lambda _: True
    if is_on_server is None:
        def is_on_server(name): 
            return not is_on_client(name)
    return is_on_client, is_on_server


def average_weights_shared(local_weights, is_on_server):
    # Copy the weights from the first model as a starting point
    global_weights = copy.deepcopy(local_weights[0])

    # Initialize the global weights for shared parameters to zero
    for key in global_weights.keys():
        if is_on_server(key):
            print(key)
            global_weights[key] = torch.zeros_like(global_weights[key])
    
    # Sum the shared parameters across all local models
    for i in range(len(local_weights)):
        for key in global_weights.keys():
            if is_on_server(key):
                global_weights[key] += local_weights[i][key]

    # Divide the summed weights by the number of local models to average them
    for key in global_weights.keys():
        if is_on_server(key):
            global_weights[key] = torch.div(global_weights[key], len(local_weights))
    
    return global_weights

def before_fl(args, train_dataset, test_dataset, user_groups, user_groups_lt):
    accs, losses, f1_scores, precision_scores, f1_macros, acc_macros = [], [], [], [], [], []
    if args.attack_type == 'none':
        if args.diff_privacy:
            file_folder = '../save2_var_'+str(args.variance)+'/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/before_fl/'
        else:
            file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/before_fl/'
    else:
        file_folder = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+'/before_fl/' 
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    file_ext = 'before_fl_'+'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
    acc_file=open(file_folder + 'acc_' + file_ext + '.txt', 'w')
    f1_file=open(file_folder + 'f1_' + file_ext + '.txt', 'w')
    macro_acc_file=open(file_folder + 'macro_acc_' + file_ext + '.txt', 'w')
    macro_f1_file=open(file_folder + 'macro_f1_' + file_ext + '.txt', 'w')
    precision_file=open(file_folder + 'precision_' + file_ext + '.txt', 'w')
    acc_byclient_byclass = []
    for user_id in range(args.num_users):
        if args.dataset == 'cicids2017':
            local_model = DenseModel(args=args)
        elif args.dataset == 'edgeiiot':
            local_model = EdgeCustomCNN(args=args)
        else:
            local_model = CustomCNN(args=args)
        model = copy.deepcopy(local_model)
        local_model = LocalUpdate(args=args, dataset=train_dataset,idx = idx, idxs=user_groups[user_id], global_round=0)
        weight, train_losss, train_acc = local_model.update_weights(args, user_id, model=model, global_round=0)
        if args.dataset == 'cicids2017':
            test_model = DenseModel(args=args)
        elif args.dataset == 'edgeiiot':
            test_model = EdgeCustomCNN(args=args)
        else:
            test_model = CustomCNN(args=args)
        test_model.load_state_dict(weight)
        #acc, loss = test_inference_metrics(args, test_model, test_dataset,)
        acc, f1, precision, recall, acc_macro, f1_macro, loss = test_inference_metrics(args, test_model, test_dataset)
        print('test acc {:.5f}, test loss {:.5f}'.format(acc, loss))
        #print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        accs.append(acc)
        f1_scores.append(f1)
        precision_scores.append(precision)
        f1_macros.append(f1_macro)
        acc_macros.append(acc_macro)
        accs.append(acc)
        losses.append(loss)

        acc_byclass = []
        for class_ in range(args.num_classes):
            acc, loss = test_inference_by_attack_server(args, test_model, test_dataset, class_)
            acc_byclass.append(acc)
        acc_byclient_byclass.append(acc_byclass)
    file_acc_byclient_byclass = file_folder + 'acc_byclient_byclass_' + file_ext + '.txt'
    with open(file_acc_byclient_byclass, 'w') as file:
        file.write(str(acc_byclient_byclass))
    acc_file.write(str(accs))
    f1_file.write(str(f1_scores))
    macro_acc_file.write(str(acc_macros))
    macro_f1_file.write(str(f1_macros))
    precision_file.write(str(precision_scores))
    acc_file.close()
    f1_file.close()
    macro_acc_file.close()
    macro_f1_file.close()
    precision_file.close()
    return accs, losses


#link scaffold https://github.com/ongzh/ScaffoldFL/blob/master/src/scaffold_main.py
def Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list):
    timestamp = time.time()
    filename = f'../save/accuracies_FL{args.dataset}_{args.ways}w{args.shots}s{args.stdev}_alpha{args.alpha}e_{args.num_users}u{timestamp}.txt'
    #create folder if not exist
    # Create folder if it doesn't exist
    if args.attack_type == 'none':
        file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
    else:
        file_folder = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+'/' + args.alg + '/'
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    
    file_ext = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg+'_num_users' + str(args.num_users)# + '_timestamp' + str(timestamp)
    # Open the file using the created file name
    accuracies_file = open(file_folder + 'acc_' + file_ext + '.txt', 'w')
    #unweighted_acc_file = open(file_folder + 'unweighted_acc_' + file_ext + '.txt', 'w')
    macro_acc_file = open(file_folder + 'macro_acc_' + file_ext + '.txt', 'w')
    f1_file = open(file_folder + 'f1_' + file_ext + '.txt', 'w')
    macro_f1_file = open(file_folder + 'macro_f1_' + file_ext + '.txt', 'w')
    precision_file = open(file_folder + 'precision_' + file_ext + '.txt', 'w')
    recall_file = open(file_folder + 'recall_' + file_ext + '.txt', 'w')
    #accuracies_file = open(filename, 'w')
    #recall_file = open(f'../save/recall_FL{args.dataset}_{args.ways}w{args.shots}s{args.stdev}_alpha{args.alpha}e_{args.num_users}u{timestamp}.txt', 'w')
    
    idxs_users = np.arange(args.num_users)

    if args.alg == 'scaffold':
        #initiliase total delta to 0 (sum of all control_delta, triangle Ci)
        delta_c = copy.deepcopy(global_model.state_dict())
        #sum of delta_y / sample size
        delta_x = copy.deepcopy(global_model.state_dict())
        #model for local control varietes
        if args.dataset == 'cicids2017':
            control_local = [DenseModel(args=args) for i in range(args.num_users)]
            control_global = DenseModel(args=args)
        elif args.dataset == 'edgeiiot':
            control_local = [EdgeCustomCNN(args=args) for i in range(args.num_users)]
            control_global = EdgeCustomCNN(args=args)
        else:
            local_controls = [CustomCNN(args=args) for i in range(args.num_users)]
            control_global = CustomCNN(args=args)
        control_weights = control_global.state_dict()
        #local_models = [cifarCNN(args=args) for i in range(args.num_users)]
        
        for net in local_controls:
            net.load_state_dict(control_weights)
    if args.alg == 'fedalt' or args.alg == 'fedsim':
        client_mode = args.client_mode
        is_on_client, is_on_server = split_server_and_client_params(client_mode)

    train_loss, train_accuracy = [], []
    accuracies = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    f1_macros = []
    acc_macros = []
    fpr_scores = []
    if args.dataset == 'cicids2017':
        local_model_list = [DenseModel(args) for i in range(args.num_users)]
    elif args.dataset == 'edgeiiot':
        local_model_list = [EdgeCustomCNN(args) for i in range(args.num_users)]
    else:
        local_model_list = [CustomCNN(args) for i in range(args.num_users)]
    local_weights_prev = []
    

    for round in tqdm(range(args.rounds)):
        local_weights,  local_losses, local_accs = [], [], []
        
        print(f'\n | Global Training Round : {round + 1} |\n')
        if args.alg == 'scaffold':
            for ci in delta_c:
                delta_c[ci] = 0.0
            for ci in delta_x:
                delta_x[ci] = 0.0

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,idx = idx, idxs=user_groups[idx], global_round=round)
            if args.alg == 'scaffold':

                weights, loss, acc, local_delta_c, local_delta, control_local_w = local_model.update_weights_scaffold(args, idx, global_model, global_round=round, c_global=control_global, c_local=local_controls[idx])
                if round != 0:
                    local_controls[idx].load_state_dict(control_local_w)
                for w in delta_c:
                    if round==0:
                        delta_x[w] += weights[w]
                    else:
                        delta_x[w] += local_delta[w]
                        delta_c[w] += local_delta_c[w]
                local_weights.append(copy.deepcopy(weights))
                local_model_list[idx].load_state_dict(copy.deepcopy(weights))
            elif args.alg == 'fedalt':
                if round == 0:
                    w, loss, acc = local_model.update_weights_fedalt(args, idx, model=copy.deepcopy(global_model),local_weights = None, global_round=round, is_on_client=is_on_client, is_on_server=is_on_server)
                    local_weights_prev.append(copy.deepcopy(w))            
                else:
                    w, loss, acc = local_model.update_weights_fedalt(args, idx, model=copy.deepcopy(global_model), local_weights =local_weights_prev[idx], global_round=round, is_on_client=is_on_client, is_on_server=is_on_server)
                    local_weights_prev[idx] = copy.deepcopy(w)
                local_weights.append(copy.deepcopy(w))
                
            elif args.alg == 'fedsim':
                if round == 0:
                    w, loss, acc = local_model.update_weights_fedsim(args, idx, model=copy.deepcopy(global_model),local_weights=None, global_round=round, is_on_client=is_on_client, is_on_server=is_on_server)
                    local_weights_prev.append(copy.deepcopy(w))
                else:
                    w, loss, acc = local_model.update_weights_fedsim(args, idx, model=copy.deepcopy(global_model), local_weights=local_weights_prev[idx], global_round=round, is_on_client=is_on_client, is_on_server=is_on_server)
                    local_weights_prev[idx] = copy.deepcopy(w)
                    
                local_weights.append(copy.deepcopy(w))
            else:
                w, loss, acc = local_model.update_weights(args, idx, model=copy.deepcopy(global_model), global_round=round)
                local_weights.append(copy.deepcopy(w))
            if args.alg != 'scaffold':
                local_model_list[idx].load_state_dict(copy.deepcopy(w))
            

            
            local_losses.append(copy.deepcopy(loss))
            local_accs.append(copy.deepcopy(acc))

                
            
        if args.alg == 'scaffold':
            #update the delta C (line 16)
            for w in delta_c:
                delta_c[w] /= args.num_users
                delta_x[w] /= args.num_users
            
            #update global control variate (line17)
            control_global_W = control_global.state_dict()
            global_weights = global_model.state_dict()
            #equation taking Ng, global step size = 1
            for w in control_global_W:
                #control_global_W[w] += delta_c[w]
                if round == 0:
                    global_weights[w] = delta_x[w]
                else:
                    global_weights[w] += delta_x[w]
                    control_global_W[w] +=  delta_c[w]

            #update global model
            control_global.load_state_dict(control_global_W)
            global_model.load_state_dict(global_weights)
        elif args.alg == 'fedalt' or args.alg == 'fedsim':
            global_weights = average_weights_shared(local_weights, is_on_server)
        elif args.alg == 'krum':
            global_weights, _ = krum(args, local_weights)
        elif args.alg == 'median':
            global_weights = median(args,local_weights)
        elif args.alg == 'trimmed_mean':
            global_weights = trimmed_mean(args, local_weights)
        elif args.alg == 'geometric_median':
            global_weights = geometric_median(args, local_weights)
        elif args.alg == 'coordinate_wise_median':
            global_weights = coordinate_wise_median(args, local_weights)
        else:
            global_weights = average_weights_(local_weights)
        # update global weights

        
        global_model_ = copy.deepcopy(global_model)
        global_model_.load_state_dict(global_weights, strict=True)
        global_model = global_model_
        #local_model_list[idx] = local_model



        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        global_protos = []
        if args.alg == 'fedalt' or args.alg == 'fedsim':
            for user in range(args.num_users):
                acc, f1, precision, recall, acc_macro, f1_macro, loss = test_inference_metrics(args, local_model_list[user], test_dataset)
                print('User {}, test acc {:.5f}, test loss {:.5f}'.format(user, acc, loss))
        
        else:
            """class_counts = [0]*args.num_classes
            for data, target in test_dataset:
                for class_ in range(args.num_classes):
                    if target == class_:
                        class_counts[class_] += 1+
            for class_ in range(args.num_classes):
                print(f'Class {class_} has {class_counts[class_]} samples')"""
            #acc, loss = test_inference(args, global_model, test_dataset, global_protos)
            acc, f1, precision, recall, acc_macro, f1_macro, loss = test_inference_metrics(args, global_model, test_dataset)
            print('test acc {:.5f}, test loss {:.5f}'.format(acc, loss))
            #print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        accuracies.append(acc)
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_macros.append(f1_macro)
        acc_macros.append(acc_macro)
        """for class_ in range(args.num_classes):
            acc, loss = test_inference_by_attack_server(args, global_model, test_dataset, class_)
            print(f'Class {class_} acc: {acc}')"""

    acc_byclient_byclass = []
    for user in range(args.num_users):
        acc_byclass = []
        for class_ in range(args.num_classes):
            acc, loss = test_inference_by_attack_server(args, local_model_list[user], test_dataset, class_)
            acc_byclass.append(acc)
        acc_byclient_byclass.append(acc_byclass)
    acc_byclass = []
    for class_ in range(args.num_classes):
        acc, loss = test_inference_by_attack_server(args, global_model, test_dataset, class_)
        acc_byclass.append(acc)

    file_acc_byclient_byclass = file_folder + 'acc_byclient_byclass_' + file_ext + '.txt'
    with open(file_acc_byclient_byclass, 'w') as file:
        file.write(str(acc_byclient_byclass))
    file.close()

    file_acc_byclass = file_folder + 'acc_byclass_' + file_ext + '.txt'
    with open(file_acc_byclass, 'w') as file:
        file.write(str(acc_byclass))
    file.close()

    accuracies_file.write(str(accuracies))
    f1_file.write(str(f1_scores))
    macro_acc_file.write(str(acc_macros))
    macro_f1_file.write(str(f1_macros))
    precision_file.write(str(precision_scores))
    recall_file.write(str(recall_scores))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)
    accuracies_file.close()
    f1_file.close()
    macro_acc_file.close()
    macro_f1_file.close()
    precision_file.close()
    recall_file.close()

    acc_file_name = file_folder + 'acc_' + file_ext + '.txt'
    f1_file_name = file_folder + 'f1_' + file_ext + '.txt'
    macro_acc_file_name = file_folder + 'macro_acc_' + file_ext + '.txt'
    macro_f1_file_name = file_folder + 'macro_f1_' + file_ext + '.txt'
    precision_file_name = file_folder + 'precision_' + file_ext + '.txt'
    recall_file_name = file_folder + 'recall_' + file_ext + '.txt'
    output_file_name = file_folder + 'metrics_plot_' + file_ext + '.pdf'
    plot_metrics(acc_file_name, f1_file_name, macro_acc_file_name, macro_f1_file_name, precision_file_name, recall_file_name, output_file_name)

    """file_name = file_folder + 'acc_' + file_ext + '.txt'
    plot_fl_accuracies(filename)
    file_name = file_folder + 'f1_' + file_ext + '.txt'
    plot_fl_accuracies(filename)
    file_name = file_folder + 'macro_acc' + file_ext + '.txt'
    plot_fl_accuracies(filename)
    file_name = file_folder + 'macro_f1' + file_ext + '.txt'
    plot_fl_accuracies(filename)"""

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def aggregate_scaffold(args, res_cache, global_params_dict, c_global):
    # Extract the delta values for model parameters and control variates from res_cache
    y_delta_cache = list(zip(*res_cache))[0]
    c_delta_cache = list(zip(*res_cache))[1]
    
    global_lr = args.lr

    # Filter parameters that require gradients
    trainable_parameters = filter(
        lambda param: param.requires_grad, global_params_dict.values()
    )

    # Update global model
    avg_weight = torch.tensor(
        [1 / args.num_users for _ in range(args.num_users)],
        device=args.device,
    )
    for param, y_del in zip(trainable_parameters, zip(*y_delta_cache)):
        x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
        param.data += global_lr * x_del

    # Update global control variates
    for c_g, c_del in zip(c_global, zip(*c_delta_cache)):
        c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
        # Since client_id_indices is equal to args.num_users, scaling factor is just 1
        c_g.data += global_lr * c_del

    return global_params_dict, c_global


def aggregate_fedopt(local_weights, global_model):
        pseudo_grads = self.get_client_pseudo_grads()
        client_num_train = 1

        pseudo_grads = [fut.wait() for fut in pseudo_grads]
        client_num_train = [fut.wait() for fut in client_num_train]
        total_train = sum(client_num_train)

        self.optimizer.zero_grad()

        # probably need to reavluate this...
        for (param_name,param) in self.model.state_dict().items():
            self.model.get_parameter(param_name).grad = torch.zeros_like(param)

            for n_train,pseudo_grad in zip(client_num_train,pseudo_grads):
                self.model.get_parameter(param_name).grad = self.model.get_parameter(param_name).grad + (n_train / total_train) * pseudo_grad[param_name]

        self.optimizer.step()

def aggregate_mapping_layers_(local_weights):
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

import torch

"""def krum(args, local_weights, f=0):
    # Convert local weights to a list of tensors
    gradients = [torch.Tensor(orderdict_tolist(gradient)) for gradient in local_weights]
    
    n = len(gradients)
    m = n - f - 2  # Number of gradients to consider for scoring
    
    # Calculate pairwise distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = torch.norm(gradients[i] - gradients[j])
    
    # Calculate scores for each gradient
    scores = []
    for i in range(n):
        sorted_distances, _ = torch.sort(distances[i])
        score = torch.sum(sorted_distances[:m + 1])  # Sum the m+1 smallest distances
        scores.append((score.item(), gradients[i]))

    # Select the gradient with the smallest score
    _, selected_gradient = min(scores, key=lambda x: x[0])

    # Return the selected gradient as a dictionary
    return list_todict(selected_gradient, args)"""

def krum(args, local_weights, f=0):
    """
    Krum aggregation method.

    Args:
        args: Argument parser or configuration object containing necessary attributes.
        local_weights (List[OrderedDict]): List of local model updates from clients.
        f (int): Number of Byzantine (malicious) clients to tolerate.

    Returns:
        OrderedDict: Aggregated gradient dictionary.
    """
    n = len(local_weights)
    m = n - f - 2  # Number of gradients to consider for scoring

    if m < 1:
        raise ValueError("Not enough clients to perform Krum with the given f.")

    # Convert OrderedDicts to flat lists
    gradients = [orderdict_tolist(weight) for weight in local_weights]

    # Convert flat lists to tensors
    gradients_tensors = [torch.tensor(gradient) for gradient in gradients]

    # Calculate pairwise Euclidean distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = torch.norm(gradients_tensors[i] - gradients_tensors[j]).item()
            distances[i, j] = distances[j, i] = distance

    # Calculate scores for each gradient
    scores = []
    for i in range(n):
        sorted_distances, _ = torch.sort(distances[i])
        score = torch.sum(sorted_distances[1:m+1]).item()  # Exclude distance to itself (0)
        scores.append((score, i, local_weights[i]))

    # Select the gradient with the smallest score
    scores.sort(key=lambda x: x[0])
    selected_score, selected_index, selected_gradient = scores[0]

    # Determine eliminated gradients
    eliminated = [i for i in range(n) if i != selected_index]

    # Print results
    print("=== Krum Aggregation ===")
    print(f"Selected Gradient: Client {selected_index + 1} with score {selected_score}")
    print("Eliminated Gradients:")
    for idx in eliminated:
        print(f"  Client {idx + 1}")
    print("========================\n")

    return selected_gradient, eliminated



def multi_krum(args, local_weights, f=0, num_selected=8):
    """
    Multi-Krum aggregation method.

    Args:
        args: Argument parser or configuration object containing necessary attributes.
        local_weights (List[OrderedDict]): List of local model updates from clients.
        f (int): Number of Byzantine (malicious) clients to tolerate.
        num_selected (int): Number of gradients to select for aggregation.

    Returns:
        OrderedDict: Aggregated gradient dictionary.
    """
    n = len(local_weights)
    m = n - f - 2  # Number of gradients to consider for scoring

    if m < 1:
        raise ValueError("Not enough clients to perform Multi-Krum with the given f.")

    # Set num_selected to max_num_selected or lower
    num_selected = min(n-f-2, num_selected) 
    """if num_selected > n - f - 2:
        raise ValueError(f"num_selected should be <= {n - f - 2}, got {num_selected}")"""

    # Convert OrderedDicts to flat lists
    gradients = [orderdict_tolist(weight) for weight in local_weights]

    # Convert flat lists to tensors
    gradients_tensors = [torch.tensor(gradient) for gradient in gradients]

    # Calculate pairwise Euclidean distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = torch.norm(gradients_tensors[i] - gradients_tensors[j]).item()
            distances[i, j] = distances[j, i] = distance

    # Calculate scores for each gradient
    scores = []
    for i in range(n):
        sorted_distances, _ = torch.sort(distances[i])
        score = torch.sum(sorted_distances[1:m+1]).item()  # Exclude distance to itself (0)
        scores.append((score, i, local_weights[i]))

    # Sort gradients based on their scores (lower is better)
    scores.sort(key=lambda x: x[0])

    # Select the top 'num_selected' gradients with the smallest scores
    selected = scores[:num_selected]
    selected_indices = [item[1] for item in selected]
    selected_gradients = [item[2] for item in selected]

    # Determine eliminated gradients
    eliminated = [i for i in range(n) if i not in selected_indices]

    # Aggregate by averaging
    # Convert selected OrderedDicts to flat lists and then to tensors
    selected_flat_lists = [orderdict_tolist(weight) for weight in selected_gradients]
    selected_tensors = [torch.tensor(flat_list) for flat_list in selected_flat_lists]
    aggregated_tensor = torch.mean(torch.stack(selected_tensors), dim=0)

    # Convert the aggregated tensor back to OrderedDict
    template = local_weights[0]  # Assuming all state_dicts have the same keys and shapes
    aggregated_state_dict = list_todict(aggregated_tensor.tolist(), args)

    # Print results
    print("=== Multi-Krum Aggregation ===")
    print("Selected Gradients:")
    for score, idx, _ in selected:
        print(f"  Client {idx + 1} with score {score}")
    print("Eliminated Gradients:")
    for idx in eliminated:
        print(f"  Client {idx + 1}")
    print("==============================")
    """print(f"Aggregated Gradient (Multi-Krum):")
    for key, tensor in aggregated_state_dict.items():
        print(f"  {key}: {tensor}")
    print("\n")"""

    return aggregated_state_dict, eliminated


def median(args, local_weights):
    # Convert local weights to a list of tensors
    gradients = [torch.Tensor(orderdict_tolist(gradient)) for gradient in local_weights]
    
    # Stack all gradients along a new dimension and take the median along that dimension
    stacked_gradients = torch.stack(gradients)
    median_gradient = torch.median(stacked_gradients, dim=0).values
    
    # Return the median gradient as a dictionary
    return list_todict(median_gradient.tolist(), args)



def trimmed_mean(args, local_weights, trim_ratio=0.1):
    # Convert local weights to a list of tensors
    gradients = [torch.Tensor(orderdict_tolist(gradient)) for gradient in local_weights]
    
    # Number of elements to trim from each end
    trim_count = int(len(gradients) * trim_ratio)
    
    # Stack all gradients along a new dimension
    stacked_gradients = torch.stack(gradients)
    
    # Sort and trim along the new dimension
    sorted_gradients, sorted_indices = torch.sort(stacked_gradients, dim=0)
    trimmed_gradients = sorted_gradients[trim_count:-trim_count]  # Trim the extremes
    print("indices in trimmed mean", sorted_indices)
    
    # Compute the mean of the trimmed gradients
    trimmed_mean_gradient = torch.mean(trimmed_gradients, dim=0)
    
    # Return the trimmed mean gradient as a dictionary
    return list_todict(trimmed_mean_gradient.tolist(), args)

def geometric_median(args, local_weights, max_iter=100, eps=1e-5):
    # Convert local weights to a list of tensors
    gradients = [torch.Tensor(orderdict_tolist(gradient)) for gradient in local_weights]
    
    # Start with the mean as an initial estimate for the geometric median
    median = torch.mean(torch.stack(gradients), dim=0)
    
    for _ in range(max_iter):
        distances = torch.stack([torch.norm(gradient - median) for gradient in gradients])
        weights = 1.0 / torch.clamp(distances, min=eps)
        weights /= weights.sum()
        
        new_median = torch.sum(torch.stack([weight * gradient for weight, gradient in zip(weights, gradients)]), dim=0)
        
        if torch.norm(new_median - median) < eps:
            break
        median = new_median
    
    # Return the geometric median as a dictionary
    return list_todict(median.tolist(), args)


def coordinate_wise_median(args, local_weights):
    # Convert local weights to a list of tensors
    gradients = [torch.Tensor(orderdict_tolist(gradient)) for gradient in local_weights]
    
    # Stack all gradients along a new dimension and take the median along each coordinate
    stacked_gradients = torch.stack(gradients)
    coordinate_wise_median = torch.median(stacked_gradients, dim=0).values
    
    # Return the coordinate-wise median gradient as a dictionary
    return list_todict(coordinate_wise_median.tolist(), args)

import torch
import os

def angular_BRSA(args,local_weights, angle_threshold=0.5):
    # Convert local weights to a list of tensors
    gradients = [torch.Tensor(orderdict_tolist(gradient)) for gradient in local_weights]
    
    n = len(gradients)
    cosine_similarities = torch.zeros((n, n))
    
    # Normalize gradients to unit vectors for cosine similarity calculation
    normalized_gradients = [g / torch.norm(g) for g in gradients]
    
    # Calculate pairwise cosine similarities
    for i in range(n):
        for j in range(i + 1, n):
            cosine_similarity = torch.dot(normalized_gradients[i], normalized_gradients[j])
            cosine_similarities[i, j] = cosine_similarities[j, i] = cosine_similarity
    
    # Identify and filter out gradients with low cosine similarity
    aligned_gradients = []
    for i in range(n):
        if torch.mean(cosine_similarities[i]) >= angle_threshold:
            aligned_gradients.append(gradients[i])
    
    # Aggregate the aligned gradients (using mean in this example)
    if aligned_gradients:
        aggregated_gradient = torch.mean(torch.stack(aligned_gradients), dim=0)
    else:
        # Fallback: if no aligned gradients, use simple mean of all gradients
        aggregated_gradient = torch.mean(torch.stack(gradients), dim=0)
    
    # Return the aggregated gradient as a dictionary
    return list_todict(aggregated_gradient.tolist(), args)




def orderdict_tolist(w):
    weight_dict = dict(w.items())
    weight_list = []
    for key in weight_dict.keys():
        weight_list.extend(torch.flatten(weight_dict[key]).tolist())
    return weight_list

def list_todict(weight_list, args):
    if args.dataset == 'cicids2017':
        model = DenseModel(args)
    elif args.dataset == 'edgeiiot':
        model = EdgeCustomCNN(args)
    else:
        model = CustomCNN(args)  # Assuming CustomCNN is your model
    state_dict = model.state_dict()
    start_index = 0
    for key, value in state_dict.items():
        num_elements = value.numel()
        reshaped_tensor = torch.tensor(weight_list[start_index:start_index + num_elements]).reshape(value.shape)
        state_dict[key] = reshaped_tensor
        start_index += num_elements
    return state_dict


def aggregate_mapping_layers(local_weights):
    # Create a deep copy of the first client's weights as the base
    aggregated_weights = copy.deepcopy(local_weights[0])
    
    # Filter out mapping layer weights that need to be aggregated
    mapping_layer_keys = [k for k in aggregated_weights.keys() if 'conv1' in k or 'conv2' in k or 'dense1' in k]

    # Aggregate only the mapping layer weights
    for key in mapping_layer_keys:
        for i in range(1, len(local_weights)):
            aggregated_weights[key] += local_weights[i][key]
        aggregated_weights[key] = torch.div(aggregated_weights[key], len(local_weights))
    
    # Return updated weights for each client
    updated_weights_list = []
    for client_weights in local_weights:
        # Deep copy to keep the original structure
        client_updated_weights = copy.deepcopy(client_weights)
        
        # Only update the aggregated mapping layer weights
        for key in mapping_layer_keys:
            client_updated_weights[key] = aggregated_weights[key]
        
        updated_weights_list.append(client_updated_weights)
    
    return updated_weights_list
import math
import numpy as np
import torch

def apply_dp_to_protos(args, protos):
    """
    Applies Differential Privacy to the prototypes by clipping and adding Gaussian noise.
    
    Instead of using a predefined clipping threshold, this function determines
    the normal range of each prototype by computing its lower (min) and upper (max)
    element values. These bounds are then used to clip the noisy prototypes.
    
    Args:
        protos (dict): Dictionary of class prototypes (each is a torch.Tensor).
        args: An object (or argument parser) with DP parameters containing:
              - epsilon: Privacy budget.
              - delta: Failure probability.
              - device: The torch device (e.g., 'cpu' or 'cuda').
    
    Returns:
        dict: Dictionary of differentially private (noisy) prototypes.
    """
    print('Applying Differential Privacy to Prototypes...')
    dp_protos = {}
    epsilon = args.epsilon
    delta = args.delta

    # Compute the noise scale sigma using the (ε, δ)-DP Gaussian mechanism formula.
    # Here we use a basic form that does not incorporate a clip threshold since
    # the sensitivity will be implicitly bounded by the range of the prototype.
    #sigma = (np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    sigma = np.sqrt(args.variance)

    for cls, proto in protos.items():
        # Determine the "normal" range for this prototype.
        # We assume each prototype is a tensor; its min and max element values define the range.
        """lower_bound = torch.min(proto)
        upper_bound = torch.max(proto)
        print(f'Class {cls}: Lower Bound = {lower_bound}, Upper Bound = {upper_bound}')"""

        # Add Gaussian noise to the prototype.
        noise = torch.normal(mean=0, std=sigma, size=proto.shape).to(args.device)
        noisy_proto = proto + noise

        # Clip the noisy prototype element-wise so that its values lie within
        # the original prototype's min and max (i.e., its normal range).
        #noisy_proto = torch.clamp(noisy_proto, min=lower_bound.item(), max=upper_bound.item())
        #noisy_proto = torch.clamp(noisy_proto, min=-1*args.clip_threshold, max=args.clip_threshold)

        #print diff between noisy_proto and proto
        print(f"\n-----------------------Class--------------- {cls}")
        print(f"Original Prototype:\n{proto}")
        print(f"Noise:\n{noise}")
        print(f"Noisy Prototype:\n{noisy_proto}")
        print(f"Difference (Noisy - Original):\n{noisy_proto - proto}")  # or just 'noise'

        

        dp_protos[cls] = noisy_proto

    return dp_protos

def apply_dp_to_model(args, model):
    """
    Applies Differential Privacy to the model parameters by clipping and adding Gaussian noise.

    This function iterates over the model's parameters and, for each parameter tensor:
      1. Clips its values element-wise between -clip_threshold and +clip_threshold.
      2. Adds Gaussian noise with a standard deviation determined by args.variance.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be modified.
        args: An object (or argument parser) with DP parameters containing:
              - variance: The variance of the Gaussian noise (we use sqrt(variance) as sigma).
              - clip_threshold: The value at which to clip the parameters.
              - device: The torch device (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The model with DP-modified parameters.
    """
    print('Applying Differential Privacy to Model Parameters...')
    sigma = np.sqrt(args.variance)

    # Iterate over each parameter in the model.
    for name, param in model.named_parameters():
        if param.requires_grad:
            """data = param.data
            param_min = data.min().item()
            param_max = data.max().item()
            param_mean = data.mean().item()
            param_std = data.std().item()
            print(f"Parameter: {name}")
            print(f"  Shape: {data.shape}")
            print(f"  Min: {param_min:.4f}, Max: {param_max:.4f}")
            print(f"  Mean: {param_mean:.4f}, Std: {param_std:.4f}\n")"""
            
            # Create Gaussian noise with the same shape as the parameter.
            noise = torch.normal(mean=0, std=sigma, size=param.data.size()).to(args.device)
            
            # Add the noise to the parameter.
            param.data = param.data + noise

            # Clip the parameter values element-wise.
            param.data = torch.clamp(param.data, min=-args.clip_threshold, max=args.clip_threshold)

            # Optional: Print or log the parameter name and stats for debugging.
            # print(f'{name}: mean={param.data.mean()}, std={param.data.std()}')
    model = model.to(args.device)

    return model




def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, aggregated = 'none', classes_distribution=None):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')
    timestamp = time.time()
    """filename_wo = f'../save/accuracies_FedProto_wo_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}_alpga{args.alpha}_e_{args.num_users}u{timestamp}.txt'
    filename_w = f'../save/accuracies_FedProto_w_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}_alpga{args.alpha}_e_{args.num_users}u{time}.txt'
    accuracies_file_wo = open (filename_wo, 'w') #open(f'../save/accuracies_FedProto_wo_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u.txt', 'w')
    accuracies_file_w = open(filename_w, 'w') #open(f'../save/accuracies_FedProto_w_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u.txt', 'w')"""
    #filename = f'../save/accuracies_FL{args.dataset}_{args.ways}w{args.shots}s{args.stdev}_alpha{args.alpha}e_{args.num_users}u{timestamp}.txt'
    #create folder if not exist
    # Create folder if it doesn't exist
    if not args.classic_eval:
        args.alg = 'FedProto_new' 
    if not os.path.exists('../save2/'):
        os.makedirs('../save2/')
    if not os.path.exists('../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/'):
        os.makedirs('../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/')
        print('Created folder')
    else:
        print('Folder exists')
    if args.attack_type == 'none':
        if args.diff_privacy:
            file_folder = '../save2_var_'+str(args.variance)+'/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
        else:
            file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
    else:
        file_folder = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+'/' + args.alg + '/'
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    
    file_ext = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg+'_num_users' + str(args.num_users)# + '_timestamp' + str(timestamp)
    # Open the file using the created file name
    accuracies_file = open(file_folder + 'acc_' + file_ext + '.txt', 'w')
    #unweighted_acc_file = open(file_folder + 'unweighted_acc_' + file_ext + '.txt', 'w')
    macro_acc_file = open(file_folder + 'macro_acc_' + file_ext + '.txt', 'w')
    f1_file = open(file_folder + 'f1_' + file_ext + '.txt', 'w')
    macro_f1_file = open(file_folder + 'macro_f1_' + file_ext + '.txt', 'w')
    precision_file = open(file_folder + 'precision_' + file_ext + '.txt', 'w')
    recall_file = open(file_folder + 'recall_' + file_ext + '.txt', 'w')
    reconstruction_loss_file = file_folder + 'reconstruction_loss_' + file_ext + '.txt'
    psnr_file = file_folder + 'psnr_' + file_ext + '.txt'
    ssim_file = file_folder + 'ssim_' + file_ext + '.txt'
    baseline_loss_file = file_folder + 'baseline_loss_' + file_ext + '.txt'
    baseline_psnr_file = file_folder + 'baseline_psnr_' + file_ext + '.txt'
    outlier_file = file_folder + 'outlier_' + args.outlier_type + '_eliminated_' + str(args.eliminate_outlier) +'_attacker' + str(args.num_attacker) + '_' + file_ext + '.txt'
    global_protos = [] 
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    global_model = copy.deepcopy(local_model_list[0])
    train_loss, train_accuracy = [], []
    accuracies = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    f1_macros = []
    acc_macros = []
    fpr_scores = []
    eliminated_results = []
    
    model_buffer_s = 1
    if args.dataset == 'cicids2017':
        local_model_list = [DenseModel(args) for i in range(args.num_users)]
    elif args.dataset == 'edgeiiot':
        local_model_list = [EdgeCustomCNN(args) for i in range(args.num_users)]
    else:
        local_model_list = [CustomCNN(args) for i in range(args.num_users)]
        prev_models = [CustomCNN(args) for i in range(args.num_users)]
        #prev_models_all = [copy.deepcopy(prev_models) for i in range(model_buffer_s)]
    local_weights_prev = []
    acc_byclient_byclass = []
    baseline_loss = []
    baseline_psnr = []
    acc_byround_byclass = []
    buffer_round = 0
    for round in tqdm(range(args.rounds)):
        local_mapping_weights, local_weights, local_losses, local_protos = [], [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        reconstruct_loss = []
        psnr_all = []
        ssim_all = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,idx = idx, idxs=user_groups[idx], global_round=round)
            
            
            #w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            #w, loss, acc, protos = local_model.update_weights_het_prox(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            if args.alg == 'moon':
                w, loss, acc, protos = local_model.update_weights_moon(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), previous_models = prev_models[idx], global_round=round)
            else:
                
                w, loss, acc, protos = local_model.update_weights_het_prox_weighted(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)

            agg_protos = agg_func(protos)
            
            local_model_list[idx].load_state_dict(copy.deepcopy(w))
            if args.diff_privacy:
                agg_protos = apply_dp_to_protos(args, agg_protos)
                #local_model_list[idx] = apply_dp_to_model(args, local_model_list[idx])


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
            projection_model = copy.deepcopy(local_model_list[idx])
            if args.inference:
                reconstruct_loss_client = []
                psnr_client = []
                ssim_client = []
                baseline_loss_client = []
                baseline_psnr_client = []
                input_shape = (args.num_features,) 
                idxs = user_groups[idx]

                
                for (label,C_i) in local_protos[idx].items():
                    subset   = DatasetSplit(train_dataset, idxs)
                    data     = torch.stack([x for x, _ in subset])
                    min_feature_value = data.min(dim=0).values 
                    max_feature_value = data.max(dim=0).values 
                    X_reconstructed = reconstruct_input(args, projection_model, C_i,  
                                        learning_rate=0.1, num_iterations=1000, 
                                        lambda_l2=1e-4, lambda_l1=1e-4,
                                        min_feature_value=min_feature_value, max_feature_value=max_feature_value)
                
                    # Move to CPU and convert to NumPy for analysis
                    X_reconstructed_np = X_reconstructed.cpu().numpy()
                    sample_size = 1
                    #sampled_original_data = sample_original_data(train_dataset, sample_size=sample_size)

                    # Evaluate the attack
                    #similarity_metrics = evaluate_reconstruction(X_reconstructed, sampled_original_data[label])
                    mse_baseline, psnr_baseline = compute_baseline_mse(args, train_dataset, label, user_groups[idx])
                    distance, psnr, ssim = evaluate_reconstruction(args, X_reconstructed, train_dataset, label,user_groups[idx])
                    reconstruct_loss_client.append(distance)
                    psnr_client.append(psnr)
                    ssim_client.append(ssim)
                    baseline_loss_client.append(mse_baseline)
                    baseline_psnr_client.append(psnr_baseline)
                
                reconstruct_loss.append(reconstruct_loss_client)
                psnr_all.append(psnr_client)
                ssim_all.append(ssim_client)
                baseline_loss.append(baseline_loss_client)
                baseline_psnr.append(baseline_psnr_client)
                    
        with open(reconstruction_loss_file, 'a') as file:
            file.write(str(reconstruct_loss))
            file.write('\n')
        with open(ssim_file, 'a') as file:
            file.write(str(ssim_all))
            file.write('\n')
        with open(psnr_file, 'a') as file:
            file.write(str(psnr_all))
            file.write('\n')
        
        with open(baseline_loss_file, 'a') as file:
            file.write(str(baseline_loss))
            file.write('\n')

        with open(baseline_psnr_file, 'a') as file:
            file.write(str(baseline_psnr))
            file.write('\n')

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
            print('No aggregation')
            """local_weights_list = local_weights

            for idx in idxs_users:
                local_model = copy.deepcopy(local_model_list[idx])
                local_model.load_state_dict(local_weights_list[idx], strict=True)
                local_model_list[idx] =  local_model"""
            global_protos = proto_aggregation(local_protos)
        elif aggregated == 'mapping_layers':
            print('Aggregating mapping layers')
            # Aggregate mapping layers
            w_list_agg = aggregate_mapping_layers(local_mapping_weights)

            for idx, local_model in enumerate(local_model_list):
                # Load the aggregated state dict back into the model
                local_model.load_state_dict(w_list_agg[idx], strict=False)
            global_model.load_state_dict(w_list_agg[0], strict=False)
            global_protos = proto_aggregation(local_protos)

            """global_mapping_weights = aggregate_mapping_layers(local_mapping_weights)

            # Update local models' mapping layers with aggregated weights
            for idx in idxs_users:
                local_model = local_model_list[idx]
                model_state_dict = local_model.state_dict()
                model_state_dict.update(global_mapping_weights)
                local_model.load_state_dict(model_state_dict, strict=False)
                local_model_list[idx] = local_model"""
        elif aggregated == 'all_layers':
            #if args.proto_robust:
            # Perform anomaly detection before aggregating prototypes
            #s = intra_client_analysis(local_protos, args)
            #print('Intra-client analysis:', s)
            """get_min_prototype_distances(local_protos, args)
            inter_client_analysis(local_protos, args)
            trusted_clients = proto_anomaly_detection(local_protos, args)

            # Update local_model_list to only include trusted clients
            # For simplicity, we will use indices of trusted clients
            trusted_idxs = [idx for idx in idxs_users if idx in trusted_clients]
            print("-----------------------------------------")
            print(f'Trusted clients: {trusted_idxs}')
            print("-----------------------------------------")
            if args.num_attacker in trusted_clients:
                print('Attacker not eliminated')
            else:
                print('Attacker eliminated')"""
            if args.outlier_type == 'intra':
                    
                results = get_min_prototype_distances_simple(local_protos, args)
                print('Min distances:', results)
                eliminated_client = [results['client_id']]
                #correct_clients = [idx for idx in idxs_users if idx not in eliminated_client]
                
                #trusted_local_weights = [local_weights[i] for i, idx in enumerate(idxs_users) if idx in trusted_clients]
                # Aggregate weights
                #global_weights = average_weights_(trusted_local_weights)
            elif args.outlier_type == 'inter_distance':
                results = inter_client_analysis_max_distance(local_protos, args)
                print('Max distances:', results)
                eliminated_client = [results['client_id']]
                
            elif args.outlier_type == 'inter_forest':
                results = inter_client_analysis_isolation_forest(local_protos, args)
                print('Isolation forest:', results)
                eliminated_client = [results['client_id']]
            elif args.outlier_type == 'multi_krum':
                global_weights_krum, eliminated_client = multi_krum(args, local_weights, 0, args.num_users-2)
                print('Eliminated:', eliminated_client)
            if args.eliminate_outlier and args.outlier_type in ['intra', 'inter_distance', 'inter_forest', 'multi_krum']:
                if args.outlier_type in ['multi_krum']:
                    global_weights = global_weights_krum
                    global_protos = proto_aggregation({idx: local_protos[idx]  for i, idx in enumerate(idxs_users) if idx not in eliminated_client})
                    eliminated_results.append( {'round': round, 'eliminated_client': eliminated_client})
                else: 
                    local_weights_correct = [local_weights[i] for i, idx in enumerate(idxs_users) if idx not in eliminated_client]
                    global_weights = average_weights_(local_weights_correct)
                    global_protos = proto_aggregation({idx: local_protos[idx]  for i, idx in enumerate(idxs_users) if idx not in eliminated_client})
                    if args.outlier_type == 'intra':
                        labels = [results['class1'], results['class2']]
                    else:
                        labels = results['class_label']
                    eliminated_results.append( {'round': round, 'eliminated_client': eliminated_client, 'eliminated_class': labels})
                    
            else:
                global_weights = average_weights_(local_weights)
                global_protos = proto_aggregation(local_protos)
                print("---------------global protos-----------------")
                print(global_protos)
            # update global weights

            """if args.outlier_detection:
                outliers_per_class = class_wise_outlier_detection( local_protos,args.num_classes)
                attacked_clients = [args.num_attacker]
                #metrics = evaluate_outlier_detection(outliers_per_class, attacked_clients, args.num_users, args.num_classes)
                outlier_status = determine_attacker_outlier_status(outliers_per_class, args.num_attacker, train_dataset, user_groups[idx])
                print('Outlier status:', outlier_status)

                # Aggregate weights
                global_weights_krum, eliminated = multi_krum(args, local_weights, 0, args.num_users-2)#average_weights_(local_weights)
                
                if  args.num_attacker in eliminated:
                    print('Attacker eliminated')
                else:
                    print('Attacker not eliminated')"""
            #global_weights = average_weights_(local_weights)
            global_model_ = copy.deepcopy(global_model)
            global_model_.load_state_dict(global_weights, strict=True)
            global_model = global_model_
            # update global weights
            local_weights_list = local_weights

            for idx in idxs_users:
                local_model = copy.deepcopy(global_model)
                local_model.load_state_dict(local_weights_list[idx], strict=True)
                prev_models[idx] = copy.deepcopy(local_model)
                for param in prev_models[idx].parameters():
                    param.requires_grad = False

                prev_models[idx].eval()
                local_model_list[idx] = global_model #local_model


            

        # update global protos
        """if args.proto_robust:
            global_protos = proto_aggregation({idx: local_protos[idx] for idx in trusted_idxs})
        else:
            global_protos = proto_aggregation(local_protos)"""

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # Verify that all clients have the same mapping layers
        """if not verify_mapping_layers_among_clients(local_model_list):
            print('Mismatch found in mapping layers among clients.')
        else:
            print('All clients have identical mapping layers.')"""
        if args.classic_eval:
            acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
        else:
            acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt_new(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, classes_distribution)
        print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
        print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
        print('For all users (with protos), mean of proto loss is {:.5f}, std of test loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
        #accuracies_file_wo.write(str(acc_list_l))
        #accuracies_file_w.write(str(acc_list_g))
        acc, f1, precision, recall, acc_macro, f1_macro, loss = 0, 0, 0, 0, 0, 0, 0
        acc_byclient_byclass = []
        
        if aggregated == 'all_layers':# or aggregated == 'mapping_layers':
            if classic_eval:
                acc, f1, precision, recall, acc_macro, f1_macro, loss, acc_by_class_ = test_inference_metrics_proto(args, global_model, test_dataset, global_protos)
                if args.alg == 'moon':
                    acc, f1, precision, recall, acc_macro, f1_macro, loss = test_inference_metrics(args, global_model, test_dataset)
            else:
                acc, f1, precision, recall, acc_macro, f1_macro, loss = test_inference_metrics_proto_new(args,idx, global_model, test_dataset, global_protos, classes_distribution)
            
            for idx in idxs_users:
                acc_byclient_byclass.append(acc_by_class_)
            acc_byround_byclass.append(acc_by_class_)
        else:
            for idx in idxs_users:
                if classic_eval:
                    acc_, f1_, precision_, recall_, acc_macro_, f1_macro_, loss_, acc_by_class_ = test_inference_metrics_proto(args, local_model_list[idx], test_dataset, global_protos)
                else:
                    acc_, f1_, precision_, recall_, acc_macro_, f1_macro_, loss_ = test_inference_metrics_proto_new(args,idx, local_model_list[idx], test_dataset, global_protos, classes_distribution)
                acc += acc_/args.num_users
                f1 += f1_/args.num_users
                precision += precision_/args.num_users
                recall += recall_/args.num_users
                acc_macro += acc_macro_/args.num_users
                f1_macro += f1_macro_/args.num_users
                loss += loss_/args.num_users
                acc_byclient_byclass.append(acc_by_class_)
            acc_byround_byclass.append(acc_by_class_)
            
        print('test acc {:.5f}, test loss {:.5f}'.format(acc, loss))
        #print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        accuracies.append(acc)
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_macros.append(f1_macro)
        acc_macros.append(acc_macro)
        """for class_ in range(args.num_classes):
            acc, loss = test_inference_by_attack_server(args, global_model, test_dataset, class_)
            print(f'Class {class_} acc: {acc}')"""

    """acc_byclient_byclass = []
    for user in range(args.num_users):
        acc_byclass = []
        for class_ in range(args.num_classes):
            if classic_eval:
                acc, loss = test_inference_by_attack_server_proto(args, local_model_list[user], test_dataset, class_)
            else: 
                acc, loss = test_inference_by_attack_server_proto_new(args, local_model_list[user], test_dataset, class_, classes_distribution)
            acc_byclass.append(acc)
        acc_byclient_byclass.append(acc_byclass)"""
    """acc_byclass = []
    for class_ in range(args.num_classes):
        acc, loss = test_inference_by_attack_server_proto(args, global_model, test_dataset, class_)
        acc_byclass.append(acc)"""

    file_acc_byclient_byclass = file_folder + 'acc_byclient_byclass_' + file_ext + '.txt'
    with open(file_acc_byclient_byclass, 'w') as file:
        file.write(str(acc_byclient_byclass))
    file.close()
    file_acc_byround_byclass = file_folder + 'acc_byround_byclass_' + file_ext + '.txt'
    with open(file_acc_byround_byclass, 'w') as file:
        file.write(str(acc_byround_byclass))
    file.close()

    with open(outlier_file, 'a') as file:
        file.write(str(eliminated_results))
        file.write('\n')
    file.close()

    """file_acc_byclass = file_folder + 'acc_byclass_' + file_ext + '.txt'
    with open(file_acc_byclass, 'w') as file:
        file.write(str(acc_byclass))
    file.close()"""

    accuracies_file.write(str(accuracies))
    f1_file.write(str(f1_scores))
    macro_acc_file.write(str(acc_macros))
    macro_f1_file.write(str(f1_macros))
    precision_file.write(str(precision_scores))
    recall_file.write(str(recall_scores))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)
    accuracies_file.close()
    f1_file.close()
    macro_acc_file.close()
    macro_f1_file.close()
    precision_file.close()
    recall_file.close()
    recall_file.close()

    acc_file_name = file_folder + 'acc_' + file_ext + '.txt'
    f1_file_name = file_folder + 'f1_' + file_ext + '.txt'
    macro_acc_file_name = file_folder + 'macro_acc_' + file_ext + '.txt'
    macro_f1_file_name = file_folder + 'macro_f1_' + file_ext + '.txt'
    precision_file_name = file_folder + 'precision_' + file_ext + '.txt'
    recall_file_name = file_folder + 'recall_' + file_ext + '.txt'
    output_file_name = file_folder + 'metrics_plot_' + file_ext + '.pdf'
    plot_metrics(acc_file_name, f1_file_name, macro_acc_file_name, macro_f1_file_name ,precision_file_name,recall_file_name, output_file_name)
    #accuracies_file_wo.write('\n')
    #accuracies_file_w.write('\n')
    """for label in range(args.num_classes):
        print("--------------------------------------------------------------------------")
        print(f'For class {label}')
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_by_attack(args, local_model_list, test_dataset, user_groups_lt, global_protos, label)
        print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
        print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
        print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
        accuracies_file_wo.write(str(acc_list_l))
        accuracies_file_w.write(str(acc_list_g))
        accuracies_file_wo.write('\n')
        accuracies_file_w.write('\n')"""
    """for idx in idxs_users:
        acc, loss = test_inference(args, local_model_list[idx], test_dataset, global_protos)
        print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
        acc = test_inference_new_het(args, local_model_list, test_dataset,global_protos)
        print('For all users, mean of test acc is {:.5f}'.format(acc))"""

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)
    
    """accuracies_file_wo.close()
    accuracies_file_w.close()
    plot_fedproto_accuracies(filename_wo)
    plot_fedproto_accuracies(filename_w)"""


def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    prev_models = []
    model_buffer_s = 1

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,idx = idx, idxs=user_groups[idx], global_round=round)
            if args.alg == 'fedprox':
                w, loss, acc, protos = local_model.update_weights_prox(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
                """elif args.alg == 'moon':
                w, loss, acc, protos = local_model.update_weights_moon(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), prev_models[idx], global_round=round)"""
            else:
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
        prev_models=copy.deepcopy(local_model_list)

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    
    acc_list_l, acc_list_g = test_inference_new_het_lt_new(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, classes_distribution)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

def FedPCL(args,  train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    global_protos = {}
    global_avg_protos = {}
    local_protos = {}
    if not os.path.exists('../save2/'):
        os.makedirs('../save2/')
    if not os.path.exists('../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/'):
        os.makedirs('../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/')
        print('Created folder')
    else:
        print('Folder exists')
    file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    
    file_ext = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg+'_num_users' + str(args.num_users)# + '_timestamp' + str(timestamp)
    # Open the file using the created file name
    accuracies_file = open(file_folder + 'acc_' + file_ext + '.txt', 'w')
    #unweighted_acc_file = open(file_folder + 'unweighted_acc_' + file_ext + '.txt', 'w')
    macro_acc_file = open(file_folder + 'macro_acc_' + file_ext + '.txt', 'w')
    f1_file = open(file_folder + 'f1_' + file_ext + '.txt', 'w')
    macro_f1_file = open(file_folder + 'macro_f1_' + file_ext + '.txt', 'w')
    precision_file = open(file_folder + 'precision_' + file_ext + '.txt', 'w')
    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    global_model = copy.deepcopy(local_model_list[0])
    train_loss, train_accuracy = [], []
    accuracies = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    f1_macros = []
    acc_macros = []
    fpr_scores = []
    
    local_weights_prev = []


    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        local_weights, local_loss1, local_loss2, local_loss_total,  = [], [], [], []
        idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset_list, idx = idx, idxs=user_groups[idx], global_round=round)
            w, w_urt, loss, protos = local_model.update_weights_fedpcl(args, idx, global_protos, global_avg_protos, backbone=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)
            

            local_weights.append(copy.deepcopy(w))
            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            #local_protos[idx] = copy.deepcopy(agg_protos)
            local_protos[idx] = {k: copy.deepcopy(v.detach()) for k, v in agg_protos.items()}




        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights[idx])

        # update global protos
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        loss_avg = sum(local_loss_total) / len(local_loss_total)
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))

        acc, f1, precision, recall, acc_macro, f1_macro, loss = 0, 0, 0, 0, 0, 0, 0
        acc_by_class_by_user = []
        with torch.no_grad():
            for idx in range(args.num_users):
                print('Test on user {:d}'.format(idx))
                local_test = LocalTest(args=args, dataset=test_dataset_list, idxs=user_groups_test[idx])
                local_model_for_test = copy.deepcopy(local_model_list[idx])
                local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                local_model_for_test.eval()
                acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                acc_, f1_, precision_, recall_, acc_macro_, f1_macro_, loss_, acc_by_class_  = local_test.test_inference_metrics(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                acc += acc_/args.num_users
                f1 += f1_/args.num_users
                precision += precision_/args.num_users
                recall += recall_/args.num_users
                acc_macro += acc_macro_/args.num_users
                f1_macro += f1_macro_/args.num_users
                loss += loss_/args.num_users
                acc_by_class_by_user.append(acc_by_class_)
            print('test acc {:.5f}, test loss {:.5f}'.format(acc, loss))
            #print('User {}, test acc {:.5f}, test loss {:.5f}'.format(idx, acc, loss))
            accuracies.append(acc)
            f1_scores.append(f1)
            recall_scores.append(recall)
            precision_scores.append(precision)
            f1_macros.append(f1_macro)
            acc_macros.append(acc_macro)
        acc_mtx = torch.zeros([args.num_users])
        loss_mtx = torch.zeros([args.num_users])
    """with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args = args, dataset = test_dataset_list, idxs = user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
            acc_mtx[idx] = acc
            loss_mtx[idx] = loss"""

    file_acc_byclient_byclass = file_folder + 'acc_byclient_byclass_' + file_ext + '.txt'
    with open(file_acc_byclient_byclass, 'w') as file:
        file.write(str(acc_by_class_by_user))
    file.close()

    """file_acc_byclass = file_folder + 'acc_byclass_' + file_ext + '.txt'
    with open(file_acc_byclass, 'w') as file:
        file.write(str(acc_byclass))
    file.close()"""

    accuracies_file.write(str(accuracies))
    f1_file.write(str(f1_scores))
    macro_acc_file.write(str(acc_macros))
    macro_f1_file.write(str(f1_macros))
    precision_file.write(str(precision_scores))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)
    accuracies_file.close()
    f1_file.close()
    macro_acc_file.close()
    macro_f1_file.close()
    precision_file.close()
    #recall_file.close()

    acc_file_name = file_folder + 'acc_' + file_ext + '.txt'
    f1_file_name = file_folder + 'f1_' + file_ext + '.txt'
    macro_acc_file_name = file_folder + 'macro_acc_' + file_ext + '.txt'
    macro_f1_file_name = file_folder + 'macro_f1_' + file_ext + '.txt'
    precision_file_name = file_folder + 'precision_' + file_ext + '.txt'
    output_file_name = file_folder + 'metrics_plot_' + file_ext + '.pdf'
    plot_metrics(acc_file_name, f1_file_name, macro_acc_file_name, macro_f1_file_name ,precision_file_name, output_file_name)

    return acc_mtx


def Federated(args):
    start_time = time.time()

    
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
        args.num_classes = 9
    elif args.dataset == '5gnidd':
        args.num_features = 34
        args.num_classes = 7
    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    print("n_list", n_list)
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
    print("k_list", k_list)

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
        elif args.dataset == 'edgeiiot':
            args.num_features = 97
            args.num_classes = 15
            local_model = EdgeCustomCNN(args=args)
            global_model = EdgeCustomCNN(args=args)
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
    file_folder = '../save2/_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) + '/' + args.alg + '/'

    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    file_ext = '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + args.num_users 
    #with open(f'../save/classes_distribution_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}_alpha{args.alpha}_e_{args.num_users}u_{time.time()}.txt', 'w') as f:    
    with open (file_folder + 'classes_distribution' + file_ext + '.txt', 'w') as f:
        for idx, user_classes in classes_distribution:
            f.write(f"User {idx}:\n")
            for label, count in user_classes.items():
                f.write(f"  Class {label}: {count} instances\n")
            f.write("\n")


    if args.alg == 'fedavg' or args.alg == 'fedprox':
        Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)
    elif args.alg == 'fedproto':
        aggregated = 'all_layers'
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,aggregated, classes_distribution)
    elif args.alg == 'fedpcl':
        backbone = Embedder(args)
        local_model_list = [Proj(args=args) for i in range(args.num_users)]
        acc_mtx = FedPCL(args, train_dataset, test_dataset, user_groups, user_groups_lt, backbone, local_model_list)
        print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_mtx),np.std(acc_mtx))   )
    elif args.alg == 'fedopt':
        print('Not implemented yet')
    else:
        Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    print (args)
    print('already flipped', args.alr_flipped)
    # set random seeds
    #args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cpu'
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
        args.num_classes = 9
    elif args.dataset == '5gnidd':
        args.num_features = 34
        args.num_classes = 7
        args.weighted_loss = True
        #args.criterion =
    elif args.dataset == 'cicids2017':
        args.num_features = 81#77#81
        args.num_classes = 25#11#25
    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    print("n_list", n_list)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'xiiotid' or args.dataset == 'ciciot' or args.dataset == '5gnidd' or args.dataset == 'cicids2017' or args.dataset == 'edgeiiot':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    print("k_list", k_list)

    #train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

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
        elif args.dataset == 'edgeiiot':
            args.num_features = 97
            args.num_classes = 15
            local_model = EdgeCustomCNN(args=args)
            global_model = EdgeCustomCNN(args=args)
        elif args.dataset == 'xiiotid':
            local_model = CustomCNN(args=args)
            global_model = CustomCNN(args=args)
        elif args.dataset == '5gnidd':
            local_model = CustomCNN(args=args)
            global_model = CustomCNN(args=args)
        elif args.dataset == 'cicids2017':
            local_model = DenseModel(args=args)
            global_model = DenseModel(args=args)
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)
        global_model.to(args.device)
        global_model.train()
    if args.dataset == 'edgeiiot':
        args.local_bs = 256
        args.lr = 0.001
        args.criterion = 'cross_entropy'
    elif args.dataset == 'cicids2017':
        args.local_bs = 256
        args.lr = 0.001
        args.criterion = 'cross_entropy'
    unique_labels = set(range(args.num_classes))
    # Save classes distribution between clients
    """classes_distribution = []
    for idx, user in user_groups.items():
        user_classes = {}
        for data_idx in user:
            label = train_dataset[int(data_idx)][1].item()  # Get the label for the data index
            if label not in user_classes:
                user_classes[label] = 0
            user_classes[label] += 1
        classes_distribution.append((idx, user_classes))"""

    # Print classes_distribution for debugging
    """print(classes_distribution)
    for alpha in [0.75]#, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]:
        args.alpha =alpha
        args.alg = 'fedproto'
        aggregated = 'none'
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,aggregated, classes_distribution)
                
        #Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)
        if args.alg != 'beforefl':
            file_folder_before = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/before_fl/'
            file_ext = 'acc_byclient_byclass_before_fl_'+'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
            file_name_before_fl = file_folder_before + file_ext + '.txt'
            file_folder_after = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
            file_ext_after = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + str(args.num_users)
            file_name_after_fl = file_folder_after + 'acc_byclient_byclass_' + file_ext_after + '.txt'
            #file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
            #file_ext = 'acc_comparaision_' + 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
            #output_file_name = file_folder + file_ext + '.pdf'

            plot_accuracy_comparison(args, file_name_before_fl, file_name_after_fl)            """
    
    # Save classes distribution to a file
    #file_path = '../save/classes_distribution_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}e_{args.num_users}u.txt'
    #with open(file_path, 'w') as f:
    """with open(f'../save/classes_distribution_{args.dataset}_{args.ways}w{args.shots}s{args.stdev}_alpha{args.alpha}_e_{args.num_users}u_{time.time()}.txt', 'w') as f:    
        for idx, user_classes in classes_distribution:
            f.write(f"User {idx}:\n")
            for label, count in user_classes.items():
                f.write(f"  Class {label}: {count} instances\n")
            f.write("\n")"""
    """for var in [10.0,1.0,0.1,0.01,0.001]:  
        args.diff_privacy = True   
        args.variance = var   """
    #for var in [50000, 10000, 5000, 1000, 500, 100, 50, 10]:  
        #args.diff_privacy = True  
        #args.diff_privacy = False 
        #args.variance = var 
    for alpha in [0.75,0.5,0.25]:#:,0.5,0.25]:#,0.5,0.25]:#,0.5,0.25]:#, 0.01, 0.001]:#, 0.1, 0.05, 0.01, 0.005]:# [ 0.05, 0.01, 0.005]:#, 0.25, 0.1]:#[0.5, 0.25, 0.1, 0.05, 0.01, 0.005]:#[0.75, #0.75, 0.5, 0.25, 0.1,
        args.alpha = alpha
        args.alr_flipped = "False"
        train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
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
        if args.attack_type == 'none':
            if args.diff_privacy:
                file_folder = '../save2_var_'+str(args.variance)+'/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) 
            else:
                file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users)
        else:
            file_folder = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)
        
        file_ext = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
        if not os.path.exists('../save_attack'):
            os.makedirs('../save_attack')
        if not os.path.exists('../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) ):
            os.makedirs('../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) )
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        with open (file_folder + 'classes_distribution_' + file_ext + '.txt', 'w') as f:
            for idx, user_classes in classes_distribution:
                f.write(f"User {idx}:\n")
                for label, count in user_classes.items():
                    f.write(f"  Class {label}: {count} instances\n")
                f.write("\n")
        

        """for alg in ['beforefl', 'fedavg', 'krum', 'median','trimmed_mean','fedprox']:#['fedproto']:#['beforefl','fedavg', 'fedprox', 'scaffold']:
            args.alg = alg
            args.attack_type = 'label-flipping'
            for attack_perc in [0.25,0.5,0.75]:
                args.flip_ratio = attack_perc
                for attackers in [2,6]:
                    args.num_attackers = attackers"""
        #for alg in ['fedproto']:#['beforefl','fedavg', 'fedprox', 'scaffold']:
        for alg in ['beforefl','fedproto']: #['beforefl','fedavg','fedprox','fedproto']:#['fedproto']:#:#, 'fedprox']:##, 'krum','median','trimmed_mean','fedavg', 'fedprox']:
            args.alg = alg
            classic_eval = True
            args.attack_type = 'none'
            #args.proto_robust = True
            """args.attack_type = 'label-flipping'
            for attack_perc in [0.1, 0.2,0.3,0.4,0.5]:#,0.5,0.75]:
                args.flip_ratio = attack_perc
                print('*****************************flip ratio********************************: ', args.flip_ratio)
                if args.alpha == 0.75:
                    attackers_ = [3]
                elif args.alpha == 0.5:
                    attackers_ = [2]
                elif args.alpha == 0.25:
                    attackers _= [9]
                for attackers in attackers_:#,6]:
                    #args.num_attackers = attackers
                    args.num_attacker = attackers
                    
                    args.eliminate_outlier = True
                    for outlier_type in [ 'intra', 'inter_distance', 'inter_forest','multi_krum']: #'intra', 'inter_distance', 'inter_forest', 
                        args.outlier_type = outlier_type"""


            print('*****************************Running algorithm********************************: ', args.alg)
            print("diff_privacy: ", args.diff_privacy)
            print("variance: ", args.variance)
            if args.alg == 'fedavg' or args.alg == 'fedprox':
                print('Running federated averaging')
                if args.dataset == 'cicids2017':
                    global_model = DenseModel(args)
                elif args.dataset == 'edgeiiot':
                    global_model = EdgeCustomCNN(args)
                else:
                    global_model = CustomCNN(args)
                Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)
            elif args.alg == 'fedproto' or args.alg == 'moon':
                aggregated = 'none' #'all_layers' #'none'#'mapping_layers' #
                args.aggregated = aggregated
                if args.dataset == 'cicids2017':
                    local_model_list = [DenseModel(args) for i in range(args.num_users)]
                elif args.dataset == 'edgeiiot':
                    local_model_list = [EdgeCustomCNN(args) for i in range(args.num_users)]
                else:
                    local_model_list = [CustomCNN(args) for i in range(args.num_users)]
                FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,aggregated, classes_distribution)
            elif args.alg == 'fedpcl':
                backbone = Embedder(args)
                local_model_list = [Proj(args=args) for i in range(args.num_users)]
                acc_mtx = FedPCL(args, train_dataset, test_dataset, user_groups, user_groups_lt, backbone, local_model_list)
                acc_mean = acc_mtx.mean().item()
                acc_std = acc_mtx.std().item()
                print(f'For all users, mean of test acc is {acc_mean:.5f}, std of test acc is {acc_std:.5f}')

            elif args.alg == 'fedopt':
                print('Not implemented yet')
            elif args.alg == 'beforefl':
                acc_mtx = before_fl(args, train_dataset, test_dataset, user_groups, user_groups_lt)
            else:
                if args.dataset == 'cicids2017':
                    global_model = DenseModel(args)
                elif args.dataset == 'edgeiiot':
                    global_model = EdgeCustomCNN(args)
                else:
                    global_model = CustomCNN(args)
                Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)
            
            #Federated_Learning(args, train_dataset, test_dataset, user_groups, user_groups_lt, global_model, classes_list)
            if args.alg != 'beforefl':
                if args.attack_type == 'none':
                    if args.diff_privacy:
                        file_folder_before = '../save2_var_'+str(args.variance)+'/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/before_fl/'
                    else:
                        file_folder_before = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/before_fl/'
                    
                else:
                    file_folder_before = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+'/before_fl/' 
                file_ext = 'acc_byclient_byclass_before_fl_'+'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
                file_name_before_fl = file_folder_before + file_ext + '.txt'
                if args.attack_type == 'none':
                    if args.diff_privacy:
                        file_folder_after = '../save2_var_'+str(args.variance)+'/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
                    else:
                        file_folder_after = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users)+ '/' + args.alg + '/'
                else:
                    file_folder_after = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+ '/' + args.alg + '/'
                #file_folder_after = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
                file_ext_after = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + str(args.num_users)
                print('file_ext_after: ', file_ext_after)
                file_name_after_fl = file_folder_after + 'acc_byclient_byclass_' + file_ext_after + '.txt'
                #file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
                #file_ext = 'acc_comparaision_' + 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
                #output_file_name = file_folder + file_ext + '.pdf'

                plot_accuracy_comparison(args, file_name_before_fl, file_name_after_fl)
                file_ext_after = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + str(args.num_users)
                file_name_after_fl = file_folder_after + 'acc_byclass_' + file_ext_after + '.txt'
                #output_file_name = file_folder + +'_global_' + file_ext + '.pdf'
                if os.path.exists(file_name_after_fl): # args.alg != 'fedproto' and args.alg != 'fedpcl':
                
                    plot_accuracy_comparison_global(args, file_name_before_fl, file_name_after_fl)