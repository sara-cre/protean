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

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, test_inference_new_het, test_inference, test_inference_new_het_by_attack, test_inference_new_het_lt_new, test_inference_new_het_lt_new_op, test_inference_metrics, test_inference_by_attack_server
from models import CNNMnist, CNNFemnist, CustomCNN
from utils import get_dataset, average_weights, average_weights_, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from plot import plot_fl_accuracies, plot_fedproto_accuracies, plot_metrics
import time
import time
from models import Proj, Embedder
from update import test_inference_all_classes, test_inference_metrics_proto, test_inference_metrics_proto_new, test_inference_by_attack_server_proto, test_inference_by_attack_server_proto_new
from plot import plot_accuracy_comparison, plot_accuracy_comparison_global
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



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
    if not os.path.exists('../save_debug/'):
        os.makedirs('../save_debug/')
    if not os.path.exists('../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/'):
        os.makedirs('../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/')
        print('Created folder')
    else:
        print('Folder exists')
    file_folder = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
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
    local_model_list = [CustomCNN(args) for i in range(args.num_users)]
    local_weights_prev = []

    for round in tqdm(range(args.rounds)):
        local_mapping_weights, local_weights, local_losses, local_protos = [], [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,idx = idx, idxs=user_groups[idx], global_round=round)
            
            w, loss, acc, protos = local_model.update_weights_het_contrastive(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)
            local_model_list[idx].load_state_dict(copy.deepcopy(w))


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
        aggregated = 'none'
        if aggregated == 'none':
            print('No aggregation')
            """local_weights_list = local_weights

            for idx in idxs_users:
                local_model = copy.deepcopy(local_model_list[idx])
                local_model.load_state_dict(local_weights_list[idx], strict=True)
                local_model_list[idx] =  local_model"""
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

        # Example usage:
        # Assuming global_protos is a dictionary containing global prototypes,
        # and local_protos_list is a list of dictionaries, each containing local prototypes for a client.
        # num_classes and num_clients should be the number of classes and clients, respectively.
        
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
        all_client_protos = []
        all_client_labels = []
        for idx in idxs_users:
            print('----------------------------User {}'.format(idx))
            if args.classic_eval:
                acc_, f1_, precision_, recall_, acc_macro_, f1_macro_, loss_,acc_by_class, all_protos, all_labels = test_inference_metrics_proto(args, local_model_list[idx], test_dataset, global_protos, idx)
                acc_byclient_byclass.append(acc_by_class)
                if args.rounds == round + 1:
                    all_client_protos.append(all_protos)
                    all_client_labels.append(all_labels)
            else:
                acc_, f1_, precision_, recall_, acc_macro_, f1_macro_, loss_ = test_inference_metrics_proto_new(args,idx, local_model_list[idx], test_dataset, global_protos, classes_distribution)
            acc += acc_/args.num_users
            f1 += f1_/args.num_users
            precision += precision_/args.num_users
            recall += recall_/args.num_users
            acc_macro += acc_macro_/args.num_users
            f1_macro += f1_macro_/args.num_users
            loss += loss_/args.num_users
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
        print('----------------------------User {}'.format(user))
        acc_byclass = []
        for class_ in range(args.num_classes):
            if args.classic_eval:
                acc, loss = test_inference_by_attack_server_proto(args, local_model_list[user], test_dataset, global_protos, class_)
            else: 
                acc, loss = test_inference_by_attack_server_proto_new(args, local_model_list[user], test_dataset, class_, classes_distribution)
            acc_byclass.append(acc)
        acc_byclient_byclass.append(acc_byclass)"""
    """acc_byclass = []
    for class_ in range(args.num_classes):
        acc, loss = test_inference_by_attack_server_proto(args, global_model, test_dataset, class_)
        acc_byclass.append(acc)"""
    visualize_prototypes(global_protos, local_protos, args.num_classes, args.num_users)

    #combined_tsne_for_clients(all_client_protos, all_client_labels, padding=5, random_state=42)
    #visualize_prototypes_interactive(global_protos, local_protos, args.num_classes, args.num_users)
    #print_and_compare_prototypes(global_protos, local_protos, num_classes=args.num_classes, num_clients=args.num_users)
    file_acc_byclient_byclass = file_folder + 'acc_byclient_byclass_' + file_ext + '.txt'
    with open(file_acc_byclient_byclass, 'w') as file:
        file.write(str(acc_byclient_byclass))
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
    #accuracies_file_wo.write('\n')
    #accuracies_file_w.write('\n')


import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
#import plotly.express as px  

def flatten_prototypes(all_client_protos):
    """
    Flattens each prototype from 3D (batch_size, sequence_length, feature_dim)
    to 2D (batch_size, sequence_length * feature_dim) for use in t-SNE.
    """
    flattened_protos = []
    
    for client_protos in all_client_protos:
        # Flatten each prototype
        client_protos_flat = [proto.flatten() for proto in client_protos]
        flattened_protos.append(np.array(client_protos_flat))
    
    return flattened_protos

def remove_inconsistent_protos_and_labels(all_client_protos, all_client_labels):
    """
    Removes prototypes and corresponding labels if the prototype shape is inconsistent with the majority.
    """
    cleaned_protos = []
    cleaned_labels = []
    
    for client_protos, client_labels in zip(all_client_protos, all_client_labels):
        # Convert to list instead of NumPy array to avoid the inhomogeneous shape issue
        client_protos = list(client_protos)
        client_labels = list(client_labels)

        # Find the most common shape among prototypes
        shapes = [proto.shape for proto in client_protos]
        common_shape = max(set(shapes), key=shapes.count)

        # Collect only prototypes and labels that match the most common shape
        consistent_protos = [proto for proto in client_protos if proto.shape == common_shape]
        consistent_labels = [label for proto, label in zip(client_protos, client_labels) if proto.shape == common_shape]

        # Append cleaned prototypes and labels
        cleaned_protos.append(np.array(consistent_protos))  # Now convert to NumPy array after filtering
        cleaned_labels.append(np.array(consistent_labels))
    
    return cleaned_protos, cleaned_labels
    
    return cleaned_protos, cleaned_labels
def combined_tsne_for_clients(all_client_protos, all_client_labels, padding=5, random_state=42):
    """
    Perform t-SNE on combined prototypes from all clients to ensure consistent t-SNE results.
    
    Parameters:
    - all_client_protos: A list of arrays, where each element contains the prototypes for a client.
    - all_client_labels: A list of arrays, where each element contains the labels for a client.
    - padding: The amount of padding to add to the axis limits.
    - random_state: Seed for random state to ensure consistency.
    """
    print("------------------------------")
    print("Performing t-SNE on combined prototypes from all clients...")
    all_client_protos, all_client_labels = remove_inconsistent_protos_and_labels(all_client_protos, all_client_labels)
    print(all_client_labels)
    for i, client_protos in enumerate(all_client_protos):
        print(f"Client {i} has {len(client_protos)} prototypes")
        for j, proto in enumerate(client_protos):
            print(f"Client {i}, Prototype {j} shape: {np.array(proto).shape}")

    # Flatten prototypes before applying t-SNE
    all_client_protos = flatten_prototypes(all_client_protos)

    # Combine the flattened prototypes into a single array
    all_client_protos = np.vstack(all_client_protos)

    # Combine all prototypes and labels into one array
    combined_protos = np.vstack(all_client_protos)
    combined_labels = np.concatenate(all_client_labels)
    
    # Apply t-SNE on the combined data
    tsne = TSNE(n_components=2, random_state=random_state)
    combined_tsne_results = tsne.fit_transform(combined_protos)
    
    # Plot the combined t-SNE results
    plot_tsne_results(combined_tsne_results, combined_labels, padding=padding)

def plot_tsne_results(tsne_results, labels, title="t-SNE Visualization", padding=5):
    plt.figure(figsize=(10, 8))

    # Get min and max values for x and y axis
    x_min, x_max = tsne_results[:, 0].min() - padding, tsne_results[:, 0].max() + padding
    y_min, y_max = tsne_results[:, 1].min() - padding, tsne_results[:, 1].max() + padding
    
    for i, label in enumerate(np.unique(labels)):
        idx = np.where(np.array(labels) == label)
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f"Class {label}")
    
    plt.legend()
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # Set axis limits dynamically based on min and max values with padding
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.show()



def visualize_prototypes(global_protos, local_protos_dict, num_classes, num_clients):
    # Collect prototypes and labels
    all_prototypes = []
    labels = []
    for class_idx in range(num_classes):
        if class_idx in global_protos:
            proto = global_protos[class_idx][0].detach().cpu().numpy()
            all_prototypes.append(proto)
            labels.append(f'Global_Class_{class_idx}')
    for client_idx, local_protos in local_protos_dict.items():
        for class_idx, proto in local_protos.items():
            proto_np = proto.detach().cpu().numpy()
            all_prototypes.append(proto_np)
            labels.append(f'Client_{client_idx}_Class_{class_idx}')
    
    # Convert to numpy array
    all_prototypes = np.array(all_prototypes)
    
    # Set perplexity dynamically based on the number of prototypes
    n_samples = len(all_prototypes)
    perplexity = min(30, n_samples - 1)  # Perplexity must be less than n_samples
    
    # Apply t-SNE with the adjusted perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    prototypes_2d = tsne.fit_transform(all_prototypes)
    
    # Plot
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(labels):
        x, y = prototypes_2d[i]
        plt.scatter(x, y)
        plt.annotate(label, (x, y))
    plt.title('t-SNE Visualization of Prototypes')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    #plt.show()
    file_folder = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
    file_ext = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    plt.savefig(file_folder + 'prototypes_tsne_' + file_ext + '.pdf')


def visualize_prototypes_interactive(global_protos, local_protos_dict, num_classes, num_clients):
    # Collect prototypes and labels
    all_prototypes = []
    labels = []
    for class_idx in range(num_classes):
        if class_idx in global_protos:
            proto = global_protos[class_idx][0].detach().cpu().numpy()
            all_prototypes.append(proto)
            labels.append(f'Global_Class_{class_idx}')
    for client_idx, local_protos in local_protos_dict.items():
        for class_idx, proto in local_protos.items():
            proto_np = proto.detach().cpu().numpy()
            all_prototypes.append(proto_np)
            labels.append(f'Client_{client_idx}_Class_{class_idx}')
    
    # Convert prototypes to numpy array
    all_prototypes = np.array(all_prototypes)
    
    # Set perplexity dynamically
    n_samples = len(all_prototypes)
    perplexity = min(30, n_samples - 1)  # Perplexity must be less than n_samples
    
    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    prototypes_2d = tsne.fit_transform(all_prototypes)
    
    # Create a DataFrame with the reduced prototypes
    prototypes_df = pd.DataFrame(prototypes_2d, columns=['x', 'y'])
    prototypes_df['label'] = labels
    print("visualize_prototypes_interactive")
    
    # Plot using Plotly
    fig = px.scatter(prototypes_df, x='x', y='y', text='label')
    fig.update_traces(textposition='top center')
    fig.update_layout(title='t-SNE Visualization of Prototypes', width=800, height=600)
    fig.show()



def print_and_compare_prototypes(global_protos, local_protos_dict, num_classes, num_clients):
    """
    Compare prototypes across classes, clients, and between global and local prototypes.

    Args:
    - global_protos (dict): Global prototypes for each class.
    - local_protos_dict (dict): Dictionary of local prototypes for each client. 
                                Keys are client indices, and values are dictionaries 
                                with class indices as keys and tensors as values.
    - num_classes (int): Number of classes.
    - num_clients (int): Number of clients.

    Returns:
    - None
    """

    file_folder = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
    file_ext_global_vs_local = 'global_vs_local_'+'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
    file_name_global_vs_local = file_folder + file_ext_global_vs_local + '.txt'
                

    
    
    def print_prototype(proto, title="Prototype"):
        print(f"{title}: {proto.detach().cpu().numpy()}")

    
    def compare_prototypes(proto1, proto2):
        return torch.norm(proto1 - proto2).item(), cosine_similarity(proto1.unsqueeze(0), proto2.unsqueeze(0)).item()

    # (i) Compare prototypes between classes within the same client
    print("\nComparing prototypes between classes within the same client:")
    for client_idx, local_protos in local_protos_dict.items():
        print(f"\nClient {client_idx + 1}:")
        for class_a in range(num_classes):
            for class_b in range(class_a + 1, num_classes):
                # Ensure class indices are Python integers
                class_a_int = int(class_a)
                class_b_int = int(class_b)
                if class_a_int in local_protos and class_b_int in local_protos:
                    proto_a = local_protos[class_a_int]
                    proto_b = local_protos[class_b_int]
                    l2_dist, cos_sim = compare_prototypes(proto_a, proto_b)
                    print(f"Class {class_a_int} vs Class {class_b_int} - L2 Distance: {l2_dist:.4f}, Cosine Similarity: {cos_sim:.4f}")
                    #print_prototype(proto_a, title=f"Class {class_a_int} Prototype")
                    #print_prototype(proto_b, title=f"Class {class_b_int} Prototype")

    # (ii) Compare prototypes for the same class across clients
    print("\nComparing prototypes for the same class across clients:")
    for class_idx in range(num_classes):
        class_idx_int = int(class_idx)
        for client_a in range(num_clients):
            for client_b in range(client_a + 1, num_clients):
                if class_idx_int in local_protos_dict.get(client_a, {}) and class_idx_int in local_protos_dict.get(client_b, {}):
                    proto_a = local_protos_dict[client_a][class_idx_int]
                    proto_b = local_protos_dict[client_b][class_idx_int]
                    l2_dist, cos_sim = compare_prototypes(proto_a, proto_b)
                    print(f"Class {class_idx_int} - Client {client_a + 1} vs Client {client_b + 1} - L2 Distance: {l2_dist:.4f}, Cosine Similarity: {cos_sim:.4f}")
                    #print_prototype(proto_a, title=f"Client {client_a + 1} Class {class_idx_int} Prototype")
                    #print_prototype(proto_b, title=f"Client {client_b + 1} Class {class_idx_int} Prototype")

    # (iii) Compare global prototypes with local prototypes
        print("\nComparing global prototypes with local prototypes:")
        for class_idx in range(num_classes):
            if class_idx in global_protos:
                global_proto = global_protos[class_idx]
                for client_idx, local_protos in local_protos_dict.items():
                    if class_idx in local_protos:
                        local_proto = local_protos[class_idx]
                        
                        l2_dist, cos_sim = compare_prototypes(global_proto[0], local_proto)
                        print(f"Class {class_idx} - Global vs Client {client_idx + 1} - L2 Distance: {l2_dist:.4f}, Cosine Similarity: {cos_sim:.4f}")
                        #print_prototype(global_proto[0], title=f"Global Class {class_idx} Prototype")
                        #print_prototype(local_proto, title=f"Client {client_idx + 1} Class {class_idx} Prototype")



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

    args.num_features = 74
    args.num_classes = 9


    n_list = np.random.randint(args.ways - args.stdev + 1 , args.ways + args.stdev + 1, args.num_users)
    # load dataset and user groups
    k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    print("k_list", k_list)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)# load model
    
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
    file_folder = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users)
    
    file_ext = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
    
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    with open (file_folder + 'classes_distribution_' + file_ext + '.txt', 'w') as f:
        for idx, user_classes in classes_distribution:
            f.write(f"User {idx}:\n")
            for label, count in user_classes.items():
                f.write(f"  Class {label}: {count} instances\n")
            f.write("\n")
    global_model = CustomCNN(args)
    global_model.to(args.device)
    local_model_list = [CustomCNN(args) for _ in range(args.num_users)]
    for idx in range(args.num_users):
        local_model_list[idx].to(args.device)
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
    
    FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, aggregated='mapping_layers', classes_distribution=classes_distribution)
    """if args.alg != 'beforefl':
                if args.attack_type == 'none':
                    file_folder_before = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/before_fl/'
                else:
                    file_folder_before = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+'/before_fl/' 
                file_ext = 'acc_byclient_byclass_before_fl_'+'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
                file_name_before_fl = file_folder_before + file_ext + '.txt'
                if args.attack_type == 'none':
                    file_folder_after = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users)+ '/' + args.alg + '/'
                else:
                    file_folder_after = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+ '/' + args.alg + '/'
                #file_folder_after = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
                file_ext_after = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + str(args.num_users)
                file_name_after_fl = file_folder_after + 'acc_byclient_byclass_' + file_ext_after + '.txt'
                #file_folder = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
                #file_ext = 'acc_comparaision_' + 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
                #output_file_name = file_folder + file_ext + '.pdf'

                plot_accuracy_comparison(args, file_name_before_fl, file_name_after_fl)
                if  args.alg != 'fedproto':
                    file_ext_after = 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + str(args.num_users)
                    file_name_after_fl = file_folder_after + 'acc_byclass_' + file_ext_after + '.txt'
                    #output_file_name = file_folder + +'_global_' + file_ext + '.pdf'

                    plot_accuracy_comparison_global(args, file_name_before_fl, file_name_after_fl)"""

