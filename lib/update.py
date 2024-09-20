#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from models import CNNFemnist
from utils import add_noise_img
from torch.optim.lr_scheduler import StepLR
from torch.optim import Optimizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from losses import ConLoss, ConLoss_op, OptimizedLoss
from poisoning import label_flipping
from utils import agg_func
from torch.nn import functional as F

def contrastive_loss(proto1, proto2, label, margin=1.0):
    """
    Contrastive loss for a pair of prototypes.
    
    Args:
    - proto1, proto2: The prototypes to compare.
    - label: 1 if the prototypes are from the same class, 0 otherwise.
    - margin: The margin value for negative pairs.
    
    Returns:
    - loss: The computed contrastive loss.
    """
    distance = torch.norm(proto1 - proto2, p=2)
    loss = label * distance**2 + (1 - label) * torch.clamp(margin - distance, min=0.0)**2
    return loss



def get_loss_function(loss_name):
    if loss_name == 'nlloss':
        return nn.NLLLoss()
    elif loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'bce':
        return nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss

def split_server_and_client_params(client_mode, layers_to_client=[], adapter_hidden_dim=-1, dropout=0.):

    assert client_mode in ['none', 'representation', 'out_layer', 'interpolate']
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

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idx, idxs, global_round, mu=0.01):
        self.args = args
        self.idx = idx
        self.global_round = global_round
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = get_loss_function(args.criterion).to(self.device)
        self.idxs = idxs
        
        self.dataset = dataset

        self.mu = args.mu


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        args = self.args
        idxs_train = idxs[:int(1 * len(idxs))]
        attack_round = [2,4,6,8,10]
        if args.attack_type == 'label-flipping' and self.global_round in attack_round and args.num_attackers > self.idx:
            dataset = label_flipping(dataset, idxs_train, args.flip_ratio)
            print('after flipping', dataset.labels[idxs_train])
        print(f'len of train dataset: {len(idxs_train)}')
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=False)
        print(f"Trainloader length: {len(trainloader)}")
        return trainloader

    def update_weights(self,args, idx, model, global_round):
        global_model = copy.deepcopy(model)
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)


        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                if args.alg == 'fedprox':
                    proximal_term = 0.0

                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    #LOSS
                    loss = self.criterion(log_probs, labels) + (self.mu / 2) * proximal_term
                else:
                    loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()


                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


    

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_scaffold(self, args, idx, model, global_round, c_global, c_local):
        global_model = copy.deepcopy(model)
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        #optimizer = ScaffoldOptimizer(model.parameters(), lr=args.lr, weight_decay=1e-4)

        lr = args.lr
        control_global_w = c_global.state_dict()
        control_local_w = c_local.state_dict()
        count = 0
        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)
                count += 1
                model.zero_grad()
                log_probs, protos = model(images)
                
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()
                #optimizer.step(c_global, c_local)
                local_weights = model.state_dict()
                for w in local_weights:
                    #line 10 in algo 
                    local_weights[w] = local_weights[w] - lr*(control_global_w[w]-control_local_w[w])
                
                #update local model params
                model.load_state_dict(local_weights)

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        control_local_w = c_local.state_dict()
        control_global_w = c_global.state_dict()
        global_weights = global_model.state_dict()
        new_control_local_w = c_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        #model_weights -> y_(i)
        model_weights = model.state_dict()
        local_delta = copy.deepcopy(model_weights)
        for w in model_weights:
            #line 12 in algo
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * args.lr)
            #line 13
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]
    

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item(),control_delta, local_delta, new_control_local_w


    def update_weights_fedalt(self, args, idx, model, local_weights, global_round, is_on_server, is_on_client):
        global_model = copy.deepcopy(model)
        if local_weights is not None:
            global_weights = model.state_dict()
            for key in global_weights.keys():
                if is_on_client(key):
                    global_weights[key] = local_weights[key]
            model.load_state_dict(global_weights)
            
        
        # Determine which parameters are on the client and which are on the server
        #client_mode = 'representation'  # or any other mode you need
        #is_on_client, is_on_server = split_server_and_client_params(client_mode)
        
        # Separate model parameters into client and server parameters
        client_params = [p for n, p in model.named_parameters() if is_on_client(n)]
        server_params = [p for n, p in model.named_parameters() if is_on_server(n)]
        
        for p in client_params:
            p.requires_grad = True
        for p in server_params:
            p.requires_grad = False

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizers for the local updates
        if self.args.optimizer == 'sgd':
            optimizer_client = torch.optim.SGD(client_params, lr=self.args.lr_v, momentum=0.5)
            optimizer_server = torch.optim.SGD(server_params, lr=self.args.lr_u, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer_client = torch.optim.Adam(client_params, lr=self.args.lr_v, weight_decay=1e-4)
            optimizer_server = torch.optim.Adam(server_params, lr=self.args.lr_u, weight_decay=1e-4)

        # Step 1: Update personalized parameters (v_i)
        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)

                loss = self.criterion(log_probs, labels)

                # Update personalized parameters
                loss.backward()
                optimizer_client.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        for p in client_params:
            p.requires_grad = False
        for p in server_params:
            p.requires_grad = True

        # Step 2: Update shared parameters (u)
        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)

                if args.alg == 'fedprox':
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = self.criterion(log_probs, labels) + (self.mu / 2) * proximal_term
                else:
                    loss = self.criterion(log_probs, labels)

                # Update shared parameters
                loss.backward()
                optimizer_server.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()



    def update_weights_fedsim(self, args, idx, model, local_weights, global_round, is_on_server, is_on_client):
        # Determine which parameters are personalized (client) and which are shared (server)
        #client_mode = args.client_mode  # Assuming client_mode is passed in args
        #is_on_client, is_on_server = split_server_and_client_params(client_mode)

        # Separate model parameters into personalized (v_i) and shared (u)
        global_model = copy.deepcopy(model)
        if local_weights is not None:
            global_weights = global_model.state_dict()
            for key in global_weights.keys():
                if is_on_client(key):
                    global_weights[key] = local_weights[key]
            model.load_state_dict(global_weights)
        
        personalized_params = [p for n, p in model.named_parameters() if is_on_client(n)]
        shared_params = [p for n, p in model.named_parameters() if is_on_server(n)]
        model.requires_grad_(True) 

        lr_v = args.lr_v
        lr_u = args.lr_u

        # Set up the optimizer; assuming we have separate learning rates and optimizers for v and u
        if args.optimizer == 'sgd':
            client_optimizer_v = torch.optim.SGD(personalized_params, lr=lr_v, momentum=0.5)
            client_optimizer_u = torch.optim.SGD(shared_params, lr=lr_u, momentum=0.5)
        elif args.optimizer == 'adam':
            client_optimizer_v = torch.optim.Adam(personalized_params, lr=args.lr_v, weight_decay=1e-4)
            client_optimizer_u = torch.optim.Adam(shared_params, lr=args.lr_u, weight_decay=1e-4)

        device = next(model.parameters()).device
        model.to(device)  # Ensure the model is on the correct device

        avg_loss = 0.0
        avg_acc = 0.0
        count = 0

        # Assuming `self.trainloader` is the client_loader
        for step in range(args.train_ep):  # Assuming num_steps corresponds to train_ep in args
            for batch_idx, (x, y) in enumerate(self.trainloader):
                x, y = x.to(device), y.to(device)

                # Zero gradients for both optimizers
                client_optimizer_v.zero_grad()
                client_optimizer_u.zero_grad()

                # Forward pass
                yhat, protos = model(x)

                # Compute loss
                loss = self.criterion(yhat, y)
                avg_loss = avg_loss * count / (count + 1) + loss.item() / (count + 1)
                avg_acc = torch.eq(yhat.argmax(dim=1), y).sum().item() / y.size(0)
                count += 1

                # Backpropagate the loss to compute gradients
                loss.backward()

                # Gradient Clipping if necessary
                if args.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(personalized_params, args.clip_value)
                    torch.nn.utils.clip_grad_norm_(shared_params, args.clip_value)

                # Update personalized parameters vi
                client_optimizer_v.step()

                # Update shared parameters ui
                client_optimizer_u.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tAccuracy: {:.2f}%'.format(
                        global_round, idx, step, batch_idx * len(x),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(), avg_acc * 100))

        # Assuming the personalized and shared parameters are updated in place
        return model.state_dict(), avg_loss, avg_acc



    def update_weights_prox(self, idx, local_weights, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        if idx in local_weights.keys():
            w_old = local_weights[idx]
        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)
                if idx in local_weights.keys():
                    loss2 = 0
                    for para in w_avg.keys():
                        loss2 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss2 /= len(local_weights)
                    loss += loss2 * 150
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_het(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        print(f"Trainloader length: {len(self.trainloader)}")
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                # Print shapes to debug
                #print(f"Batch {batch_idx} - images shape: {images.shape}")
                #print(f"Batch {batch_idx} - labels shape: {labels.shape}")
                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)
                    

                loss = loss1 + loss2 * args.ld
                # Debug prints
                """print(f"Batch Index: {batch_idx}, Iteration: {iter}")
                print(f"loss1: {loss1.item()}, loss2: {loss2.item()}, combined loss: {loss.item()}")
                print(f"labels: {labels}, log_probs: {log_probs}, protos: {protos}")"""
                #print(f"proto_new: {proto_new}, global_protos: {global_protos}")

                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def update_weights_het_prox(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        global_model = copy.deepcopy(model)
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        print(f"Trainloader length: {len(self.trainloader)}")
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                # Print shapes to debug
                #print(f"Batch {batch_idx} - images shape: {images.shape}")
                #print(f"Batch {batch_idx} - labels shape: {labels.shape}")
                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                log_probs, protos = model(images)
                proximal_term = 0.0

                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                #LOSS
                loss_prox = (self.mu / 2) * proximal_term

                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)
                    

                loss = loss1 + loss2 * args.ld + loss_prox 
                # Debug prints
                """print(f"Batch Index: {batch_idx}, Iteration: {iter}")
                print(f"loss1: {loss1.item()}, loss2: {loss2.item()}, combined loss: {loss.item()}")
                print(f"labels: {labels}, log_probs: {log_probs}, protos: {protos}")"""
                #print(f"proto_new: {proto_new}, global_protos: {global_protos}")

                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    """print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))"""
                    print(f"| Global Round: {global_round} | User: {idx} | Epoch: {iter} | [{batch_idx * len(images)}/{len(self.trainloader.dataset)} ({100. * batch_idx / len(self.trainloader):.0f}%)] "
                        f"| Loss1: {loss1.item():.3f} | Loss2: {loss2.item():.3f} | LossProx: {loss_prox.item():.3f} | Combined: {loss.item():.3f} | Acc: {acc_val.item():.3f}")

                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label



    def update_weights_lg(self, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[]}
        loss_mse = nn.MSELoss().to(args.device)
        criterion_CL = ConLoss(temperature=0.07)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'1':[],'2':[],'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(self.device), labels.to(self.device)

                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)

                # compute supervised contrastive loss
                model.zero_grad()
                log_probs, features = model(reps)
                bsz = labels.shape[0]
                lp1, lp2 = torch.split(log_probs, [bsz, bsz], dim=0)
                loss1 = self.criterion_CE(lp1, labels)

                # compute regularized loss term
                loss2 = 0 * loss1
                if len(global_protos) == args.num_users:
                    if args.alg == 'fedproto':
                        # compute global proto-based distance loss
                        num, xdim = features.shape
                        features_global = torch.zeros_like(features)
                        for i, label in enumerate(labels):
                            features_global[i, :] = copy.deepcopy(global_protos[label.item()].data)
                        loss2 = loss_mse(features_global, features) / num * args.ld
                    elif args.alg == 'fedpcl':
                        # compute global proto based CL loss
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                        for i in range(args.num_users):
                            for label in global_avg_protos.keys():
                                if label not in global_protos[i].keys():
                                    global_protos[i][label] = global_avg_protos[label]
                            loss2 += criterion_CL(features, labels, global_protos[i])

                loss = loss2

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tLoss2: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss.item(),
                                    loss2.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        agg_protos_label = {}
        model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), label_g.to(self.device)

            with torch.no_grad():
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return model.state_dict(), [], epoch_loss, agg_protos_label




    def update_weights_lg_prox(self, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round=round):
        # Set mode to train model
        if idx in local_weights.keys():
            w_old = local_weights[idx]
        w_avg = model.state_dict()
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[]}
        loss_mse = nn.MSELoss().to(args.device)
        criterion_CL = ConLoss(temperature=0.07)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'1':[],'2':[],'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(self.device), labels.to(self.device)

                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)

                # compute supervised contrastive loss
                model.zero_grad()
                log_probs, features = model(reps)
                bsz = labels.shape[0]
                lp1, lp2 = torch.split(log_probs, [bsz, bsz], dim=0)
                loss1 = self.criterion_CE(lp1, labels)

                # compute regularized loss term
                loss2 = 0 * loss1
                if len(global_protos) == args.num_users:
                    if args.alg == 'fedproto':
                        # compute global proto-based distance loss
                        num, xdim = features.shape
                        features_global = torch.zeros_like(features)
                        for i, label in enumerate(labels):
                            features_global[i, :] = copy.deepcopy(global_protos[label.item()].data)
                        loss2 = loss_mse(features_global, features) / num * args.ld
                    elif args.alg == 'fedpcl':
                        # compute global proto based CL loss
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                        for i in range(args.num_users):
                            for label in global_avg_protos.keys():
                                if label not in global_protos[i].keys():
                                    global_protos[i][label] = global_avg_protos[label]
                            loss2 += criterion_CL(features, labels, global_protos[i])

                loss = loss2

                if idx in local_weights.keys():
                    loss3 = 0
                    for para in w_avg.keys():
                        loss3 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss3 /= len(local_weights)
                    loss += loss3 * 150

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tLoss2: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss.item(),
                                    loss2.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        agg_protos_label = {}
        model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), label_g.to(self.device)

            with torch.no_grad():
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return model.state_dict(), [], epoch_loss, agg_protos_label

    def update_weights_het_contrastive(self, args, idx, global_protos, model, global_round=round, local_protos=[]):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                            weight_decay=1e-4)
        print(f"Trainloader length: {len(self.trainloader)}")
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                # Print shapes to debug
                #print(f"Batch {batch_idx} - images shape: {images.shape}")
                #print(f"Batch {batch_idx} - labels shape: {labels.shape}")
                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss1 = self.criterion(log_probs, labels)
                loss3 = 0*loss1
                loss_mse = nn.MSELoss()
                loss_mse3 = nn.MSELoss(reduction='none')
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                    
                else:
                    instances_byclass = [torch.sum(labels == i).item() for i in range(args.num_classes)]
                    all_instances = sum(instances_byclass)
                    proto_new = copy.deepcopy(protos.data)
                    proto_diff = [copy.deepcopy(protos.data) for _ in range(len(global_protos.keys()))]
    
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data 
                            for class_ in global_protos.keys():
                                if class_ != label.item():
                                    #global_proto = global_protos[class_][0].clone()  # Clone to ensure no in-place modifications
                                    # Calculate the Euclidean distance between current proto and global proto of another class
                                    proto_diff[class_][i, :] = global_protos[class_][0].data
                                    #dist = torch.norm(protos - global_proto, p=2)
                                    #dist = torch.clamp(dist, min=0.1)
                                    # Penalize small distances (closer prototypes)
                                    #loss3 += torch.log(1.0 / (dist + 1e-6)) 
                                    
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                    margin = torch.tensor([loss2]*len(protos)).to(self.device)
                    # Now calculate the loss3 based on distances
                    for class_idx, class_ in enumerate(global_protos.keys()):
                        if class_ != label.item():  # Only consider classes that are different
                            dist = torch.norm(proto_diff[class_idx] - protos, p=2, dim=1)
                            #dist = torch.clamp(dist, min=0.1)  # Clamp to avoid very small distances
                            #scaling_factor = 0.001  # Adjust the scaling factor if needed
                            #loss3 += scaling_factor * torch.sum(torch.log(1.0 / (dist + 1e-6)))
                            #loss3 -= torch.sum(dist) 
                            #loss3 -= loss_mse(proto_diff[class_idx], protos)
                            dist_ = loss_mse3 (protos, proto_diff[class_idx]).mean(dim=1) #(protos - proto_diff[class_idx]) **2
                           
                            clamp = torch.clamp(margin - dist_, min=0.0)
                            loss3 +=  (clamp **2).mean()
                    
                    
                    loss3 = loss3 / len(global_protos.keys()) #*0.05/ (len(labels)*len(global_protos.keys()) * 2)
                

                #loss3 = torch.tensor(0.0, device=self.device)  # Initialize loss3 on the correct device

                

                # Iterate over pairs of prototypes
                """for class_ in global_protos.keys():
                    if class_ != label_g[i].item():
                        global_proto = global_protos[class_][0].clone()  # Clone to ensure no in-place modifications
                        loss3 += contrastive_loss(protos, global_proto, label=1)  # Positive pair

                    # Compare with prototypes of different classes
                    for other_class in local_protos.keys():
                        if other_class != class_:
                            other_proto = local_protos[other_class].clone()  # Clone to ensure no in-place modifications
                            loss3 += contrastive_loss(local_proto, other_proto, label=0)  # Negative pair"""




                """if len(global_protos) == args.num_users:
                    for i in range(args.num_users):
                        for label in global_protos.keys():
                            if label not in local_protos[i].keys():
                                local_protos[i][label] = global_protos[label][0]
                        loss3 += criterion_CL(features, labels, global_protos, local_protos[i])"""
                    
                
                loss = loss1 + loss2 * args.ld + loss3
                # Debug prints
                """print(f"Batch Index: {batch_idx}, Iteration: {iter}")
                print(f"loss1: {loss1.item()}, loss2: {loss2.item()}, combined loss: {loss.item()}")
                print(f"labels: {labels}, log_probs: {log_probs}, protos: {protos}")"""
                #print(f"proto_new: {proto_new}, global_protos: {global_protos}")

                loss.backward(retain_graph=True)
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]



                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(f"loss1: {loss1.item()}, loss2: {loss2.item()}, loss3: {loss3.item()}, total_loss: {loss.item()}")

                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def update_weights_fedpcl(self, args, idx, global_protos, global_avg_protos, backbone, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[]}
        criterion_CL = ConLoss(temperature=0.07)#OptimizedLoss(temperature=0.07)#
        torch.autograd.set_detect_anomaly(True)
        

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            # In update.py, before the error line
            #print(f"Accessing index {self.idxs[idx]} in dataset of length {len(self.dataset)}")

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                pretrained = False
                if pretrained:
                    with torch.no_grad():
                        reps = backbone(images)
                else:
                    reps = backbone(images)
                model.zero_grad()
                log_probs, features = model(reps)
                loss1 = self.criterion(log_probs, labels)
                loss2 = 0*loss1
                loss3 = 0*loss1

                if len(global_protos) == args.num_users:
                    f1 = features.unsqueeze(1)
                    features = f1
                    for i in range(args.num_users):
                        for label in global_avg_protos.keys():
                            if label not in global_protos[i].keys():
                                global_protos[i][label] = global_avg_protos[label]

                        
                        loss2 += criterion_CL(features, labels, global_protos[i])#criterion_CL(features, labels, global_avg_protos, global_protos)

                
                #print("loss2 in update_weights_fedpcl: ", loss2)
                loss = loss2 * args.ld + loss1 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tLoss2: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss.item(),
                                    loss2.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        agg_protos_label = {}
        model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images, labels = images.to(self.device), label_g.to(self.device)

            with torch.no_grad():
                reps = backbone(images)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return model.state_dict(), [], epoch_loss, agg_protos_label


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        print(f"| Validation | Accuracy: {accuracy}")
        return accuracy, loss




    

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        #idxs_test = idxs[:int(1 * len(idxs))]
        size = int(len(dataset))#/self.args.num_users)
        print(f"Size of testloader: {size}")
        idxs_test = np.random.choice(range(len(dataset)), size=size, replace=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                     batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[: , 0 : args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = get_loss_function(args.criterion).to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


    def test_inference_twoway(self, idx, args, global_protos, local_protos, backbone_list, local_model):
        device = args.device
        criterion = get_loss_function(args.criterion).to(device)
        loss_mse = nn.MSELoss()

        model = local_model
        model.to(args.device)
        loss, total, correct = 0.0, 0.0, 0.0

        # test (only use local model)
        print("len of testloader: ", len(self.testloader))
        
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)

            # generate representations by different backbone
            reps = backbone_list(images)

            probs, features = model(reps)

            # compute the dist between features and input protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
            for i in range(images.shape[0]):
                for j in range(args.num_classes):
                    if j in local_protos.keys():
                        d = loss_mse(features[i, :], local_protos[j]) # compare with local protos
                        dist[i, j] = d
            _, pred_labels = torch.min(dist, 1)

            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        loss /= (batch_idx + 1)
        print('| User: {} | Test Acc: {:.5f} | Test Loss: {:.5f}'.format(idx, acc, loss))

        return acc, loss

    def test_inference_metrics(self, idx, args, global_protos, local_protos, backbone_list, local_model):
        device = args.device
        criterion = get_loss_function(args.criterion).to(device)
        loss_mse = nn.MSELoss()

        model = local_model
        model.to(args.device)
        loss, total, correct = 0.0, 0.0, 0.0
        all_preds = []
        all_labels = []

        # test (only use local model)
        print("len of testloader: ", len(self.testloader))
        
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)

            # generate representations by different backbone
            reps = backbone_list(images)

            probs, features = model(reps)

            # compute the dist between features and input protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
            for i in range(images.shape[0]):
                for j in range(args.num_classes):
                    if j in local_protos.keys():
                        d = loss_mse(features[i, :], local_protos[j]) # compare with local protos
                        dist[i, j] = d
            _, pred_labels = torch.min(dist, 1)

            pred_labels = pred_labels.view(-1)
        
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_labels.cpu().numpy())

        print("len of total_w: ", total)
        print("len of correct_w: ", correct)
        acc = correct / total
        print('| Test Acc with protos: {:.5f}'.format( acc))
        #acc_list_g.append(acc_global)
        #loss_list.append(loss_proto)




        # Calculate other metrics using sklearn
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        # Calculate other metrics using sklearn with macro averaging
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision_macro = precision_score(all_labels, all_preds, average='macro')
        recall_macro = recall_score(all_labels, all_preds, average='macro')

        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Calculate FPR for each class
        fpr_per_class = {}
        unweighted_accuracy_sum = 0.0
        print('Confusion Matrix:', cm)
        acc_by_class = []
        for i in range(len(cm)):
            # False Positives (FP) and True Negatives (TN)
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fpr_per_class[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

            # Calculate accuracy for class i
            class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            unweighted_accuracy_sum += class_accuracy 
            acc_by_class.append(class_accuracy)
            print('Class: {} | Accuracy: {:.5f}'.format(i, class_accuracy))
        unweighted_accuracy = unweighted_accuracy_sum / len(cm)

        # Normalize loss
        loss = loss / len(self.testloader)
        
        print('Test Loss: {:.5f}'.format(loss))
        print('Test Accuracy: {:.5f}'.format(acc))
        print('F1 Score: {:.5f}'.format(f1))
        print('Precision: {:.5f}'.format(precision))
        print('Recall: {:.5f}'.format(recall))
        print('UnWeighted Accuracy: {:.5f}'.format(unweighted_accuracy))
        print('False Positive Rate per class:', fpr_per_class)

        print('F1 Score (Macro): {:.5f}'.format(f1_macro))
        print('Precision (Macro): {:.5f}'.format(precision_macro))
        print('Recall (Macro): {:.5f}'.format(recall_macro) )
        return acc, f1, precision, recall, unweighted_accuracy, f1_macro, loss, acc_by_class





def test_inference(args, model, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs, protos = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    
    accuracy = correct/total
    print(f"| Test | Loss: {loss} | Accuracy: {accuracy}")
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(10):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc

def test_inference_new_cifar(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        cnt = np.zeros(100)
        for i in range(100):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(100):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(args.num_classes):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct/total

    return acc


def test_inference_new_het_by_attack(args, local_model_list, test_dataset,user_groups_gt, global_protos=[], attack_label=0):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()
    device = args.device
    criterion = get_loss_function(args.criterion).to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        loss, total, correct = 0.0, 0.0, 0.0
        model = local_model_list[idx]
        model.to(args.device)
        test_loader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        test_loader_filtered = [(images, labels) for images, labels in test_loader if (labels == attack_label).any().item()]
        #print("len test_loader_filtered: ", len(test_loader_filtered))
        for batch_idx, (images, labels) in enumerate(test_loader_filtered):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        if total == 0:
            acc = 0
        else:
            acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

        # test (use global proto)
        if global_protos!=[]:
            loss, total, correct = 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(test_loader_filtered):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs, protos = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    #print("len classes list in test inf",len(classes_list))
                    for j in range(args.num_classes):
                        if j in global_protos.keys(): # and j in classes_list[idx]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()
            if total == 0:
                acc = 0
            else:
                acc = correct / total
            print('len of total: ', total)
            print('len of correct: ', correct)
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list


def test_inference_new_het_by_attack_new(args, local_model_list, test_dataset,user_groups_gt, global_protos=[], attack_label=0):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()
    device = args.device
    criterion = get_loss_function(args.criterion).to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        loss, total, correct = 0.0, 0.0, 0.0
        model = local_model_list[idx]
        model.to(args.device)
        test_loader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        test_loader_filtered = [(images, labels) for images, labels in test_loader if (labels == attack_label).any().item()]
        print("len test_loader_filtered: ", len(test_loader_filtered))
        for batch_idx, (images, labels) in enumerate(test_loader_filtered):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        if total == 0:
            acc = 0
        else:
            acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

    # test (use global proto)
    if global_protos:
        total_w, correct_w = 0, 0
        loss_proto = 0.0
        idx = 1
        test_loader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        test_loader_filtered = [(images, labels) for images, labels in test_loader if (labels == attack_label).any().item()]
        print("len test_loader_filtered: ", len(test_loader_filtered))
        
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            protos=[]
            dists = []
            pred_labels = []
            weights = []
            for user in range(args.num_users):
                _, proto = local_model_list[user](images)
                protos.append(proto)
            
                # Compute the distance between protos and global_protos
                dist = torch.full((images.shape[0], args.num_classes), 100.0).to(device)
                distrib = torch.full((images.shape[0],), 0.0).to(device)   
                
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos:
                            d = loss_mse(proto[i, :], global_protos[j][0])
                            dist[i, j] = d
            
                
                # Prediction
                di, pred_label = torch.min(dist, 1)
                #print(di)
                dists.append(di)
                pred_labels.append(pred_label)
                #print("pred_label: ", pred_label)
                for i in range(images.shape[0]):
                    distrib[i] = distributions[user][pred_label[i].item()]
                    #print("distrib[i]: ", distrib[i])
                    #print(pred_label[i].item())
                #print("length of distrib",(distrib.size()))
                #print("length of di",(di.size()))
                weight=torch.div(distrib, di)
                weights.append(weight)
                #print("weight: ", weight)
            #print(weights)
            max_weight_value = float('-inf')
            max_weight_index = [-1]*len(weights)
            pred = torch.full((images.shape[0],), 0.0).to(device)   

            prots = torch.full((images.shape[0],), 0.0).to(device)   
                
            for i in range(images.shape[0]):
                weight_sum=[0]*args.num_classes
                for label in range(args.num_classes):
                    for j in range(args.num_users):
                        if label == pred_labels[j][i]:
                            weight_sum[label]+=weights[j][i]
                label = weight_sum.index(max(weight_sum))
                #print("max_weight_index: ", max_weight_index)
                pred[i]=label

                #prots[i]=protos[pred_labels[j][i]==label]
            #print("(((((((weights))))))): ", pred)
                

            """for i, weight in enumerate(weights):
                print(i)
                for j in range(len(weight)):
                    if weight[j]>max_weight_value:
                        max_weight_value = weight[j]
                        max_weight_index = i
            pred = pred_labels[max_weight_index]"""
            proto = protos[1] # index to be corrected
            correct_w += torch.sum(torch.eq(pred, labels)).item()
            total_w += len(labels)
            
            # Compute loss
            proto_new = copy.deepcopy(proto.data)
            for i, label in enumerate(labels):
                if label.item() in global_protos:
                    proto_new[i, :] = global_protos[label.item()][0].data
            
            loss2 = loss_mse(proto_new, proto)
            loss_proto += loss2.cpu().detach().numpy() if args.device == 'cuda' else loss2.detach().numpy()
        print("len of total_w: ", total_w)
        print("len of correct_w: ", correct_w)
        acc_global = correct_w / total_w
        print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_global))
        acc_list_g.append(acc_global)
        loss_list.append(loss_proto)

    return acc_list_l, acc_list_g, loss_list

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []

    for idx in range(args.num_users):
        print("User: ", idx)
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        print("len testloader: ", len(testloader))

        # Initialize accuracy tracking for local model
        total, correct = 0, 0
        loss = 0.0
        
        # Local model evaluation
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)
            
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        acc_local = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_local))
        acc_list_l.append(acc_local)
        
        # Initialize accuracy tracking for global prototypes
        if global_protos:
            total_w, correct_w = 0, 0
            loss_proto = 0.0
            
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                _, protos = model(images)
                
                # Compute the distance between protos and global_protos
                dist = torch.full((images.shape[0], args.num_classes), 100.0).to(device)
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d
                
                # Prediction
                _, pred_labels = torch.min(dist, 1)
                correct_w += torch.sum(torch.eq(pred_labels, labels)).item()
                total_w += len(labels)
                
                # Compute loss
                proto_new = copy.deepcopy(protos.data)
                for i, label in enumerate(labels):
                    if label.item() in global_protos:
                        proto_new[i, :] = global_protos[label.item()][0].data
                
                loss2 = loss_mse(proto_new, protos)
                loss_proto += loss2.cpu().detach().numpy() if args.device == 'cuda' else loss2.detach().numpy()
            print("len of total_w: ", total_w)
            print("len of correct_w: ", correct_w)
            acc_global = correct_w / total_w
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_global))
            acc_list_g.append(acc_global)
            loss_list.append(loss_proto)
    
    return acc_list_l, acc_list_g, loss_list


def test_inference_new_het_lt_new(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos, data_distribution):
    """ Returns the test accuracy and loss.
    """
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    # Initialize the output table
    distributions = [[0] * args.num_classes for _ in range(len(data_distribution))]
    sum_distributions = [0] * args.num_users
    # Populate the table
    for user_id, class_dict in data_distribution:
        for class_id, count in class_dict.items():
            distributions[user_id][class_id] = count
            sum_distributions[user_id] += count

    print("Distributions: ", distributions)

    for idx in range(args.num_users):
        print("User: ", idx)
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        print("len testloader: ", len(testloader))

        # Initialize accuracy tracking for local model
        total, correct = 0, 0
        loss = 0.0
        
        # Local model evaluation
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)
            
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        acc_local = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_local))
        acc_list_l.append(acc_local)
        
    # Initialize accuracy tracking for global prototypes
    if global_protos:
        beta = 1
        total_w, correct_w = 0, 0
        loss_proto = 0.0
        idx = 1
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        print("len testloader: ", len(testloader))

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            protos=[]
            dists = []
            pred_labels = []
            weights = []
            for user in range(args.num_users):
                #print("----------------User---------------: ", user)
                _, proto = local_model_list[user](images)
                protos.append(proto)
            
                # Compute the distance between protos and global_protos
                dist = torch.full((images.shape[0], args.num_classes), 100.0).to(device)
                distrib = torch.full((images.shape[0],), 0.0).to(device)   
                
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos:
                            d = loss_mse(proto[i, :], global_protos[j][0])
                            dist[i, j] = d
            
                #print("dist: ", dist)
                # Prediction
                di, pred_label = torch.min(dist, 1)
                #print(di)
                dists.append(di)
                pred_labels.append(pred_label)
                #print("pred_label: ", pred_label)
        
                for i in range(images.shape[0]):
                    distrib[i] = distributions[user][pred_label[i].item()]
                    distrib[i] = torch.div(distrib[i], sum_distributions[user])
                    #print("distrib[i]: ", distrib[i])
                    #print(pred_label[i].item())
                #print("length of distrib",(distrib.size()))
                #print("length of di",(di.size()))
                #print("di",di)
                #print("distrib",distrib)
                mean_dis = torch.mean(dist, dim=1)
                abs_diff = torch.abs(di - mean_dis)
                di = torch.div(di, abs_diff)
                weight = torch.div(distrib, di**beta)#di#
                weights.append(weight)
                #print("weight: ", weight)
            #print(weights)
            max_weight_value = float('-inf')
            max_weight_index = [-1]*len(weights)
            pred = torch.full((images.shape[0],), 0.0).to(device)  
             

            prots = torch.full((images.shape[0],), 0.0).to(device)   
                
            for i in range(images.shape[0]):
                weight_sum=[0]*args.num_classes
                for classe in range(args.num_classes):
                    for j in range(args.num_users):
                        if classe == pred_labels[j][i]:
                            weight_sum[classe]+=weights[j][i]
                pred[i]= weight_sum.index(max(weight_sum))
                #print("max_weight_index: ", max_weight_index)
                

                #prots[i]=protos[pred_labels[j][i]==label]
            #print("(((((((weights))))))): ", pred)
                

            """for i, weight in enumerate(weights):
                print(i)
                for j in range(len(weight)):
                    if weight[j]>max_weight_value:
                        max_weight_value = weight[j]
                        max_weight_index = i
            pred = pred_labels[max_weight_index]"""
            #print("pred: ", pred)
            #print("labels: ", labels)
            proto = protos[1] # index to be corrected
            correct_w += torch.sum(torch.eq(pred, labels)).item()
            total_w += len(labels)
            
            # Compute loss
            proto_new = copy.deepcopy(proto.data)
            for i, label in enumerate(labels):
                if label.item() in global_protos:
                    proto_new[i, :] = global_protos[label.item()][0].data
            
            loss2 = loss_mse(proto_new, proto)
            loss_proto += loss2.cpu().detach().numpy() if args.device == 'cuda' else loss2.detach().numpy()
        print("len of total_w: ", total_w)
        print("len of correct_w: ", correct_w)
        acc_global = correct_w / total_w
        print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_global))
        acc_list_g.append(acc_global)
        loss_list.append(loss_proto)
    
    return acc_list_l, acc_list_g, loss_list


def test_inference_new_het_lt_new_op(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos, data_distribution):
    """ Returns the test accuracy and loss.
    """
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    # Initialize the output table
    distributions = [[0] * args.num_classes for _ in range(len(data_distribution))]

    # Populate the table
    for user_id, class_dict in data_distribution:
        for class_id, count in class_dict.items():
            distributions[user_id][class_id] = count

    print("Distributions: ", distributions)

    for idx in range(args.num_users):
        print("User: ", idx)
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        print("len testloader: ", len(testloader))

        # Initialize accuracy tracking for local model
        total, correct = 0, 0
        loss = 0.0
        
        # Local model evaluation
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)
            
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        acc_local = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_local))
        acc_list_l.append(acc_local)
        
    # Initialize accuracy tracking for global prototypes
    if global_protos:
        total_w, correct_w = 0, 0
        loss_proto = 0.0
        
        
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            protos = []
            weights = []
            dists = []  # Initialize dists list here
            pred_labels = []  # Initialize pred_labels list here
            
            # Compute prototypes for all users in one pass
            for user in range(args.num_users):
                _, proto = local_model_list[user](images)
                protos.append(proto)  # Ensure proto is a tensor
            
            # Convert protos list to a tensor
            protos = torch.stack(protos)  # Shape: (num_users, batch_size, feature_dim)
            
            # Initialize distance and distribution tensors
            dist = torch.full((args.num_users, images.shape[0], args.num_classes), 100.0).to(device)
            distrib = torch.full((args.num_users, images.shape[0]), 0.0).to(device)
            
            # Compute distances in a vectorized manner
            for user in range(args.num_users):
                for j in global_protos.keys():
                    global_proto_tensor = global_protos[j][0].to(device)  # Ensure the global_proto is a tensor
                    dist[user, :, j] = torch.norm(protos[user] - global_proto_tensor, dim=1)  # Calculate distance
                
                # Predictions
                di, pred_label = torch.min(dist[user], 1)
                dists.append(di)  # Append distances to the dists list
                pred_labels.append(pred_label.cpu().tolist())  # Convert tensor to list before appending
                
                for i in range(images.shape[0]):
                                distrib[user, i] = distributions[user][pred_label[i].item()]  # Get distribution value for each prediction
                            

                weight = distrib[user] / (di + 1e-10)  # Added epsilon to avoid division by zero
                weights.append(weight)
            
            # Combine weights and make final predictions
            weights = torch.stack(weights)
            pred = torch.zeros(images.shape[0]).to(device)
            
            for i in range(images.shape[0]):
                weight_sum = torch.zeros(args.num_classes).to(device)
                for user in range(args.num_users):
                    condition = pred_labels[user] == torch.arange(args.num_classes).unsqueeze(0).to(device)
                    weight_sum += torch.where(condition, weights[user, i].unsqueeze(0), torch.tensor(0.0, device=device))
                pred[i] = torch.argmax(weight_sum)


                    #prots[i]=protos[pred_labels[j][i]==label]
                #print("(((((((weights))))))): ", pred)
                     

                """for i, weight in enumerate(weights):
                    print(i)
                    for j in range(len(weight)):
                        if weight[j]>max_weight_value:
                            max_weight_value = weight[j]
                            max_weight_index = i
                pred = pred_labels[max_weight_index]"""
                proto = protos[1] # index to be corrected
                correct_w += torch.sum(torch.eq(pred, labels)).item()
                total_w += len(labels)
                
                # Compute loss
                proto_new = copy.deepcopy(proto.data)
                for i, label in enumerate(labels):
                    if label.item() in global_protos:
                        proto_new[i, :] = global_protos[label.item()][0].data
                
                loss2 = loss_mse(proto_new, proto)
                loss_proto += loss2.cpu().detach().numpy() if args.device == 'cuda' else loss2.detach().numpy()
            print("len of total_w: ", total_w)
            print("len of correct_w: ", correct_w)
            acc_global = correct_w / total_w
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_global))
            acc_list_g.append(acc_global)
            loss_list.append(loss_proto)
    
    return acc_list_l, acc_list_g, loss_list



def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")

def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    dist = loss_mse(protos[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:,i], labels)).item()
        total += len(labels)

        cnt+=1
        if cnt==20:
            break

    acc = correct/total

    return acc


def test_inference_by_attack_all_clients(args, local_model_list, test_dataset, attack_label=0):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()
    device = args.device
    criterion = get_loss_function(args.criterion).to(device)

    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        loss, total, correct = 0.0, 0.0, 0.0
        model = local_model_list[idx]
        model.to(args.device)
        test_loader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        test_loader_filtered = [(images, labels) for images, labels in test_loader if (labels == attack_label).any().item()]
        print("len test_loader_filtered: ", len(test_loader_filtered))
        for batch_idx, (images, labels) in enumerate(test_loader_filtered):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        if total == 0:
            acc = 0
        else:
            acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

        
    return acc_list_l, loss_list

def test_inference_by_attack_server(args, model, test_dataset, attack_label=0):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()
    device = args.device
    criterion = get_loss_function(args.criterion).to(device)


    loss, total, correct = 0.0, 0.0, 0.0
    model.to(args.device)
    model.eval()
    test_dataset_filtered = [(images, labels) for images, labels in test_dataset if (labels == attack_label).any().item()]
    if len(test_dataset_filtered) == 0:
        return 0, 0
    test_loader = DataLoader(test_dataset_filtered, batch_size=64, shuffle=True)
    #test_loader_filtered = [(images, labels) for images, labels in test_loader if (labels == attack_label).any().item()]
    #print("len test_loader_filtered: ", len(test_loader))
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        torch.no_grad()
        outputs, protos = model(images)

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    if total == 0:
        acc = 0
    else:
        acc = correct / total
    #print('| class{}: | Global Test Acc : {:.3f}'.format(attack_label, acc))


    return acc,loss

def test_inference_all_classes(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    accs = []
    all_labels, all_preds = [], []
    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    print("len testloader: ", len(testloader))
    model.to(device)

    model.eval()
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        outputs, protos = model(images)

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:', cm)
    for class_ in range(args.num_classes):
        accs.append(cm[class_][class_]/sum(cm[class_])) 
    print(accs)
    return accs

def test_inference_metrics(args, model, test_dataset):
    """ Returns the test accuracy, F1 score, precision, recall, per-class FPR, weighted accuracy, and loss.
    """

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    print("len testloader: ", len(testloader))
    model.to(device)

    model.eval()
    
    loss, correct, total = 0.0, 0.0, 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Ensure no gradients are computed
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # Store predictions and labels for metric computation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_labels.cpu().numpy())

    # Accuracy
    acc = correct / total

    # Calculate other metrics using sklearn
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Calculate other metrics using sklearn with macro averaging
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate FPR for each class
    fpr_per_class = {}
    unweighted_accuracy_sum = 0.0
    print('Confusion Matrix:', cm)

    for i in range(len(cm)):
        # False Positives (FP) and True Negatives (TN)
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fpr_per_class[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Calculate accuracy for class i
        class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        unweighted_accuracy_sum += class_accuracy 
        print('Class: {} | Accuracy: {:.5f}'.format(i, class_accuracy))
    unweighted_accuracy = unweighted_accuracy_sum / len(cm)

    # Normalize loss
    loss = loss / len(testloader)
    
    print('Test Loss: {:.5f}'.format(loss))
    print('Test Accuracy: {:.5f}'.format(acc))
    print('F1 Score: {:.5f}'.format(f1))
    print('Precision: {:.5f}'.format(precision))
    print('Recall: {:.5f}'.format(recall))
    print('UnWeighted Accuracy: {:.5f}'.format(unweighted_accuracy))
    print('False Positive Rate per class:', fpr_per_class)

    print('F1 Score (Macro): {:.5f}'.format(f1_macro))
    print('Precision (Macro): {:.5f}'.format(precision_macro))
    print('Recall (Macro): {:.5f}'.format(recall_macro) )
    return acc, f1, precision, recall, unweighted_accuracy, f1_macro, loss


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
def test_inference_metrics_proto(args, model, test_dataset, global_protos=[], user_id=0):
    """ Returns the test accuracy, F1 score, precision, recall, per-class FPR, weighted accuracy, and loss.
    """
    print("************test_inference_metrics_proto************")

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs)
    print("len testloader: ", len(testloader))
    model.to(device)

    model.eval()
    
    loss, correct, total = 0.0, 0.0, 0.0
    all_labels = []
    all_preds = []
    all_protos = []

    total_w, correct_w = 0, 0
    loss_proto = 0.0
    loss_mse = nn.MSELoss()
    acc_by_class = []

    
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        _, protos = model(images)
        
        # Compute the distance between protos and global_protos
        dist = torch.full((images.shape[0], args.num_classes), 100.0).to(device)
        for i in range(images.shape[0]):
            for j in range(args.num_classes):
                if j in global_protos:
                    d = loss_mse(protos[i, :], global_protos[j][0])
                    dist[i, j] = d
        
        # Prediction
        _, pred_labels = torch.min(dist, 1)
        """print("-------------------pred_labels: ", pred_labels)
        print("-------------------labels: ", labels)
        print('--------------------dist: ', dist)"""
        correct_w += torch.sum(torch.eq(pred_labels, labels)).item()
        total_w += len(labels)
        
        # Compute loss
        proto_new = copy.deepcopy(protos.data)
        for i, label in enumerate(labels):
            if label.item() in global_protos:
                proto_new[i, :] = global_protos[label.item()][0].data
        
        loss2 = loss_mse(proto_new, protos)
        loss_proto += loss2.cpu().detach().numpy() if args.device == 'cuda' else loss2.detach().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_labels.cpu().numpy())
        all_protos.append(protos.cpu().detach().numpy())

    print("len of total_w: ", total_w)
    print("len of correct_w: ", correct_w)
    acc = correct_w / total_w
    print('| Test Acc with protos: {:.5f}'.format( acc))
    #acc_list_g.append(acc_global)
    #loss_list.append(loss_proto)




    # Calculate other metrics using sklearn
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Calculate other metrics using sklearn with macro averaging
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate FPR for each class
    fpr_per_class = {}
    unweighted_accuracy_sum = 0.0
    print('Confusion Matrix:', cm)

    for i in range(len(cm)):
        # False Positives (FP) and True Negatives (TN)
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fpr_per_class[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Calculate accuracy for class i
        class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        unweighted_accuracy_sum += class_accuracy 
        acc_by_class.append(class_accuracy)
        print('Class: {} | Accuracy: {:.5f}'.format(i, class_accuracy))
    unweighted_accuracy = unweighted_accuracy_sum / len(cm)

    # Normalize loss
    loss = loss / len(testloader)


    # Flatten prototype array for t-SNE
    """all_protos_flat = np.concatenate(all_protos, axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(all_protos_flat)
    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(all_labels)):
        idx = np.where(np.array(all_labels) == label)
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f"Class {label}")
    plt.legend()
    plt.title("t-SNE Visualization of Test Data Prototypes")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    file_folder = '../save_debug/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
    file_ext = 'user' + str(user_id) +'_data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    plt.savefig(file_folder + 'prototypes_tsne_' + file_ext + '.pdf')"""

    
    print('Test Loss: {:.5f}'.format(loss))
    print('Test Accuracy: {:.5f}'.format(acc))
    print('F1 Score: {:.5f}'.format(f1))
    print('Precision: {:.5f}'.format(precision))
    print('Recall: {:.5f}'.format(recall))
    print('UnWeighted Accuracy: {:.5f}'.format(unweighted_accuracy))
    print('False Positive Rate per class:', fpr_per_class)

    print('F1 Score (Macro): {:.5f}'.format(f1_macro))
    print('Precision (Macro): {:.5f}'.format(precision_macro))
    print('Recall (Macro): {:.5f}'.format(recall_macro) )
    return acc, f1, precision, recall, unweighted_accuracy, f1_macro, loss_proto, acc_by_class#, all_protos, all_labels




def test_inference_metrics_proto_new(args, user, model, test_dataset, global_protos=[], data_distribution=[]):
    """ Returns the test accuracy, F1 score, precision, recall, per-class FPR, weighted accuracy, and loss.
    """

    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    print("len testloader: ", len(testloader))
    model.to(device)

    model.eval()
    
    loss, correct, total = 0.0, 0.0, 0.0
    all_labels = []
    all_preds = []

    total_w, correct_w = 0, 0
    loss_proto = 0.0
    loss_mse = nn.MSELoss()

    # Initialize the output table
    distributions = [[0] * args.num_classes for _ in range(len(data_distribution))]
    sum_distributions = [0] * args.num_users
    # Populate the table
    for user_id, class_dict in data_distribution:
        for class_id, count in class_dict.items():
            distributions[user_id][class_id] = count
            sum_distributions[user_id] += count

    print("Distributions: ", distributions)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        _, protos = model(images)
        
        # Compute the distance between protos and global_protos
        dist = torch.full((images.shape[0], args.num_classes), 100.0).to(device)
        for i in range(images.shape[0]):
            for j in range(args.num_classes):
                if j in global_protos:
                    d = loss_mse(protos[i, :], global_protos[j][0])
                    dist[i, j] = d
        
        # Prediction
        # Predictions
        di, pred_label = torch.min(dist[user], 1)
        dists.append(di)  # Append distances to the dists list
        pred_labels.append(pred_label.cpu().tolist())  # Convert tensor to list before appending
        
        for i in range(images.shape[0]):
                        distrib[user, i] = distributions[user][pred_label[i].item()]  # Get distribution value for each prediction
                    

        weight = distrib[user] / (di + 1e-10)  # Added epsilon to avoid division by zero
        weights.append(weight)
    
        # Combine weights and make final predictions
        weights = torch.stack(weights)
        pred = torch.zeros(images.shape[0]).to(device)
        
        for i in range(images.shape[0]):
            weight_sum = torch.zeros(args.num_classes).to(device)
            for user in range(args.num_users):
                condition = pred_labels[user] == torch.arange(args.num_classes).unsqueeze(0).to(device)
                weight_sum += torch.where(condition, weights[user, i].unsqueeze(0), torch.tensor(0.0, device=device))
            pred[i] = torch.argmax(weight_sum)


                #prots[i]=protos[pred_labels[j][i]==label]
            #print("(((((((weights))))))): ", pred)
                    

            """for i, weight in enumerate(weights):
                print(i)
                for j in range(len(weight)):
                    if weight[j]>max_weight_value:
                        max_weight_value = weight[j]
                        max_weight_index = i
            pred = pred_labels[max_weight_index]"""
        proto = protos[1] # index to be corrected
        correct_w += torch.sum(torch.eq(pred, labels)).item()
        total_w += len(labels)
        
        # Compute loss
        proto_new = copy.deepcopy(protos.data)
        for i, label in enumerate(labels):
            if label in global_protos:
                loss_proto += loss_mse(protos[i, :], global_protos[label][0])
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_labels.cpu().numpy())
        # Perform inference
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Update metrics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.tolist())
        all_preds.extend(predicted.tolist())
        
        # Calculate loss
        loss += criterion(outputs, labels).item()
        
        # Calculate weighted accuracy
        for i in range(len(data_distribution)):
            distributions[i][predicted[i]] += 1
            sum_distributions[i] += 1
        
        # Calculate loss with global prototypes
        if len(global_protos) > 0:
            for i in range(len(images)):
                dists = torch.cdist(images[i].unsqueeze(0), global_protos)
                min_dist, _ = torch.min(dists, dim=1)
                loss_proto += min_dist.item()
                total_w += 1
                if predicted[i] == labels[i]:
                    correct_w += 1
        
        # Calculate accuracy and other metrics
        acc = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision_macro = precision_score(all_labels, all_preds, average='macro')
        recall_macro = recall_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        fpr_per_class = {}
        unweighted_accuracy_sum = 0.0
        for i in range(len(cm)):
            fpr_per_class[i] = cm[i].sum() - cm[i][i]
            unweighted_accuracy_sum += cm[i][i]
        unweighted_accuracy = unweighted_accuracy_sum / len(cm)
        loss = loss / len(testloader)
        loss_proto = loss_proto / total_w
        
        # Print and return the metrics
        print('Test Loss: {:.5f}'.format(loss))
        print('Test Accuracy: {:.5f}'.format(acc))
        print('F1 Score: {:.5f}'.format(f1))
        print('Precision: {:.5f}'.format(precision))
        print('Recall: {:.5f}'.format(recall))
        print('UnWeighted Accuracy: {:.5f}'.format(unweighted_accuracy))
        print('False Positive Rate per class:', fpr_per_class)
        print('F1 Score (Macro): {:.5f}'.format(f1_macro))
        print('Precision (Macro): {:.5f}'.format(precision_macro))
        print('Recall (Macro): {:.5f}'.format(recall_macro))
        print('Loss with Global Prototypes: {:.5f}'.format(loss_proto))
        
        return acc, f1, precision, recall, unweighted_accuracy, loss, fpr_per_class, f1_macro, precision_macro, recall_macro, loss_proto, distributions, sum_distributions


def test_inference_by_attack_server_proto(args, model, test_dataset, attack_label=0, global_protos=[]):
    """ Returns the test accuracy and loss for a specific attack label using global prototypes.
    """
    print("*****************test_inference_by_attack_server_proto*****************")
    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    test_dataset_f = [(images, labels) for images, labels in test_dataset if (labels == attack_label).any().item()]
    testloader = DataLoader(test_dataset_f, batch_size=args.local_bs, shuffle=False)
    print("len testloader: ", len(testloader))
    model.to(device)

    model.eval()
    
    loss, correct, total = 0.0, 0.0, 0.0
    all_labels = []
    all_preds = []

    total_w, correct_w = 0, 0
    loss_proto = 0.0
    loss_mse = nn.MSELoss()

    
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        _, protos = model(images)
        
        # Compute the distance between protos and global_protos
        dist = torch.full((images.shape[0], args.num_classes), 100.0).to(device)
        for i in range(images.shape[0]):
            for j in range(args.num_classes):
                if j in global_protos:
                    d = loss_mse(protos[i, :], global_protos[j][0])
                    dist[i, j] = d
        
        # Prediction
        _, pred_labels = torch.min(dist, 1)
        correct_w += torch.sum(torch.eq(pred_labels, labels)).item()
        total_w += len(labels)
        
        # Compute loss
        proto_new = copy.deepcopy(protos.data)
        for i, label in enumerate(labels):
            if label.item() in global_protos:
                proto_new[i, :] = global_protos[label.item()][0].data
        
        loss2 = loss_mse(proto_new, protos)
        loss_proto += loss2.cpu().detach().numpy() if args.device == 'cuda' else loss2.detach().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_labels.cpu().numpy())

    print("confusion matrix: ", confusion_matrix(all_labels, all_preds))

    print("len of total_w: ", total_w)
    print("len of correct_w: ", correct_w)
    acc = correct_w / total_w
    print('| Test Acc with protos: {:.5f}'.format( acc))

    
    return acc, loss_proto





def test_inference_by_attack_server_proto_new (args, model, test_dataset, attack_label, global_protos=[], data_distribution=[]):
    """ Returns the test accuracy and loss for a specific attack label using global prototypes.
    """
     # Initialize the output table
    distributions = [[0] * args.num_classes for _ in range(len(data_distribution))]
    sum_distributions = [0] * args.num_users
    # Populate the table
    for user_id, class_dict in data_distribution:
        for class_id, count in class_dict.items():
            distributions[user_id][class_id] = count
            sum_distributions[user_id] += count

    print("Distributions: ", distributions)
    loss_mse = nn.MSELoss()
    device = args.device
    criterion = get_loss_function(args.criterion).to(device)
    loss, total, correct = 0.0, 0, 0
    model.to(args.device)
    test_dataset_filtered = [(images, labels) for images, labels in test_dataset if (labels == attack_label).any().item()]
    test_loader = DataLoader(test_dataset_filtered, batch_size=64, shuffle=True)
    for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            protos = []
            weights = []
            dists = []  # Initialize dists list here
            pred_labels = []  # Initialize pred_labels list here
            
            # Compute prototypes for all users in one pass
            for user in range(args.num_users):
                _, proto = local_model_list[user](images)
                protos.append(proto)  # Ensure proto is a tensor
            
            # Convert protos list to a tensor
            protos = torch.stack(protos)  # Shape: (num_users, batch_size, feature_dim)
            
            # Initialize distance and distribution tensors
            dist = torch.full((args.num_users, images.shape[0], args.num_classes), 100.0).to(device)
            distrib = torch.full((args.num_users, images.shape[0]), 0.0).to(device)
            
            # Compute distances in a vectorized manner
            for user in range(args.num_users):
                for j in global_protos.keys():
                    global_proto_tensor = global_protos[j][0].to(device)  # Ensure the global_proto is a tensor
                    dist[user, :, j] = torch.norm(protos[user] - global_proto_tensor, dim=1)  # Calculate distance
                
                # Predictions
                di, pred_label = torch.min(dist[user], 1)
                dists.append(di)  # Append distances to the dists list
                pred_labels.append(pred_label.cpu().tolist())  # Convert tensor to list before appending
                
                for i in range(images.shape[0]):
                                distrib[user, i] = distributions[user][pred_label[i].item()]  # Get distribution value for each prediction
                            

                weight = distrib[user] / (di + 1e-10)  # Added epsilon to avoid division by zero
                weights.append(weight)
            
            # Combine weights and make final predictions
            weights = torch.stack(weights)
            pred = torch.zeros(images.shape[0]).to(device)
            
            for i in range(images.shape[0]):
                weight_sum = torch.zeros(args.num_classes).to(device)
                for user in range(args.num_users):
                    condition = pred_labels[user] == torch.arange(args.num_classes).unsqueeze(0).to(device)
                    weight_sum += torch.where(condition, weights[user, i].unsqueeze(0), torch.tensor(0.0, device=device))
                pred[i] = torch.argmax(weight_sum)

                proto = protos[1] # index to be corrected
                correct_w += torch.sum(torch.eq(pred, labels)).item()
                total_w += len(labels)
                
                # Compute loss
                proto_new = copy.deepcopy(proto.data)
                for i, label in enumerate(labels):
                    if label.item() in global_protos:
                        proto_new[i, :] = global_protos[label.item()][0].data
                
                loss2 = loss_mse(proto_new, proto)
                loss_proto += loss2.cpu().detach().numpy() if args.device == 'cuda' else loss2.detach().numpy()
            print("len of total_w: ", total_w)
            print("len of correct_w: ", correct_w)
            acc_global = correct_w / total_w
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_global))
            acc_list_g.append(acc_global)
            loss_list.append(loss_proto)
    
    return  acc_list_g, loss_list

        