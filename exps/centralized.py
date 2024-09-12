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

from data_load_split import load_data_cicids2017
from models import CustomCNN, DenseModel
from options import args_parser

import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import random
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

#https://github.com/mahendradata/cicids2017-ml/blob/master/6.2%20Dense.ipynb


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
        self.criterion = nn.CrossEntropyLoss().to(self.device) #nn.NLLLoss().to(self.device)
        self.idxs = idxs
        
        self.dataset = dataset

        self.mu = mu


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        args = self.args
        idxs_train = idxs[:int(1 * len(idxs))]
        attack_round = [2,4,6,8,10]
        
        print(f'len of train dataset: {len(idxs_train)}')
        trainloader = DataLoader(dataset, batch_size=1024,
                            shuffle=False)
        print(f"Trainloader length: {len(trainloader)}")
        return trainloader

    def update_weights(self,args, idx, model, global_round):
        global_model = copy.deepcopy(model)
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                         weight_decay=1e-4)


        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
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


def test_inference(args, model, test_dataset, criterion):
    """ Returns the test accuracy and loss.
    """

    # Set the model to eval
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    testloader = DataLoader(test_dataset, batch_size=1024,
                            shuffle=False)
    all_preds = []
    all_labels = []
    print(f"Testloader length: {len(testloader)}")
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(args.device), labels.to(args.device)
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


    acc = 100.*correct/total
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    return test_loss, acc

args = args_parser()

train_dataset, test_dataset  = load_data_cicids2017(args)
args.dataset = 'cicids2017'
#args.num_classes = 25
#args.num_features = 77
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#model = CustomCNN(args)
model = DenseModel(args)
idx = 0
idxs = np.arange(0, 10)
global_round = 0
local_update = LocalUpdate(args, train_dataset, idx, idxs, global_round, mu=0.01)
model_state, loss, acc = local_update.update_weights(args, idx, model, global_round)
test_model = copy.deepcopy(model)
test_model.load_state_dict(model_state)
test_model.eval()
test_model.to(args.device)
criterion = nn.CrossEntropyLoss().to(args.device)
test_loss, test_acc = test_inference(args, test_model, test_dataset, criterion)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


