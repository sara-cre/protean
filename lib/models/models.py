#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNNFemnist(nn.Module):
    def __init__(self, args):
        super(CNNFemnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(16820/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, args.num_classes)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x1))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1


class Lenet(nn.Module):
    def __init__(self, args):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), x1



class DenseModel_(nn.Module):
    def __init__(self, args):
        super(DenseModel_, self).__init__()
        # Define the layers
        input_size = args.num_features
        output_size = args.num_classes
        self.dense1 = nn.Linear(input_size, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output_size)
    
    def forward(self, x):
        # Forward pass with ReLU activations
        x = F.relu(self.dense1(x))
        x1 = F.relu(self.dense2(x))

        x = F.relu(self.dense3(x1))
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x, x1

# Define the Improved Model
class DenseModel(nn.Module):
    def __init__(self, args):
        input_size = args.num_features
        output_size = args.num_classes
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, output_size)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x1 = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x1)
        x = self.fc4(x)
        return x, x1

class CustomCNN(nn.Module):
    def __init__(self, args):
        super(CustomCNN, self).__init__()
        num_features = args.num_features
        num_instances = args.num_classes
        if args.dataset == 'ciciot':
            feature_map = 9
        elif args.dataset == '5gnidd':
            feature_map = 6
        elif args.dataset == 'cicids2017':
            feature_map = 17
        else:
            feature_map = 16
        self.reshape_layer = nn.Identity()  # No reshape needed as PyTorch will handle the shape during forward pass
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128 * feature_map, 128)  # Adjust in_features based on the input size after convolutions
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, num_instances)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape to (batch_size, 1, num_features)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x1 = F.relu(self.dense1(x))  # Save intermediate output x1
        x = self.dropout2(x1)
        x = self.dense2(x)

        return F.log_softmax(x, dim=1), x1 


class EdgeCustomCNN(nn.Module):
    def __init__(self, input_features, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)  # Conv1D(32, 3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)                               # MaxPooling1D(2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3) # Conv1D(64, 3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)                                # MaxPooling1D(2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3) # Conv1D(128, 3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)                                 # MaxPooling1D(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 10, 64)  # Adjusted based on input size after convs and pools
        self.fc2 = nn.Linear(64, num_classes)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Ensure input shape is (batch_size, 1, 96)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Return raw logits; no activation function here
        return x


class Proj(nn.Module):
    def __init__(self, args):
        super(Proj, self).__init__()
        if args.dataset == 'ciciot':
            feature_map = 9
        elif args.dataset == '5gnidd':
            feature_map = 6
        else:
            feature_map = 16
        self.fc1 = nn.Linear(128 * feature_map, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1 = F.normalize(x1, dim=1)
        x = F.relu(self.fc2(x1))
        x = self.dropout2(x1)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class Embedder (nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()
        num_features = args.num_features
        print("num_features", num_features)
        num_instances = args.num_classes
        if args.dataset == 'ciciot':
            feature_map = 9
        elif args.dataset == '5gnidd':
            feature_map = 6
        else:
            feature_map = 16
        self.reshape_layer = nn.Identity()  # No reshape needed as PyTorch will handle the shape during forward pass
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()


    def forward(self, x):
        #print("x shape_________before__", x.shape)
        x = x.view(x.size(0), 1, -1)  # Reshape to (batch_size, 1, num_features)
        #print("x shape___________", x.shape)
        #batch_size, num_images, num_features = x.size()
        #x = x.view(batch_size * num_images, 1, num_features)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
        #print("x shape___________", x.shape)
        return x