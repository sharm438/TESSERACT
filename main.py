# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:53:34 2021

@author: sharm438
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import numpy as np
import sys
import pdb
from copy import deepcopy
import aggregation
import attack

#import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--net", help="net", default='dnn', type=str, choices=['mlr', 'dnn', 'dnn2', 'resnet18', 'dnn_femnist'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.003, type=float)
    parser.add_argument("--nworkers", help="# workers", default=10, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=20, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=-1, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=2, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='full_trim', type=str,
                        choices=['benign', 'partial_trim', 'full_trim', 'full_krum'])
    parser.add_argument("--aggregation", help="aggregation rule", default='trim', type=str)
    parser.add_argument("--cmax", help="FLAIR's notion of c_max", default=2, type=int)
    parser.add_argument("--decay", help="Decay rate", default=2.0, type=float)
    return parser.parse_args()

class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #self.bn1   = nn.BatchNorm2d(in_channels)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        #out = F.relu(self.bn1(x))
        #out = self.conv1(out)
        #out = self.conv2(F.relu(self.bn2(out)))
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x

        return out + shortcut


class ResNet18(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 256, num_blocks[3], stride=2),
        )

        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = x.half()
        x = self.prep(x)

        x = self.layers(x)

        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)

        x = torch.cat([x_avg, x_max], dim=-1)

        x = self.classifier(x)

        return x

class DNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 50, 3, padding=1)
        self.fc1 = nn.Linear(50*7*7, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DNN_femnist(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class LR_har(nn.Module):
    
    def __init__(self):
        super(LR_har, self).__init__()
        self.linear = nn.Linear(1152, 6)
        
    def forward(self, x):
        x = self.linear(x)
        return x
        
    
def get_lr(epoch, num_epochs):

    mu = num_epochs/4
    sigma = num_epochs/4
    max_lr = 0.1
    if (epoch < num_epochs/4):
        return max_lr*(1-np.exp(-25*(epoch/num_epochs)))
    else:
        #return 0.1*(np.exp(-7.5*(epoch - num_epochs/4)/num_epochs))
        return max_lr*np.exp(-0.5*(((epoch-mu)/sigma)**2))
    
    #return max_lr*np.exp(-0.5*(((epoch-mu)/sigma)**2))
    

    
# STUFF FOR FEMNIST --------------------------------------------------------

import numpy as np
import os
import sys
import random

import random
import warnings

import json
from collections import defaultdict
import copy

class Client:
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}):
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        
    def return_data(self):
        return self.train_data, self.eval_data
    
    def return_id(self):
        return self.id
    

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

def return_data(dataset, use_val_set=False):
    eval_set = 'test' if not use_val_set else 'val'
    
    train_data_dir = os.path.join('data', 'femnist', 'data', 'train')
    test_data_dir = os.path.join('data', 'femnist', 'data', eval_set)
    
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    
    uid = []
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    for user in train_data:
        uid.append(user)
        train_x.append(torch.tensor(train_data[user]['x']).view(-1,1,28,28))
        train_y.append(torch.tensor(train_data[user]['y']))
        test_x.append(torch.tensor(test_data[user]['x']).view(-1,1,28,28))
        test_y.append(torch.tensor(test_data[user]['y']))
    
    return uid, train_x, train_y, torch.cat(test_x, dim=0), torch.cat(test_y)

    
import matplotlib.pyplot as plt           
def main(args):
    
    '''
    current_lr = []
    for i in range(50000):
        current_lr.append(get_lr(i, 50000))
    plt.plot(current_lr)
    plt.show()
    '''
    ####Load arguments
    
    num_workers = args.nworkers
    num_epochs = args.nepochs
    
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    batch_size = args.batch_size
    lr = args.lr
    
    ###Load datasets
    if (args.dataset == 'mnist'):
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]) 
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download='True', transform=transform)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset        
        num_inputs = 28 * 28
        num_outputs = 10
        
    elif (args.dataset == 'fmnist'):
        num_inputs = 28 * 28
        num_outputs = 10
    elif args.dataset == 'chmnist':
        num_inputs = 64*64
        num_outputs = 8
    elif args.dataset == 'bcw':
        num_inputs = 30
        num_outputs = 2
    elif args.dataset == 'cifar10':
        num_inputs = 32*32*3
        num_outputs = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])        
        #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset
    
    elif args.dataset == 'femnist':
        num_inputs = 28 * 28
        num_outputs = 62
        uid, train_x, train_y, test_x, test_y = return_data("femnist", False)
        
        for i in range(len(train_x)):
            if i >= num_workers:
                ind = i % num_workers
                train_x[ind] = torch.cat((train_x[ind], train_x[i]), dim=0)
                train_y[ind] = torch.cat((train_y[ind], train_y[i]), dim=0)
                
        train_x = train_x[:num_workers]
        train_y = train_y[:num_workers]
        
        from torch.utils.data import TensorDataset
        testset = TensorDataset(test_x, test_y)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del testset
        
        print("Length of train:", len(train_x))
        
    else:
        sys.exit('Not Implemented Dataset!')
        
    ####Load models
    if (args.net == 'resnet18'):
        net = ResNet18()
    elif(args.net == 'dnn'):
        net = DNN()
    elif(args.net == 'dnn_femnist'):
        net = DNN_femnist()
        
    net.to(device) # --------------------------------------------------------------------------------------------------------------------------------------
    
    if args.byz_type == 'benign':
        byz = attack.benign
    elif args.byz_type == 'full_trim':
        byz = attack.full_trim
    elif args.byz_type == 'full_krum':
        byz = attack.full_krum
    
    if args.dataset != 'femnist':
        ####Distribute data samples
        bias_weight = args.bias
        other_group_size = (1-bias_weight) / (num_outputs-1)
        worker_per_group = num_workers / (num_outputs)
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)] 
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                upper_bound = (y.item()) * (1-bias_weight) / (num_outputs-1) + bias_weight
                lower_bound = (y.item()) * (1-bias_weight) / (num_outputs-1)
                rd = np.random.random_sample()
                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size)+y.item()+1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.item()

                # assign a data point to a worker
                rd = np.random.random_sample()
                selected_worker = int(worker_group*worker_per_group + int(np.floor(rd*worker_per_group)))
                if (args.bias == 0): selected_worker = np.random.randint(num_workers)
                each_worker_data[selected_worker].append(x.to(device))
                each_worker_label[selected_worker].append(y.to(device))

        # concatenate the data for each worker
        each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data] 
        each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]
    
    else:
        each_worker_data = train_x
        each_worker_label = train_y
    
    # random shuffle the workers
    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    criterion = nn.CrossEntropyLoss()
    test_acc = np.empty(num_epochs)
    
    P = 0
    for param in net.parameters():
        if param.requires_grad:
            P = P + param.nelement()
    direction = torch.zeros(P).to(device)
    susp = torch.zeros(num_workers).to(device)
    decay = args.decay
    
    batch_idx = np.zeros(num_workers)
    
    for epoch in range(num_epochs):
        grad_list = []
        if (args.aggregation == 'flair'):
            susp = susp/decay
        if (args.aggregation == 'cifar10'):
            lr = get_lr(epoch, num_epochs)
        for worker in range(num_workers):
            net_local = deepcopy(net) # --------------------------------------------------------------------------------------------------------------------------------------
            net_local.train()
            #optimizer = optim.SGD(net_local.parameters(), lr=lr)
            optimizer = optim.Adam(net_local.parameters(), lr=lr)
            optimizer.zero_grad()
            if args.dataset == 'mnist':
                if (batch_idx[worker]+batch_size < each_worker_data[worker].shape[0]):
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),int(batch_idx[worker])+batch_size)))
                    batch_idx[worker] = batch_idx[worker] + batch_size
                else: 
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),each_worker_data[worker].shape[0]))) 
                    batch_idx[worker] = 0
            else: 
                minibatch = np.random.choice(list(range(each_worker_data[worker].shape[0])), size=batch_size, replace=False)
            output = net_local(each_worker_data[worker][minibatch].to(device))
            loss = criterion(output, each_worker_label[worker][minibatch].to(device))
            loss.backward()
            optimizer.step()
                    
            grad_list.append([(x-y).detach() for x, y in zip(net_local.parameters(), net.parameters()) if x.requires_grad != 'null'])
            
            del net_local, output, loss
            torch.cuda.empty_cache()
            
        # print("Before:", net.conv1.weight, net.conv1.bias, net.fc2.weight, net.fc2.bias)
            
        if (args.aggregation == 'mean'):
            net = aggregation.mean(device, byz, grad_list, net) 
        elif (args.aggregation == 'flair'):
            net, direction, susp, flip_local = aggregation.flair(device, byz, grad_list, net, direction, susp, mod=True)
        elif (args.aggregation == 'krum'):
            net = aggregation.krum(device, byz, grad_list, net)         
        elif (args.aggregation == 'trim'):
            net = aggregation.trim(device, byz, grad_list, net, args.nbyz)
            
        # print("After:", net.conv1.weight, net.conv1.bias,  net.fc2.weight, net.fc2.bias)
        
        del grad_list
        torch.cuda.empty_cache()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            test_acc[epoch] = correct/total                
            print ('Epoch: %d, test_acc: %f, lr: %f' %(epoch, test_acc[epoch], lr))      
        
    np.save('Test_acc.npy', test_acc)

            
if __name__ == "__main__":
    args = parse_args()
    main(args)
