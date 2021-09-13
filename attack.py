# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:08:11 2021

@author: sharm438
"""

import torch
from copy import deepcopy
import pdb
import math
import numpy as np
import copy

def benign(device, param_list, cmax=0):

    return param_list

def label_flip(device, param_list, cmax=0):

    return param_list

def full_trim(device, param_list, cmax=0):

    max_dim = torch.max(param_list, axis=0)[0]
    min_dim = torch.min(param_list, axis=0)[0]
    direction = torch.sign(torch.sum(param_list, axis=0)).to(device)
    directed_dim = (direction > 0) * min_dim + (direction < 0) * max_dim
    
    for i in range(cmax):
        random_12 = 1 + torch.rand(len(param_list[0])).to(device)
        param_list[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
           
    return param_list

def full_krum(device, v, f):

    prob = 1
    adapt = torch.tensor(np.random.choice(2, len(v[0]), p=[1-prob, prob])).to(device)
    if (f==0):
        return v
    e = 0.0001/len(v[0])
    direction = torch.sign(torch.sum(v, axis=0))
    #pdb.set_trace()
    l_max = lambda_max(device, v, f)
    l = find_lambda(l_max, v, direction, len(v), f)
    #print ("Lambda selected: ", l)
    if (l>0):
        v[0] = -(direction*l)
        for i in range(1, f):
            noise = torch.FloatTensor(1, len(v[0])).uniform_(-e, e).to(device)
            v[i] = v[0] + noise
    #pdb.set_trace()
    return v

def lambda_max(device, v, f): #(m, c, params, global_param):

    m = len(v)
    dist = torch.zeros((m,m)).to(device)
    for i in range (0, m):
        for j in range(0, i):
            dist[i][j] = torch.norm(v[i] - v[j])
            dist[j][i] = dist[i][j]   
    sorted_benign_dist = torch.sort(dist[f:,f:])
    sum_benign_dist = torch.sum((sorted_benign_dist[0][:, :(m-f-1)])**2, axis=1)
    min_distance = torch.min(sum_benign_dist).item()
    
    dist_global = torch.zeros(m-f)
    for i in range(f, m):
        dist_global[i-f] = torch.norm(v[i])
    max_global_dist = torch.max(dist_global).item()
    scale = 1.0/(len(v[0]))
    return (math.sqrt(scale/(m-2*f-1))*min_distance) + math.sqrt(scale)*max_global_dist


def find_lambda(lambda_current, params, s, m, c):
    
    #print ("Current lambda: ", lambda_current)
    if (lambda_current <= 0.00001):
        return 0.0
 
    params_local = params.detach().clone()
    params_local[0][:] = -(lambda_current)*s #[:] is important
    for i in range(1, c):
        params_local[i] = params_local[0]
    model_selected = local_krum(params_local, c)
    if (model_selected <= c):
        del params_local
        return lambda_current
    else:
        del params_local
        return find_lambda(lambda_current*0.5, params, s, m, c)
    
def local_krum(param_list, f):

    k = len(param_list) - f - 2
    dist = torch.zeros((len(param_list),len(param_list)))
    for i in range (0, len(param_list)):
        for j in range(0, i):
            dist[i][j] = torch.norm(param_list[i] - param_list[j])
            dist[j][i] = dist[i][j]      
    sorted_dist = torch.sort(dist)
    sum_dist = torch.sum(sorted_dist[0][:,:k+1], axis=1)
    model_selected = torch.argmin(sum_dist).item()        
    #print("Local krum selected model: ", model_selected)
    
    return model_selected    
        
    
