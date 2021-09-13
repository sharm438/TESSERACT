# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:08:29 2021

@author: sharm438
"""
import pdb
import torch
import numpy as np
import torch.nn as nn

def mean(device, byz, grad_list, net):
    
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, param_list, 2) 
    global_params = torch.mean(param_list, dim=0)
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list, global_params

    return net

def flair(device, byz, grad_list, net, old_direction, susp, cmax=0, mod=True):
    
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, param_list, cmax)
    flip_local = torch.zeros(len(param_list)).to(device)
    penalty = 1.0 - cmax/len(param_list)
    reward = 1.0 - penalty

    if mod == True:
        for i in range(len(param_list)):
            direction = torch.sign(param_list[i])
            flip = torch.sign(direction*(direction-old_direction.reshape(-1)))
            flip_local[i] = torch.sum(flip*(param_list[i]**2))
            del direction, flip
    else:
        for i in range(len(param_list)):
            direction = torch.sign(param_list[i])
            flip = torch.sign(direction*(direction-old_direction.reshape(-1)))
            flip_local[i] = 0.5*(torch.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).item()
            del direction, flip

    argsorted = torch.argsort(flip_local).to(device)
    print (flip_local)
    if (cmax > 0):
        susp[argsorted[:-cmax]] = susp[argsorted[:-cmax]] + reward
        susp[argsorted[-cmax:]] = susp[argsorted[-cmax:]] - penalty  
    argsorted = torch.argsort(susp)

    weights = torch.exp(susp)/torch.sum(torch.exp(susp))
    global_params = torch.matmul(torch.transpose(param_list, 0, 1), weights.reshape(-1,1))
    global_direction = torch.sign(global_params)

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list, global_params

    return net, global_direction, susp, flip_local

def krum(device, byz, grad_list, net, cmax=0):
    
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    
    k = len(param_list)-cmax-2
    dist = torch.zeros((len(param_list), len(param_list)))
    for i in range(len(param_list)):
        for j in range(i):
            dist[i][j] = torch.norm(param_list[i]-param_list[j])
            dist[j][i] = dist[i][j]       
    sorted_dist = torch.sort(dist)
    sum_dist = torch.sum(sorted_dist[0][:,:k+1], axis=1)
    model_selected = torch.argmin(sum_dist).item()
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += param_list[model_selected][idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list
    return net   

def fltrust(device, byz, grad_list, net, tau):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    server_params = param_list[0]
    server_norm = torch.norm(server_params)
    param_list = (param_list[1:])[np.random.permutation(tau)]
    ts = torch.zeros((len(param_list)))
    for i in range(len(param_list)):
        ts[i] = max(cos(server_params, param_list[i]), 0)
        param_list[i] = (server_norm/torch.norm(param_list[i])) * param_list[i] * ts[i]
    global_params = torch.sum(param_list, dim=0) / torch.sum(ts)
    del param_list
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del global_params
    return net   

def trim(device, byz, grad_list, net, cmax=0): 

    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    param_list = byz(device, param_list, cmax)
    sorted_array = torch.sort(param_list, axis=0)
    trimmed = torch.mean(sorted_array[0][cmax:len(param_list)-cmax,:], axis=0)

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += trimmed[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
                
    del param_list, sorted_array, trimmed
    return net  
    
    
