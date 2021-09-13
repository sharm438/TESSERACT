#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

import pdb
import aggregation
import attack
from copy import deepcopy
import numpy as np

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=10)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)

def random_training_set(device, worker, num_workers, chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, int((worker+1)*(file_len/num_workers)) - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    inp = inp.to(device)
    target = target.to(device)
    return inp, target

def train(device, decoder, decoder_optimizer, inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    hidden = hidden.to(device)
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()
    return loss.data / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(net, save_filename)
    np.save('test_loss_tessk_l2_m10c2.npy', all_losses)
    np.save('flip_score_tessk_l2_m10c2.npy', flip_score)
    #print('Saved as %s' % save_filename)

# Initialize models and start training

device = torch.device('cuda')
net = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
net = net.to(device)
#decoder_optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()


start = time.time()
all_losses = np.zeros(args.n_epochs)

P = 0
for param in net.parameters():
    if param.requires_grad:
        P = P + param.nelement()
#print (P)
num_workers = 11 
byz = attack.full_krum
flip_score = np.empty((args.n_epochs, num_workers-1))

try:
    print("Training for %d epochs..." % args.n_epochs)
    direction = torch.zeros(P).to(device)
    susp = torch.zeros(num_workers-1).to(device)
    for epoch in range(1, args.n_epochs + 1):
        loss_avg = 0
        grad_list = []
        susp = susp/2
        for worker in range (num_workers-1):
            net_local = deepcopy(net)
            net_local.train()
            optimizer = torch.optim.Adam(net_local.parameters(), lr=args.learning_rate, weight_decay=0.001)
            optimizer.zero_grad()
            inp, target = random_training_set(device, worker, num_workers, args.chunk_len, args.batch_size)
            hidden = net_local.init_hidden(args.batch_size).to(device)
            loss = 0
            for c in range(args.chunk_len):
                output, hidden = net_local(inp[:,c], hidden)
                loss += criterion(output.view(args.batch_size, -1), target[:,c])
            loss = loss/args.chunk_len
            loss.backward()
            optimizer.step()
            #loss = train(net_local, optimizer, *random_training_set(worker, num_workers, args.chunk_len, args.batch_size))
            loss_avg += loss
            grad_list.append([(x-y) for x, y in zip(net_local.parameters(), net.parameters()) if x.requires_grad != 'null'])
        #net = aggregation.mean(device, byz, grad_list, net)
        net, direction, susp, flip = aggregation.flair(device, byz, grad_list, net, direction, susp, 2)
        flip_score[epoch-1] = flip.detach().cpu().numpy()
        loss_avg = loss_avg/(num_workers-1)
        inp, target = random_training_set(device, worker, num_workers-1, args.chunk_len, args.batch_size)
        net.eval()
        with torch.no_grad():
            hidden = net.init_hidden(args.batch_size).to(device)
            test_loss = 0
            for c in range(args.chunk_len):
                output, hidden = net(inp[:,c], hidden)
                test_loss += criterion(output.view(args.batch_size, -1), target[:,c])
            test_loss = test_loss/args.chunk_len

        #print (test_loss)
        all_losses[epoch-1] = test_loss
        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss_avg, test_loss))
            #print(generate(device, net, 'Wh', 100), '\n')

            #print("Saving...")
            save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

