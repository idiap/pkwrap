#!/usr/bin/env python
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

# In this test we time the memory copy operation happening during padding
import torch
import time
import torch.nn.functional as F

input = torch.zeros(32, 150, 40)
input.uniform_()
context_len = 3
mb, T, D = input.shape
subsampling_factor = 1

l = context_len
N = T-l+1
time_list = []
W = torch.zeros(D*context_len, 1024)
W = W.uniform_()
for i in range(0, 20):
    start_time = time.time()
    padded_input = torch.zeros(mb, N, D*context_len, device=input.device)
    start_d = 0
    for i in range(l):
        end_d = start_d + D
        padded_input[:,:,start_d:end_d] = input[:,i:i+N,:]
        start_d = end_d
    if subsampling_factor>1:
        padded_input = padded_input[:,::subsampling_factor,:]
    x = padded_input.matmul(W)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_list.append(elapsed_time)

avg_time = sum(time_list)/len(time_list)
print("Avg time taken for padding + mul = {}".format(avg_time))    

time_list = []
W = W.permute(1, 0).reshape(-1, D, context_len)
for i in range(0, 20):
    start_time = time.time()
    x = F.conv1d(input.permute(0, 2, 1), W)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_list.append(elapsed_time)

avg_time = sum(time_list)/len(time_list)
print("Avg time taken for conv1d = {}".format(avg_time))     