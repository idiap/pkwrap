#!/usr/bin/env python
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

# In this test we time the memory copy operation happening in the GPU
import torch

mb = 32
fps = 50
num_outputs = 3000
x = torch.zeros(mb, fps,  num_outputs)
x = x.uniform_()
x = x.cuda()

# run 20 times
elapsed_times = []
for ridx in range(0, 20):    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y = torch.zeros_like(x)
    for i in range(0, mb):
        y[i::mb,:] = x[i,:,:]
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    elapsed_times.append(start.elapsed_time(end))
print("Creating and copying takes ",sum(elapsed_times)/len(elapsed_times))

elapsed_times = []
for ridx in range(0, 20):    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y = x.permute(1, 0, 2).reshape(-1, num_outputs).contiguous()
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    elapsed_times.append(start.elapsed_time(end))
print("Permuting takes ",sum(elapsed_times)/len(elapsed_times))

## Results on one of the GPUs
## Creating and copying takes  1.1409071922302245                                                                                                                   
## Permuting takes  0.18044959977269173