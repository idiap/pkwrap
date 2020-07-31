#!/usr/bin/env python
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

from dask import delayed
from dask.distributed import Client
n_workers = 5
from time import sleep


def scale_to_sge(n_workers):
    queue="q_gpu"
    queue_resource_spec="q_gpu=TRUE"
    memory="4GB"
    sge_log= "./logs"
    from dask_jobqueue import SGECluster
    cluster = SGECluster(queue=queue, memory=memory, cores=1, processes=1,
              log_directory=sge_log,
              local_directory=sge_log,
              resource_spec=queue_resource_spec
              )
    cluster.scale_up(n_workers)
    return Client(cluster)  # start local workers as threads


#### SWITCH THIS IF YOU WANT TO RUN LOCALLY OR IN OUR SGE GRID ###

# Local client
client = scale_to_sge(2)

def inc(x):
    sleep(0.5)
    return x + 1

def add(x, y):
    sleep(0.5)
    return x + y

# x = inc(1)
# y = inc(2)
# z = add(x, y)
data = range(10)

results = []
for x in data:
    y = delayed(inc)(x)
    y = delayed(add)(y, y)
    results.append(y)
    
total = delayed(sum)(results)
#total
total.compute(scheduler=client)
