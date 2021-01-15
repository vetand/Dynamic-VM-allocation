import json
import random
import sys
import numpy as np
random.seed(int(sys.argv[1]))
np.random.seed(int(sys.argv[1]))

AVERAGE_TIME_BETWEEN_EVENTS = 10

def VM_random(node, id, min_cpu, max_cpu, min_ram, max_ram, min_bw, max_bw):
    node['VM'] = dict()
    node['VM']['id'] = id
    node['VM']['CPU'] = random.randrange(min_cpu, max_cpu + 1)
    node['VM']['RAM'] = random.randrange(min_ram, max_ram + 1)
    node['VM']['bandwidth'] = random.randrange(min_bw, max_bw + 1)

# https://www.huaweicloud.com/intl/en-us/product/ecs.html
# General Computing-plus C6 ECS
def VM_real(node, id, time_end, poisson = "", weights_param = [10, 1]):
    ratio = random.choices([2, 4], weights = weights_param, k = 1)[0]
    cpu = random.choices([2, 4, 8, 12, 16, 24, 32], weights = [1, 1, 1, 1, 1, 1, 1], k = 1)[0]
    ram = ratio * cpu
    if poisson == "poisson":
        cpu = max(int(np.random.poisson(cpu, 1)[0]), 1)
        ram = max(int(np.random.poisson(ram, 1)[0]), 1)
    node['VM'] = dict()
    node['VM']['id'] = id
    node['VM']['CPU'] = cpu
    node['VM']['RAM'] = ram
    node['VM']['bandwidth'] = int(np.random.poisson(40, 1)[0])
    node['VM']['time end'] = time_end

def event_add_VM(node, prev_time, id, weights = [2, 1]):
    node.append(dict())
    current_time = prev_time + max(int(np.random.poisson(AVERAGE_TIME_BETWEEN_EVENTS, 1)[0]), 1)
    running_time = random.randrange(1, 25) * 500
    end_time = current_time + running_time
    node[-1]['time'] = current_time
    node[-1]['type'] = "add"
    VM_real(node[-1], id, end_time, sys.argv[2], weights)
    return current_time, end_time

def event_remove_VM(node, time_end, id):
    node.append(dict())
    node[-1]['time'] = time_end
    node[-1]['type'] = "remove"
    node[-1]['VM'] = dict()
    node[-1]['VM']['id'] = id

# https://www.huaweicloud.com/intl/en-us/pricing/index.html?tab=detail#/deh
# General Computing s3 | 144 vCPUs | 320 GB
def host(node):
    node['PM'] = dict()
    node['PM']['CPU'] = 144 # vCPUs
    node['PM']['RAM sockets'] = 1
    node['PM']['RAM size'] = 320 # GB
    node['PM']['bandwidth'] = 10000 # Gbit/sec

def general(PMs, ADDs, file_name):
    global AVERAGE_TIME_BETWEEN_EVENTS
    node = dict()
    node['number of PMs'] = PMs
    host(node)
    node['number of events'] = ADDs * 2
    node['events'] = []
    current_time = 0
    weights = [10, 1] # CPU / RAM ratio weights
    for i in range(ADDs):
        current_time, end_time = event_add_VM(node['events'], current_time, i, weights)
        event_remove_VM(node['events'], end_time, i)

        # scenario one - the VMs flow speed changes instantly from 1 / 5 to 1 / 15
        if i > ADDs / 2 and AVERAGE_TIME_BETWEEN_EVENTS == 10:
            AVERAGE_TIME_BETWEEN_EVENTS = 20

        # scenario two - CPU / RAM ratio weights change instantly from [10, 1] to [1, 1] (workload increases due to worse fit ratio)
        #if i > ADDs / 2 and AVERAGE_TIME_BETWEEN_EVENTS == 10:
        #   weights = [1, 1]

        # scenario three - CPU / RAM ratio weights change instantly from [1, 1] to [10, 1] (workload decreases due to fine fit ratio)
        #if i > ADDs / 2 and AVERAGE_TIME_BETWEEN_EVENTS == 10:
        #    weights = [10, 1]
    
    node['events'] = sorted(node['events'], key = lambda x: x['time'])
    with open(file_name, 'w') as file:
        json.dump(node, file, indent = 4)

general(500, 8000, 'input.json')


