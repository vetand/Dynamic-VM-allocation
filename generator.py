import json
import random
import sys
import numpy as np
random.seed(int(sys.argv[1]))
np.random.seed(int(sys.argv[1]))

MAX_TIME_BETWEEN_EVENTS = 10

def VM_random(node, id, min_cpu, max_cpu, min_ram, max_ram, min_bw, max_bw):
    node['VM'] = dict()
    node['VM']['id'] = id
    node['VM']['CPU'] = random.randrange(min_cpu, max_cpu + 1)
    node['VM']['RAM'] = random.randrange(min_ram, max_ram + 1)
    node['VM']['bandwidth'] = random.randrange(min_bw, max_bw + 1)

# https://www.huaweicloud.com/intl/en-us/product/ecs.html
# General Computing-plus C6 ECS
def VM_real(node, id, time_end, poisson = ""):
    ratio = random.choices([2, 4], weights = [2, 1], k = 1)[0]
    cpu = random.choices([2, 4, 8, 12, 16, 24, 32, 64], weights = [3, 3, 3, 3, 2, 2, 1, 1], k = 1)[0]
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

def event_add_VM(node, prev_time, id):
    node.append(dict())
    current_time = prev_time + random.randrange(1, MAX_TIME_BETWEEN_EVENTS + 1)
    running_time = random.choices([1000, 3000, 5000, 7000, 9000], weights = [1, 1, 1, 1, 1], k = 1)[0]
    end_time = current_time + running_time
    node[-1]['time'] = current_time
    node[-1]['type'] = "add"
    VM_real(node[-1], id, end_time, sys.argv[2])
    return current_time, end_time

def event_remove_VM(node, time_end, id):
    node.append(dict())
    node[-1]['time'] = time_end
    node[-1]['type'] = "remove"
    node[-1]['VM'] = dict()
    node[-1]['VM']['id'] = id

# https://www.huaweicloud.com/intl/en-us/pricing/index.html?tab=detail#/deh
# General Computing s6_pro | 264 vCPUs | 702 GB
def host(node):
    node['PM'] = dict()
    node['PM']['CPU'] = 264 # vCPUs
    node['PM']['RAM sockets'] = 1
    node['PM']['RAM size'] = 702 # GB
    node['PM']['bandwidth'] = 10000 # Gbit/sec

def general(PMs, ADDs, file_name):
    global MAX_TIME_BETWEEN_EVENTS
    node = dict()
    node['number of PMs'] = PMs
    host(node)
    node['number of events'] = ADDs * 2
    node['events'] = []
    current_time = 0
    for i in range(ADDs):
        current_time, end_time = event_add_VM(node['events'], current_time, i)
        event_remove_VM(node['events'], end_time, i)
        if i > ADDs / 2 and MAX_TIME_BETWEEN_EVENTS == 10:
            MAX_TIME_BETWEEN_EVENTS = 20
    node['events'] = sorted(node['events'], key = lambda x: x['time'])
    with open(file_name, 'w') as file:
        json.dump(node, file, indent = 4)

general(100, 20000, 'input.json')


