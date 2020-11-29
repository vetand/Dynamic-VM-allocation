import json
import random
import sys
random.seed(int(sys.argv[1]))

MAX_TIME_BETWEEN_EVENTS = 10

def VM_random(node, id, min_cpu, max_cpu, min_ram, max_ram, min_bw, max_bw):
    node['VM'] = dict()
    node['VM']['id'] = id
    node['VM']['CPU'] = random.randrange(min_cpu, max_cpu + 1)
    node['VM']['RAM'] = random.randrange(min_ram, max_ram + 1)
    node['VM']['bandwidth'] = random.randrange(min_bw, max_bw + 1)

# https://www.huaweicloud.com/intl/en-us/product/ecs.html
# General Computing-plus C6 ECS
def VM_real(node, id):
    ratio = random.choices([2, 4], weights = [2, 1], k = 1)[0]
    cpu = random.choices([2, 4, 8, 12, 16, 24, 32, 64], weights = [3, 3, 3, 3, 2, 2, 1, 1], k = 1)[0]
    ram = ratio * cpu
    node['VM'] = dict()
    node['VM']['id'] = id
    node['VM']['CPU'] = cpu
    node['VM']['RAM'] = ram
    node['VM']['bandwidth'] = 40

def event_add_VM(node, prev_time, id):
    node.append(dict())
    current_time = prev_time + random.randrange(1, MAX_TIME_BETWEEN_EVENTS + 1)
    node[-1]['time'] = current_time
    node[-1]['type'] = "add"
    VM_real(node[-1], id)
    return current_time

# https://www.huaweicloud.com/intl/en-us/pricing/index.html?tab=detail#/deh
# General Computing s6_pro | 264 vCPUs | 702 GB
def host(node):
    node['PM'] = dict()
    node['PM']['CPU'] = 264 # vCPUs
    node['PM']['RAM sockets'] = 1
    node['PM']['RAM size'] = 702 # GB
    node['PM']['bandwidth'] = 700 # Gbit/sec

def general(PMs, ADDs, file_name):
    node = dict()
    node['number of PMs'] = PMs
    host(node)
    node['number of events'] = ADDs
    node['events'] = []
    current_time = 0
    for i in range(ADDs):
        current_time = event_add_VM(node['events'], current_time, i)
    with open(file_name, 'w') as file:
        json.dump(node, file, indent = 4)

general(1000, 10000, 'input.json')


