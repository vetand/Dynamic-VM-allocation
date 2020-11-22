import json
import random
random.seed(47)

MAX_TIME_BETWEEN_EVENTS = 10

def VM_random(node, id, min_cpu, max_cpu, min_ram, max_ram, min_bw, max_bw):
    node['VM'] = dict()
    node['VM']['id'] = id
    node['VM']['CPU'] = random.randrange(min_cpu, max_cpu + 1)
    node['VM']['RAM'] = random.randrange(min_ram, max_ram + 1)
    node['VM']['bandwidth'] = random.randrange(min_bw, max_bw + 1)

def event_add_VM(node, prev_time, id):
    node.append(dict())
    current_time = prev_time + random.randrange(1, MAX_TIME_BETWEEN_EVENTS + 1)
    node[-1]['time'] = current_time
    node[-1]['type'] = "add"
    VM_random(node[-1], id, 400, 2500, 1500, 8000, 500, 500)
    return current_time

def host(node):
    node['PM'] = dict()
    node['PM']['CPU'] = 5000
    node['PM']['RAM sockets'] = 2
    node['PM']['RAM size'] = 8192
    node['PM']['bandwidth'] = 10000

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

general(1000, 2000, 'input.json')

