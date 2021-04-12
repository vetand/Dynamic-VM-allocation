import json
import random
import sys
import numpy as np
import copy
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
def VM_real(node, id, time_end, poisson = "", weights_param = [2, 4, 2, 1]):
    ratio = random.choices([1, 2, 4, 8], weights = weights_param, k = 1)[0]
    cpu = random.choices([1, 2, 4, 8, 15, 40], weights = [17, 37, 32, 9, 4, 1], k = 1)[0]
    ram = ratio * cpu
    if poisson == "poisson" and cpu >= 15:
        cpu = max(int(np.random.poisson(cpu, 1)[0]), 1)
        ram = max(int(np.random.poisson(ram, 1)[0]), 1)
    node['VM'] = dict()
    node['VM']['id'] = id
    node['VM']['CPU'] = cpu
    node['VM']['RAM'] = ram
    node['VM']['bandwidth'] = int(np.random.poisson(40, 1)[0])
    node['VM']['time end'] = time_end

def event_add_VM(node, current_time, id, running_time, weights = [2, 4, 2, 1],):
    node.append(dict())
    running_time = max(int(np.random.poisson(running_time, 1)[0]), 1)
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

def day_night_scenario(PMs, sim_time, file_name):
    node = dict()
    node['number of PMs'] = PMs
    host(node)
    node['number of events'] = 0
    node['events'] = []
    current_time = 0
    weights = [8, 8, 4, 1] # CPU / RAM ratio weights
    currentID = 0
    for i in range(sim_time):
        if i % 1440 <= 480 or i % 1440 >= 1320: # NIGHT
            num_requests_this_minute = abs(int(np.random.poisson(2, 1)[0]))
        else: #DAY
            num_requests_this_minute = abs(int(np.random.poisson(8, 1)[0]))
        
        #short-time machines
        for j in range(num_requests_this_minute):
            running_time = random.choices([random.randrange(5, 15), 
                                   random.randrange(15, 100)],
                                   weights = [1, 1], k = 1)[0]
            current_time = i
            _, end_time = event_add_VM(node['events'], current_time, 
                                       currentID, running_time, weights)
            event_remove_VM(node['events'], end_time, currentID)
            currentID += 1
        
        # long-time machine
        running_time = random.choices([
            random.randrange(100, 360),
            random.randrange(360, 1500),
            random.randrange(1500, 4000)], 
            weights = [3, 1, 1], k = 1)[0]
        current_time = i
        _, end_time = event_add_VM(node['events'], current_time,
                                    currentID, running_time, weights)
        event_remove_VM(node['events'], end_time, currentID)
        currentID += 1
    
    node['events'] = sorted(node['events'], key = lambda x: x['time'])
    with open(file_name, 'w') as file:
        json.dump(node, file, indent = 4)

def simple(PMs, ADDs, file_name, SCENARIO=1):
    global AVERAGE_TIME_BETWEEN_EVENTS
    node = dict()
    node['number of PMs'] = PMs
    host(node)
    node['number of events'] = ADDs * 2
    node['events'] = []
    current_time = 0
    if SCENARIO in [1, 2, 5]:
        weights = [8, 8, 4, 1] # CPU / RAM ratio weights
    elif SCENARIO == 3:
        weights = [8, 2, 1, 0] # CPU / RAM ratio weights
    elif SCENARIO == 4:
        weights = [1, 1, 2, 1] # CPU / RAM ratio weights

    if SCENARIO == 2:
        AVERAGE_TIME_BETWEEN_EVENTS = 30
    else:
        AVERAGE_TIME_BETWEEN_EVENTS = 10

    for i in range(ADDs):
        running_time = random.choices([random.randrange(100, 3000), 
                                        random.randrange(300, 2000),
                                        random.randrange(2000, 12000),
                                        random.randrange(12000, 30000),
                                        random.randrange(30000, 80000),
                                        random.randrange(80000, 200000)],
                                        weights = [5, 5, 2, 2, 1, 1], k = 1)[0]
        _, end_time = event_add_VM(node['events'], current_time, i, running_time, weights)
        current_time += np.random.poisson(AVERAGE_TIME_BETWEEN_EVENTS)
        event_remove_VM(node['events'], end_time, i)

        # scenario one - the VMs flow speed changes instantly from 1 / 10 to 1 / 30
        if SCENARIO == 1:
            if i >= ADDs * 0.7 and AVERAGE_TIME_BETWEEN_EVENTS == 10:
                AVERAGE_TIME_BETWEEN_EVENTS = 30

        if SCENARIO == 2:
            if i >= ADDs * 0.3 and AVERAGE_TIME_BETWEEN_EVENTS == 30:
                AVERAGE_TIME_BETWEEN_EVENTS = 20

        # scenario two - CPU / RAM ratio weights change instantly from [8, 2, 1, 0] to [1, 1, 2, 1] (workload increases due to worse fit ratio)
        if SCENARIO == 3:
            if i > ADDs / 2 and AVERAGE_TIME_BETWEEN_EVENTS == 10:
                weights = [1, 1, 2, 1]

        # scenario three - CPU / RAM ratio weights change instantly from [1, 1, 2, 1] to [8, 2, 1, 0] (workload decreases due to fine fit ratio)
        if SCENARIO == 4:
            if i > ADDs / 2 and AVERAGE_TIME_BETWEEN_EVENTS == 10:
                weights = [8, 2, 1, 0]
    
    node['events'] = sorted(node['events'], key = lambda x: x['time'])
    with open(file_name, 'w') as file:
        json.dump(node, file, indent = 4)

def full_poisson(PMs, sim_time, file_name):
    global AVERAGE_TIME_BETWEEN_EVENTS
    node = dict()
    node['number of PMs'] = PMs
    host(node)
    node['number of events'] = 0
    node['events'] = []
    current_time = 0
    weights = [8, 8, 4, 1] # CPU / RAM ratio weights
    currentID = 0

    speed_rate_short = 6
    speed_rate_long = 1
    weights_short = [1, 1]
    weights_long = [3, 1, 1]

    for i in range(sim_time):
        if i % 2000 == 0:
            speed_rate_short = min(max(np.random.poisson(speed_rate_short), 1), 9)
            speed_rate_long = min(max(np.random.poisson(speed_rate_long), 1), 4)
            for i in range(len(weights_short)):
                weights_short[i] = max(np.random.poisson(weights_short[i]), 1)
            for i in range(len(weights_long)):
                weights_long[i] = max(np.random.poisson(weights_long[i]), 1)
            
        #short-time machines
        for j in range(speed_rate_short):
            running_time = random.choices([np.random.poisson(10), 
                                   random.randrange(15, 100)],
                                   weights = weights_short, k = 1)[0]
            current_time = i
            random_weights = copy.deepcopy(weights)
            for k in range(len(weights)):
                random_weights[k] = np.random.poisson(weights[k])
            _, end_time = event_add_VM(node['events'], current_time, 
                                       currentID, running_time, random_weights)
            event_remove_VM(node['events'], end_time, currentID)
            currentID += 1
        
        # long-time machine
        for j in range(speed_rate_long):
            running_time = random.choices([
                random.randrange(100, 360),
                random.randrange(360, 1500),
                random.randrange(1500, 4000)], 
                weights = weights_long, k = 1)[0]
            current_time = i
            _, end_time = event_add_VM(node['events'], current_time,
                                        currentID, running_time, weights)
            event_remove_VM(node['events'], end_time, currentID)
            currentID += 1
    
    node['events'] = sorted(node['events'], key = lambda x: x['time'])
    with open(file_name, 'w') as file:
        json.dump(node, file, indent = 4)

# MODE 1 - workload decreases abruptly
#simple(200, 40000, 'input.json', 1)

# MODE 2 - workload increases abruptly
#simple(100, 40000, 'input.json', 2)

# MODE 3 - type distribution changes abruptly
#simple(100, 40000, 'input.json', 3)

# MODE 4 - type distribution changes abruptly
simple(100, 40000, 'input.json', 4)

# MODE 5 - absolutly constant distributions and speed rate
#simple(100, 40000, 'input.json', 5) 

# MODE 6 - day and night workload periodical cycle
#day_night_scenario(90, 20000, 'input.json')

# MODE 7 - distribution and speed poisson process
#full_poisson(170, 40000, 'input.json')


