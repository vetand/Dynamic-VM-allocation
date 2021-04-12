import json
import copy
import sys
import timeit
import math
import random
random.seed(47)

EPS = 0.00000001
SLEEP_MODE_ENERGY = 0.0
MIN_ENERGY_CONSUMPTION = 0.4
MAX_ENERGY_CONSUMPTION = 1.0
ADAPTIVE_BEST_FIT = True

LOWER_THRESHOLD = 0.6

def format_e(n):
    a = '%E' % n
    return str(round(float(a.split('E')[0].rstrip('0').rstrip('.')), 3)) + '^' + a.split('E')[1]

class VM:
    def __init__(self, id, cpu, ram, bandwidth, time_end):
        self.id = id
        self.cpu = cpu
        self.ram = ram
        self.bandwidth = bandwidth
        self.time_end = time_end
        self.host = None

class Host:
    def __init__(self, cpu, ram_sockets, ram_size, bandwidth):
        self.cpu = cpu
        self.ram_sockets = ram_sockets
        self.ram_size = ram_size
        self.bandwidth = bandwidth

        self.cpu_available = cpu
        self.ram_available = [ram_size] * ram_sockets
        self.bw_available = bandwidth

        self.vm_list = []
        self.cpu_utilization = 0
        self.ram_utilization = 0
        self.energy = SLEEP_MODE_ENERGY

    def add_vm(self, vm, socket):
        vm.socket = socket
        self.vm_list.append(vm)
        self.vm_list.sort(key = lambda x: x.time_end)
        self.cpu_available -= vm.cpu
        self.bw_available -= vm.bandwidth
        self.ram_available[socket] -= vm.ram
        self.cpu_utilization = (self.cpu - self.cpu_available) / self.cpu
        self.ram_utilization = (self.ram_size - self.ram_available[socket]) / self.ram_size
        prev_energy = self.energy
        self.energy = MIN_ENERGY_CONSUMPTION + \
                     (MAX_ENERGY_CONSUMPTION - MIN_ENERGY_CONSUMPTION) * (2 * self.cpu_utilization - self.cpu_utilization ** 1.4)
        return self.energy - prev_energy

    def remove_vm(self, id):
        for vm in self.vm_list:
            if vm.id == id:
                self.cpu_available += vm.cpu
                self.bw_available += vm.bandwidth
                self.ram_available[vm.socket] += vm.ram
                self.cpu_utilization = (self.cpu - self.cpu_available) / self.cpu
                self.ram_utilization = (self.ram_size - self.ram_available[0]) / self.ram_size
                prev_energy = self.energy
                if self.cpu_utilization < EPS:
                    self.energy = SLEEP_MODE_ENERGY
                else:
                    self.energy = MIN_ENERGY_CONSUMPTION + \
                                  (MAX_ENERGY_CONSUMPTION - MIN_ENERGY_CONSUMPTION) \
                                  * (2 * self.cpu_utilization - self.cpu_utilization ** 1.4)
                self.vm_list.remove(vm)
                return prev_energy - self.energy
        return 0

    def get_cpu_util(self):
        return self.cpu_utilization

def next_fit(datacentre, host, socket, sockets, vm):
    cycle = 0
    while True:
        if datacentre.can_add(vm, host, socket):
            return host, socket
        socket += 1
        if socket == sockets:
            host = (host + 1) % len(datacentre.hosts)
            if cycle > 0:
                return -1, -1
            if host == 0:
                cycle += 1
            socket = 0
        if host == len(datacentre.hosts):
            return -1, -1

def first_fit(datacentre, host, socket, sockets, vm):
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                return host, socket
    return -1, -1

def best_fit(datacentre, host, socket, sockets, vm):
    best_util = 0.0 - EPS
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                available_ram = datacentre.hosts[host].ram_available[socket] - vm.ram
                full_ram = datacentre.hosts[host].ram_size
                available_bw = datacentre.hosts[host].bw_available - vm.bandwidth
                full_bw = datacentre.hosts[host].bandwidth
                new_util = (((full_cpu - available_cpu) / full_cpu) ** 2 + 
                            ((full_ram - available_ram) / full_ram) ** 2) ** 0.5
                if best_util < new_util:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
    return best_host, best_socket

def min_add_time_BF(datacentre, host, socket, sockets, vm):
    best_add_time = 1 / EPS
    best_util = 0.0 - EPS
    best_host = -1
    best_socket = -1
    cpu_global = datacentre.total_cpu_util()
    ram_global = datacentre.total_ram_util()
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                available_ram = datacentre.hosts[host].ram_available[socket] - vm.ram
                full_ram = datacentre.hosts[host].ram_size
                available_bw = datacentre.hosts[host].bw_available - vm.bandwidth
                full_bw = datacentre.hosts[host].bandwidth

                if not ADAPTIVE_BEST_FIT:
                    new_util = (((full_cpu - available_cpu) / full_cpu) ** 2 + 
                                ((full_ram - available_ram) / full_ram) ** 2) ** 0.5
                else:
                    if cpu_global > ram_global + 0.1:
                        new_util = (full_cpu - available_cpu) / full_cpu
                    elif ram_global > cpu_global + 0.1:
                        new_util = (full_ram - available_ram) / full_ram
                    else:
                        new_util = (((full_cpu - available_cpu) / full_cpu) ** 2 + 
                                ((full_ram - available_ram) / full_ram) ** 2) ** 0.5

                if len(datacentre.hosts[host].vm_list) == 0:
                    add_time = 1 / EPS - 1
                else:
                    add_time = max(0, vm.time_end - max([cvm.time_end for cvm in datacentre.hosts[host].vm_list]))
                if add_time < best_add_time:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
                    best_add_time = add_time
                elif add_time == best_add_time and best_util < new_util:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
                    best_add_time = add_time
    return best_host, best_socket

def min_add_time2(datacentre, host, socket, sockets, vm):
    best_add_time = 1 / EPS + 1
    best_util = 0.0 - EPS
    best_host = -1
    best_socket = -1
    min_diff_time = 1 / EPS + 1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                available_ram = datacentre.hosts[host].ram_available[socket] - vm.ram
                full_ram = datacentre.hosts[host].ram_size
                available_bw = datacentre.hosts[host].bw_available - vm.bandwidth
                full_bw = datacentre.hosts[host].bandwidth

                new_util = (((full_cpu - available_cpu) / full_cpu) ** 2 + 
                            ((full_ram - available_ram) / full_ram) ** 2) ** 0.5

                if len(datacentre.hosts[host].vm_list) == 0:
                    add_time = 1 / EPS - 1
                    diff_time = 1 / EPS
                else:
                    add_time = max(0, vm.time_end - max([cvm.time_end for cvm in datacentre.hosts[host].vm_list]))
                    diff_time = max(0, max([cvm.time_end for cvm in datacentre.hosts[host].vm_list]) - vm.time_end)
                if add_time < best_add_time:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
                    best_add_time = add_time
                    min_diff_time = diff_time
                elif add_time == 0 and best_add_time == 0:
                    if min_diff_time > diff_time:
                        best_util = new_util
                        best_host = host
                        best_socket = socket
                        best_add_time = add_time
                        min_diff_time = diff_time
                elif add_time == best_add_time and best_util < new_util:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
                    best_add_time = add_time
                    min_diff_time = diff_time
    return best_host, best_socket

def min_add_time3(datacentre, host, socket, sockets, vm):
    best_add_time = 1 / EPS + 1
    best_util = 0.0 - EPS
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.hosts[host].cpu_utilization > EPS:
                server_end_time = max([cvm.time_end for cvm in datacentre.hosts[host].vm_list])
            else:
                server_end_time = datacentre.current_time
            if datacentre.hosts[host].cpu_utilization < EPS:
                continue
            if datacentre.can_add(vm, host, socket) and (vm.time_end - datacentre.current_time) * 3 // 2 > server_end_time - datacentre.current_time:
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                available_ram = datacentre.hosts[host].ram_available[socket] - vm.ram
                full_ram = datacentre.hosts[host].ram_size
                available_bw = datacentre.hosts[host].bw_available - vm.bandwidth
                full_bw = datacentre.hosts[host].bandwidth

                new_util = (((full_cpu - available_cpu) / full_cpu) ** 2 + 
                            ((full_ram - available_ram) / full_ram) ** 2) ** 0.5

                if len(datacentre.hosts[host].vm_list) == 0:
                    add_time = 1 / EPS - 1
                else:
                    add_time = max(0, vm.time_end - max([cvm.time_end for cvm in datacentre.hosts[host].vm_list]))
                if add_time < best_add_time:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
                    best_add_time = add_time
                elif add_time == best_add_time and best_util < new_util:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
                    best_add_time = add_time
    if best_host != -1:
        return best_host, best_socket
    else:
        return min_add_time_BF(datacentre, host, socket, sockets, vm)

def min_add_time_FF(datacentre, host, socket, sockets, vm):
    best_add_time = 1 / EPS
    best_util = 0.0 - EPS
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                available_ram = datacentre.hosts[host].ram_available[socket] - vm.ram
                full_ram = datacentre.hosts[host].ram_size
                available_bw = datacentre.hosts[host].bw_available - vm.bandwidth
                full_bw = datacentre.hosts[host].bandwidth
                new_util = (((full_cpu - available_cpu) / full_cpu) ** 2 + 
                            ((full_ram - available_ram) / full_ram) ** 2) ** 0.5
                if len(datacentre.hosts[host].vm_list) == 0:
                    add_time = 1 / EPS - 1
                else:
                    add_time = max(0, vm.time_end - max([cvm.time_end for cvm in datacentre.hosts[host].vm_list]))
                if add_time < best_add_time:
                    best_util = new_util
                    best_host = host
                    best_socket = socket
                    best_add_time = add_time
    return best_host, best_socket

def min_server_endtime(datacentre, host, socket, sockets, vm):
    best_endtime = -1
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            end_time = 0
            for vm_ in datacentre.hosts[host].vm_list:
                end_time = max(end_time, vm_.time_end)
            if datacentre.can_add(vm, host, socket) and end_time > best_endtime:
                best_endtime = end_time
                best_host = host
                best_socket = socket
    return best_host, best_socket

def best_fit_cpu(datacentre, host, socket, sockets, vm):
    best_cpu_util = 0.0 - EPS
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                new_util = (full_cpu - available_cpu) / full_cpu
                if best_cpu_util < new_util:
                    best_cpu_util = new_util
                    best_host = host
                    best_socket = socket
    return best_host, best_socket

# http://zhenxiao.com/papers/tpds2012.pdf
def min_skewness(datacentre, host, socket, sockets, vm):
    best_skew = 2 + EPS
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                available_ram = datacentre.hosts[host].ram_available[socket] - vm.ram
                full_ram = datacentre.hosts[host].ram_size
                available_bw = datacentre.hosts[host].bw_available - vm.bandwidth
                full_bw = datacentre.hosts[host].bandwidth
                average_util = (((full_cpu - available_cpu) / full_cpu) + 
                               ((full_ram - available_ram) / full_ram)) / 2
                new_skew = ((((full_cpu - available_cpu) / full_cpu) / average_util - 1) ** 2 + \
                            (((full_ram - available_ram) / full_ram) / average_util - 1) ** 2) ** 0.5
                if best_skew > new_skew:
                    best_skew = new_skew
                    best_host = host
                    best_socket = socket
    return best_host, best_socket

def worst_fit(datacentre, host, socket, sockets, vm):
    best_cpu_util = 1.0
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.hosts[host].cpu_utilization > 0 and datacentre.can_add(vm, host, socket):
                available = datacentre.hosts[host].cpu_available - vm.cpu
                full = datacentre.hosts[host].cpu
                new_util = (full - available) / full
                if best_cpu_util > new_util:
                    best_cpu_util = new_util
                    best_host = host
                    best_socket = socket
    if best_host == -1:
        for host in range(len(datacentre.hosts)):
            if datacentre.hosts[host].cpu_utilization == 0:
                return host, 0
    return best_host, best_socket

# http://ilanrcohen.droppages.com/pdfs/FracOnVBinPack.pdf
def f_res(x, y):
    return abs(x - y) < 0.3

def first_fit_f_res(datacentre, host, socket, sockets, vm):
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            cpu = (datacentre.hosts[host].cpu - datacentre.hosts[host].cpu_available + vm.cpu) \
                                                                    / datacentre.hosts[host].cpu
            ram = (datacentre.hosts[host].ram_size - sum(datacentre.hosts[host].ram_available) \
                                                     + vm.ram) / datacentre.hosts[host].ram_size
            bw = (datacentre.hosts[host].bandwidth - datacentre.hosts[host].bw_available + vm.bandwidth) \
                                                                       / datacentre.hosts[host].bandwidth
            if f_res(cpu, ram):
                if datacentre.can_add(vm, host, socket):
                     return host, socket
    return first_fit(datacentre, host, socket, sockets, vm)

# http://ilanrcohen.droppages.com/pdfs/FracOnVBinPack.pdf
class RandomizedSlidingWindowAssignment:
    def __init__(self, dim_number = 2):
        self.allocated = [0] * dim_number
        self.factor = 0.69 # = 1 / (1 + eps)
        
    def _update(self, host, vm):
        self.allocated[0] += (vm.cpu / host.cpu) * self.factor
        self.allocated[1] += (vm.ram / (host.ram_size * host.ram_sockets)) * self.factor

    def fit(self, datacentre, host, socket, sockets, vm):
        probs = [0.0] * len(datacentre.hosts)
        if max(self.allocated) == 0:
            probs[0] = 1.0
            answers = [i for i in range(len(datacentre.hosts))]
            bucket = random.choices(answers, weights = probs)[0]
            if not datacentre.can_add(vm, bucket, 0):
                host, socket = first_fit(datacentre, host, socket, sockets, vm)
            else:
                host, socket = bucket, 0
            self._update(datacentre.hosts[host], vm)
            return host, socket
        min_index = math.ceil(max(self.allocated))
        max_index = math.floor(math.ceil(max(self.allocated)) * math.e)
        for j in range(min_index, max_index + 1):
            probs[j] = math.log((j + 1) / j)
        max_index = math.ceil(math.ceil(max(self.allocated)) * math.e)
        probs[max_index] = math.log(math.ceil(math.ceil(max(self.allocated)) * math.e) / max_index)
        answers = [i for i in range(len(datacentre.hosts))]
        bucket = random.choices(answers, weights = probs)[0]
        while datacentre.hosts[bucket - 1].cpu_utilization == 0:
            bucket -= 1
        if not datacentre.can_add(vm, bucket, 0):
            host, socket = first_fit(datacentre, host, socket, sockets, vm)
        else:
            host, socket = bucket, 0
        self._update(datacentre.hosts[host], vm)
        return host, socket

class Datacentre:
    def __init__(self, host_type, num_hosts):
        self.hosts = []
        for i in range(num_hosts):
            self.hosts.append(copy.deepcopy(host_type))
        self.cum_energy = 0
        self.current_energy = num_hosts * SLEEP_MODE_ENERGY
        self.current_time = 0
        self.prev_host = 0
        self.prev_socket = 0
        self.vm_count = 0
        self.prev_migration = 0

    def can_add(self, vm, host_id, socket):
        if self.hosts[host_id].cpu_available < vm.cpu:
            return False
        if self.hosts[host_id].bw_available < vm.bandwidth:
            return False
        if self.hosts[host_id].ram_available[socket] < vm.ram:
            return False
        return True

    def add_vm(self, vm, host_id, socket):
        approve = self.can_add(vm, host_id, socket)
        if approve:
            added = self.hosts[host_id].add_vm(vm, socket)
            self.current_energy += added
            self.prev_host = host_id
            self.prev_socket = socket
            self.vm_count += 1

    def remove_vm(self, vm_id):
        for host in range(len(self.hosts)):
            freed = self.hosts[host].remove_vm(vm_id)
            self.current_energy -= freed
            if freed > 0:
                self.vm_count -= 1
                return host

    def migrateVM(self, vm_id, host):
        self.remove_vm(vm_id)
        self.add_vm(vm_id, hostB)

    def update_time(self, new_time):
        delta = new_time - self.current_time
        if new_time >= 250000 and new_time <= 500000:
            self.cum_energy += delta * self.current_energy
        self.current_time = new_time

    def total_cpu_util(self):
        answer = 0
        cnt = 0
        for host in self.hosts:
            if host.cpu_utilization > EPS:
                answer += host.cpu_utilization
                cnt += 1
        if cnt == 0:
            return 0.0
        return answer / cnt

    def total_ram_util(self):
        free = 0
        cnt = 0
        for host in self.hosts:
            if host.cpu_utilization > EPS:
                cnt += 1
                for in_socket in host.ram_available:
                    free += in_socket
        total = self.hosts[0].ram_size * len(self.hosts[0].ram_available) * cnt
        if total == 0:
            return 0.0
        return (total - free) / total

    def total_bandwidth_util(self):
        free = 0
        cnt = 0
        for host in self.hosts:
            if host.cpu_utilization > EPS:
                cnt += 1
                free += host.bw_available
        total = self.hosts[0].bandwidth * cnt
        if total == 0:
            return 0.0
        return (total - free) / total

    def total_active(self):
        cnt = 0
        for host in self.hosts:
            if host.cpu_utilization > EPS:
                cnt += 1
        return cnt

    def ideal_result(self):
        cnt = 0
        for host in self.hosts:
            if host.cpu_utilization > EPS:
                cnt += 1
        return int(cnt * max(self.total_cpu_util(), self.total_ram_util(), self.total_bandwidth_util())) + 1

# select %percent% least loaded hosts and migrate all VMs from them
def simple_migration_policy(datacentre, policy, rate = 40000, percent = 0.15):
    if datacentre.current_time - datacentre.prev_migration < rate:
        return
    datacentre.prev_migration = datacentre.current_time
    workloads = []
    for hostID in range(len(datacentre.hosts)):
        if datacentre.hosts[hostID].get_cpu_util() > EPS:
            workloads.append(datacentre.hosts[hostID].get_cpu_util())
    workloads.sort()
    if len(workloads) == 0:
        return
    threshold = workloads[int(len(workloads) * percent)]
    vm_migration_list = []
    for hostID in range(len(datacentre.hosts)):
        if datacentre.hosts[hostID].get_cpu_util() <= threshold:
            for vm in datacentre.hosts[hostID].vm_list:
                vm_migration_list.append(vm)
                datacentre.remove_vm(vm.id)

    def compare(vm):
        return vm.cpu

    vm_migration_list = sorted(vm_migration_list, key = compare, reverse = True)
    for vm in vm_migration_list:
        host, socket = policy(datacentre, datacentre.prev_host,
                    datacentre.prev_socket,
                    len(datacentre.hosts[0].ram_available), vm)
        if host != -1:
            datacentre.add_vm(vm, host, socket)
        else:
            print("Cannot reallocate some VM!!!")

class Simulation:
    def __init__(self, function_name,
                       file_name,
                       pack_policy,
                       migration_policy,
                       mode = "fill",
                       logs = "False"):
        self.function_name = function_name
        self.data = json.load(open(file_name))
        host = Host(self.data['PM']['CPU'], self.data['PM']['RAM sockets'], \
                    self.data['PM']['RAM size'], self.data['PM']['bandwidth'])
        self.datacentre = Datacentre(host, self.data['number of PMs'])
        self.migration_policy = migration_policy
        self.pack_policy = pack_policy
        self.vms_allocated = 0
        self.mode = mode
        self.fails = 0
        self.logs = (logs == "logs")

    def extract_VM(self, event):
        vm_json = event['VM']
        return VM(vm_json['id'], vm_json['CPU'], vm_json['RAM'], vm_json['bandwidth'], vm_json['time end'])

    def start_sim(self):
        start_time = timeit.default_timer()
        answer_data = []
        count = 0
        for event in self.data['events']:
            time = event['time']
            #if time == 20000:
            #    for host in self.datacentre.hosts:
            #        print(host.cpu_utilization, host.ram_utilization)
            self.datacentre.update_time(time)
            type = event['type']
            answer_data.append(dict())
            answer_data[-1]['active hosts'] = self.datacentre.total_active()
            answer_data[-1]['time'] = event['time']
            answer_data[-1]['cpu'] = self.datacentre.total_cpu_util()
            answer_data[-1]['ram'] = self.datacentre.total_ram_util()
            answer_data[-1]['ideal result'] = self.datacentre.ideal_result()
            answer_data[-1]['current energy'] = self.datacentre.current_energy
            count += 1
            #if count % 1000 == 0:
            #    print("Processed event #{}".format(count))

            self.migration_policy(self.datacentre, self.pack_policy)

            if type == "add":
                vm = self.extract_VM(event)
                host, socket = self.pack_policy(self.datacentre, self.datacentre.prev_host,
                                                self.datacentre.prev_socket,
                                                len(self.datacentre.hosts[0].ram_available), vm)
                if host != -1:
                    self.datacentre.add_vm(vm, host, socket)
                    if self.logs:
                        print("Time = {}, new event: add VM with id = {} to PM #{}, new CPU utulization = {}" \
                           .format(time, vm.id, host, self.datacentre.hosts[host].cpu_utilization))
                    self.vms_allocated += 1
                else:
                    if self.logs:
                        print("Time = {}, failed to allocate new VM".format(time))
                    if self.mode == "fill":
                        self.fails += 1
                        if self.fails == 100:
                            break
            else:
                id = event['VM']['id']
                host = self.datacentre.remove_vm(id)
                if self.logs:
                    print("Time = {}, new event: remove VM with id = {} from PM #{}, new CPU utulization = {}" \
                        .format(time, id, host, self.datacentre.hosts[host].cpu_utilization))
        with open('output.json', 'w') as outfile:
            json.dump(answer_data, outfile)
        
        print(self.function_name, ":", sep = "")
        print("Total time: {}s".format(round(timeit.default_timer() - start_time, 3)))
        print("Total cumulative energy: {}".format(format_e(round(self.datacentre.cum_energy, 2))))
        print("Total hosts active: {}".format(self.datacentre.total_active()))
        print("Total VMs allocated: {}".format(self.vms_allocated))
        print("CPU global utilization: {}".format(round(self.datacentre.total_cpu_util(), 5)))
        print("RAM global utilization: {}".format(round(self.datacentre.total_ram_util(), 5)))
        print("bandwidth global utilization: {}".format(round(self.datacentre.total_bandwidth_util(), 5)))
        print()
        print()

def EMPTY_FUNC(*args):
    pass

RSWA_policy = RandomizedSlidingWindowAssignment(2)
func_match = dict()
func_match['next-fit'] = next_fit
func_match['first-fit'] = first_fit
func_match['first-fit-f-res'] = first_fit_f_res
func_match['best-fit'] = best_fit
func_match['best-fit-cpu'] = best_fit_cpu
func_match['worst-fit'] = worst_fit
func_match['min-skew'] = min_skewness
func_match['RSWA'] = RSWA_policy.fit
func_match['min-add-time-bf'] = min_add_time_BF
func_match['min-add-time-ff'] = min_add_time_FF
func_match['min-add-time-2'] = min_add_time2
func_match['min-add-time-3'] = min_add_time3
func_match['min-server-endtime'] = min_server_endtime

func_match['no-migration'] = EMPTY_FUNC
func_match['migration-simple'] = simple_migration_policy

sim = Simulation(sys.argv[1] + ", " + sys.argv[2],
                "input.json",
                func_match[sys.argv[1]],
                func_match[sys.argv[2]],
                "no-fill",
                "no-logs")
ADAPTIVE_BEST_FIT = (sys.argv[3] == 'adaptive')
sim.start_sim()