import json
import copy
import sys
import timeit

EPS = 0.00001
SLEEP_MODE_ENERGY = 0.1
MIN_ENERGY_CONSUMPTION = 0.4
MAX_ENERGY_CONSUMPTION = 1.0

def format_e(n):
    a = '%E' % n
    return str(round(float(a.split('E')[0].rstrip('0').rstrip('.')), 3)) + '^' + a.split('E')[1]

class VM:
    def __init__(self, id, cpu, ram, bandwidth):
        self.id = id
        self.cpu = cpu
        self.ram = ram
        self.bandwidth = bandwidth
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
        self.energy = SLEEP_MODE_ENERGY

    def add_vm(self, vm, socket):
        vm.socket = socket
        self.vm_list.append(vm)
        self.cpu_available -= vm.cpu
        self.bw_available -= vm.bandwidth
        self.ram_available[socket] -= vm.ram
        self.cpu_utilization = (self.cpu - self.cpu_available) / self.cpu
        prev_energy = self.energy
        self.energy = MIN_ENERGY_CONSUMPTION + \
                     (MAX_ENERGY_CONSUMPTION - MIN_ENERGY_CONSUMPTION) * self.cpu_utilization
        return self.energy - prev_energy

    def remove_vm(self, id):
        for vm in self.vm_list:
            if vm.id == id:
                self.cpu_available += vm.cpu
                self.bw_available += vm.bandwidth
                self.ram_available[vm.socket] += vm.ram
                self.cpu_utilization = (self.cpu - self.cpu_available) / self.cpu
                prev_energy = self.energy
                if self.cpu_utilization < EPS:
                    self.energy = SLEEP_MODE_ENERGY
                else:
                    self.energy = MIN_ENERGY_CONSUMPTION + \
                                  (MAX_ENERGY_CONSUMPTION - MIN_ENERGY_CONSUMPTION) \
                                  * self.cpu_utilization
                self.vm_list.remove(vm)
                return prev_energy - self.energy
        return 0

def next_fit(datacentre, host, socket, sockets, vm):
    while True:
        if datacentre.can_add(vm, host, socket):
            return host, socket
        socket += 1
        if socket == sockets:
            host += 1
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
    best_cpu_util = 0.0
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available_cpu = datacentre.hosts[host].cpu_available - vm.cpu
                full_cpu = datacentre.hosts[host].cpu
                available_ram = datacentre.hosts[host].ram_available[socket] - vm.ram
                full_ram = datacentre.hosts[host].ram_size
                new_util = (((full_cpu - available_cpu) / full_cpu) ** 2 + 
                            ((full_ram - available_ram) / full_ram) ** 2) ** 0.5
                if best_cpu_util < new_util:
                    best_cpu_util = new_util
                    best_host = host
                    best_socket = socket
    return best_host, best_socket

def worst_fit(datacentre, host, socket, sockets, vm):
    best_cpu_util = 1.0
    best_host = -1
    best_socket = -1
    for host in range(len(datacentre.hosts)):
        for socket in range(len(datacentre.hosts[0].ram_available)):
            if datacentre.can_add(vm, host, socket):
                available = datacentre.hosts[host].cpu_available - vm.cpu
                full = datacentre.hosts[host].cpu
                new_util = (full - available) / full
                if best_cpu_util > new_util:
                    best_cpu_util = new_util
                    best_host = host
                    best_socket = socket
    return best_host, best_socket

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

    def remove_vm(self, vm_id):
        for host in range(len(self.hosts)):
            freed = self.hosts[host].remove_vm(vm_id)
            self.current_energy -= freed
            if freed > 0:
                return host

    def update_time(self, new_time):
        delta = new_time - self.current_time
        self.cum_energy += delta * self.current_energy
        self.current_time = new_time

    def total_cpu_util(self):
        answer = 0
        cnt = 0
        for host in self.hosts:
            if host.cpu_utilization > EPS:
                answer += host.cpu_utilization
                cnt += 1
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
        return (total - free) / total

    def total_active(self):
        free = 0
        cnt = 0
        for host in self.hosts:
            if host.cpu_utilization > EPS:
                cnt += 1
        return cnt

class Simulation:
    def __init__(self, file_name, pack_policy, mode = "fill", logs = "False"):
        self.data = json.load(open(file_name))
        host = Host(self.data['PM']['CPU'], self.data['PM']['RAM sockets'], \
                    self.data['PM']['RAM size'], self.data['PM']['bandwidth'])
        self.datacentre = Datacentre(host, self.data['number of PMs'])
        self.pack_policy = pack_policy
        self.vms_allocated = 0
        self.mode = mode
        self.fails = 0
        self.logs = (logs == "logs")

    def extract_VM(self, event):
        vm_json = event['VM']
        return VM(vm_json['id'], vm_json['CPU'], vm_json['RAM'], vm_json['bandwidth'])

    def start_sim(self):
        start_time = timeit.default_timer()
        for event in self.data['events']:
            time = event['time']
            self.datacentre.update_time(time)
            type = event['type']
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
        print("Total time: {}s".format(round(timeit.default_timer() - start_time, 3)))
        print("Total cumulative energy: {}".format(format_e(round(self.datacentre.cum_energy, 2))))
        print("Total hosts active: {}".format(self.datacentre.total_active()))
        print("Total VMs allocated: {}".format(self.vms_allocated))
        print("CPU global utilization: {}".format(round(self.datacentre.total_cpu_util(), 5)))
        print("RAM global utilization: {}".format(round(self.datacentre.total_ram_util(), 5)))


func_match = dict()
func_match['next-fit'] = next_fit
func_match['first-fit'] = first_fit
func_match['best-fit'] = best_fit
func_match['worst-fit'] = worst_fit

if len(sys.argv) >= 3:
    sim = Simulation("input.json", func_match[sys.argv[1]], "fill", sys.argv[2])
else:
    sim = Simulation("input.json", func_match[sys.argv[1]], "fill", "nologs")
sim.start_sim()