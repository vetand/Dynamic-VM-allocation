/* Copyright (c) 2007-2021. The SimGrid Team. All rights reserved.          */

/* This program is free software; you can redistribute it and/or modify it
 * under the terms of the license (GNU LGPL) which comes with this package. */

#include "simgrid/s4u.hpp"
#include "simgrid/plugins/live_migration.h"
#include "simgrid/s4u/VirtualMachine.hpp"
#include "simgrid/plugins/energy.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

using json = nlohmann::json;

XBT_LOG_NEW_DEFAULT_CATEGORY(s4u_cloud_migration, "Messages specific for this example");

const int NUMBER_OF_HOSTS = 30;
const int HOST_CPU_SIZE = 144;
const double EPS = (double)1 / 1000 / 1000;

struct VirtualMachine {
  int id_;
  int cpu_;
  int ram_size_;
  int bandwidth_;
  int time_end_;
};

struct Event {
  int time_;
  std::string type_;
  VirtualMachine vm_;
};

static void worker(double computation_amount, bool use_bound, double bound)
{
  double clock_start = simgrid::s4u::Engine::get_clock();

  simgrid::s4u::ExecPtr exec = simgrid::s4u::this_actor::exec_init(computation_amount);

  if (use_bound) {
    if (bound < 1e-12) /* close enough to 0 without any floating precision surprise */
      XBT_INFO("bound == 0 means no capping (i.e., unlimited).");
    exec->set_bound(bound);
  }
  exec->start();
  exec->wait();
  double clock_end     = simgrid::s4u::Engine::get_clock();
  double duration      = clock_end - clock_start;
  double flops_per_sec = computation_amount / duration;
  XBT_INFO("Started at %f, ended at %f", clock_start, clock_end);

  if (use_bound)
    XBT_INFO("bound to %f => duration %f (%f flops/s)", bound, duration, flops_per_sec);
  else
    XBT_INFO("not bound => duration %f (%f flops/s)", duration, flops_per_sec);
}

class Simulation {
public:
  Simulation(const std::string& file_name) {
    for (int i = 0; i < NUMBER_OF_HOSTS; ++i) {
      std::stringstream ss;
      ss << "MyHost" << i + 1;
      hosts_.push_back(simgrid::s4u::Host::by_name(ss.str()));
      available_cpu_.push_back(1.0);
    }
    next_event_ = 0;
    current_time_ = 0;

    std::ifstream input(file_name);
    json root;
    input >> root;
    input.close();

    const auto events = root["events"];
    for (const auto& event: events) {
      const auto type = event["type"].get<std::string>();
      if (type == "add") {
        VirtualMachine vm{
          event["VM"]["id"].get<int>(),
          event["VM"]["CPU"].get<int>(),
          event["VM"]["RAM"].get<int>(),
          event["VM"]["bandwidth"].get<int>(),
          event["VM"]["time end"].get<int>()
        };
        events_.push_back({
          event["time"].get<int>(),
          event["type"].get<std::string>(),
          vm
        });
      } else {
        VirtualMachine vm{event["VM"]["id"].get<int>(), 0, 0, 0, 0};
        events_.push_back({
          event["time"].get<int>(),
          event["type"].get<std::string>(),
          vm
        });
      }
    }
  }

  simgrid::s4u::Host* GetHostByNumber(int number) const {
    return hosts_[number];
  }

  Event GetNextEvent() {
    if (next_event_ >= events_.size()) {
      return Event{-1, "", {}};
    }
    return events_[next_event_++];
  }

  simgrid::s4u::VirtualMachine* PackViaFirstFit(const Event& event) {
    const auto vm = event.vm_;
    std::stringstream ss;
    ss << vm.id_;
    simgrid::s4u::this_actor::sleep_for(event.time_ - current_time_);
    current_time_ = event.time_;

    for (int i = 0; i < NUMBER_OF_HOSTS; ++i) {
      const auto host = GetHostByNumber(i);
      double available = available_cpu_[i];
      double demand = (double)vm.cpu_ / HOST_CPU_SIZE;
      if (demand <= available + EPS) {
        const double computation_amount = (event.vm_.time_end_ - event.time_) * host->get_speed();

        auto* vm = new simgrid::s4u::VirtualMachine(ss.str(), host, 1);
        vm->set_ramsize(1e9); // 1Gbytes
        vm->start();

        simgrid::s4u::Actor::create(ss.str(), host, worker, computation_amount, false, 0);
        available_cpu_[i] -= demand;
        return vm;
      }
    }
    return NULL;
  }

private:
  std::vector<simgrid::s4u::Host*> hosts_;
  std::vector<Event> events_;
  std::vector<double> available_cpu_;
  int next_event_;
  int current_time_;
};

static void master_main()
{
  auto sim = Simulation("input.json");
  int number = 0;

  while (true) {
    const auto event = sim.GetNextEvent();
    if (event.time_ == -1) {
      break;
    }
    if (event.type_ == "add") {
      const auto vm = sim.PackViaFirstFit(event);
    }
  }
}

int main(int argc, char* argv[])
{
  /* Get the arguments */
  simgrid::s4u::Engine e(&argc, argv);
  sg_vm_live_migration_plugin_init();
  sg_host_energy_plugin_init();


  /* load the platform file */
  e.load_platform(argv[1]);

  simgrid::s4u::Actor::create("master_", simgrid::s4u::Host::by_name("MyHost0"), master_main);

  e.run();

  XBT_INFO("Bye (simulation time %g)", simgrid::s4u::Engine::get_clock());

  return 0;
}
