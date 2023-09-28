from typing import List, Any

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.utils import common
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tf_agents.environments import utils
from tf_agents.environments import wrappers
from params import *
import copy
import math
import random

reward_weight = 0.5 # it MUST be 0~1

#for MD         #tail                         #head
layers_size   = [ 100,    100,     50,    50,     20,     20,     10,      10,     20] #GFLOP
e_consumption = [   1,      1,     0.5,   0.5,    0.2,    0.2,    0.1,     0.1,    0.2] #*0.1 Watt/Task
                #tail # input
output_size   = [10,  20] # MByte intermediate size

min_md_comp = 15 #* 10 GFLOPS
max_md_comp = 20 #* 10 GFLOPS

min_edge_comp = 30 #* 10 GFLOPS
max_edge_comp = 40 #* 10 GFLOPS

# min_channel_gain = 0
# max_channel_gain = 100

#transmission parameter (gain follows )
noise = -110 # -110dB
user_tp = -7 # 23dBm
gain_mean = -90 # -90dB
min_gain   = -95
max_gain   = -75
bandwidth = 20
Watt = 0.2


action_static = 2



class SC_MTL_State():
    def __init__(self, layers_size=layers_size):
        # self.channel_gain #H                                                                                                     #for reward
        self.state_dimension = (max_edge_comp - min_edge_comp + 1) + (max_md_comp - min_md_comp + 1) + (max_gain - min_gain + 1) + 2
        self.layers_size     = layers_size
        self.e_consumption   = np.array(self.layers_size)*0.01
        self.output_size     = output_size
        self.bandwidth       = bandwidth

        self.edge_comp = random.randint(min_edge_comp, max_edge_comp) #C_E
        self.md_comp   = random.randint(min_md_comp,   max_md_comp)   #C_M

        self.gain       = np.clip(int(-90 + (np.random.rayleigh(0.1) - 0.1) * 50), -95, -75)      # todo: this value should be calculated rather than directly using

    def set_parameter(self, bandwidth, layers_size, output_size):
        self.bandwidth = bandwidth
        self.layers_size = layers_size
        self.output_size = output_size

    def new_state(self):
        self.edge_comp = random.randint(min_edge_comp, max_edge_comp) #C_E
        self.md_comp   = random.randint(min_md_comp,   max_md_comp)   #C_M
        self.gain       = np.clip(int(-90 + (np.random.rayleigh(0.1) - 0.1) * 50), -95, -75)      # todo: this value should be calculated rather than directly using


    def gain_to_transmission_power(self, gain):
        Mbps     = self.bandwidth*math.log2(1 + (user_tp+gain-noise))
        return Mbps/8


    def make_state(self):
        state = [0,0]
        edge_state = [0 for i in range(max_edge_comp-min_edge_comp+1)]
        edge_state[self.edge_comp-min_edge_comp] = 1
        md_state = [0 for i in range(max_md_comp - min_md_comp + 1)]
        md_state[self.md_comp - min_md_comp] = 1
        gain_state = [0 for i in range(max_gain - min_gain + 1)]
        gain_state[self.gain - min_gain] = 1
        state.extend(edge_state)
        state.extend(md_state)
        state.extend(gain_state)
        return state


    def make_min_state(self):
        state = [-10000,-10000]
        edge_state = [0 for i in range(max_edge_comp-min_edge_comp+1)]
        md_state = [0 for i in range(max_md_comp - min_md_comp + 1)]
        gain_state = [0 for i in range(max_gain - min_gain + 1)]
        state.extend(edge_state)
        state.extend(md_state)
        state.extend(gain_state)
        return state


    def make_max_state(self):
        state = [0,0]
        edge_state = [1 for i in range(max_edge_comp-min_edge_comp+1)]
        md_state = [1 for i in range(max_md_comp - min_md_comp + 1)]
        gain_state = [1 for i in range(max_gain - min_gain + 1)]
        state.extend(edge_state)
        state.extend(md_state)
        state.extend(gain_state)
        return state


    def make_initial_input(self):
        return [self.make_min_state(), self.make_max_state(), self.state_dimension, self.make_state(), len(self.layers_size)]



class SC_MTL_Env(py_environment.PyEnvironment):
    def __init__(self, initial_input, state, env_type, action_dim, action_min, action_max, reward_weight):
        [state_min, state_max, state_dimension, initial_state, n_layers] = initial_input
        self.action_min = action_min
        self.action_max = action_max
        self.action_dim = action_dim
        self._action_spec      = array_spec.BoundedArraySpec(
                                 shape=(self.action_dim,), dtype=np.float32, minimum=action_min, maximum=action_max, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(state_dimension,), dtype=np.float32, minimum=state_min,
                                 maximum=state_max, name='observation')
        self._init_state    = initial_state
        self._state         = initial_state
        self.state          = state
        # self.period         = period
        self.n_layers       = n_layers
        self._episode_ended = False
        self.env_type       = env_type
        self.weight = reward_weight

    def action_spec(self):
        return self._action_spec


    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        self.state.new_state()
        self._state         = self.state.make_state()
        self._episode_ended = False
        return ts.restart(np.array(self._state,dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            return self._reset()
        if action.any() == b'Edge_Only':
            action_index = self.action_dim-1
        elif action.any() == b'MD_Only':
            action_index = self.action_dim-2
        elif action.any() == b'Head_Only':
            action_index = 0
        elif action.any() == b'Random':
            action_index = random.randint(0,self.action_dim-1)
        else:
            action_index = np.argmax(action)

        action_list = self.action_index_to_list(action_index)
        # calculate cost
        cost, energy_consump, task_comple = self.cost_function(self.state, action_list)

        self.state.new_state()
        self._state = self.state.make_state()

        self._state[0] = -energy_consump
        self._state[1] = -task_comple

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), -cost)
        else:
            return ts.transition(np.array(self._state,  dtype=np.float32), -cost, discount= 0.99)

    def action_index_to_list(self, action_index):
        if action_index == self.action_dim-1:
            action_list = [0]
            return action_list # Edge_only
        else:
            action_list = [0 for i in range(len(self.state.layers_size) - 1)]
            index = 0
            while action_index != 0:
                action_list[index] = action_index % 2
                action_index = action_index // 2
                index += 1
            if action_index == self.action_dim-2:
                print(action_list)
            return action_list

    def cost_function(self, state, action):
        energy_consump = self.energy_consumption(state, action)
        task_comple    = self.task_completion_time(state, action)
        cost = self.weight*energy_consump + \
               (1-self.weight)*task_comple
        return cost, energy_consump, task_comple


    def energy_consumption(self, state, action):
        energy = 0
        if action == [0]:
            return state.output_size[-1]/state.gain_to_transmission_power(state.gain)*Watt
        else:
            if len(action) == len(self.state.layers_size)-1:
                energy += state.e_consumption[-1]
                if np.all(action) == False:
                    energy += Watt*state.output_size[0]/state.gain_to_transmission_power(state.gain)
                for i in range(len(action)):
                    energy += action[i]*(state.e_consumption[i])
            else:
                RuntimeError('action error')
            return energy


    def task_completion_time(self, state, action):
        latency = [0 for i in range(len(self.state.layers_size)-1)]
        if action == [0]:
            edge_tail = len(self.state.layers_size)-1
            for i in range(len(self.state.layers_size)-1):
                latency[i] += state.output_size[-1]/state.gain_to_transmission_power(state.gain) # x input send.
                latency[i] += state.layers_size[-1]/state.edge_comp #calculate head
                latency[i] +=(state.layers_size[i]/state.edge_comp)*edge_tail

        else:
            for i in range(len(action)):
                latency[i] += state.layers_size[-1]/state.md_comp #calculate head
            edge_tail = 0
            md_tail = 0
            for i in range(len(action)):
                if action[i] == 0:
                    edge_tail+=1
                elif action[i] == 1:
                    md_tail +=1
            for i in range(len(action)):
                if action[i] == 0:
                    latency_edge = state.output_size[0] / state.gain_to_transmission_power(state.gain)
                    latency_edge += (state.layers_size[i] / state.edge_comp)*edge_tail
                    latency[i] += latency_edge
                elif action[i] == 1:
                    latency_md = (state.layers_size[i]/state.md_comp)*md_tail
                    latency[i] += latency_md
        avg_latency = np.mean(latency)

        return avg_latency

def action_index_to_list(action_index):
    action_dim = 2**(len(layers_size)-1)+1 
    if action_index == action_dim-1:
        action_list = [0]
        return action_list # Edge_only
    else:
        action_list = [0 for i in range(len(layers_size) - 1)]
        index = 0
        while action_index != 0:
            action_list[index] = action_index % 2
            action_index = action_index // 2
            index += 1
        if action_index == action_dim-2:
            print(action_list)
        return action_list

def compute_avg_return(environment, policy, num_step, num_episodes=10, return_action=False):

    total_return = []
    total_ec = []
    total_tc = []
    action_list = []
    ec_list = []
    tc_list = []

    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return =[]

        for j in range(num_step):
            action_step = policy.action(time_step)
            time_step= environment.step(action_step.action)
            if return_action == True:
                action_list.append(action_index_to_list(np.argmax(action_step.action)))
            ec_list.append(time_step.observation.numpy()[0][0])
            tc_list.append(time_step.observation.numpy()[0][1])
            # print(action_list)
            episode_return.append(time_step.reward)
        total_ec.append(ec_list)
        total_tc.append(tc_list)
        total_return.append(episode_return)

    # avg_return = total_return / num_episodes
    if return_action == False:
        return total_return, total_ec, total_tc
    else:
        return total_return, total_ec, total_tc, action_list, tc_list
def compute_avg_return_for_static(environment, policy, num_step, num_episodes=10):

    total_return = []
    total_ec = []
    total_tc = []
    ec_list = []
    tc_list = []
    for _ in range(num_episodes):
        episode_return =[]
        for j in range(num_step):
            time_step= environment.step(policy)
            episode_return.append(time_step.reward)
            ec_list.append(time_step.observation.numpy()[0][0])
            tc_list.append(time_step.observation.numpy()[0][1])
        total_ec.append(ec_list)
        total_tc.append(tc_list)
        total_return.append(episode_return)


    return total_return,total_ec, total_tc


def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

