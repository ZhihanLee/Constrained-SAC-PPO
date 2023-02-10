import os
import sys
import time
from copy import deepcopy

import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import numpy.random as rd
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

is_training = True

"""net.py"""

class ActorPPO(nn.Module):
    def __init__(self, mid_dim, hidden_dim, state_dim, action_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_hidden = None

        self.net_mlp1_1 = nn.Linear(state_dim, mid_dim)
        self.net_mlp1_2 = nn.Linear(mid_dim, hidden_dim)
        self.net_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.net_mlp2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.net_mlp2_2 = nn.Linear(hidden_dim, action_dim)
        
        # layer_norm(self.net_mlp2_2[-1], std=0.1)  # output layer for action
        layer_norm(self.net_mlp2_2, std=0.1)  # output layer for action

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        
        orthogonal_init(self.net_lstm)
        orthogonal_init(self.net_mlp1_1)
        orthogonal_init(self.net_mlp1_2)
        orthogonal_init(self.net_mlp2_1)
        orthogonal_init(self.net_mlp2_2)

    def forward(self, state):
        self.net_lstm.flatten_parameters()   
        s = F.relu(self.net_mlp1_1(state))
        s = F.relu(self.net_mlp1_2(s))
        lstm_out, self.lstm_hidden = self.net_lstm(s)
        output = F.relu(self.net_mlp2_1(lstm_out))
        output = F.relu(self.net_mlp2_2(lstm_out))
        
        return output.tanh()  # action.tanh()

    def get_action(self, state):
        state = state.unsqueeze(0)
        a_avg = self.forward(state)
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)

        action = a_avg[0] + noise * a_std
        return action.squeeze(0), noise.squeeze(0)

    def get_logprob_entropy(self, state, action):
   
        # a_avg = torch.stack([self.forward(state[i,:,:].unsqueeze(0)) for i in range(0, state.size(0))], dim=0)
        a_avg = self.forward(state)
        a_avg = torch.squeeze(a_avg) # shape = [32,128,3] stands for [batch_size, sequence_length, action_dim]
        a_std = self.a_logstd.exp()
        
        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(2)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy

        return logprob, dist_entropy # should be [batch_size, sequence_length]

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(2)  # old_logprob

class CriticAdv(nn.Module):
    def __init__(self, mid_dim, hidden_dim, state_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_hidden = None
        self.net_mlp1 = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, hidden_dim), nn.ReLU(), )
        self.net_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.net_mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Hardswish(),
                                      nn.Linear(hidden_dim, 1), )
        layer_norm(self.net_mlp2[-1], std=0.5)  # output layer for Q value
        
        orthogonal_init(self.net_lstm)
        orthogonal_init(self.net_mlp1)
        orthogonal_init(self.net_mlp2)

    def forward(self, state):
        self.net_lstm.flatten_parameters() 
        s = self.net_mlp1(state)
        lstm_out, self.lstm_hidden = self.net_lstm(s)
        output = self.net_mlp2(lstm_out)
        
        return output.tanh()  # Q value


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    
def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


"""agent.py"""


class AgentPPO_L:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.08  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None

        self.state = None
        self.device = None
        self.criterion = None
        self.act = self.act_optimizer = None
        self.cri = self.cri_optimizer = self.cri_target = None
        self.safety_cri = self.safety_cri_target = None
        
        self.sequence_length = None # for RNN

    def init(self, net_dim, hidden_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        
        # Initialize Lagrangian Multiplier
        self.constraint_dim = 2 # althrough it is two, but we only have one lambda
        self.Lambda = torch.tensor(
            np.random.rand(1),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        ) # shape = [1, constrain_dim, 1]
        self.UPDATE_INTERVAL = 2**2
        self.lambda_interval = 0 # counting variable
        constraint_learning_rate = 1e-4

        # constraint hyper-parameter
        d1 = 0.05 # constraint cost 1 Jc1 = Expectation(sigma t :gamma^t*ct) < d1
        d2 = 0.05 # constraint cost 2
        self.d = torch.tensor([d1, d2])
        self.lambda_optim = torch.optim.Adam((self.Lambda,), lr=constraint_learning_rate)
        
        # Initialize Neural Network
        self.act = ActorPPO(net_dim, hidden_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, hidden_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri) if self.cri_target is True else self.cri
        self.safety_cri = CriticAdv(net_dim, hidden_dim, state_dim).to(self.device)
        self.safety_cri_target = deepcopy(self.safety_cri) if self.safety_cri_target is True else self.safety_cri
        print("actor network: ", self.act)
        print("critc network: ", self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.safety_cri_optimizer = torch.optim.Adam(self.safety_cri.parameters(), lr=learning_rate) 

    def select_action(self, state):
        state = np.array(state)
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)  # plan to be get_action_a_noise
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        ##### edit this according to your own enviroment #####
#         trajectory_list = list()
#         episode_all = 0
#         episode_all_spd = 0
#         episode_all_fuel = 0
#         episode_all_suc = 0
#         episode_all_soc = 0
#         episode_all_ill = 0
#         episode_all_pwt = 0
#         episode_all_safe = 0
#         episode_all_action = 0
#         cost_engine = 0
#         state = env.reset()
#         for _ in range(target_step):
#             action, noise = self.select_action(state)
#             next_state, reward, done, info = env.step(np.tanh(action))
#             env.out_info(info)
#             other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise, info['con_cost'][0], info['con_cost'][1]) # need perfection
#             trajectory_list.append((state, other))
#             state = next_state

#             episode_all += info['r']
#             episode_all_spd += info['r_moving']
#             episode_all_fuel += info['r_fuel']
#             episode_all_suc += info['r_suc']
#             episode_all_soc += info['r_soc']
#             episode_all_ill += info['r_ill']
#             episode_all_pwt += info['r_pwt']
#             episode_all_safe += info['r_safe']
#             episode_all_action += info['r_action']
#             cost_engine += info['fuel_cost_L']

#         final_state_list = np.array([item[0] for item in trajectory_list], dtype=np.float32)
#         final_state_list = final_state_list[::-1] # reverse
#         self.state = final_state_list[:self.sequence_length,:]

#         mean_reward = episode_all/target_step
#         env.mean_reward_list.append(mean_reward)
#         mean_reward_spd = episode_all_spd/target_step
#         env.mean_spd_list.append(mean_reward_spd)
#         mean_reward_fuel = episode_all_fuel/target_step
#         env.mean_fuel_list.append(mean_reward_fuel)
#         mean_reward_suc = episode_all_suc/target_step
#         env.mean_suc_list.append(mean_reward_suc)
#         mean_reward_soc = episode_all_soc/target_step
#         env.mean_soc_list.append(mean_reward_soc)
#         mean_reward_ill = episode_all_ill/target_step
#         env.mean_ill_list.append(mean_reward_ill)
#         mean_reward_pwt = episode_all_pwt/target_step
#         env.mean_pwt_list.append(mean_reward_pwt)
#         mean_reward_safe = episode_all_safe/target_step
#         env.mean_safe_list.append(mean_reward_safe)
#         mean_reward_action = episode_all_action/target_step
#         env.mean_action_list.append(mean_reward_action)
#         print("FuelCost per 100km is :",cost_engine * 100/(env.travellength/1000))

#         return trajectory_list
        pass

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau, eps_step):
        buffer.update_now_len()
        # buf_len = buffer.now_len
        buf_state, buf_action, buf_r_sum, buf_r_c_sum, buf_logprob, buf_advantage, buf_advantage_c, cost = self.prepare_buffer(buffer, eps_step, batch_size) # buf_r_c_sum and buf_advantage_c is a list for multi-constraints
        buffer.empty_buffer()

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = logprob = obj_critic_c = None
        for _ in range(repeat_times):         
            # dont need indices, because the random sample has been deployed in sample_seg() function
            state = buf_state # shape = [batch_size, sequence_length, state_dim]
            action = buf_action
            r_sum = buf_r_sum
            r_c_sum = buf_r_c_sum
            logprob = buf_logprob
            advantage = buf_advantage
            advantage_c = buf_advantage_c

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip) # [batch_size, sequence_length]
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean() 
            obj_surrogate_c = self.Lambda * (advantage_c * ratio).mean()

            # rewrite the actor loss as the original loss minus each cost advantage
            obj_actor_lag = (obj_surrogate - obj_surrogate_c) / (1 + self.Lambda)
            obj_actor = obj_actor_lag + obj_entropy * self.lambda_entropy
            
            self.optim_update(self.act_optimizer, obj_actor)

            # update critic network
            value = torch.stack([self.cri(state[i,:,:].unsqueeze(0)) for i in range(0, state.size(0))], dim=0)
            value = torch.squeeze(value)
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            self.optim_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None
            
            # update safety critic network
            value_c = torch.stack([self.safety_cri(state[i,:,:].unsqueeze(0)) for i in range(0, state.size(0))], dim=0)
            value_c = torch.squeeze(value_c)
            obj_critic_c = self.criterion(value_c, r_c_sum) / (r_c_sum.std() + 1e-6)

            self.optim_update(self.safety_cri_optimizer, obj_critic_c)
            self.soft_update(self.safety_cri_target, self.safety_cri, soft_update_tau) if self.safety_cri_target is not self.safety_cri else None
        
        '''Lagrangian multiplier update by Primal Dual Optimization'''
        self.lambda_interval += 1
        
        if self.lambda_interval == self.UPDATE_INTERVAL :
            # delayed update
            expectation_buffer = []
            
            for i in range(self.constraint_dim):
                J_c_i = (self.d[i] - cost[:,i]).mean()
                expectation_buffer.append(J_c_i)
                
            expectation_buffer = torch.as_tensor(expectation_buffer, dtype=torch.float32, device=self.device)
            expectation_buffer = expectation_buffer.reshape(self.constraint_dim,1)
            # obj_lambda = torch.matmul(self.Lambda, expectation_buffer) # for vector Lambda
            obj_lambda = self.Lambda * expectation_buffer[0] + self.Lambda * expectation_buffer[1]
            self.optim_update(self.lambda_optim, obj_lambda)
            
            # projection onto the dual space, making lambda >= 0
            with torch.no_grad():
                self.Lambda[:] = self.Lambda.clamp(0.,16.).detach()
                
            self.lambda_interval = 0

        return obj_critic.item(), obj_actor.item(), logprob.mean().item()  # logging_tuple

    def prepare_buffer(self, buffer, eps_step, batch_size):
        # buf_len = buffer.now_len
        buf_len = batch_size

        with torch.no_grad():  # compute reverse reward
            reward, mask, action, a_noise, cost, state = buffer.sample_batch_seg(eps_step, batch_size, buf_len, self.sequence_length) # cost is [buf_len, constraint_dim]
            cost_sum = cost.sum(2) # shape = 32,128 (before sum is 32,128,2
            # state shape = [32 128 22]
            
            logprob = self.act.get_old_logprob(action, a_noise)
            
            # state value and advantage
            value = torch.stack([self.cri_target(state[i,:,:].unsqueeze(0)) for i in range(0, state.size(0))],dim=0)
            value = torch.squeeze(value) # [32, 128]
            
            pre_state = state[:, -1, :]
            pre_state = pre_state.unsqueeze(0)
            pre_r_sum = self.cri(pre_state).detach() 
            pre_r_sum = torch.squeeze(pre_r_sum) # [32] which is equal to batch_size
            r_sum, advantage = self.get_reward_sum(self, buf_len, reward, mask, value, pre_r_sum) 
            
            # safety cost value and advantage
            value_c = torch.stack([self.safety_cri_target(state[i,:,:].unsqueeze(0)) for i in range(0, state.size(0))], dim=0)
            value_c = torch.squeeze(value)
            
            pre_r_c_sum = self.safety_cri(pre_state).detach()
            pre_r_c_sum = torch.squeeze(pre_r_c_sum)
            r_c_sum, advantage_c = self.get_reward_sum(self, buf_len, cost_sum, mask, value_c, pre_r_c_sum)

            # r_sum.shape = [batch_size, sequence_length], adv = [batch_size]
            
        return state, action, r_sum, r_c_sum, logprob, advantage, advantage_c, cost

    @staticmethod
    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def get_reward_sum_gae(self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, buf_reward.size(1), dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, buf_reward.size(1), dtype=torch.float32, device=self.device)  # advantage value
        
        # calculating adv, for a 2dim matrix: calculate each batch(32) with a sequence adv with length of 128
        for i in range(buf_len - 1, -1, -1):
            for j in range(buf_reward.size(1)-1, -1, -1):
                pre_advantage = 0  # advantage value of previous step
                buf_r_sum[i][j] = buf_reward[i][j] + buf_mask[i][j] * pre_r_sum[i]
                pre_r_sum[i] = buf_r_sum[i][j]

                buf_advantage[i][j] = buf_reward[i][j] + buf_mask[i][j] * (pre_advantage - buf_value[i][j])  # fix a bug here
                pre_advantage = buf_value[i][j] + buf_advantage[i][j] * self.lambda_gae_adv
            buf_advantage[i] = (buf_advantage[i] - buf_advantage[i].mean()) / (buf_advantage[i].std() + 1e-5) # advantage norm
        return buf_r_sum, buf_advantage # [batch_size, sequence_length], [batch_size]

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        # objective.backward()
        # objective.requires_grad_(True)
        torch.autograd.set_detect_anomaly(True)
        objective.backward(retain_graph=True)
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))


"""replay.py"""


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, constraint_dim, if_discrete):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = 1 if if_discrete else action_dim  # for self.sample_all(
        self.state_dim = state_dim
        self.tuple = None
        self.np_torch = torch

        other_dim = 1 + 1 + self.action_dim + action_dim + constraint_dim # for PPO-L, the buffer stores r, mask, action, noise, cost
        # other = (reward, mask, action, a_noise) for continuous action
        # other = (reward, mask, a_int, a_prob) for discrete action
        max_len = int(max_len)
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        state_ary = np.array([item[0] for item in trajectory_list], dtype=np.float32)
        other_ary = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        self.extend_buffer(state_ary, other_ary)

    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_all(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2+self.action_dim:2+2*self.action_dim],  # action_noise or action_prob
                all_other[:, 2+2*self.action_dim:], # constraint cost
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state
    
    def sample_batch_seg(self, eps_step, batch_size, buf_len, sequence_length):
        full_seg = int(buf_len/eps_step) # equal to rollout times
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        all_state = torch.as_tensor(self.buf_state[:self.now_len], device=self.device)
        # init empty data tensor
        r_tensor = torch.ones([batch_size, sequence_length], device=self.device)
        mask_tensor = torch.ones([batch_size, sequence_length], device=self.device)
        a_tensor = torch.ones([batch_size, sequence_length, self.action_dim], device=self.device)
        a_n_tensor = torch.ones([batch_size, sequence_length, self.action_dim], device=self.device)
        cost_tensor = torch.ones([batch_size, sequence_length, 2], device=self.device) # need perfect
        s_tensor = torch.ones([batch_size, sequence_length, self.state_dim], device=self.device)
        # adding elements
        for i in range(full_seg):
            for j in range(int(batch_size/full_seg)):
                indices = rd.randint(i*eps_step, (i+1)*eps_step - sequence_length)
                r_tensor[j,:] = all_other[indices:indices+sequence_length, 0]
                mask_tensor[j,:] = all_other[indices:indices+sequence_length, 1]
                a_tensor[j,:,:] = all_other[indices:indices+sequence_length, 2:2+self.action_dim]
                a_n_tensor[j,:,:] = all_other[indices:indices+sequence_length, 2+self.action_dim:2+2*self.action_dim]
                cost_tensor[j,:,:] = all_other[indices:indices+sequence_length, 2+2*self.action_dim:]
                s_tensor[j,:,:] = all_state[indices:indices+sequence_length,:]
        return (r_tensor, mask_tensor, a_tensor, a_n_tensor, cost_tensor, s_tensor)  # state
                
        

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


'''env.py'''


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super(PreprocessEnv, self).__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

        self.reset = self.reset_type
        self.step = self.step_type

    def reset_type(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step_type(self, action) -> (np.ndarray, float, bool, dict):
        state, reward, done, info = self.env.step(action * self.action_max)
        return state.astype(np.float32), reward, done, info


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    """get information of a standard OpenAI gym env.

    The DRL algorithm AgentXXX need these env information for building networks and training.
    env_name: the environment name, such as XxxXxx-v0
    state_dim: the dimension of state
    action_dim: the dimension of continuous action; Or the number of discrete action
    action_max: the max action of continuous action; action_max == 1 when it is discrete action space
    if_discrete: Is this env a discrete action space?
    target_return: the target episode return, if agent reach this score, then it pass this game (env).
    max_step: the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step

    :env: a standard OpenAI gym environment, it has env.reset() and env.step()
    :bool if_print: print the information of environment. Such as env_name, state_dim ...
    """
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id

    # state_shape = env.observation_space.shape
    # state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list
    state_dim = env.s_dim

    target_return = getattr(env, 'target_return', None)
    target_return_default = getattr(env.spec, 'reward_threshold', None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2 ** 16

    # max_step = getattr(env, 'max_step', None)
    # max_step_default = getattr(env, '_max_episode_steps', None)
    # if max_step is None:
    #     max_step = max_step_default
    # if max_step is None:
    #     max_step = 2 ** 10
    max_step = env.travellength / env.STEP_SIZE

    # if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if_discrete = 0
    action_dim = env.a_dim
    action_max = 1.0
    # if if_discrete:  # make sure it is discrete action space
    #     action_dim = env.action_space.n
    #     action_max = int(1)
    # elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
    #     action_dim = env.action_space.shape[0]
    #     action_max = float(env.action_space.high[0])
    #     assert not any(env.action_space.high + env.action_space.low)
    # else:
    #     raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action space if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return


def deepcopy_or_rebuild_env(env):
    try:
        env_eval = deepcopy(env)
    except Exception as error:
        print('| deepcopy_or_rebuild_env, error:', error)
        env_eval = PreprocessEnv(env.env_name, if_print=False)
    return env_eval


'''run.py'''


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training (off-policy)'''
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 8  # the network width
            self.hidden_dim = 2**6
            self.batch_size = 2**6  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            # self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.target_step = 0 # 训练中一个episode里面的步数
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = True  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim # num of transitions sampled from replay buffer.
            # self.target_step = 2 ** 10  # collect target_step, then update network
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 5  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        self.break_step = 2 ** 27  # break training after 'total_step > break_step'
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

    def init_before_training(self, process_id=0):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set None value automatically'''
        if self.gpu_id is None:  # set gpu_id as '0' in default
            self.gpu_id = '0'

        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{self.env.env_name}_{agent_name}'

        if process_id == 0:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        gpu_id = self.gpu_id[process_id] if isinstance(self.gpu_id, list) else self.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id
    env_eval = args.env_eval

    '''training arguments'''
    net_dim = args.net_dim
    hidden_dim = args.hidden_dim
    # max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    learning_rate = args.learning_rate
    if_per_or_gae = args.if_per_or_gae
    if_break_early = args.if_allow_break

    gamma = args.gamma
    reward_scale = args.reward_scale
    soft_update_tau = args.soft_update_tau

    '''evaluating arguments'''
    show_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.s_dim
    action_dim = env.a_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    rollout_times = 4 # using current policy, rollout n times
    agent.init(net_dim, hidden_dim, state_dim, action_dim, learning_rate, if_per_or_gae)
    agent.sequence_length = 64

    buffer_len = (target_step + max_step)*rollout_times
    buffer = ReplayBuffer(max_len=buffer_len, state_dim=state_dim, action_dim=action_dim, constraint_dim=agent.constraint_dim,
                          if_discrete=if_discrete)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    # agent.state = env.reset() # dont need it, it will be reseted in explore_env()
    total_step = 0

    '''start training'''
    if_train = True
    episode_num = 0
    glosa_list = []
    lambda_list = []
    while if_train:
        for _ in range(rollout_times) : # can be multiple thread in the future
            with torch.no_grad():
                trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
            steps = len(trajectory_list)
            total_step += steps

            buffer.extend_buffer_from_list(trajectory_list)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau, target_step)
        lambda_list.append(np.array(agent.Lambda.detach().cpu()))
        
        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple, episode_num)
            if_train = not ((if_break_early and if_reach_goal)
                            or total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
        
        episode_num = episode_num + 1
        if episode_num%10 == 0 :
            env.write_info(episode_num, is_training = True)

    env.write_mean_reward()
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    print("Finish Time:", time.strftime("%Y-%m-%d %X",time.localtime()))


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # a early time
        print(f"{'ID':>2} {'Step':>8} {'MaxR':>8} |"
              f"{'avgR':>8} {'stdR':>8} |{'avgS':>5} {'stdS':>4} |"
              f"{'objC':>8} {'etc.':>8}")

    def evaluate_save(self, act, steps, log_tuple, episode_num) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()
            if_reach_goal = False
            episode_return, episode_step = get_episode_return_and_step(self.env, act, self.device)
            if episode_step == self.env.max_step :
                rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                                        range(self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

                if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                    rewards_steps_list += [get_episode_return_and_step(self.env, act, self.device)
                                            for _ in range(self.eval_times2 - self.eval_times1)]
                    r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
                if r_avg > self.r_max:  # save checkpoint with highest episode return
                    self.r_max = r_avg  # update max reward (episode return)

                    '''save policy network in *.pth'''
                    act_save_path = f'{self.cwd}/actor{episode_num}.pth'
                    torch.save(act.state_dict(), act_save_path)
                    print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |")  # save policy and print

                self.recorder.append((self.total_step, r_avg, r_std, *log_tuple))  # update recorder

                if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
                if if_reach_goal and self.used_time is None:
                    self.used_time = int(time.time() - self.start_time)
                    print(f"{'ID':>2} {'Step':>8} {'TargetR':>8} |{'avgR':>8} {'stdR':>8} |"
                            f"  {'UsedTime':>8}  ########\n"
                            f"{self.agent_id:<2} {self.total_step:8.2e} {self.target_return:8.2f} |"
                            f"{r_avg:8.2f} {r_std:8.2f} |"
                            f"  {self.used_time:>8}  ########")

                # plan to
                # if time.time() - self.print_time > self.show_gap:
                # print('another debug')
                print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |"
                        f"{r_avg:8.2f} {r_std:8.2f} |{s_avg:5.0f} {s_std:4.0f} |"
                        f"{' '.join(f'{n:8.2f}' for n in log_tuple)}")
        else:
            if_reach_goal = False

        return if_reach_goal

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = int(env.max_step)
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, info = env.step(action)
        episode_return += reward
        if done:
            break
    GLOSA = env.GLOSA
    episode_step = GLOSA + episode_step
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


'''DEMO'''


def demo_continuous_action():
    # # activate env
    env_name = 'my_env_sac-v0'
    env = PreprocessEnv(env=gym.make(env_name))

    # # init agent
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.env = env
    args.agent = AgentPPO_L()
    args.agent.cri_target = True  # True
    args.target_step = int(env.travellength/env.STEP_SIZE)

    '''train and evaluate'''
    if is_training:
        train_and_evaluate(args) # there is a while loop in it
    else:
        pass

if __name__ == '__main__':
    demo_continuous_action()
