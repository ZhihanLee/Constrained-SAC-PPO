from copy import deepcopy
import sys
sys.path.append('D:\\RL\\ElegantRL-master\\SAC_Prius')
import numpy as np
import numpy.random as rd
import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorSAC, CriticTwin, ShareSPG, CriticMultiple


class AgentSAC(AgentBase):  # [ElegantRL.2021.11.11]
    """
    Bases: ``AgentBase``

    Soft Actor-Critic algorithm. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”. Tuomas Haarnoja et al.. 2018.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=True,
        env_num=1,
        gpu_id=0,
    ):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing.
        """
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )

        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)


        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select actions given an array of states.

        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        alpha = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.optim_update(self.cri_ogptim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            """objective of alpha (temperature parameter automatic adjustment)"""
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            """objective of actor"""
            obj_actor = -(self.cri(state, action_pg) + logprob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.0
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )

            next_a, next_log_prob = self.act.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.0
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentModSAC(AgentSAC):  # [ElegantRL.2021.11.11]
    """
    Bases: ``AgentSAC``

    Modified SAC with introducing of reliable_lambda, to realize “Delayed” Policy Updates.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self):
        AgentSAC.__init__(self)
        self.__name__ = 'SAC'
        self.ClassCri = CriticMultiple  # REDQ ensemble (parameter sharing)
        # self.ClassCri = CriticEnsemble  # REDQ ensemble  # todo ensemble
        self.state = None
        self.device = None
        self.if_use_cri_target = True
        self.if_use_act_target = True

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.005 * obj_critic.item()
            )  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            """objective of alpha (temperature parameter automatic adjustment)"""
            obj_alpha = (
                self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.optim_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha).mean()  # todo ensemble
                self.optim_update(self.act_optim, obj_actor)
                if self.if_use_act_target:
                    self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()
    
    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True
            )[
                0
            ]  # multiple critics

            # todo ensemble
            q_label = reward + mask * (next_q + next_log_prob * alpha)
            q_labels = q_label * torch.ones(
                (1, self.cri.q_values_num), dtype=torch.float32, device=self.device
            )
        q_values = self.cri.get_q_values(state, action)  # todo ensemble

        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        print("===========================")
        print("Using PER")
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True
            )[
                0
            ]  # multiple critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
            q_labels = q_label * torch.ones(
                (1, self.cri.q_values_num), dtype=torch.float32, device=self.device
            )
        q_values = self.cri.get_q_values(state, action)

        print("reward shape = ", reward.shape)
        print("next_q.shape = ", next_q.shape)
        print("next_log_prob.shape = ", next_log_prob.shape)
        print("q_labels = ", q_labels)
        print("q_labels shape = ", q_labels.shape)

        # obj_critic = self.criterion(q_values, q_labels)
        td_error = self.criterion(q_values, q_labels).mean(dim=1, keepdim=True)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

class AgentConstrainedSAC(AgentModSAC):
    '''
    the SAC-Lagrangian method without Lagrangian multiplier updating by NN
    '''
    def __init__(self):
        AgentSAC.__init__(self)
        self.__name__ = 'SAC'
        self.ClassCri = CriticMultiple  # REDQ ensemble (parameter sharing)
        # self.ClassCri = CriticEnsemble  # REDQ ensemble  # todo ensemble
        self.state = None
        self.device = None
        self.if_use_cri_target = True
        self.if_use_act_target = True

        # Initialize Lagrangian Multiplier
        self.constraint_dim = 2
        self.Lambda = torch.tensor(
            np.random.rand(self.constraint_dim),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        ) # shape = [1, constrain_dim, 1]
        constraint_learning_rate = 1e-4

        # constraint hyper-parameter
        d1 = 0.05 # constraint cost 1 Jc1 = Expectation(sigma t :gamma^t*ct) < d1
        d2 = 0.05 # constraint cost 2
        self.d = torch.tensor([d1, d2])
        self.lambda_optim = torch.optim.Adam((self.Lambda,), lr=constraint_learning_rate)

    def explore_one_env(self, env, target_step: int) -> list:
        """
        ################ A constrained explore_env method in AgentConstrainedSAC #################

        actor explores in single Env, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step()
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        state = env.reset()
        traj = []
        con_cost_list = []
        trajectory_list = list()
        episode_all = 0
        episode_all_spd = 0
        episode_all_fuel = 0
        episode_all_suc = 0
        episode_all_soc = 0
        episode_all_ill = 0
        episode_all_pwt = 0
        episode_all_safe = 0
        episode_all_action = 0
        cost_engine = 0

        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()
            next_s, reward, done, info = env.step(action)
            con_cost_list.append(info['con_cost'])
            env.out_info(info)

            ten_other = torch.empty(2 + self.action_dim)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2:] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s

            # # 绘图用数据
            episode_all += info['r']
            episode_all_spd += info['r_moving']
            episode_all_fuel += info['r_fuel']
            episode_all_suc += info['r_suc']
            episode_all_soc += info['r_soc']
            episode_all_ill += info['r_ill']
            episode_all_pwt += info['r_pwt']
            episode_all_safe += info['r_safe']
            episode_all_action += info['r_action']
            cost_engine += info['fuel_cost_L']

        # self.states[0] = state

        mean_reward = episode_all/target_step
        env.mean_reward_list.append(mean_reward)
        mean_reward_spd = episode_all_spd/target_step
        env.mean_spd_list.append(mean_reward_spd)
        mean_reward_fuel = episode_all_fuel/target_step
        env.mean_fuel_list.append(mean_reward_fuel)
        mean_reward_suc = episode_all_suc/target_step
        env.mean_suc_list.append(mean_reward_suc)
        mean_reward_soc = episode_all_soc/target_step
        env.mean_soc_list.append(mean_reward_soc)
        mean_reward_ill = episode_all_ill/target_step
        env.mean_ill_list.append(mean_reward_ill)
        mean_reward_pwt = episode_all_pwt/target_step
        env.mean_pwt_list.append(mean_reward_pwt)
        mean_reward_safe = episode_all_safe/target_step
        env.mean_safe_list.append(mean_reward_safe)
        mean_reward_action = episode_all_action/target_step
        env.mean_action_list.append(mean_reward_action)
        print("FuelCost per 100km is :",cost_engine * 100/(env.travellength/1000))

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [
            (traj_state, traj_other),
        ]     

        con_cost_list = np.array(con_cost_list)

        return self.convert_trajectory(traj_list), con_cost_list  # [traj_env_0, ]

    def update_net(self, buffer, expectation_buffer, rollout_buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.

        Update the lagrangian multiplier Lambda by on-policy data from expectation_buffer and rollout_buffer 

        """
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()
            Lambda = self.Lambda

            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, rollout_buffer, batch_size, alpha, Lambda)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.005 * obj_critic.item()
            )  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient

            """objective of alpha (temperature parameter automatic adjustment)"""
            obj_alpha = (
                self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.optim_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            """objective of lambda"""
            expectation_buffer = torch.as_tensor(expectation_buffer, dtype=torch.float32)
            expectation_buffer = expectation_buffer.reshape(self.constraint_dim,1)
            obj_lambda = torch.matmul(self.Lambda, expectation_buffer)  # dont need mean(), it has been taken in run.py 
            self.optim_update(self.lambda_optim, obj_lambda)

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha)  # todo ensemble
                # Adding lagrangian part
                for i in range(self.constraint_dim):
                    cost_list = torch.as_tensor(rollout_buffer[i]).reshape(len(rollout_buffer[i]),1)
                    Lag = Lambda[i] * cost_list
                    obj_actor -= Lag 
                obj_actor = obj_actor.mean()
                self.optim_update(self.act_optim, obj_actor)
                if self.if_use_act_target:
                    self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()
    
    def get_obj_critic_raw(self, buffer, rollout_buffer, batch_size, alpha, Lambda):
        """
        Modified Q value computation with lagrangian multiplier

        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True
            )[
                0
            ]  # multiple critics

            # todo ensemble
            q_label = reward + mask * (next_q + next_log_prob * alpha)

            # Adding lagrangian part
            for i in range(self.constraint_dim):
                cost_list = torch.tensor(rollout_buffer[i])
                Lag = Lambda[i] * cost_list
                q_label += Lag 

            q_labels = q_label * torch.ones(
                (1, self.cri.q_values_num), dtype=torch.float32, device=self.device
            )
        q_values = self.cri.get_q_values(state, action)  # todo ensemble

        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, rollout_buffer, batch_size, alpha, Lambda):
        """
        Modified Q value computation with lagrangian multiplier

        rollout_buffer is a [[con_cost1 list], [con_cost2 list]], using constraint_dim to visit different cost list

        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True
            )[
                0
            ]  # multiple critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)

            # Adding lagrangian part
            for i in range(self.constraint_dim):
                cost_list = torch.as_tensor(rollout_buffer[i]).reshape(len(rollout_buffer[i]),1)
                Lag = Lambda[i] * cost_list
                q_label += Lag 

            q_labels = q_label * torch.ones(
                (1, self.cri.q_values_num), dtype=torch.float32, device=self.device
            )
        q_values = self.cri.get_q_values(state, action)

        # obj_critic = self.criterion(q_values, q_labels)
        td_error = self.criterion(q_values, q_labels).mean(dim=1, keepdim=True)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
    

class AgentShareSAC(AgentSAC):  # Integrated Soft Actor-Critic
    def __init__(self):
        AgentSAC.__init__(self)
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.cri_optim = None

        self.target_entropy = None
        self.alpha_log = None

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing.
        """
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.target_entropy = np.log(action_dim)
        self.act = self.cri = ShareSPG(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.cri_optim = torch.optim.Adam(
            [
                {"params": self.act.enc_s.parameters(), "lr": learning_rate * 1.5},
                {
                    "params": self.act.enc_a.parameters(),
                },
                {"params": self.act.net.parameters(), "lr": learning_rate * 1.5},
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.dec_d.parameters(),
                },
                {
                    "params": self.act.dec_q1.parameters(),
                },
                {
                    "params": self.act.dec_q2.parameters(),
                },
                {"params": (self.alpha_log,)},
            ],
            lr=learning_rate,
        )

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> tuple:  # 1111
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            alpha = self.alpha_log.exp()

            """objective of critic"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.0025 * obj_critic.item()
            )  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda

            """objective of alpha (temperature parameter automatic adjustment)"""
            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log
                * (logprob - self.target_entropy).detach()
                * reliable_lambda
            ).mean()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(
                    *self.act_target.get_q1_q2(state, a_noise_pg)
                ).mean()  # twin critics
                obj_actor = -(
                    q_value_pg + logprob * alpha.detach()
                ).mean()  # policy gradient

                obj_united = obj_critic + obj_alpha + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic + obj_alpha

            self.optim_update(self.cri_optim, obj_united)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()
