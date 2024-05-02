'''
This code is mainly based off https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
'''

import numpy as np
import torch as T
import torch.nn as nn
from dueling_deep_q_network import DuelingDeepQNetwork
from replay_memory import ReplayBuffer

class D3QNAgent(object):
    def __init__(self, n_actions, input_dims, eps_start, eps_min, eps_dec, save_load_dir, fc1_out=1024, fc2_out=512, num_rel_positions=2, gamma=0.99, lr=0.00025, batch_size=32, mem_size=50000, replace=1000):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.num_rel_positions = num_rel_positions
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.replace_target_cnt = replace
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.save_load_dir = save_load_dir
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, self.input_dims)

        self.q_eval = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions,
                        input_dims=self.input_dims,
                        fc1_out = self.fc1_out,
                        fc2_out = self.fc2_out,
                        num_rel_positions =self.num_rel_positions,
                        save_load_file=self.save_load_dir+'D3QN_eval.pth')

        self.q_next = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions,
                        input_dims=self.input_dims,
                        fc1_out = self.fc1_out,
                        fc2_out = self.fc2_out,
                        num_rel_positions =self.num_rel_positions,
                        save_load_file=self.save_load_dir+'D3QN_next.pth')

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = np.array([observation], copy=False, dtype=np.float32)
            state_tensor = T.tensor(state).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(state_tensor)

            action = T.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):        
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
