'''
This code uses code from https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
'''

import numpy as np
import torch as T
from deep_q_network_learner import DeepQNetworkLearner
from replay_memory import ReplayBuffer
from eval_d3qn_agent import D3QNAgent
import torch.nn.functional as F

class DQNAgentLearner(object):
    def __init__(self, n_actions, input_dims, eps_start, eps_min, eps_dec, save_load_dir, num_rel_positions=2, gamma=0.99, lr=0.00025, batch_size=32, mem_size=50000, replace=1000):
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

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(self.mem_size, self.input_dims)

        self.q_eval = DeepQNetworkLearner(lr=self.lr, n_actions=self.n_actions,
                        input_dims=self.input_dims,
                        num_rel_positions =self.num_rel_positions,
                        save_load_file=self.save_load_dir+'DQN_eval_learner.pth')

        self.q_next = DeepQNetworkLearner(lr=self.lr, n_actions=self.n_actions,
                        input_dims=self.input_dims,
                        num_rel_positions =self.num_rel_positions,
                        save_load_file=self.save_load_dir+'DQN_next_learner.pth')
                        
	######################################################################
        self.teacher_agent = D3QNAgent(lr=self.lr, n_actions=self.n_actions, input_dims=self.input_dims,
                                       num_rel_positions=self.num_rel_positions, load_dir = self.save_load_dir)
        self.teacher_agent.load_models()
        for param in self.teacher_agent.q_eval.parameters():
            param.requires_grad = False #ensure weights of D3QN model frozen
        
        self.temperature = 3
        #####################################################################

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

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

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        
        q_values = self.q_eval.forward(states)#New
        q_pred = q_values[indices, actions]#Changed
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

	###################################################################
        teacher_V_s, teacher_A_s = self.teacher_agent.q_eval.forward(states)
        teacher_q_values = T.add(teacher_V_s, (teacher_A_s - teacher_A_s.mean(dim=1, keepdim=True)))
        teacher_soft = F.softmax(teacher_q_values / self.temperature, dim=-1)

        # Compute distillation loss using a soft target and KL divergence
        distillation_loss = F.kl_div(F.log_softmax(q_values / self.temperature, dim=1), teacher_soft, reduction='batchmean')

        total_loss = loss + distillation_loss
        total_loss.backward()
        ###################################################################        

        #loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
