'''
This code is mainly based off https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
'''

import numpy as np
import torch as T
from dueling_deep_q_network import DuelingDeepQNetwork
from replay_memory import ReplayBuffer

class D3QNAgent(object):
	def __init__(self, lr, n_actions, input_dims, num_rel_positions, load_dir, fc1_out=1024, fc2_out=512):
		self.lr = lr
		self.n_actions = n_actions
		self.input_dims = input_dims
		self.num_rel_positions = num_rel_positions
		self.load_dir = load_dir
		self.fc1_out = fc1_out
		self.fc2_out = fc2_out

		self.q_eval = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions,
		                input_dims=self.input_dims,
		                fc1_out = self.fc1_out,
		                fc2_out = self.fc2_out,
		                num_rel_positions =self.num_rel_positions,
		                save_load_file=self.load_dir+'D3QN_eval.pth')

	def choose_action(self, observation):
		state = np.array([observation], copy=False, dtype=np.float32)
		state_tensor = T.tensor(state).to(self.q_eval.device)
		_, advantages = self.q_eval.forward(state_tensor)

		action = T.argmax(advantages).item()

		return action

	def load_models(self):
		print('.... loading models ....')
		self.q_eval.load_checkpoint()
		print('.... finished loading models ....')
