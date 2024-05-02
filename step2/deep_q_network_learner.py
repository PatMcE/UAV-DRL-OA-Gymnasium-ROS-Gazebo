'''
This code is mainly based off https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
'''

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetworkLearner(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_out=128, fc2_out=32, num_rel_positions=2, save_load_file='~/'):
        super(DeepQNetworkLearner, self).__init__()
        self.save_load_file = save_load_file
        self.image_input_dims = (input_dims[0] - 1,) + input_dims[1:]

        self.conv1 = nn.Conv2d(self.image_input_dims[0], 16, 8, stride=4)#32/64/128/16 failed
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(self.image_input_dims)

        self.fc1 = nn.Linear(fc_input_dims, fc1_out)
        self.fc2 = nn.Linear(fc1_out+num_rel_positions, fc2_out)
        self.fc3 = nn.Linear(fc2_out, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        image_state = state[:,0:-1,:,:]

        conv1 = F.relu(self.conv1(image_state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        
        #flat2 = F.relu(self.fc2(flat1))
        xyz_state = T.stack([state[:,-1,0,0], state[:,-1,0,1]], dim=-1).float()
        cat = T.cat((flat1, xyz_state), dim=1)
        flat2 = F.relu(self.fc2(cat))
        
        actions = self.fc3(flat2)

        return actions

    def save_checkpoint(self, save_load_file='~/'):
        print('... saving checkpoint ...')
        if save_load_file == '~/':
            T.save(self.state_dict(), self.save_load_file)
        else:
            T.save(self.state_dict(), save_load_file)

    def load_checkpoint(self, save_load_file='~/'):
        print('... loading checkpoint ...')
        if save_load_file == '~/':
            #self.load_state_dict(T.load(self.save_load_file))
            self.load_state_dict(T.load(self.save_load_file, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(save_load_file, map_location=T.device('cpu')))
