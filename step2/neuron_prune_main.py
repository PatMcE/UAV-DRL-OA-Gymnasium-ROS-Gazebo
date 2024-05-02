#! /usr/bin/env python3

from train_d3qn_agent import D3QNAgent
from utils import plot_learning_curve
import eval_drone_gym_gazebo_env #defines 'DroneGymGazeboEnv-v0' custom environment
import gymnasium
import numpy as np
import rospy
import argparse
import time
import torch

'''Neuron pruning and weight transfer for a D3QN model in ROS Gazebo navigation.'''
    
def compute_keep_neurons_index(model, layers, pruning_percentage):
    indices = {}
    for layer_name in layers:
        l1_norms = torch.norm(getattr(model, layer_name).weight.data, p=1, dim=1) #compute L1 norm of all neurons in the specified layer
        num_neurons_to_keep = int(l1_norms.shape[0] * (1 - pruning_percentage)) #determine the number of neurons to keep based on pruning_percentage
        _, keep_neurons_index = torch.topk(l1_norms, num_neurons_to_keep) #get indices of neurons with the highest L1 norms
        indices[layer_name] = keep_neurons_index.cpu().numpy().tolist() #make sure this is a list
        print(f"{layer_name}: keeping {num_neurons_to_keep} out of {l1_norms.shape[0]} neurons.")
    return indices

def transfer_weights(original_model, pruned_model, keep_neurons_indices):
    for name, module in pruned_model.named_modules():
        original_module = getattr(original_model, name, None)
        if original_module is not None:
            if hasattr(module, 'weight') and name in keep_neurons_indices:
                if name == 'fc1':
                    out_indices = keep_neurons_indices['fc1']
                    module.weight.data = original_module.weight.data[out_indices, :]
                    if module.bias is not None:
                        module.bias.data = original_module.bias.data[keep_neurons_indices['fc1']]
                    print('fc1 original module weight matrix shape = ' + str(original_module.weight.data.shape))
                    print('fc1 pruned module weight matrix shape = ' + str(module.weight.data.shape))
                elif name == 'fc2':   
                    in_indices = keep_neurons_indices['fc1'] + [len(keep_neurons_indices['fc1']), len(keep_neurons_indices['fc1']) + 1]
                    out_indices = keep_neurons_indices['fc2']
                    module.weight.data = original_module.weight.data[out_indices][:, in_indices]
                    if module.bias is not None:
                        module.bias.data = original_module.bias.data[out_indices]
                    print('fc2 original module weight matrix shape = ' + str(original_module.weight.data.shape))
                    print('fc2 pruned module weight matrix shape = ' + str(module.weight.data.shape))

            elif hasattr(module, 'weight'):
                if name in ['V', 'A']:
                    module.weight.data = original_module.weight.data[:, keep_neurons_indices['fc2']]
                    print('V/A original module weight matrix shape = ' + str(original_module.weight.data.shape))
                    print('V/A pruned module weight matrix shape = ' + str(module.weight.data.shape))
                    if module.bias is not None:
                        module.bias.data = original_module.bias.data
                else:#all other layers
                    module.weight.data = original_module.weight.data
                    module.bias.data = original_module.bias.data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='D3QN Model Neuron Pruning')
	parser.add_argument('-pruning_percentage', type=float, default=0.01, help='Percentage of neurons to prune')
	parser.add_argument('-acceptable_twelve_ep_avg_reward', type=int, default=40)
	parser.add_argument('-root_path', type=str, default='/home/pm/catkin_ws/src/mavros-px4-vehicle/', help='Root path for saving/loading models')
	args = parser.parse_args()

	# Initialize ROS node and gym environment
	rospy.init_node('drone_node', anonymous=True)
	env = gymnasium.make('DroneGymGazeboEnv-v0')

	# Setup for saving/loading
	save_load_dir = args.root_path + 'models/'

	# Create agent and load the original model
	original_agent = D3QNAgent(n_actions=env.action_space.n,
		               input_dims=(env.observation_space.shape),
		               eps_start=0.0,
		               eps_min=0.0,
		               eps_dec=0.0,
		               save_load_dir=save_load_dir,
		               fc1_out=1024,
		               fc2_out=512)

	original_agent.load_models()
	
	# Compute indices of neurons to keep for both fc1 and fc2
	keep_neurons_indices = compute_keep_neurons_index(original_agent.q_eval, ['fc1', 'fc2'], args.pruning_percentage)
	
	# Ensure that num_neurons_to_keep_fc1 and num_neurons_to_keep_fc2 are correctly retrieved from the dictionary
	num_neurons_to_keep_fc1 = len(keep_neurons_indices['fc1'])
	num_neurons_to_keep_fc2 = len(keep_neurons_indices['fc2'])
	
	agent = D3QNAgent(n_actions=env.action_space.n,
		             input_dims=(env.observation_space.shape),
		             eps_start=0.0,
		             eps_min=0.0,
		             eps_dec=0.0,
		             save_load_dir=save_load_dir,
		             fc1_out=num_neurons_to_keep_fc1,
		             fc2_out=num_neurons_to_keep_fc2)

	# Transfer weights from the original model to the pruned model
	transfer_weights(original_agent.q_eval, agent.q_eval, keep_neurons_indices)
	transfer_weights(original_agent.q_next, agent.q_next, keep_neurons_indices)

	# Initialize various variables:
	total_steps = 0
	episode = 1 #start with episode = 1
	start_time = time.time()

	#Reset some variables:
	twelve_ep_avg_reward = 0
	episode_reward_array = []
	while ((episode<=12) or (twelve_ep_avg_reward < args.acceptable_twelve_ep_avg_reward)):
		print('##########################')
		print('      Episode = (' + str(episode) + ')')
		print('##########################')
		done = False #reset done flag at start of each episode
		episode_reward = 0 #reset episode_reward at start of each episode
		episode_steps = 0 #reset episode_steps at start of each episode
		observation, _ = env.reset() #reset environment at start of each episode
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated #gymnasium (and gym==0.26) uses terminated and truncated instead of done
			print('Episode ' + str(episode) + ', episode steps = ' + str(episode_steps) + ', total steps = ' + str(total_steps) + ', x_obs = ' + str(observation[-1, 0, 0]) + ', y_obs = ' + str(observation[-1, 0, 1]))

			episode_reward += reward
			print('Episode reward = ' + str(episode_reward))

			agent.store_transition(observation, action, reward, observation_, int(done))
			agent.learn()

			observation = observation_
			total_steps += 1
			episode_steps += 1
		
		print('##### Episode ' + str(episode) + ' finised #####')
		episode_reward_array.append(episode_reward)
		#print(episode_reward_array)
		print(episode_reward_array[-12:])
		episode += 1
		twelve_ep_avg_reward = np.mean(episode_reward_array[-12:])
		print('12 episode average reward = ' + str(twelve_ep_avg_reward))

	agent.save_models()

	# At end print total time taken in easy to read format:
	total_seconds = time.time() - start_time
	hours, remainder = divmod(total_seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	print("Total time taken: {:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))
	
	# Land the drone:
	env.land_drone()
	rospy.logwarn('finished :)')
