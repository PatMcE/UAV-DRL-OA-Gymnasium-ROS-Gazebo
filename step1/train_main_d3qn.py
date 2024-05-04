#! /usr/bin/env python3

'''
This code is mainly based off https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
'''

from train_d3qn_agent import D3QNAgent
from utils import plot_learning_curve
import new_drone_gym_gazebo_env #defines 'DroneGymGazeboEnv-v0' custom environment

import gymnasium
import numpy as np
import rospy
import argparse
import time

from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    # Allow the following arguments to be passed:
    parser = argparse.ArgumentParser(description = 'Drone StableBaselines')
    parser.add_argument('-load_checkpoints', type=bool, default=False)
    parser.add_argument('-render', type=bool, default=False)
    parser.add_argument('-eps_start', type=float, default=1, help='What epsilon starts at')    
    parser.add_argument('-eps_min', type=float, default=0.05, help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-exploration_steps', type=float, default=120000, help='Number of steps before epsilon levels out at eps_min')
    parser.add_argument('-max_episodes', type=int, default=4000)
    parser.add_argument('-min_steps', type=int, default=200000)
    parser.add_argument('-acceptable_avg_reward', type=int, default=75)
    ############### Change the bellow line to match your system ###############
    parser.add_argument('-root_path', type=str, default='/home/pm/catkin_ws/src/mavros-px4-vehicle/', help='root path for saving/loading')   
    #Tensorboard log code: tensorboard --logdir=/home/pm/catkin_ws/src/mavros-px4-vehicle/tb_logs     
    ##############################################################################
    args = parser.parse_args()

    # Initialize/create ROS node and custom gym environment:
    rospy.init_node('drone_node', anonymous=True)
    env = gymnasium.make('DroneGymGazeboEnv-v0') #'DroneGymGazeboEnv-v0' is the custom gymnasium environment from drone_gym_gazebo_env

    # Setup files/directories for saving/loading:
    figure_file = args.root_path + 'plots/D3QN.png' 
    save_load_dir = args.root_path + 'models/'   
    tb_log_dir = args.root_path + 'tb_logs/D3QN'

    # Create a writer for the tensorboard plots:
    writer = SummaryWriter(log_dir=tb_log_dir)

    #Determine epsilon decay rate:
    eps_decay_rate = (args.eps_start-args.eps_min)/args.exploration_steps 
    
    #Create agent from argument values:
    agent = D3QNAgent(n_actions=env.action_space.n,
                     input_dims=(env.observation_space.shape),
                     eps_start=args.eps_start, eps_min=args.eps_min, eps_dec=eps_decay_rate,
                     save_load_dir = save_load_dir)

    #Load models if load_checkpoints argument is True:
    if args.load_checkpoints:
        agent.load_models()

    # Initialize various variables:
    best_avg_reward = -np.inf
    avg_reward = 0
    total_steps = 0
    episode = 1 #start with episode = 1
    episode_reward_array, eps_history_array, steps_array = [], [], []
    start_time = time.time()

    #Keep training until the number of min steps have been exceeded and the average reward is >= args.acceptable_avg_reward:
    while (episode<=args.max_episodes):
        print('######################################################################')
        print('                           Episode = ' + str(episode))
        print('######################################################################')

        done = False #reset done flag at start of each episode
        episode_reward = 0 #reset episode_reward at start of each episode
        episode_steps = 0 #reset episode_steps at start of each episode
        observation, _ = env.reset() #reset environment at start of each episode
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated #gymnasium (and gym==0.26) uses terminated and truncated instead of done
            print('Episode ' + str(episode) + ', episode steps = ' + str(episode_steps) + ', total steps = ' + str(total_steps) + ', x_obs = ' + str(observation[-1, 0, 0]) + ', y_obs = ' + str(observation[-1, 0, 1])) #observation[-1,0,0 or 1] = relative position observation

            episode_reward += reward

            if args.render:
                env.render()
            
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()

            observation = observation_
            total_steps += 1
            episode_steps += 1
        
        episode_reward_array.append(episode_reward)
        steps_array.append(total_steps)
        eps_history_array.append(agent.epsilon)
        avg_reward = np.mean(episode_reward_array[-100:])
        print('Episode', episode, ' Summary: ep reward', episode_reward, ', average ep reward %.1f' % avg_reward, ', best avg reward %.2f' % best_avg_reward, ', epsilon %.2f' % agent.epsilon, ', ep steps ', episode_steps, ', tot steps ', total_steps)

        # If best average reward has been beat, save current model and update best average reward to current average reward:
        if avg_reward > best_avg_reward:
            agent.save_models()
            best_avg_reward = avg_reward

        # Log values for tensorboard:
        writer.add_scalar('Average Reward vs Episodes', avg_reward, global_step=episode)
        writer.add_scalar('Average Reward vs Number of Steps', avg_reward, global_step=total_steps)
        writer.add_scalar('Epsilon vs Number of Steps', agent.epsilon, global_step=total_steps)

        # Increment episodes:
        episode += 1

    # At end print total time taken in easy to read format:
    total_seconds = time.time() - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("Total time taken: {:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))

    # Close writer used for tensorboard stuff:
    writer.close()

    # Plot learning curve with function from utils file (in case tensorboard fails):
    x = [i+1 for i in range(len(episode_reward_array))]
    plot_learning_curve(steps_array, episode_reward_array, eps_history_array, figure_file)

    # Land the drone:
    env.land_drone()
    rospy.logwarn('finished :)')
