#! /usr/bin/env python3

import eval_drone_gym_gazebo_env #defines 'DroneGymGazeboEnv-v0' custom environment
import gymnasium
import numpy as np
import rospy
import argparse
import time
import tensorrt as trt
from trt_utils import allocate_buffers, do_inference
import psutil
from jtop import jtop

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = '/home/jetson/catkin_ws/src/mavros-px4-vehicle/models/D3QN_eval.trt'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Drone StableBaselines')
	parser.add_argument('-episodes', type=int, default=100)
	args = parser.parse_args()

	# Initialize/create ROS node and custom gym environment:
	rospy.init_node('drone_node', anonymous=True)
	env = gymnasium.make('DroneGymGazeboEnv-v0') #'DroneGymGazeboEnv-v0' is the custom gym environment from drone_gym_gazebo_env

	with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read()) #Load/deserialize TensorRT engine   
	context = engine.create_execution_context() #Create execution context
	inputs, outputs, bindings, stream = allocate_buffers(engine) #Prepare buffers
   
	total_steps = 0
	successful_steps = 0
	num_exceeded_workspace = 0
	num_collided = 0
	num_reached_des_point = 0
	accumulated_inf_time = 0
	with jtop() as jetson:       
		for episode in range(1, args.episodes+1): #add one because python range() goes up to but not including the second value
			print('######################################################################')
			print('                           Episode = ' + str(episode))
			print('######################################################################')

			done = False #reset done flag at start of each episode
			episode_reward = 0 #reset episode_reward at start of each episode
			episode_steps = 0 #reset episode_steps at start of each episode
			observation, _ = env.reset() #reset environment at start of each episode
			while not done:
				start_time = time.time()
				
				preprocessed_obs = np.ascontiguousarray(observation.reshape((1,) + observation.shape)) #Preprocess observation/state
				np.copyto(inputs[0].host, preprocessed_obs.ravel()) #Copy data to buffer
				trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) #Run inference
				trt_outputs_V = trt_outputs[0]
				trt_outputs_A = trt_outputs[1]
				Qvals = trt_outputs_V + (trt_outputs_A - np.mean(trt_outputs_A)) # Compute Q-values
				action = np.argmax(Qvals) #Get action from the output of TRT engine
				
				accumulated_inf_time += time.time() - start_time	
				
				observation_, reward, terminated, truncated, info = env.step(action)
				done = terminated or truncated #gymnasium (and gym==0.26) uses terminated and truncated instead of done
				print('Episode ' + str(episode) + ', episode steps = ' + str(episode_steps) + ', total steps = ' + str(total_steps) + ', x_obs = ' + str(observation[-1, 0, 0]) + ', y_obs = ' + str(observation[-1, 0, 1])) #observation[-1,0,0 or 1] = relative position observation

				episode_reward += reward
				observation = observation_
				total_steps += 1
				episode_steps += 1
				
			if env.get_has_drone_exceeded_workspace():
				num_exceeded_workspace += 1
			if env.get_has_drone_collided():
				num_collided += 1
			if env.get_has_reached_des_point():
				num_reached_des_point += 1
				successful_steps += episode_steps	

			print('Episode', episode, ' Summary: episode steps', episode_steps, ', episode_reward', episode_reward, ', total steps', total_steps, ', exceeded workspace', num_exceeded_workspace, ', collided', num_collided, ', reached dest', num_reached_des_point)

	# After all episodes
	print(f'Success Rate: {(num_reached_des_point/args.episodes)*100:.2f}%')
	print(f'Average Inference Time: {(accumulated_inf_time/total_steps)*1000:.2f} ms')
	print(f'Average (Successfull Episode) Steps: {successful_steps / num_reached_des_point}')
	
	# Land the drone:
	env.land_drone()
	rospy.logwarn('finished :)')
