# UAV-DRL-OA-Gymnasium-ROS-Gazebo
## Overview 
#### The code in this repository is connected to the yet to be published paper 'Towards Latency Efficient DRL Inference: Improving UAV Obstacle Avoidance at the Edge Through Model Compression'.
#### The bellow steps provide instructions to 1) setup and train a D3QN baseline model on a Ubuntu 20.04 machine (e.g. desktop), 2) apply model compression techniques to that baseline model and then convert to onnx and 3) setup the nvidia jetson orin nano or nvidia jetson nano to run the models (as TensorRT engines)
## 1: Setup and train a D3QN baseline model on a Ubuntu 20.04 machine (e.g. desktop)
### Step 1.1 - Ensure you have a Ubuntu 20.04 Machine and install ROS Noetic (http://wiki.ros.org/noetic/Installation/Ubuntu)
### Step 1.2 - Run various updates/installs and create a catkin workspace:
```
sudo apt update
sudo apt-get install python3-catkin-tools -y
sudo apt install git
sudo snap install sublime-text --classic
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin build
source devel/setup.bash
echo $ROS_PACKAGE_PATH
```
### Step 1.3 - Get the Gazebo Model for the Uvify IFO-S (https://github.com/decargroup/ifo_gazebo):
#### Step 1.3a - Execute the following commands:
```
cd ~/catkin_ws/src
git clone https://github.com/decarsg/ifo_gazebo.git --recursive
cd ..
catkin config --blacklist px4
catkin build
catkin build
cd ..
bash ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot/Tools/setup/ubuntu.sh
```
#### Step 1.3b - Relogin or reboot and execute the following commands:
```
sudo apt install python3-pip
```
```
pip3 install pyulog
pip3 install future
sudo apt upgrade -y
```
```
cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make distclean

cd ~
pip3 install --user empy
pip3 install --user packaging
pip3 install --user toml
pip3 install --user numpy
pip3 install --user jinja2

cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make px4_sitl gazebo
#if gazebo black screen then ctrl+c and run make command again
```
```
#ctrl+c
cd ~/catkin_ws/src/ifo_gazebo
rm -r real*
git clone https://github.com/pal-robotics/realsense_gazebo_plugin.git
cd ~/catkin_ws
catkin build
#run catkin build again if previous catkin build returns with a warning
```
#### Step 1.3c - execute more commands:
```
cd ~
nano ubuntu_sim_ros_noetic.sh
#fill ubuntu_sim_ros_noetic.sh with the contents of https://gist.githubusercontent.com/ekaktusz/a1065a2a452567cb04b919b20fdb57c4/raw/8be54ed561db7e3a2ce61c9c7b1fb9fec72501f4/ubuntu_sim_ros_noetic.sh
#exit and save ubuntu_sim_ros_noetic.sh
bash ubuntu_sim_ros_noetic.sh
#answer 'y' for any prompts
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/src/ifo_gazebo/setup_ifo_gazebo.bash suppress" >> ~/.bashrc
cd ~/catkin_ws
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
### Step 1.4 - Get the ROS package that allows a user to communicate with PX4 autopilot using MAVROS by executing the following commands (based off https://github.com/troiwill/mavros-px4-vehicle):
```
#cnrl+c
cd ~/catkin_ws/src
git clone https://github.com/troiwill/mavros-px4-vehicle.git
chmod +x mavros-px4-vehicle/scripts/*.py
chmod +x mavros-px4-vehicle/test/*.py
ln -s mavros-px4-vehicle mavros-px4-vehicle
cd ~/catkin_ws
catkin build
source devel/setup.bash
```
### Step 1.5 - Install scipy, gymnasium and torch:
```
pip3 install scipy
pip3 install gymnasium
pip3 install torch
```
### Step 1.6 - Copy the files from this github repository into the appropriate places as outlined bellow
#### Step 1.6a - Open sublime text
```
cd ~/catkin_ws
subl .
```
#### Step 1.6b - Add 6 python scripts to '~/catkin_ws/src/mavros-px4-vehicle/scripts'
##### Step 1.6bi - Navigate to '~/catkin_ws/src/mavros-px4-vehicle/scripts' and create 6 empty files
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
touch {dueling_deep_q_network,new_drone_gym_gazebo_env,train_d3qn_agent,train_main_d3qn,replay_memory,utils}.py
```
##### Step 1.6bii - Copy and paste the contents of the 6 python files with the 6 names from the touch command into their respective files on your computer with Sublime Text (and save before exiting files)
##### Step 1.6biii - Change 2 lines in 'train_main_d3qn.py' that are specific to your computer (highlighted by comments)
##### Step 1.6biv - Ensure the python scripts are executable
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
chmod +x *.py
```
#### Step 1.6c - Replace 2 of your launch files
##### Step 1.6ci - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_empty_world_1.launch' and replace the contents with the file in this repo with the same name (and save). At line 9, change training_world.world to test_world.world if you want to launch test_world instead of training_world.
##### Step 1.6cii - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_spawn_single.launch' and replace the contents with the file in this repo with the same name (and save). At lines 5 and 6 change default x and y values to change drone start position.
#### Step 1.6d - Create a few folders and add two world files to one of them
##### Step 1.6di -  Create folders at '~/catkin_ws/src/mavros-px4-vehicle'
```
cd ~/catkin_ws/src/mavros-px4-vehicle
mkdir {models,plots,worlds}
```
##### Step 1.6dii - Create 'training_world.world' and 'test_world.world' files at 'worlds' folder
```
cd ~/catkin_ws/src/mavros-px4-vehicle/worlds
touch {training_world,test_world}.world
```
##### Step 1.6diii - Copy and paste the contents of 'training_world.world' and 'test_world.world' in this repo into the files with the same name on your computer with Sublime Text (and save)
### Step 1.7 - Launch world and start training (you may have to restart machine before this step)
#### Step 1.7a - Run the bellow commands
```
cd ~
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
#### Step 1.7b - Open another terminal tab and run the bellow commands
```
cd ~/catkin_ws
source ~/.bashrc
rosrun mavros_px4_vehicle train_main_d3qn.py
```

## 2: Apply model compression techniques to the baseline model and then convert to onnx
### Step 2.1 - Copy the files from this github repository into the appropriate places as outlined bellow
#### Step 2.1a - Open sublime text
```
cd ~/catkin_ws
subl .
```
#### Step 2.1b - Add 9 python scripts to '~/catkin_ws/src/mavros-px4-vehicle/scripts'
##### Step 2.1bi - Navigate to '~/catkin_ws/src/mavros-px4-vehicle/scripts' and create 9 empty files
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
touch {deep_q_network_learner, dueling_deep_q_network, eval_d3qn_agent, kd_dqn_agent, kd_main_dqn, eval_prune_drone_gym_gazebo_env, neuron_prune_main, onnx_to_onnxHalf, pytorch_to_onnx}.py
```
##### Step 2.1bii - Copy and paste the contents of the 9 python files with the 9 names from the touch command into their respective files on your computer with Sublime Text (and save before exiting files)
##### Step 2.1biii - Change lines in 'kd_main_dqn.py' and 'neuron_prune_main.py' that are specific to your computer (highlighted by comments)
##### Step 2.1biv - Ensure the python scripts are executable
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
chmod +x *.py
```
### Step 2.2 - Neuron prune
#### Step 2.2a - Run the bellow commands
```
cd ~
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
#### Step 2.2b - Open another terminal tab and run the bellow command
```
rosrun mavros_px4_vehicle neuron_prune_main.py
```

### Step 2.3 - Knowledge Distillation
#### Step 2.3a - Run the bellow commands
```
cd ~
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
#### Step 2.3b - Open another terminal tab and run the bellow command
```
rosrun mavros_px4_vehicle kd_main_dqn.py
```

### Step 2.4 - Pytorch to ONNX:
#### Step 2.4a - change to the scripts folder directory and run the bellow command
```
python3 pytorch_to_onnx.py
```

### Step 2.5 - ONNX to ONNX_half:
#### Step 2.5a - change to the scripts folder directory and run the bellow command
```
python3 onnx_to_onnxHalf.py
```

## 3: Setup the nvidia jetson orin nano or nvidia jetson nano to run the models (as TensorRT engines)
### Step 3.1 - Install Ubuntu 20.04 and ROS Noetic on jetson orin nano/jetson nano (for jetson nano install ubuntu 20.04 by following https://qengineering.eu/install-ubuntu-20.04-on-jetson-nano.html)
### Step 3.2 - Execute the following commands:
```
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
roscore
#cntrl+c
sudo apt update
sudo apt-get install python3-catkin-tools -y
sudo apt install git
```
### Step 3.3 - Create catkin workspace:
```
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin build
source devel/setup.bash
echo $ROS_PACKAGE_PATH
```
### Step 3.4 - More commands:
```
cd ~
nano ubuntu_sim_ros_noetic.sh
#fill ubuntu_sim_ros_noetic.sh with the contents of https://gist.githubusercontent.com/ekaktusz/a1065a2a452567cb04b919b20fdb57c4/raw/8be54ed561db7e3a2ce61c9c7b1fb9fec72501f4/ubuntu_sim_ros_noetic.sh
#exit and save ubuntu_sim_ros_noetic.sh
bash ubuntu_sim_ros_noetic.sh
#answer 'y' for any prompts
```
### Step 3.5 - Get the ROS package that allows a user to communicate with PX4 autopilot using MAVROS by executing the following commands (based off https://github.com/troiwill/mavros-px4-vehicle):
```
#cnrl+c
cd ~/catkin_ws/src
git clone https://github.com/troiwill/mavros-px4-vehicle.git
chmod +x mavros-px4-vehicle/scripts/*.py
chmod +x mavros-px4-vehicle/test/*.py
ln -s mavros-px4-vehicle mavros-px4-vehicle
cd ~/catkin_ws
catkin build
source devel/setup.bash
```
### Step 3.6 - Install scipy, gymnasium, torch and re-install numpy so it v1.21:
```
pip3 install scipy
pip3 install gymnasium
pip3 install torch
pip3 uninstall numpy
pip3 install numpy=1.21
```
### Step 3.7 - Copy the files from this github repository into the appropriate places as outlined bellow
#### Step 3.7a - Open sublime text
```
cd ~/catkin_ws
subl .
```
#### Step 3.7b - Add 6 python scripts to '~/catkin_ws/src/mavros-px4-vehicle/scripts'
##### Step 3.7bi - Navigate to '~/catkin_ws/src/mavros-px4-vehicle/scripts' and create 6 empty files (first set of instructions if jetson orin nano, second set if jetson nano)
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
touch {eval_main_d3qn_trt_orin_nano, eval_main_dqn_trt_orin_nano, eval_drone_gym_gazebo_env, onnxHalf_to_trtHalf_orin_nano, onnx_to_trt_orin_nano, trt_utils}.py
```
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
touch {eval_main_d3qn_trt_nano, eval_main_dqn_trt_nano, eval_drone_gym_gazebo_env, onnxHalf_to_trtHalf_nano, onnx_to_trt_nano, trt_utils}.py
```
##### Step 3.7bii - Copy and paste the contents of the 6 python files with the 6 names from the touch command into their respective files on your jetson device with Sublime Text (and save before exiting files). Some file paths may need to be changed.
##### Step 3.7biii - Ensure the python scripts are executable
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
chmod +x *.py
```
### Step 3.8 - ONNX to TRT:
#### Step 3.8a - change to the scripts folder directory and run the bellow command
```
python3 onnx_to_trt_orin_nano.py #onnx_to_trt_nano.py if jetson nano
```

### Step 3.9 - ONNX_half to TRT_half:
#### Step 3.9a - change to the scripts folder directory and run the bellow command
```
python3 onnxHalf_to_trtHalf_orin_nano.py #onnxHalf_to_trtHalf_nano.py if jetson nano
```

### Step 3.10 - Run TRT engines on jetson orin nano/jetson nano:
#### Step 3.10a - Setup LAN setup as shown in paper
#### Step 3.10b - On Desktop where 192.168.8.107:11311 will be different for your machine (run ifconfig):
```
edit .bashrc file at ~ 
export ROS_MASTER_URI=http://192.168.8.107:11311
export ROS_IP=192.168.8.107
sudo reboot
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
#### Step 3.10c - On jetson orin nano/jetson nano where 192.168.8.134 will be different for you jetson device (run ifconfig):
```
edit .bashrc file at ~ 
export ROS_MASTER_URI=http://192.168.8.107:11311
export ROS_IP=192.168.8.134
sudo reboot
source ~/.bashrc
rosrun mavros_px4_vehicle eval_main_d3qn_trt_orin_nano.py #or eval_main_d3qn_trt_nano.py or eval_main_dqn_trt_orin_nano.py or eval_main_dqn_trt_nano.py
```
