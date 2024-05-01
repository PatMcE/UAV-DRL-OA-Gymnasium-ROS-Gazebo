# UAV-DRL-OA-Gymnasium-ROS-Gazebo
## Overview 
#### The code in this repository is connected to the yet to be published paper 'Towards Latency Efficient DRL Inference: Improving UAV Obstacle Avoidance at the Edge Through Model Compression'.
#### The bellow steps provide instructions to 1) setup and train a D3QN baseline model on a Ubuntu 20.04 machine (e.g. desktop), 2) apply model compression techniques to that baseline model and then convert to onnx and 3) setup the nvidia jetson orin nano or nvidia jetson nano to run the models (as tensorrt engines)
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
##### Step 1.6ci - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_empty_world_1.launch' and replace the contents with the file in this repo with the same name (and save). At line 9, change world1.world to world2.world if you want to launch world2 instead of world1.
##### Step 1.6cii - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_spawn_single.launch' and replace the contents with the file in this repo with the same name (and save)
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
rosrun mavros_px4_vehicle training_world.py
```
## 2: Apply model compression techniques to that baseline model and then convert to onnx
