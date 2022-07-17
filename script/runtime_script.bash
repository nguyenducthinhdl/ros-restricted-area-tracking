#!/bin/bash
roscore &
sleep 3
rosrun web_video_server web_video_server &
roslaunch rosbridge_server rosbridge_websocket.launch & 
roslaunch /launch/video_file.launch &

catkin_make -DCMAKE_BUILD_TYPE=Release
source /catkin_workspace/devel/setup.bash
roslaunch /launch/darknet_ros.launch