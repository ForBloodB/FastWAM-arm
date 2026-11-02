#!/usr/bin/env bash
# Read-only graph introspection. Never sends a goal or a motor command.
set -eu
source /opt/ros/jazzy/setup.bash
source /home/aup/b601_ros2_client_ws/install/setup.bash
ros2 node list
ros2 action list -t
ros2 topic list -t
