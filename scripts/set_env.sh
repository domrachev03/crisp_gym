#!/bin/bash
# ROS2 environment configuration for crisp_gym (must match the robots / iris-panda-ros2).
export ROS_DOMAIN_ID=100
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Rig configs (panda env, RealSense cameras) ship in crisp_gym/config and are found by
# default; set CRISP_CONFIG_PATH only to add external config folders.
