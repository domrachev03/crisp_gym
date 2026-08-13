#!/bin/bash
# Default domain for ordinary single-context tools. The mixed-FR3 runner creates
# explicit contexts for leader domain 78 and follower domain 79 instead.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Rig configs (panda env, RealSense cameras) ship in crisp_gym/config and are found by
# default; set CRISP_CONFIG_PATH only to add external config folders.
