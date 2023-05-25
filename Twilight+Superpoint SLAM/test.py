#!/usr/bin/env python3

import sys, os
from datetime import datetime
import numpy as np

# # load times and poses
# pose_path = "/home/billymazotti/Documents/Twilight_SuperPoint_SLAM/datasets/TartanAir/abandonedfactory_sample_P001/P001/pose_left.txt"
# time_path = "/home/billymazotti/Documents/Twilight_SuperPoint_SLAM/datasets/Formatted_TartanAir/abandonedfactory_sample_P001/P001/times.txt"
# poses = np.loadtxt(pose_path)
# times = np.loadtxt(time_path).reshape(-1,1)

# # stack times and poses
# new_poses = np.hstack([times,poses])

# # save times and poses
# new_poses_path = "/home/billymazotti/Documents/Twilight_SuperPoint_SLAM/test.txt"
# np.savetxt(new_poses_path,new_poses)

queue = ["a"]*3 + ['b']*5


print(queue)
