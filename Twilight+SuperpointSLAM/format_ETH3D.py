#!/usr/bin/env python3

import sys, os
from datetime import datetime
import numpy as np

# input
rgb_txt_path = "/home/billymazotti/Documents/Twilight_SuperPoint_SLAM/datasets/ETH3D/sfm_lab_room_2/rgb.txt"

# outdir
outdir = "/home/billymazotti/Documents/Twilight_SuperPoint_SLAM/datasets/ETH3D/sfm_lab_room_2"

# outputs/created files
times_txt_path = outdir + "/times.txt"
images_txt_path = outdir + "/images.txt"

times = np.loadtxt(rgb_txt_path,dtype=str)[:,0].astype(np.double).reshape(-1,1)
images = np.loadtxt(rgb_txt_path,dtype=str)[:,1]

images = np.array([image_name.split(sep ="/")[-1] for image_name in images]).reshape(-1,1)
print(times)

np.savetxt(times_txt_path,times,fmt='%s')
np.savetxt(images_txt_path,images,fmt='%s')



