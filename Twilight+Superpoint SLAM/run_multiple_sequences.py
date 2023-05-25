import os, sys
import shutil
from tqdm import tqdm
from datetime import datetime
import stat


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


twilight_superpoint_slam = os.getcwd()

# paths from the SuperPoint-SLAM Directory
vocabuylary_path = "./vocabulary/superpoint_voc.yml.gz"


########################## SuperPoint-SLAM settings ##########################
# KITTI Dataset 
mono_kitti_path = "./Examples/Monocular/mono_kitti"     # monocular kitti executable path
kitti_yaml_path = "./Examples/Monocular/KITTI04-12.yaml"  # kitti .ymal file for sequqnce 4-12

# ETH3D 
mono_eth3d_path = "./Examples/Monocular/mono_eth3d"


########################## Sequence Selection ##########################
# KITTI sequences
kitti_sequences = []
sequences_04  = ["04","04_bread","04_dual","04_egan","04_zero"]
sequences_06  = ["06_bread","06_dual","06_egan","06_zero"]
sequences_07  = ["07","07_bread","07_dual","07_egan","07_zero"]
kitti_sequences = sequences_04 * 4 + sequences_06 * 4 + sequences_07 * 4


# ETH3D
eth3d_sequences = []
sfm_house_loop = ["sfm_house_loop","sfm_house_loop_bread","sfm_house_loop_dual","sfm_house_loop_egan","sfm_house_loop_zero"]
sfm_lab_room_2 = ["sfm_lab_room_2","sfm_lab_room_2_bread","sfm_lab_room_2_dual","sfm_lab_room_2_egan","sfm_lab_room_2_zero"]
# eth3d_sequences = sfm_lab_room_2 * 5


for sequence in tqdm(kitti_sequences):
    print(f"\n\nTESTING SEQUENCE: {sequence}")
    
    # set directory to SuperPoint-SLAM
    os.chdir(os.path.join(os.getcwd(),'thirdparty/SuperPoint_SLAM'))
    sequence_path = "../../datasets/KITTI/data_odometry_color/dataset/sequences/" + sequence

    # execute sequence
    print(f"\n{mono_kitti_path} {vocabuylary_path} {kitti_yaml_path} {sequence_path}")
    os.system(f"{mono_kitti_path} {vocabuylary_path} {kitti_yaml_path} {sequence_path}")

    # reutrn to Twilight-SLAM directory and save trajectory
    os.chdir(twilight_superpoint_slam)
    original = 'thirdparty/SuperPoint_SLAM/KeyFrameTrajectory.txt'
    now = datetime.now()
    timestamp = f"Y{now.year}_M{now.month}_D{now.day}_H{now.hour}_M{now.minute}_S{now.second}"
    target = f'evaluations/estimated_trajectories/KITTI/{sequence}_KeyFrameTrajectory_{timestamp}.txt'

    if os.path.isfile(original):
        shutil.copyfile(original, target)
        while True:
            print("Copying Trajecotry File...")
            if(os.path.isfile(target)):
                print("Copying Trajectory File COMPLETE!")
                os.remove(original)
                break
        
    else:
        print("\nSEQEUENCE DID NOT FINISH, KeyFrameTrajectory.txt FILE WAS NOT GENERATED\n")
    
for sequence in tqdm(eth3d_sequences):
    print(f"\n\nTESTING SEQUENCE: {sequence}")
    
    # set directory to SuperPoint-SLAM
    os.chdir(os.path.join(os.getcwd(),'thirdparty/SuperPoint_SLAM'))
    sequence_path = "../../datasets/ETH3D/" + sequence

    # path to yaml file
    yaml_path = sequence_path + "/ETH3D.yaml"

    # execute sequence
    print(f"\n{mono_eth3d_path} {vocabuylary_path} {yaml_path} {sequence_path}")
    os.system(f"{mono_eth3d_path} {vocabuylary_path} {yaml_path} {sequence_path}")

    # reutrn to Twilight-SLAM directory and save trajectory
    os.chdir(twilight_superpoint_slam)
    original = 'thirdparty/SuperPoint_SLAM/KeyFrameTrajectory.txt'
    now = datetime.now()
    timestamp = f"Y{now.year}_M{now.month}_D{now.day}_H{now.hour}_M{now.minute}_S{now.second}"
    target = f'evaluations/estimated_trajectories/ETH3D/{sequence}_KeyFrameTrajectory_{timestamp}.txt'

    if os.path.isfile(original):
        shutil.copyfile(original, target)
        while True:
            print("Copying Trajecotry File...")
            if(os.path.isfile(target)):
                print("Copying Trajectory File COMPLETE!")
                os.remove(original)
                break
        
    else:
        print("\nSEQEUENCE DID NOT FINISH, KeyFrameTrajectory.txt FILE WAS NOT GENERATED\n")

    


