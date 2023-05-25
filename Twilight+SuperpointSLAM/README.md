# Twilight_SuperPoint_SLAM
Primary Objective: To investigate hypothesized performance improvements of SuperPoint-SLAM, a monocular visual SLAM technique combining superpoint's neural network-based feature detection and description with the ORB-SLAM2 architecture, employing image enhancing modules for low light or "twilight" monocular SLAM dataset sequences. 

Secondary Objectve: To provide tools for streamlining evaluation of multiple datasets.

Experiment Overview: We initially test against against the KITTI benchmark to confirm pipeline success. Then, we test against ETH3D benchmark data sequences which were qualitatively found to have more underexposed images on average or in general througout the entire sequence. Twilight dataset sequences that were too underexposed to initiate SuperPoint-SLAM wihout image enhancment modules were not considered.

# 1. OS and Hardware Prerequisites
We tested this project in **Ubuntu 18.04**. Additionally, the original repository reports testing for **Ubuntu 12.04**, **14.04**, and **16.04**. Using a HP-360x-Spectre with 4 cores, Intel Core i7, and no GPU. Thus we  we achieved SuperPoint-SLAM at 5 seconds per frame.


# 2. Intial Steps
```bash
cd # to your preferred directory
git clone https://github.com/TwilightSLAM/Twilight_SuperPoint_SLAM.git
cd Twilight_SuperPoint_SLAM
```


# 3. Third Party Installations
To streamline the installation process, we provide the library versions that we used, links to the official respositories, and recommended installation steps/guides based on what worked for us.

## Pangolin v0.6
### Official Repository: https://github.com/stevenlovegrove/Pangolin
### Recommended Steps
``` bash
# build in third party directory (recommended)
cd thirdparty

# build Pangolin
git clone 
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout v0.6
./scripts/install_prerequisites.sh recommended
mkdir build && cd build
cmake ..
make -j2
make install

# return to thirdparty directory
cd ../..
```

## OpenCV 3.2.0
### Official Repository: https://github.com/opencv/opencv.git
### Recommended Guide: https://gist.github.com/syneart/3e6bb68de8b6390d2eb18bff67767dcb

## Eigen3 3.3.4-4
### Official Repository: http://eigen.tuxfamily.org/
### Recommended Steps
You should have already installed libeigen via the recommended instruction for building OpenCV. If not, try installing with
```bash
sudo apt-get install libeigen3-dev
```

## Libtorch
### Official Repository:
### Recommended Steps (per SuperPoint_SLAM repo):
```bash
git clone --recursive -b v1.0.1 https://github.com/pytorch/pytorch
cd pytorch && mkdir build && cd build
python ../tools/build_libtorch.py
```

## SuperPoint-SLAM
### Official Repository: https://github.com/KinglittleQ/SuperPoint_SLAM.git
### Recommended Steps: 
```bash
git clone https://github.com/KinglittleQ/SuperPoint_SLAM.git
cd SuperPoint_SLAM

# edit CMakeLists.txt, changing set(Torch_DIR "/home/deng/SP_SLAM/Thirdparty/libtorch/share/cmake/Torch") to the directory containing the libtorch torch config cmake file, e.g. set(Torch_DIR "/home/billymazotti/Documents/Twilight_SuperPoint_SLAM/thirdparty/pytorch/torch/lib/tmp_install/share/cmake/Torch")
gedit CMakeLists.txt

# build SuperPoint_SLAM and it's thirdparty modules (DBoW3 and g2o)
chmod +x build.sh
./build.sh
```


# 4. Install KITTI and ETH3D SLAM Datasets


# 5. Prepare and format ETH3D SLAM Datasets
Match the same dataset format as KITTI datasets


# 6. Enhance Images
Follow steps in [INSERT REPO HERE] to download visual slam datasets (KITTI and ETH3D) and generate new images using low light image enhancement modules (Bread, Dual, EnlightenGAN and ZeroDCE).

Once images are downloaded, use the recommended direcotry formatting to store you images in order to use `run_multiple_sequences.py` and `evaluate_multiple_sequences.py`.


# 7. Run SuperPoint-SLAM
The dataset sequences chosen for this study take 20 minutes to 3 hours to run each, therefore we edit and run the following python file to run SuperPoint-SLAM on and thereby generate estimated trajectories for multiple dataset sequences overnight.
```bash
python3 run_multiple_sequences.py
```


# 8. Evaluate 
To speed up the evaluation process, use the following python script ti generate SLAM trajectory metrics and plots for for each estimated trajecotry.
```bash
python3 evaluate_multiple_sequences.py
```
