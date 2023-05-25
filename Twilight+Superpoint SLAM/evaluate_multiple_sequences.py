# This script contains code created for eval_kitti.py in SuperPoint-SLAM at https://github.com/KinglittleQ/SuperPoint_SLAM
# and alterned for use in Twilight-SuperPoint-SLAM at https://github.com/TwilightSLAM/Twilight_SuperPoint_SLAM


import numpy as np
import matplotlib.pyplot as plt
import sys, os
from tqdm import tqdm

def gen_data(ground_time, res_time, ground_data):
	ground_time = ground_time
	res_time = res_time
	ground_data = ground_data

	time_mark = 0
	time = []

	data_1 = []

	for num in range(len(ground_data)):

		data_1.append(np.concatenate(([ground_time[num]], ground_data[num])))
			

	data_2 = []


	for num in range(len(res_time)):
		while not np.allclose(data_1[time_mark][0], res_time[num][0]):
			time_mark+=1
		data_2.append(data_1[time_mark])

	return data_2


def get_coo(data):
	points = [[],[],[]]
	for num in range(len(data)):
		points[0].append(data[num][4])
		points[1].append(data[num][8])
		points[2].append(data[num][12])
	return points


def get_points(data):
	points = [[],[],[]]
	for num in range(len(data)):
		points[0].append(data[num][1])
		points[1].append(data[num][2])
		points[2].append(data[num][3])
	return points


def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    np.set_printoptions(precision=3,suppress=True)
    model_mean=[[model.mean(1)[0]], [model.mean(1)[1]], [model.mean(1)[2]]]
    data_mean=[[data.mean(1)[0]], [data.mean(1)[1]], [data.mean(1)[2]]]
    model_zerocentered = model - model_mean
    data_zerocentered = data - data_mean
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = np.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    

    # print ("scale: %f " % s) 
    
    trans = data_mean - s*rot * model_mean
    
    model_aligned = s*rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error, s




if __name__ == '__main__':

	# evaluate KITTI seuqences
	kitti_trajectories_path = os.getcwd() + "/evaluations/estimated_trajectories/KITTI/"

	print("\nEVALUATING ESTIMATED TRAJECTORIES ON KITTI DATASET SEQUENCES")
	for path in tqdm(os.listdir(kitti_trajectories_path)): #Path to the KeyFrameTrajectory.txt file
		
		#Path to the times.txt in KITTI dataset
		sequence = path.split("_")[0]
		ground_time = np.loadtxt(os.getcwd() + '/datasets/KITTI/data_odometry_color/dataset/sequences/'+sequence+'/times.txt')
		res_time = np.loadtxt(kitti_trajectories_path + path)
	
		#Path to the ground truth file
		ground_data = np.loadtxt(os.getcwd() + '/datasets/KITTI/data_odometry_color/dataset/poses/'+sequence+'.txt')
		data= gen_data(ground_time, res_time, ground_data)
		ground_points = np.asarray(get_coo(data))
		re_points = np.asarray(get_points(res_time))
		rot,trans,trans_error,s = align(re_points, ground_points)
		re_fpoints = s*rot*re_points+trans

		# save plot of ground truth and estimated trajectories
		name = path.split(".")[0]
		
		# auto save plots
		plt.figure(figsize=(11.5, 6.5))
		plt.axis('equal')
		ground_truth = plt.scatter(ground_points[0], ground_points[2], s=5.0, c='blue')
		estimated = plt.scatter(list(re_fpoints[0]), list(re_fpoints[2]), s=5.0, c='red')
		plt.legend([ground_truth,estimated],["ground truth","estimated"], markerscale=5, fontsize=22)
		plt.xlabel('x(m)',fontsize=24)
		plt.ylabel('y(m)',fontsize=24)
		plt.savefig(os.getcwd() + '/evaluations/plots/KITTI/'+name+'.png')

	
		# save metrics txt file
		with open(os.getcwd() + '/evaluations/metrics/ETH3D/'+name+'.txt',"w+") as times_file:
			times_file.write("compared_pose_pairs %d pairs\r\n"%(len(trans_error)))
			times_file.write("absolute_translational_error.rmse %f m\r\n"%np.sqrt(np.dot(trans_error,trans_error) / len(trans_error)))
			times_file.write("absolute_translational_error.mean %f m\r\n"%np.mean(trans_error))
			times_file.write("absolute_translational_error.median %f m\r\n"%np.median(trans_error))
			times_file.write("absolute_translational_error.std %f m\r\n"%np.std(trans_error))
			times_file.write("absolute_translational_error.min %f m\r\n"%np.min(trans_error))
			times_file.write("absolute_translational_error.max %f m\r\n"%np.max(trans_error))	
		

	print("\nEVALUATING ESTIMATED TRAJECTORIES ON ETH3D DATASET SEQUENCES")
	# evaluate ETH3D sequences
	eth3d_trajectories_path = os.getcwd() + "/evaluations/estimated_trajectories/ETH3D/"
	for path in tqdm(os.listdir(eth3d_trajectories_path)): #Path to the KeyFrameTrajectory.txt file
		
		#Path to the times.txt in ETH3D dataset
		sequence = "".join(path.split("_KeyFrameTrajectory_")[0])
		estimated_trajectory_path = eth3d_trajectories_path + path
	
		#Path to the ground truth file
		ground_truth_path = os.getcwd() + '/datasets/ETH3D/'+sequence+'/groundtruth.txt'
		plot_path = (os.getcwd() + "/evaluations/plots/ETH3D/"+path).split(".")[0] + ".png"
		txt_file_path = os.getcwd() + "/evaluations/metrics/ETH3D/"+path

		os.system("python2.7 eval_eth3d.py "+ ground_truth_path +" "+ estimated_trajectory_path + " --plot="+plot_path+ " --txt_file="+txt_file_path)


