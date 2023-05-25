#!/usr/bin/python2.7

# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys, os
import numpy
import argparse
import associate
import numpy as np

def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    s -- scale
    """


    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = numpy.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)      
        
    return rot, s

def plot_traj(ax,stamps,traj,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            #ax.plot(x,y,style,color=color,label=label)
            ax.scatter(x,y,color=color,label=label,s=5)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        #ax.plot(x,y,style,color=color,label=label)
        ax.scatter(x,y,color=color,label=label,s=5)
        
    return x,y
        
            
def eval_metric(traj_gt, traj_est):
    trans_error_mat = abs(traj_gt - traj_est)
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(trans_error_mat,trans_error_mat),axis=0))
    
    return trans_error
    
if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 10000000 ns)',default=20000000)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    parser.add_argument('--verbose2', help='print scale eror and RMSE absolute translational error in meters after alignment with and without scale correction', action='store_true')
    parser.add_argument('--txt_file',help='save verbose to txt file (format)')
    args = parser.parse_args()

    first_list = associate.read_file_list(args.first_file, False)
    second_list = associate.read_file_list(args.second_file, False)
    
    for key in first_list:
    	a = float(first_list[key][0]) + 5.1011
    	first_list[key][0] = str(a)
    	b = float(first_list[key][1]) - 2.3219
    	first_list[key][1] = str(b)
    	c = float(first_list[key][2]) + 6.0578
    	first_list[key][2] = str(c)
    	
      	
    matches = associate.associate(first_list, second_list,float(args.offset),float(args.max_difference))    
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")
    first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
    dictionary_items = second_list.items()
    sorted_second_list = sorted(dictionary_items)
    
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale) for value in sorted_second_list[i][1][0:3]] for i in range(len(sorted_second_list))]).transpose() # sorted_second_list.keys()]).transpose()
    rot, scale = align(second_xyz,first_xyz)
    
    second_xyz_aligned = scale * rot * second_xyz 
    second_xyz_notscaled = rot * second_xyz 
    second_xyz_notscaled_full = rot * second_xyz_full
    first_stamps = first_list.keys()
    first_stamps.sort()
    first_xyz_full = numpy.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
    
    second_stamps = second_list.keys()
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = scale * rot * second_xyz_full
    
    trans_error = eval_metric(first_xyz, second_xyz_full_aligned)
    if args.verbose:
        print("compared_pose_pairs %d pairs"%(trans_error.shape[1]))
        print("absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error.transpose()) / trans_error.shape[1]))
        print("absolute_translational_error.mean %f m"%numpy.mean(trans_error))
        print( "absolute_translational_error.median %f m"%numpy.median(trans_error,axis=1))
        print( "absolute_translational_error.std %f m"%numpy.std(trans_error))
        print( "absolute_translational_error.min %f m"%numpy.min(trans_error))
        print( "absolute_translational_error.max %f m"%numpy.max(trans_error))
        print( "max idx: %i" %numpy.argmax(trans_error))

    if args.save_associations:
        file = open(args.save_associations,"w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f"%(a,x1,y1,z1,b,x2,y2,z2) for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A)]))
        file.close()
        
    if args.save:
        file = open(args.save,"w")
        file.write("\n".join(["%f "%stamp+" ".join(["%f"%d for d in line]) for stamp,line in zip(second_stamps,second_xyz_notscaled_full.transpose().A)]))
        file.close()

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        from matplotlib.patches import Ellipse
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x,y = plot_traj(ax,first_stamps,first_xyz_full.transpose().A,"blue","ground truth")
        _,_ = plot_traj(ax,second_stamps,second_xyz_full_aligned.transpose().A,"red","estimated")
        w = 2488
        h = 1342
        my_dpi = 100
        fig.set_size_inches(w/my_dpi,h/my_dpi)
        ax.legend(markerscale=5, fontsize=22)
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('blue')
        leg.legendHandles[1].set_color('red')
        
        ax.set_xlabel('x(m)', fontsize=24)
        ax.set_ylabel('y(m)', fontsize=24)
        
        plt.setp(ax.get_xticklabels(), fontsize=20)
        plt.setp(ax.get_yticklabels(), fontsize=20)
        plt.axis('equal')
        plt.savefig(args.plot,format="png")

    if args.txt_file is not None:

        with open(args.txt_file,"w+") as times_file:
            times_file.write("compared_pose_pairs %d pairs"%(trans_error.shape[1]))
            times_file.write("absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error.transpose()) / trans_error.shape[1]))
            times_file.write("absolute_translational_error.mean %f m"%numpy.mean(trans_error))
            times_file.write( "absolute_translational_error.median %f m"%numpy.median(trans_error,axis=1))
            times_file.write( "absolute_translational_error.std %f m"%numpy.std(trans_error))
            times_file.write( "absolute_translational_error.min %f m"%numpy.min(trans_error))
            times_file.write( "absolute_translational_error.max %f m"%numpy.max(trans_error))
            times_file.write( "max idx: %i" %numpy.argmax(trans_error))


        
