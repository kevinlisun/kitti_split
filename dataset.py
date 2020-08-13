from scipy.io import loadmat
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# loop-closure GTs
kitti_gt_dir = "/home/kevin/KITTI_GroundTruth"
# robot car ground truth poses
kitti_pose_dir = "/home/kevin/kitti_odometry/poses"

seq = "02"

x = loadmat(os.path.join(kitti_gt_dir, "kitti"+seq+"GroundTruth.mat"))

query_set = []
margin = 50

visualise = True

if visualise:
	mat = np.transpose(np.array(x['truth']))
	cv2.imshow('image', mat*255.0)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

num = mat.shape[0]
print(num)
for i in range(num):
    print("processing "+str(i)+" of "+str(num))
    curr = mat[i].flatten()
    index = np.nonzero(curr==1)[0]

    if(len(index) == 0):
        continue

    ref = i
    # print(ref)
    for idx in index:
        if idx-ref > margin:
            print("dataset ref "+str(ref)+" query "+str(idx))
            query_set.append(idx)

query_set = np.unique(query_set)
database_set = np.delete(range(num), query_set)
print(query_set)
print("number of query set: "+str(len(query_set)))

# read the KITTI pose file
poses = np.loadtxt(os.path.join(kitti_pose_dir, seq+".txt"))

database_x = []
database_y = []
for i in database_set:
    pose = poses[i]
    pose = pose.reshape((3, 4))
    database_x.append(pose[0][3]) # x coordinate
    database_y.append(pose[2][3]) # z coordinate

query_x = []
query_y = []

for i in query_set:
    pose = poses[i]
    pose = pose.reshape((3, 4))
    query_x.append(pose[0][3])
    query_y.append(pose[2][3])

plt.plot(database_x, database_y, 'b.') # plot database as blue dots
plt.plot(query_x, query_y, 'r.') # plot query as red points
plt.show()








