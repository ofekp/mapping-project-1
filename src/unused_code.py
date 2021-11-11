

# For debugging purposes only

# other datasets
# MAIN_DATASET_PATH = "../../Data2/kitti_data/2011_09_26"
# DRIVE_NUMBER = "0005"
# MAIN_DATASET_PATH = "../../Data3/kitti_data/2011_09_26"
# DRIVE_NUMBER = "0061"


# extreme rotation and translation error
# trans = np.identity(4)
# trans[0:2, 0:2] = np.array([[0.9950042, -0.0998334], [0.0998334,  0.9950042]])
# velodyne_frame_filtered_copy = (np.dot(trans, velodyne_frame_filtered_copy.T) + np.array([-0.5, -0.4, 0, 0]).reshape(4, 1)).T
# trans_inv = np.linalg.inv(trans)