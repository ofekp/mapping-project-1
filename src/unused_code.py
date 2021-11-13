

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


# def cross_cov_xy(P, Q):
#     assert P.shape[0] > P.shape[1]
#     assert Q.shape[0] > Q.shape[1]
#     W = np.zeros((3,3))
#     for i in range(P.shape[0]):
#         mat = np.dot(Q[i, 0:3].reshape(3, 1), P[i, 0:3].reshape(1, 3))
#         assert mat.shape[0] == 3 and mat.shape[1] == 3
#         W += mat
#     return W / P.shape[0]


# velodyne_frame = np.array([
#     [20, 20, 0, 0],
#     [10, 10, 2, 0],
#     [10, 20, 2, 0],
#     [-10, -5, 2, 0],
#     [5, 5, 2, 0],
#     [17, 9, 2, 0],
#     [19, -5, 2, 0]
# ], dtype=float)


# in load_velodyne() I removed the following since it is in a list anyway
# frame_idx = int(os.path.basename(path_str).split('.')[0])

# some transformations on the prev velodyne frame
# velodyne_frame_prev = velodyne["frames_raw"][0].copy()
# velodyne_frame_prev = np.dot(velo_to_enu, velodyne_frame_prev.T).T
# velodyne_frame_prev_filtered = filter_velodyne_data(velodyne_frame_prev.copy())
# theta = -yaw_from_config[frame_idx - 1]
# R = np.identity(4)
# R[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).reshape(2, 2)
# t_e_diff = e_noise_icp[-1]
# t_n_diff = n_noise_icp[-1]
# # t_n_diff = enu_from_config[frame_idx - 1, 1]
# t = np.array([t_e_diff, t_n_diff, 0.0, 0.0]).reshape(4, 1)
# velodyne_frame_prev_filtered = (np.dot(R, velodyne_frame_prev_filtered.T) + t).T
# print("Expecting {} {} {}".format(enu_noise[frame_idx, 0] - enu[frame_idx, 0], enu_noise[frame_idx, 1] - enu[frame_idx, 1], yaw_noise[frame_idx] - yaw[frame_idx]))
# print("noise {} {} {}".format(enu_noise[frame_idx, 0] - enu_noise[frame_idx - 1, 0], enu_noise[frame_idx, 1] - enu_noise[frame_idx - 1, 1], yaw_noise[frame_idx] - yaw_noise[frame_idx - 1]))
# print("Totals {} {} {}".format(enu[frame_idx, 0] - enu[frame_idx - 1, 0], enu[frame_idx, 1] - enu[frame_idx - 1, 1], yaw[frame_idx] - yaw[frame_idx - 1]))


# with open(CALIB_IMU_TO_VELO_PATH, 'r') as calib_imu_to_velo_file:
#     lines = calib_imu_to_velo_file.readlines()
#     R = np.array([float(a) for a in lines[1].strip().replace("R: ", "").split(" ")]).reshape(3, 3)
#     R_inv = np.linalg.inv(R)
#     t = np.array([float(a) for a in lines[2].strip().replace("T: ", "").split(" ")]).reshape(1, 3)
#     R_inv4 = np.identity(4)
#     t4 = np.zeros(4).reshape(1, 4)
#     t4[0, 0:3] = t
#     R_inv4[0:3, 0:3] = R_inv
#     velodyne_frame_calibrated_to_imu = (np.dot(velodyne_frame, R_inv4) - t4)



# def transform_velodyne(velodyne_frame, curr_e, prev_e, curr_n, prev_n, curr_yaw, prev_yaw):
#     velodyne_frame_copy = velodyne_frame.copy()
#     theta_diff = -(curr_yaw - prev_yaw)  # yaw_from_config[frame_idx - 1]
#     # theta = -yaw_noise_icp[-1]  # yaw_from_config[frame_idx - 1]
#     R = np.identity(4)
#     R[0:2, 0:2] = np.array([[np.cos(theta_diff), -np.sin(theta_diff)], [np.sin(theta_diff), np.cos(theta_diff)]]).reshape(2, 2)
#     print(R)
#     print(curr_yaw)
#     print(theta_diff)
#     # radius = np.linalg.norm([(enu_from_config[frame_idx, 0] - e_noise_icp[-1]), (enu_from_config[frame_idx, 1] - n_noise_icp[-1])])
#     e_diff = (curr_e - prev_e)
#     n_diff = (curr_n - prev_n)
#     print(e_diff)
#     print(n_diff)
#     # t = -np.dot(R, np.array([e_diff, n_diff, 0, 0])).reshape(4, 1)
#     t_x_diff = np.cos(theta_diff)*n_diff + np.sin(theta_diff)*(e_diff) #(enu_from_config[frame_idx, 0] - e_noise_icp[-1])  # enu_from_config[frame_idx - 1, 0])
#     t_y_diff = -np.sin(theta_diff)*n_diff + np.cos(theta_diff)*(e_diff)  #(enu_from_config[frame_idx, 1] - n_noise_icp[-1])  # enu_from_config[frame_idx - 1, 1])
#     # t_e_diff = enu_from_config[frame_idx, 0] - e_noise_icp[-1]
#     # t_n_diff = enu_from_config[frame_idx, 1] - n_noise_icp[-1]
#     # t = np.array([t_x_diff, -t_y_diff, 0.0, 0.0]).reshape(4, 1)
#     t = np.array([-t_x_diff, t_y_diff, 0, 0]).reshape(4, 1)
#     print(t)
#     # print("going to apply t {}".format(t))
#     # velodyne_frame = np.rot90(velodyne_raw_bev, 2)
#     velodyne_frame_copy = (np.dot(R, velodyne_frame_copy.T) + t).T
#     return velodyne_frame_copy


# redundant print
# print("x: {}..{}".format(velodyne["min_x"], velodyne["max_x"]))
# print("y: {}..{}".format(velodyne["min_y"], velodyne["max_y"]))
# print("z: {}..{}".format(velodyne["min_z"], velodyne["max_z"]))


# fig, (ax) = plt.subplots(1, 1)
# ax.plot(lon, lat, 'b')
# # ax.set_xlim(((lon[0] + lon[-1]) / 2) - 0.001, ((lon[0] + lon[-1]) / 2) + 0.001)
# # ax.set_ylim(((lat[0] + lat[-1]) / 2) - 0.001, ((lat[0] + lat[-1]) / 2) + 0.001)
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel('Longitude [degrees]', fontsize=20)
# ax.set_ylabel('Latitude [degrees]', fontsize=20)
# plt.show()