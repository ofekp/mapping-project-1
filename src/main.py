import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import random
from pathlib import Path
from datetime import datetime
# note that to install pymap3d on pycharm when using Conda, the "conda-forge" repo must be added in the settings of the interpreter
# TODO: ask if it's is ok to use this
import pymap3d as pm
import pandas as pd
import struct
import pickle
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
import multiprocessing as mp
from multiprocessing import Pool
import time
import math
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.neighbors import KDTree
# import imageio
# TODO https://github.com/panagelak/RoboND-Occupancy_Grid_Mapping-/blob/master/Inverse_Sensor_Model/solution.cpp

random.seed(11)
np.random.seed(17)

# MAIN_DATASET_PATH = "../../Data/kitti_data/2011_09_26"
# DRIVE_NUMBER = "0013"
MAIN_DATASET_PATH = "../../Data2/kitti_data/2011_09_26"
DRIVE_NUMBER = "0005"
OXTS_DATASET_PATH = MAIN_DATASET_PATH + "/2011_09_26_drive_" + DRIVE_NUMBER + "_sync/oxts"
VELODYNE_DATASET_PATH = MAIN_DATASET_PATH + "/2011_09_26_drive_" + DRIVE_NUMBER + "_sync/velodyne_points"
COLOR_IMAGE_DATASET_PATH = MAIN_DATASET_PATH + "/2011_09_26_drive_" + DRIVE_NUMBER + "_sync/image_02"
CALIB_IMU_TO_VELO_PATH = MAIN_DATASET_PATH + "/calib_imu_to_velo.txt"
RESULTS_PATH = "../../Results"
VELODYNE_HEIGHT_METERS = 1.73  # the height of the LiDAR sensor
ABOVE_GROUND_THRESHOLD = 0.3  # we ignore Velodyne sample below 30cm
MAX_RANGE_RADIUS_METERS = 30
MIN_RANGE_RADIUS_METERS = 1
MAP_RESOLUTION = 0.2
# MAP_SATURATION_MARGIN = 0.05  # clamping probabilities to the range of [0.05, 0.95]
MAP_FREE_THRESHOLD = 0.2  # threshold below which cells are considered obstacle-free.
SENSOR_ALPHA = MAP_RESOLUTION * 2  # thickness of obstacle
SENSOR_BETA_RAD = 0.002  # opening angle of the beam in radians (0.09 deg)
# SENSOR_BETA_RAD = 0.2  # opening angle of the beam in radians
MAP_MARGIN_METERS = 1  # just for aesthetics
l_0 = 0  # in our case no information means a 0.5 prior, and in log-odds it evaluates to 0


def make_gif(folder):
    with imageio.get_writer(os.path.join(folder, 'movie.gif'), mode='I', duration=0.5) as writer:
        # oxts_data_path = os.path.join(OXTS_DATASET_PATH, 'data')
        pathlist = Path(folder).glob('**/*.png')
        for path in pathlist:
            path_str = str(path)
            image = imageio.imread(path_str)
            writer.append_data(image)


def convert_lla_to_enu(lat, lon, alt):
    assert lat.shape[0] == lon.shape[0]
    assert lon.shape[0] == alt.shape[0]
    frame_count = lat.shape[0]
    lat_t0 = lat[0]
    lon_t0 = lon[0]
    alt_t0 = alt[0]
    enu = np.zeros((frame_count, 3))
    for i in range(frame_count):
        lat_curr = lat[i]
        lon_curr = lon[i]
        alt_curr = alt[i]
        enu[i, 0:3] = np.array(pm.geodetic2enu(lat_curr, lon_curr, alt_curr, lat_t0, lon_t0, alt_t0)).reshape(1, 3)
    return enu


def print_rand(log_line, prob=0.01):
    if random.uniform(0, 1) < prob:
        print(log_line)


def log_odds(x):
    return math.log(x / (1 - x))


def inverse_sensor_model_async(car_pos_x, car_pos_y, curr_cell_center_x, curr_cell_center_y, yaw, velodyne_filtered, l_occ, l_free, u, v):
    l = inverse_sensor_model(car_pos_x, car_pos_y, curr_cell_center_x, curr_cell_center_y, yaw, velodyne_filtered, l_occ, l_free)
    return u, v, l


def inverse_sensor_model(car_pos_x, car_pos_y, curr_cell_center_x, curr_cell_center_y, yaw, velodyne_filtered, l_occ, l_free):
    r = np.linalg.norm(np.array([curr_cell_center_x - car_pos_x, curr_cell_center_y - car_pos_y]))
    # phi = np.arctan2(-(curr_cell_center_y - car_pos_y), curr_cell_center_x - car_pos_x) + (yaw + (math.pi / 2))
    phi = np.arctan2(curr_cell_center_y - car_pos_y, curr_cell_center_x - car_pos_x) - yaw
    phi = phi - math.pi if phi > math.pi else phi + math.pi
    min_angle_diff = None
    min_angle_diff_sample = None
    min_angle_diff_sample_angle = None
    # indices = np.where(abs(phi - velodyne_filtered[:, 4]) <= (SENSOR_BETA_RAD / 2))
    # indices = indices[0]  # since a tuple was returned in the previous line
    indices = np.searchsorted(velodyne_filtered[:, 4], phi) + [-1, 0, 1]
    indices = indices % velodyne_filtered.shape[0]  # treat the array as circular
    for point in velodyne_filtered[indices, :]:
        sample_angle = point[4]
        if min_angle_diff is None or abs(phi - sample_angle) < min_angle_diff:
            min_angle_diff = abs(phi - sample_angle)
            min_angle_diff_sample = point
            min_angle_diff_sample_angle = sample_angle

    # if min_angle_diff is None:
    #     return l_0

    Zk = np.linalg.norm(min_angle_diff_sample[0:2])
    if (r > min(MAX_RANGE_RADIUS_METERS, Zk + SENSOR_ALPHA / 2)) or abs(phi - min_angle_diff_sample_angle) > SENSOR_BETA_RAD / 2 or Zk > MAX_RANGE_RADIUS_METERS or Zk < MIN_RANGE_RADIUS_METERS:
        return l_0
    elif Zk < MAX_RANGE_RADIUS_METERS and abs(r - Zk) < SENSOR_ALPHA / 2:
        return l_occ
    elif r <= Zk:
        return l_free

    assert False  # should not reach here


def filter_velodyne_data(velodyne_raw):
    """
    to ease the computation we first clean up non interesting points
      - points that are further than MAX_RANGE_RADIUS_METERS from the sensor
      - points that are below 30cm off the ground
    """
    velodyne_filtered = np.array([]).reshape(0, 4)
    for point in velodyne_raw:
        # only considering the x and y (ignoring the z value)
        dist = np.linalg.norm(point[0:2])
        if dist > MAX_RANGE_RADIUS_METERS or dist < MIN_RANGE_RADIUS_METERS:
            continue
        height_val = point[2]
        # we're looking for points that are 30cm above the ground, keeping in mind that the sensor is positioned
        # on top of the car's roof at a height of VELODYNE_HEIGHT_METERS
        if height_val <= (ABOVE_GROUND_THRESHOLD - VELODYNE_HEIGHT_METERS):
            continue
        velodyne_filtered = np.concatenate((velodyne_filtered, point.reshape(1, 4)), axis=0)
    return velodyne_filtered


def find_center_of_mass_xy(X):
    """
    :param X: of shape NxK where N is the number of samples and K is the dimension of the data
    :return: the mean of X of shape 1xK
    """
    return np.concatenate((X[:, 0:2].mean(axis=0), np.zeros(X.shape[1] - 2)))
    # return X[:, 0:2].mean(axis=0)


def cross_cov_xy(P, Q):
    assert P.shape[0] > P.shape[1]
    assert Q.shape[0] > Q.shape[1]
    W = np.zeros((3,3))
    for i in range(P.shape[0]):
        mat = np.dot(Q[i, 0:3].reshape(3, 1), P[i, 0:3].reshape(1, 3)) # * (2 + Q[i, 2] if Q[i, 2] > -0.75 and P[i, 2] > -0.75 else 1)
        assert mat.shape[0] == 3 and mat.shape[1] == 3
        W += mat
    return W


def update_occupancy_map(velodyne_frame_filtered, velodyne_frame_prev_filtered, frame_idx, occupancy_map, car_pos_x, car_pos_y, e, n, yaw, l_occ, l_free, is_async=True):
    velodyne_frame_filtered_copy = velodyne_frame_filtered.copy()

    e_icp_diff = 0
    n_icp_diff = 0
    u_icp_diff = 0
    e_orig = e
    n_orig = n
    yaw_icp_diff = 0
    if velodyne_frame_prev_filtered is not None:
        velodyne_frame_prev_filtered_copy = velodyne_frame_prev_filtered.copy()

        # extreme rotation and translation error
        # trans = np.identity(4)
        # trans[0:2, 0:2] = np.array([[0.9950042, -0.0998334], [0.0998334,  0.9950042]])
        # velodyne_frame_filtered_copy = (np.dot(trans, velodyne_frame_filtered_copy.T) + np.array([-0.5, -0.4, 0, 0]).reshape(4, 1)).T
        # trans_inv = np.linalg.inv(trans)

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        #
        # _, bev_Q, _ = process_frame(velodyne_raw_prev, frame_idx - 1)
        # _, bev_P, _ = process_frame(velodyne_raw, frame_idx)
        #
        # map_size = int((2 * MAX_RANGE_RADIUS_METERS) / MAP_RESOLUTION)
        # ax1.set_xticks([0, (map_size // 2), map_size - 1])
        # ax1.set_xticklabels([-MAX_RANGE_RADIUS_METERS, 0.0, MAX_RANGE_RADIUS_METERS], fontsize=20)
        # ax1.set_yticks([0, (map_size // 2), map_size - 1])
        # ax1.set_yticklabels([MAX_RANGE_RADIUS_METERS, 0.0, -MAX_RANGE_RADIUS_METERS], fontsize=20)
        # ax1.set_xlabel('X [meters]', fontsize=20)
        # ax1.set_ylabel('Y [meters]', fontsize=20)
        # ax1.set_title('Instantaneous Point Cloud')
        # ax1.imshow(bev_Q, vmin=0.0, vmax=1.0)
        #
        # ax2.set_xticks([0, (map_size // 2), map_size - 1])
        # ax2.set_xticklabels([-MAX_RANGE_RADIUS_METERS, 0.0, MAX_RANGE_RADIUS_METERS], fontsize=20)
        # ax2.set_yticks([0, (map_size // 2), map_size - 1])
        # ax2.set_yticklabels([MAX_RANGE_RADIUS_METERS, 0.0, -MAX_RANGE_RADIUS_METERS], fontsize=20)
        # ax2.set_xlabel('X [meters]', fontsize=20)
        # ax2.set_ylabel('Y [meters]', fontsize=20)
        # ax2.set_title('Instantaneous Point Cloud')
        # ax2.imshow(bev_P, vmin=0.0, vmax=1.0)
        #
        # plt.show()

        # fig, (ax1) = plt.subplots(1, 1)
        # _, bev_Q, _ = process_frame(Q, frame_idx - 1)
        # map_size = int((2 * MAX_RANGE_RADIUS_METERS) / MAP_RESOLUTION)
        # ax1.set_xticks([0, (map_size // 2), map_size - 1])
        # ax1.set_xticklabels([-MAX_RANGE_RADIUS_METERS, 0.0, MAX_RANGE_RADIUS_METERS], fontsize=20)
        # ax1.set_yticks([0, (map_size // 2), map_size - 1])
        # ax1.set_yticklabels([MAX_RANGE_RADIUS_METERS, 0.0, -MAX_RANGE_RADIUS_METERS], fontsize=20)
        # ax1.set_xlabel('X [meters]', fontsize=20)
        # ax1.set_ylabel('Y [meters]', fontsize=20)
        # ax1.set_title('Instantaneous Point Cloud')
        # ax1.imshow(bev_Q, vmin=0.0, vmax=1.0)
        # plt.show()

        # we are looking for R and t to apply to P such that it will best fit Q
        iterations = 40
        max_dist = 2
        Q = velodyne_frame_prev_filtered_copy[:, 0:3].copy()
        # Q = Q[:, [1,0,2]]  # velodyne_frame_prev_filtered_copy[1] is east due to velodyne's local axis (y point to the right, x points down)
        Q = Q[np.linalg.norm(Q[:, 0:2], axis=1) > 1.5]
        Q = Q[Q[:, 2] > -0.3]
        # Q = Q[Q[:, 2] < -0.3]
        # idx = ((Q[:, 1] > -30) & (Q[:, 1] < 20)) & (((Q[:, 0] > -20) & (Q[:, 0] < -5)) | (Q[:, 0] > 5) & (Q[:, 0] < 20))
        # idx = ((Q[:, 1] < 30) & (Q[:, 1] > 5))
        # idx = ((Q[:, 0] > -20) & (Q[:, 0] < -5))
        # Q = Q[idx]
        # Q = Q[Q[:, 0] > -20]
        # Q = Q[Q[:, 0] < -5]
        # Q = Q[Q[:, 1] < 20]
        # Q = Q[Q[:, 1] > -30]
        Q_center = find_center_of_mass_xy(Q).reshape(1, 3)
        Q_centered = Q - Q_center
        tree = KDTree(Q_centered)
        P = velodyne_frame_filtered_copy[:, 0:3].copy()
        # P = P[:, [1, 0, 2]]
        P = P[np.linalg.norm(P[:, 0:2], axis=1) > 1.5]
        P = P[P[:, 2] > -0.3]
        # P = P[P[:, 2] < -0.3]
        # idx = ((P[:, 1] > -30) & (P[:, 1] < 20)) & (((P[:, 0] > -20) & (P[:, 0] < -5)) | (P[:, 0] > 5) & (P[:, 0] < 20))
        # idx = ((P[:, 1] < 30) & (P[:, 1] > 5))
        # idx = ((P[:, 0] > -20) & (P[:, 0] < -5))
        # P = P[idx]
        # P = P[P[:, 0] > -20]
        # P = P[P[:, 0] < -5]
        # P = P[P[:, 1] < 20]
        # P = P[P[:, 1] > -30]
        # random_indices = np.random.choice(P.shape[0], size=1000, replace=False)
        # P = P[random_indices, :]
        tt = np.zeros(3).reshape(3, 1)
        RR = np.identity(3).reshape(3, 3)
        P_center = None
        P_centered = None
        t = None
        Pt = P.copy()
        norm_values = []
        for i in range(iterations):
        # while len(norm_values) == 0 or np.linalg.norm(P_centered[0:2000, :] - Q_centered[0:2000, :]) > 50:
            P_center = find_center_of_mass_xy(P).reshape(1, 3)
            P_centered = P - P_center

            # pick the best points
            dist_, ind_ = tree.query(P_centered, k=1)
            dist, ind = dist_.squeeze(), ind_.squeeze()
            # consider only close pairs
            ind = ind[dist < max_dist]
            P_matching = P_centered[dist < max_dist, :].copy()
            Q_matching = Q_centered[ind, :].copy()
            assert P_matching.shape == Q_matching.shape

            norm_values.append(np.linalg.norm(P_centered[0:1000, :] - Q_centered[0:1000, :]))

            W = cross_cov_xy(P_matching, Q_matching)
            assert W.shape == (3, 3)
            # W = np.dot(np.transpose(P_matching), Q_matching)
            U, S, V_T = np.linalg.svd(W)
            R = np.dot(U, V_T)
            assert R.shape == (3, 3)
            # iden = np.identity(3)
            # iden[0:2, 0:2] = R
            # R = iden
            t = Q_center.reshape(3, 1) - np.dot(R, P_center.reshape(3, 1))
            # ttt = np.dot(R, np.array([e, n, 0]).reshape((3, 1))) + t.reshape(3, 1)
            # e = float(ttt[0])
            # n = float(ttt[1])
            if i == 0:
                tt = t
                RR = R
            else:
                # tt = np.dot(R, tt) + t
                # RR = np.dot(R, RR)
                tt += t
                RR = np.dot(R, RR)
            # print(t)
            # print('----')
            P = np.dot(R, np.transpose(P)) + t.reshape(3, 1)
            P = np.transpose(P)

        dummy_vec = np.transpose(np.array([1, 0, 0]))
        dummy_vec_rotated = np.dot(RR, dummy_vec)
        yaw += np.arctan2(dummy_vec_rotated[1], dummy_vec_rotated[0])
        yaw_icp_diff += np.arctan2(dummy_vec_rotated[1], dummy_vec_rotated[0])

        # diff = np.dot(R - np.identity(3), np.array([car_pos_x, car_pos_y, 0.0]).reshape(3, 1)) + t
        # e_icp_diff = Q_center[0, 0] - (P_center[0, 0] + tt[0])
        # n_icp_diff = Q_center[0, 1] - (P_center[0, 1] + tt[1])
        e_icp_diff = tt[1] #+ 0.2069537  # velodyne_frame_prev_filtered_copy[1] is east due to velodyne's local axis (y point to the right, x points down)
        n_icp_diff = tt[0] #+ 0.35403833  # same here for north values
        u_icp_diff = tt[2]  # same here for north values
        # print("x_diff: {} y_diff {}".format(e_icp_diff, n_icp_diff))

        print("start norm [{}] end norm [{}]".format(norm_values[0], norm_values[-1]))

        print("frame [{}] after ICP: e_icp_diff {} n_icp_diff {} u_icp_diff {} yaw_icp_diff {}".format(frame_idx, e_icp_diff, n_icp_diff, u_icp_diff, yaw_icp_diff))

        if frame_idx > 0 and frame_idx % 5 == 0:
            _, velodyne_raw_bev_P, _ = process_frame(P, frame_idx - 1)
            _, velodyne_raw_bev_Q, _ = process_frame(Q, frame_idx)
            _, velodyne_filtered_prev_bev_P, _ = process_frame(velodyne_frame_filtered_copy, frame_idx - 1)
            _, velodyne_filtered_bev_Q, _ = process_frame(velodyne_frame_prev_filtered_copy, frame_idx)
            fig, (ax1, ax2) = plt.subplots(1, 2)

            map_size = int((2 * MAX_RANGE_RADIUS_METERS) / MAP_RESOLUTION)
            ax1.set_xticks([0, (map_size // 2), map_size - 1])
            ax1.set_xticklabels([-MAX_RANGE_RADIUS_METERS, 0.0, MAX_RANGE_RADIUS_METERS], fontsize=20)
            ax1.set_yticks([0, (map_size // 2), map_size - 1])
            ax1.set_yticklabels([MAX_RANGE_RADIUS_METERS, 0.0, -MAX_RANGE_RADIUS_METERS], fontsize=20)
            ax1.set_xlabel('X [meters]', fontsize=20)
            ax1.set_ylabel('Y [meters]', fontsize=20)
            ax1.set_title('Instantaneous Point Cloud')
            ax1.imshow(velodyne_filtered_bev_Q + velodyne_filtered_prev_bev_P, vmin=0.0, vmax=1.0)

            ax2.set_xticks([0, (map_size // 2), map_size - 1])
            ax2.set_xticklabels([-MAX_RANGE_RADIUS_METERS, 0.0, MAX_RANGE_RADIUS_METERS], fontsize=20)
            ax2.set_yticks([0, (map_size // 2), map_size - 1])
            ax2.set_yticklabels([MAX_RANGE_RADIUS_METERS, 0.0, -MAX_RANGE_RADIUS_METERS], fontsize=20)
            ax2.set_xlabel('X [meters]', fontsize=20)
            ax2.set_ylabel('Y [meters]', fontsize=20)
            ax2.set_title('Instantaneous Point Cloud')
            ax2.imshow(velodyne_raw_bev_Q + velodyne_raw_bev_P, vmin=0.0, vmax=1.0)

            plt.show()

        car_pos_x += e_icp_diff
        car_pos_y += n_icp_diff

    return e_icp_diff, n_icp_diff, yaw_icp_diff

    velodyne_filtered = np.concatenate((velodyne_filtered, np.arctan2(velodyne_filtered[:, 0], velodyne_filtered[:, 1]).reshape(velodyne_filtered.shape[0], 1)), axis=1)
    # velodyne_filtered = np.concatenate((velodyne_filtered, np.arctan2(-velodyne_filtered[:, 1], velodyne_filtered[:, 0]).reshape(velodyne_filtered.shape[0], 1)), axis=1)
    velodyne_filtered = velodyne_filtered[velodyne_filtered[:, 4].argsort()]
    max_range_uv = int(MAX_RANGE_RADIUS_METERS // MAP_RESOLUTION)
    car_pos_u = int(car_pos_x // MAP_RESOLUTION)
    car_pos_v = int(car_pos_y // MAP_RESOLUTION)
    u_range = range(int(car_pos_u - max_range_uv // 2), int(car_pos_u + max_range_uv // 2))
    v_range = range(int(car_pos_v - max_range_uv // 2), int(car_pos_v + max_range_uv // 2))

    p = Pool(mp.cpu_count() - 1)
    job_args = []

    for u in u_range:
        for v in v_range:
            # print_rand("Progress [{}/{}]".format(count, len(u_range) * len(v_range)), prob=0.001)
            curr_cell_center_x = u * MAP_RESOLUTION + MAP_RESOLUTION / 2
            curr_cell_center_y = v * MAP_RESOLUTION + MAP_RESOLUTION / 2
            if np.linalg.norm(np.array([curr_cell_center_x - car_pos_x, curr_cell_center_y - car_pos_y])) <= MAX_RANGE_RADIUS_METERS:
                if is_async:
                    job_args.append((car_pos_x, car_pos_y, curr_cell_center_x, curr_cell_center_y, yaw, velodyne_filtered, l_occ, l_free, u, v))
                else:
                    occupancy_map[v, u] += inverse_sensor_model(car_pos_x, car_pos_y, curr_cell_center_x, curr_cell_center_y, yaw, velodyne_filtered, l_occ, l_free) - l_0
    for u, v, l in p.starmap(inverse_sensor_model_async, job_args):
        occupancy_map[v, u] += l - l_0

    p.close()


def render_frame(img, velodyne_raw_bev, occupancy, frame_idx, folder):
    fig, ((axtop1, axtop2), (ax1, ax2)) = plt.subplots(2, 2)
    fig.subplots_adjust(
        top=0.978,
        bottom=0.075,
        left=0.009,
        right=0.991,
        hspace=0.112,
        wspace=0.018
    )
    gs = axtop1.get_gridspec()
    axtop1.remove()
    axtop2.remove()
    axbig = fig.add_subplot(gs[0, 0:2])
    axbig.set_axis_off()
    axbig.imshow(img)

    map_size = int((2 * MAX_RANGE_RADIUS_METERS) / MAP_RESOLUTION)

    ax1.set_xticks([0, (map_size // 2), map_size - 1])
    ax1.set_xticklabels([-MAX_RANGE_RADIUS_METERS, 0.0, MAX_RANGE_RADIUS_METERS], fontsize=20)
    ax1.set_yticks([0, (map_size // 2), map_size - 1])
    ax1.set_yticklabels([MAX_RANGE_RADIUS_METERS, 0.0, -MAX_RANGE_RADIUS_METERS], fontsize=20)
    ax1.set_xlabel('X [meters]', fontsize=20)
    ax1.set_ylabel('Y [meters]', fontsize=20)
    ax1.set_title('Instantaneous Point Cloud')
    ax1.imshow(velodyne_raw_bev, vmin=0.0, vmax=1.0)

    ax2.set_xticks([0, (map_size // 2), map_size - 1])
    ax2.set_xticklabels([-MAX_RANGE_RADIUS_METERS, 0.0, MAX_RANGE_RADIUS_METERS], fontsize=20)
    ax2.set_yticks([0, (map_size // 2), map_size - 1])
    ax2.set_yticklabels([MAX_RANGE_RADIUS_METERS, 0.0, -MAX_RANGE_RADIUS_METERS], fontsize=20)
    ax2.set_xlabel('X [meters]', fontsize=20)
    ax2.set_ylabel('Y [meters]', fontsize=20)
    ax2.set_title('Occupancy Map')
    cmap = ListedColormap(['white', 'black', 'lightgray', 'red'], N=4)
    # cmap = { 0: 'white', 1: 'black', 2: 'lightgray', 3: 'red'}
    # cmap_binary = ListedColormap(['b', 'w', 'r'], N=3)
    # u = np.unique(occupancy, return_inverse=True)
    # print(u)
    # norm = colors.BoundaryNorm(np.arange(len(u) + 1) - .5, len(u))
    # ax2.imshow(occupancy, interpolation='nearest', cmap=cmap, norm=norm)
    ax2.imshow(occupancy, interpolation='nearest', cmap=cmap)
    # plt.show()
    fig.set_size_inches(15, 15)

    ram = io.BytesIO()
    plt.savefig(ram, format='png', dpi=100)
    ram.seek(0)
    im = Image.open(ram)
    im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
    im2.save("../../../Results/{}/{:04d}.png".format(folder, frame_idx), format='PNG')

    # fig.savefig("../../../Results/{}/{:04d}.png".format(folder, frame_idx), dpi=100)
    plt.close(fig)
    plt.close('all')
    plt.cla()
    plt.clf()


def process_frame(velodyne_frame, frame_idx):
    velodyne_frame_copy = velodyne_frame.copy()
    map_size = int((2 * MAX_RANGE_RADIUS_METERS) // MAP_RESOLUTION)
    velodyne_frame_copy[:, 0] += MAX_RANGE_RADIUS_METERS
    velodyne_frame_copy[:, 0] //= MAP_RESOLUTION
    velodyne_frame_copy[:, 1] += MAX_RANGE_RADIUS_METERS
    velodyne_frame_copy[:, 1] //= MAP_RESOLUTION
    velodyne_raw_bev = np.zeros((map_size, map_size))
    min_z = np.min(velodyne_frame_copy[:, 2])
    max_z = np.max(velodyne_frame_copy[:, 2])
    assert (max_z - min_z) > 0
    for point in velodyne_frame_copy:
        if point[0] >= map_size or point[0] < 0 or point[1] >= map_size or point[1] < 0:
            continue
        height_val = (point[2] - min_z) / (max_z - min_z)
        velodyne_raw_bev[int(point[0]), int(point[1])] = max(velodyne_raw_bev[int(point[0]), int(point[1])], height_val)
    velodyne_raw_bev = np.rot90(velodyne_raw_bev, 2)

    occupancy = np.zeros((map_size, map_size), dtype=int)
    for point in velodyne_frame_copy:
        if point[0] >= map_size or point[0] < 0 or point[1] >= map_size or point[1] < 0:
            continue
        height_val = point[2]
        # we're looking for points that are 30cm above the ground, keeping in mind that the sensor is positioned
        # on top of the car's roof at a height of VELODYNE_HEIGHT_METERS
        if height_val > (ABOVE_GROUND_THRESHOLD - VELODYNE_HEIGHT_METERS):
            occupancy[int(point[0]), int(point[1])] = 1
    car_center = int(MAX_RANGE_RADIUS_METERS // MAP_RESOLUTION)
    # I am taking into account the position of the sensor in relation to the car
    occupancy[(car_center - 4):(car_center - 4 + 14), (car_center - 4):(car_center + 4)] = 2
    occupancy = np.flip(occupancy, 1)  # Velodyne's y axis is flipped

    img_path = os.path.join(COLOR_IMAGE_DATASET_PATH, 'data', "{:010d}.png".format(frame_idx))
    img = mpimg.imread(img_path)

    return img, velodyne_raw_bev, occupancy


def load_velodyne():
    velodyne_data_path = os.path.join(VELODYNE_DATASET_PATH, 'data')
    assert os.path.isdir(velodyne_data_path)
    pathlist = Path(velodyne_data_path).glob('**/*.bin')
    velodyne = {
        "frames_raw": [],
        "min_x": float("inf"),
        "max_x": -float("inf"),
        "min_y": float("inf"),
        "max_y": -float("inf"),
        "min_z": float("inf"),
        "max_z": -float("inf"),
    }
    for path in pathlist:
        path_str = str(path)
        print(path_str)
        frame_idx = int(os.path.basename(path_str).split('.')[0])
        vals = []
        with open(path_str, 'rb') as frame_data_file:
            # in Velodyne binary files every 4 bytes is a float, each set of 4 floats represent (x, y, z, reflectance)
            bytes = frame_data_file.read(4 * 4)
            while bytes:
                # convert each set of 4 bytes to float
                (x,) = struct.unpack('f', bytes[0:4])
                (y,) = struct.unpack('f', bytes[4:8])
                (z,) = struct.unpack('f', bytes[8:12])
                (r,) = struct.unpack('f', bytes[12:16])
                velodyne["min_x"] = min(velodyne["min_x"], x)
                velodyne["max_x"] = max(velodyne["max_x"], x)
                velodyne["min_y"] = min(velodyne["min_y"], y)
                velodyne["max_y"] = max(velodyne["max_y"], y)
                velodyne["min_z"] = min(velodyne["min_z"], z)
                velodyne["max_z"] = max(velodyne["max_z"], z)
                vals.append([x, y, z, r])
                bytes = frame_data_file.read(4 * 4)
        velodyne["frames_raw"].append(np.array(vals).reshape(-1, 4))
    return velodyne


def convert_time_to_epoch(datetime_string):
    """
    convert time from the following format to epoch in millis
    2011-09-26 13:10:51.175862102
    :return:
    """
    without_fraction = datetime.strptime(datetime_string.split('.')[0], "%Y-%m-%d %H:%M:%S").timestamp()
    with_fraction = float(without_fraction) + float('0.' + datetime_string.split('.')[1])
    return with_fraction


def count_frames_in_dataset():
    """
    :return: the number of frames in the dataset as and integer
    """
    oxts_data_path = os.path.join(OXTS_DATASET_PATH, 'data')
    file_list = os.listdir(oxts_data_path)
    return len(file_list)


def load_oxts():
    """
    load latitude, longitude, altitude, roll, pitch, yaw, num_of_satellites, navigation_status_code, vel_accuracy, pos_accuracy
    note that:
      roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
      pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
      yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
    return:
        numpy matrix with the relevant data as columns
    """
    oxts_data_path = os.path.join(OXTS_DATASET_PATH, 'data')
    assert os.path.isdir(oxts_data_path)
    pathlist = Path(oxts_data_path).glob('**/*.txt')
    frame_count = count_frames_in_dataset()
    # timestamp, lat, lon, alt, rol, pit, yaw
    res = np.zeros((frame_count, 11))
    for path in pathlist:
        path_str = str(path)
        frame_idx = int(os.path.basename(path_str).split('.')[0])
        with open(path_str, 'r') as frame_data_file:
            lines = frame_data_file.readlines()
            data_split = lines[0].split(' ')
            res[frame_idx][1:7] = [float(val.strip()) for val in data_split[0:6]]
            res[frame_idx][7:9] = [float(val.strip()) for val in data_split[23:25]]
            res[frame_idx][9:11] = [float(val.strip()) for val in data_split[25:27]]
        print(res[frame_idx])

    # add timestamps to the data
    timestamps_file_path = os.path.join(OXTS_DATASET_PATH, 'timestamps.txt')
    with open(timestamps_file_path, 'r') as frame_data_file:
        lines = frame_data_file.readlines()
        for i, line in enumerate(lines):
            res[i][0] = convert_time_to_epoch(line)

    return res


# def plot_graph():
#     plt.plot(range(1, n + 1), error_graph_train, color='blue', marker='o', linestyle='solid', linewidth=1, markersize=6)
#     plt.plot(range(1, n + 1), error_graph_test, color='green', marker='o', linestyle='solid', linewidth = 1, markersize = 6)
#     plt.xlabel('Iteration #')
#     plt.ylabel('Error [%]')
#     plt.legend(["Train", "Test"], prop={"size":20}, loc="upper right")
#     plt.show()


def main():
    np.set_printoptions(precision=9, formatter={'all': lambda x: str(x)}, threshold=100000, linewidth=1000)
    font = {'size': 20}
    plt.rc('font', **font)

    frame_count = count_frames_in_dataset()
    oxts = load_oxts()
    print("oxts at start point {}".format(oxts[0, :]))
    print("oxts at end point   {}".format(oxts[143, :]))

    # world coord system - plotting lat, long and alt against the frames
    print("LLA Graph")
    lat = oxts[0:frame_count, 1].reshape(-1, 1)
    lon = oxts[0:frame_count, 2].reshape(-1, 1)
    alt = oxts[0:frame_count, 3].reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(range(0, frame_count), lat, color='blue', linestyle='solid', markersize=1)
    ax1.plot(range(0, frame_count), lon, color='green', linestyle='solid', markersize=1)
    ax1.set_xlabel('Frame Index', fontsize=20)
    ax1.set_ylabel('LLA', fontsize=20)
    ax1.legend(["Latitude [deg]", "Longitude [deg]"], prop={"size":20}, loc="best")

    ax2.plot(range(0, frame_count), alt, color='red', linestyle='solid', markersize=1)
    ax2.set_xlabel('Frame Index', fontsize=20)
    ax2.set_ylabel('Altitude [meters]', fontsize=20)
    plt.show()

    fig, (ax) = plt.subplots(1, 1)
    ax.plot(lon, lat, 'b')
    # ax.set_xlim(((lon[0] + lon[-1]) / 2) - 0.001, ((lon[0] + lon[-1]) / 2) + 0.001)
    # ax.set_ylim(((lat[0] + lat[-1]) / 2) - 0.001, ((lat[0] + lat[-1]) / 2) + 0.001)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude [degrees]', fontsize=20)
    ax.set_ylabel('Latitude [degrees]', fontsize=20)
    plt.show()

    # local coord system - plotting ENU graph and pitch/yaw/roll graph
    print("ENU Graph")
    enu = convert_lla_to_enu(lat, lon, alt)
    with open(CALIB_IMU_TO_VELO_PATH, 'r') as calib_imu_to_velo_file:
        lines = calib_imu_to_velo_file.readlines()
        R = np.array([float(a) for a in lines[1].strip().replace("R: ", "").split(" ")]).reshape(3, 3)
        t = np.array([float(a) for a in lines[2].strip().replace("T: ", "").split(" ")]).reshape(3, 1)
        enu_corrected = (np.dot(R, enu.T) + t).T
    # enu = enu_corrected

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(range(0, frame_count), enu[:, 0], color='blue', linestyle='solid', markersize=1)
    ax1.plot(range(0, frame_count), enu[:, 1], color='green', linestyle='solid', markersize=1)
    ax1.set_xlabel('Frame Index', fontsize=20)
    ax1.set_ylabel('Difference from reference point at t0', fontsize=20)
    ax1.legend(["East [meters]", "North [meters]"], prop={"size":20}, loc="best")

    ax2.plot(range(0, frame_count), enu[:, 2], color='red', linestyle='solid', markersize=1)
    ax2.set_xlabel('Frame Index', fontsize=20)
    ax2.set_ylabel('Difference from reference point at t0 - Up [meters]', fontsize=20)

    ax3.plot(range(0, frame_count), oxts[:, 4], color='blue', linestyle='solid', markersize=1)
    ax3.plot(range(0, frame_count), oxts[:, 5], color='green', linestyle='solid', markersize=1)
    ax3.plot(range(0, frame_count), oxts[:, 6], color='red', linestyle='solid', markersize=1)
    ax3.set_xlabel('Frame Index', fontsize=20)
    ax3.set_ylabel('Roll, Pitch and Yaw in ENU coord system', fontsize=20)
    ax3.legend(["Roll [rad]", "Pitch [rad]", "Yaw [rad]"], prop={"size":20}, loc="best")
    plt.show()

    fig, (ax) = plt.subplots(1, 1)
    ax.plot(enu[:, 0], enu[:, 1], 'b')
    # ylim = ax.get_ylim()
    # ax.set_xlim(ylim[0] - 100, ylim[1] - 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('East [meters]', fontsize=20)
    ax.set_ylabel('North [meters]', fontsize=20)
    plt.show()

    # local coord system - plotting NED and pitch/yaw/roll graph
    print("NED Graph")
    ned = np.zeros((frame_count, 3))
    ned[:, 0] = enu[:, 1]
    ned[:, 1] = enu[:, 0]
    ned[:, 2] = -enu[:, 2]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(range(0, frame_count), ned[:, 0], color='blue', linestyle='solid', markersize=1)
    ax1.plot(range(0, frame_count), ned[:, 1], color='green', linestyle='solid', markersize=1)
    ax1.set_xlabel('Frame Index', fontsize=20)
    ax1.set_ylabel('Difference from reference point at t0', fontsize=20)
    ax1.legend(["North [meters]", "East [meters]"], prop={"size":20}, loc="best")

    ax2.plot(range(0, frame_count), ned[:, 2], color='red', linestyle='solid', markersize=1)
    ax2.set_xlabel('Frame Index', fontsize=20)
    ax2.set_ylabel('Difference from reference point at t0 - Down [meters]', fontsize=20)

    # convert roll pitch and yaw from ENU to NED
    rpy_enu = np.zeros((frame_count, 3))
    rpy_enu[:, :] = oxts[:, [4, 5, 6]]
    # transform = np.array([
    #     [1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, -1]
    # ])
    # rpy_in_ned = np.matmul(transform, np.transpose(rpy_enu))
    # rpy_in_ned = np.transpose(rpy_in_ned)
    rpy_in_ned = rpy_enu
    rpy_in_ned[:, 2] = -rpy_in_ned[:, 2] + math.pi/2
    ax3.plot(range(0, frame_count), rpy_in_ned[:, 0], color='blue', linestyle='solid', markersize=1)
    ax3.plot(range(0, frame_count), rpy_in_ned[:, 1], color='green', linestyle='solid', markersize=1)
    ax3.plot(range(0, frame_count), rpy_in_ned[:, 2], color='red', linestyle='solid', markersize=1)
    ax3.set_xlabel('Frame Index', fontsize=20)
    ax3.set_ylabel('Roll, Pitch and Yaw in NED coord system', fontsize=20)
    ax3.legend(["Roll [rad]", "Pitch [rad]", "Yaw [rad]"], prop={"size":20}, loc="best")
    plt.show()

    print("GPS Quality")
    pos_accuracy = oxts[:, 7]
    vel_accuracy = oxts[:, 8]
    nav_status_code = oxts[:, 9]
    num_of_satellites = oxts[:, 10]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(range(0, frame_count), num_of_satellites, color='blue', linestyle='solid', markersize=1)
    ax1.plot(range(0, frame_count), nav_status_code, color='red', linestyle='solid', markersize=1)
    ax1.set_xlabel('Frame Index', fontsize=20)
    ax1.set_ylabel('# Connected Satellites & Navigation Status Code', fontsize=20)
    ax1.legend(["# Connected Satellites", "Navigation Status Code"], prop={"size":20}, loc="best")

    ax2.plot(range(0, frame_count), vel_accuracy, color='blue', linestyle='solid', markersize=1)
    ax2.plot(range(0, frame_count), pos_accuracy, color='red', linestyle='solid', markersize=1)
    ax2.set_xlabel('Frame Index', fontsize=20)
    ax2.set_ylabel('# Connected Satellites & Navigation Status Code', fontsize=20)
    ax2.legend(["# Velocity Accuracy [north/east in m/s]", "Position Accuracy [north/east in m]"], prop={"size":20}, loc="best")
    plt.show()

    print("Creating csv file for Google My Maps")
    long_lat_path = os.path.join(RESULTS_PATH, "long_lat.csv")
    df = pd.DataFrame(data=oxts[:, [1, 2]], columns=["lat", "long"])
    df.to_csv(long_lat_path, index=True)

    # Probabilistic Occupancy Grid
    print("Probabilistic Occupancy Grid")
    # load Velodine data for all the frames, I'm using pickle, if we already all the binary files parsed as a
    # pickle file, loading the Velodine data will be much faster
    velodyne_file_path = os.path.join(VELODYNE_DATASET_PATH, 'velodyne.pickle')
    if os.path.isfile(velodyne_file_path):
        velodyne_file = open(velodyne_file_path, 'rb')
        velodyne = pickle.load(velodyne_file)
        velodyne_file.close()
    else:
        velodyne = load_velodyne()
        velodyne_file = open(velodyne_file_path, 'wb')
        pickle.dump(velodyne, velodyne_file)
        velodyne_file.close()

    # TODO: this is a redundant print, should probably remove it
    print("x: {}..{}".format(velodyne["min_x"], velodyne["max_x"]))
    print("y: {}..{}".format(velodyne["min_y"], velodyne["max_y"]))
    print("z: {}..{}".format(velodyne["min_z"], velodyne["max_z"]))

    # calculate the size of the complete map, I'm ignoring the altitude here
    # notice that (x,y) and (e,n) are in meters while (u,v) are the integer coordinates of the occupancy map

    enu_min_e = np.min(enu[:, 0])
    enu_min_n = np.min(enu[:, 1])
    enu_max_e = np.max(enu[:, 0])
    enu_max_n = np.max(enu[:, 1])
    print("Min east point [{}] min north point [{}]".format(enu_min_e, enu_min_n))
    print("Max east point [{}] max north point [{}]".format(enu_max_e, enu_max_n))
    map_size_meters_e = enu_max_e - enu_min_e + 2 * (MAX_RANGE_RADIUS_METERS + MAP_MARGIN_METERS)
    map_size_meters_n = enu_max_n - enu_min_n + 2 * (MAX_RANGE_RADIUS_METERS + MAP_MARGIN_METERS)
    # moving to matrix coords, where x is from left to right and y is from top to bottom
    start_point_x = MAP_MARGIN_METERS + MAX_RANGE_RADIUS_METERS + (abs(enu_min_e) if enu_min_e < 0 else 0)
    start_point_y = map_size_meters_n - (MAX_RANGE_RADIUS_METERS + MAP_MARGIN_METERS + (abs(enu_min_n) if enu_min_n < 0 else 0))
    print("Start point position in meters ({}, {})".format(start_point_x, start_point_y))
    map_size_u = int(map_size_meters_e // MAP_RESOLUTION)
    map_size_v = int(map_size_meters_n // MAP_RESOLUTION)
    print("Occupancy map size is ({}, {})".format(map_size_u, map_size_v))
    print("Occupancy map size in meters is ({}, {})".format(map_size_meters_e, map_size_meters_n))
    yaw = rpy_enu[:, 2]  # for all frames

    # setup noisy samples for section 3
    enu_noise = enu + np.concatenate((np.random.normal(0, 0.5, (enu.shape[0], 2)), np.zeros((enu.shape[0], 1))), axis=1)
    # enu_noise = enu.copy() # + np.concatenate(( np.zeros((enu.shape[0], 1)), 0.5 * np.ones((enu.shape[0], 1)), np.zeros((enu.shape[0], 1)) ), axis=1)
    yaw_noise = yaw + np.random.normal(0, 0.01, yaw.shape[0])
    # yaw_noise = yaw.copy()
    # enu_noise[1, 1] += 1.5
    # enu_noise[2, 1] += 0.5
    # print("enu_noise:")
    # print(enu_noise[0:3, :])
    enu_noise[0, 0:2] = enu[0, 0:2]

    print(enu_noise[0:5, :])
    print(yaw_noise[0:5])

    print("Iterative Closest Point (ICP)")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(enu[:, 0], enu[:, 1], color='b', linestyle='solid', markersize=1)
    ax1.plot(enu_noise[:, 0], enu_noise[:, 1], color='r', linestyle='solid', markersize=1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('Longitude [degrees]', fontsize=20)
    ax1.set_ylabel('Latitude [degrees]', fontsize=20)
    ax1.legend(["Raw lon/lat samples", "Samples with Gaussian noise sigma=0.000005 deg"], prop={"size": 20}, loc="best")

    ax2.plot(range(0, frame_count), yaw, color='b', linestyle='solid', markersize=1)
    ax2.plot(range(0, frame_count), yaw_noise, color='r', linestyle='solid', markersize=1)
    ax2.set_xlabel('Yaw [rad]', fontsize=20)
    ax2.set_ylabel('Frame Index', fontsize=20)
    ax2.legend(["Raw yaw samples", "yaw with Gaussian noise, sigma=0.01 rad"], prop={"size": 20}, loc="best")
    plt.show()

    configs = [
        # {
        #     "folder": "occupancy_map_1_hit_0.7_miss_0.4_occ_threshold_0.8",
        #     "hit": 0.7,
        #     "miss": 0.4,
        #     "occ_threshold": 0.8,
        #     "enu_vec": enu,
        #     "yaw_vec": yaw,
        #     "is_animation": True,
        #     "apply_icp": False
        # },
        # {
        #     "folder": "occupancy_map_2_hit_0.9_miss_0.1_occ_threshold_0.8",
        #     "hit": 0.9,
        #     "miss": 0.1,
        #     "occ_threshold": 0.8,
        #     "enu_vec": enu,
        #     "yaw_vec": yaw,
        #     "is_animation": False,
        #     "apply_icp": False
        # },
        # {
        #     "folder": "occupancy_map_3_hit_0.6_miss_0.4_occ_threshold_0.8",
        #     "hit": 0.6,
        #     "miss": 0.4,
        #     "occ_threshold": 0.8,
        #     "enu_vec": enu,
        #     "yaw_vec": yaw,
        #     "is_animation": False,
        #     "apply_icp": False
        # },
        # {
        #     "folder": "occupancy_map_4_hit_0.7_miss_0.4_occ_threshold_0.75",
        #     "hit": 0.7,
        #     "miss": 0.4,
        #     "occ_threshold": 0.75,
        #     "enu_vec": enu,
        #     "yaw_vec": yaw,
        #     "is_animation": False,
        #     "apply_icp": False
        # },
        # {
        #     "folder": "occupancy_map_5_hit_0.7_miss_0.4_occ_threshold_0.9",
        #     "hit": 0.7,
        #     "miss": 0.4,
        #     "occ_threshold": 0.9,
        #     "enu_vec": enu,
        #     "yaw_vec": yaw,
        #     "is_animation": False,
        #     "apply_icp": False
        # },
        # {
        #     "description": "section 3b - noisy east, north and yaw",
        #     "folder": "occupancy_map_6_hit_0.7_miss_0.4_occ_threshold_0.8_noise",
        #     "hit": 0.7,
        #     "miss": 0.4,
        #     "occ_threshold": 0.8,
        #     "enu_vec": enu_noise,
        #     "yaw_vec": yaw_noise,
        #     "is_animation": True,
        #     "apply_icp": False
        # },
        # {
        #     "description": "section 3b - noisy east, north and yaw",
        #     "folder": "occupancy_map_7_hit_0.7_miss_0.4_occ_threshold_0.8_noise_icp",
        #     "hit": 0.7,
        #     "miss": 0.4,
        #     "occ_threshold": 0.8,
        #     "enu_vec": enu_noise,
        #     "yaw_vec": yaw_noise,
        #     "is_animation": True,
        #     "apply_icp": True
        # },
        # {
        #     "description": "section 3b - noisy east, north and yaw",
        #     "folder": "occupancy_map_8_hit_0.7_miss_0.4_occ_threshold_0.8_noise",
        #     "hit": 0.7,
        #     "miss": 0.4,
        #     "occ_threshold": 0.8,
        #     "enu_vec": enu,
        #     "yaw_vec": yaw,
        #     "is_animation": True,
        #     "apply_icp": False
        # },
        {
            "description": "section 3b - noisy east, north and yaw",
            "folder": "occupancy_map_9_hit_0.7_miss_0.4_occ_threshold_0.8_noise_icp",
            "hit": 0.7,
            "miss": 0.4,
            "occ_threshold": 0.8,
            "enu_vec": enu_noise,
            "yaw_vec": yaw_noise,
            "is_animation": True,
            "apply_icp": True
        },
    ]

    e_noise_icp = [float(enu_noise[0, 0])]
    n_noise_icp = [float(enu_noise[0, 1])]
    yaw_noise_icp = [float(yaw_noise[0])]

    # enu_noise[1, 0] += 0.5
    enu_noise[1, 1] += 1.7
    # yaw_noise[1] += 0.03

    for config in configs:
        occupancy_map = np.zeros((map_size_v, map_size_u), dtype=float)
        occ_threshold_log_odds = log_odds(config['occ_threshold'])
        free_threshold_log_odds = log_odds(1 - config['occ_threshold'])
        enu_from_config = config['enu_vec']
        yaw_from_config = config['yaw_vec']
        l_occ = log_odds(config['hit'])
        l_free = log_odds(config['miss'])
        is_animation = config['is_animation']
        apply_icp = config['apply_icp']
        print("Now processing [{}]...".format(config['folder']))
        for frame_idx in range(frame_count):  #  range(frame_count if not apply_icp else frame_count - 1)
            ts = int(time.time())
            velodyne_frame = velodyne["frames_raw"][frame_idx].copy()
            # velodyne_frame = np.array([
            #     [20, 20, 0, 0],
            #     [10, 10, 2, 0],
            #     [10, 20, 2, 0],
            #     [-10, -5, 2, 0],
            #     [5, 5, 2, 0],
            #     [17, 9, 2, 0],
            #     [19, -5, 2, 0]
            # ], dtype=float)
            img, velodyne_raw_bev, _ = process_frame(velodyne_frame, frame_idx)
            velodyne_frame_filtered = filter_velodyne_data(velodyne_frame.copy())
            car_pos_x = start_point_x + enu_from_config[frame_idx, 0]
            car_pos_y = start_point_y - enu_from_config[frame_idx, 1]
            # if we apply ICP algorithm on the velodyne data the method update_occupancy_map will
            # use velodyne data from both frame_idx and from (frame_idx + 1)
            # print("real data:")
            # print(start_point_x + enu[frame_idx, 0])
            # print(start_point_y + enu[frame_idx, 1])
            # print(yaw[frame_idx])
            velodyne_frame_prev_filtered = None
            velodyne_frame_prev = None
            t_e_diff = None
            t_n_diff = None
            if apply_icp and frame_idx > 0:
                theta_diff = yaw_from_config[frame_idx] - yaw_noise_icp[-1]  # yaw_from_config[frame_idx - 1]
                R = np.identity(4)
                R[0:2, 0:2] = np.array([[np.cos(-theta_diff), -np.sin(-theta_diff)], [np.sin(-theta_diff), np.cos(-theta_diff)]]).reshape(2, 2)
                # print(R)
                t_e_diff = -(enu_from_config[frame_idx, 0] - e_noise_icp[-1])  # enu_from_config[frame_idx - 1, 0])
                t_n_diff = (enu_from_config[frame_idx, 1] - n_noise_icp[-1])  # enu_from_config[frame_idx - 1, 1])
                t = np.array([t_n_diff, t_e_diff, 0.0, 0.0]).reshape(4, 1)
                print(t)
                # print("going to apply t {}".format(t))
                velodyne_frame = (np.dot(R, velodyne_frame.T) + t).T
                velodyne_frame_filtered = (np.dot(R, velodyne_frame_filtered.T) + t).T

                velodyne_frame_prev = velodyne["frames_raw"][frame_idx - 1].copy()
                # velodyne_frame_prev = np.array([
                #     [20, 20, 0, 0],
                #     [10, 10, 2, 0],
                #     [10, 20, 2, 0],
                #     [-10, -5, 2, 0],
                #     [5, 5, 2, 0],
                #     [17, 9, 2, 0],
                #     [19, -5, 2, 0]
                # ], dtype=float)
                velodyne_frame_prev_filtered = filter_velodyne_data(velodyne_frame_prev.copy())
                # theta = -yaw_from_config[frame_idx - 1]
                # R = np.identity(4)
                # R[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).reshape(2, 2)
                # t_e_diff = e_noise_icp[-1]
                # t_n_diff = n_noise_icp[-1]
                # # t_n_diff = enu_from_config[frame_idx - 1, 1]
                # t = np.array([t_e_diff, t_n_diff, 0.0, 0.0]).reshape(4, 1)
                # velodyne_frame_prev_filtered = (np.dot(R, velodyne_frame_prev_filtered.T) + t).T
                print("Expecting {} {} {}".format(enu_noise[frame_idx, 0] - enu[frame_idx, 0], enu_noise[frame_idx, 1] - enu[frame_idx, 1], yaw_noise[frame_idx] - yaw[frame_idx]))
            e_icp_diff, n_icp_diff, yaw_icp_diff = update_occupancy_map(velodyne_frame, velodyne_frame_prev, frame_idx, occupancy_map, car_pos_x, car_pos_y, float(enu_from_config[frame_idx, 0]), float(enu_from_config[frame_idx, 1]), float(yaw_from_config[frame_idx]), l_occ, l_free)

            # print("before: e {} n {}".format(float(enu_from_config[frame_idx, 0]), float(enu_from_config[frame_idx, 1])))
            if frame_idx > 0:
                # e_noise_icp.append(float(enu_from_config[frame_idx, 0]) + float(e_icp_diff))
                # n_noise_icp.append(float(enu_from_config[frame_idx, 1]) + float(n_icp_diff))
                # yaw_noise_icp.append(float(yaw_from_config[frame_idx]) + float(yaw_icp_diff))
                e_noise_icp.append(float(enu_from_config[frame_idx, 0]) + float(e_icp_diff))
                n_noise_icp.append(float(enu_from_config[frame_idx, 1]) - float(n_icp_diff))
                # e_noise_icp.append(float(enu_from_config[frame_idx, 0]) - float(e_icp_diff))
                # n_noise_icp.append(float(enu_from_config[frame_idx, 1]) - float(n_icp_diff))
                #         yaw_noise_icp.append(float(yaw_from_config[frame_idx]) - float(yaw_icp_diff))
                yaw_noise_icp.append(float(yaw_from_config[frame_idx]) - float(yaw_icp_diff))
                # e_noise_icp.append(float(e_icp_diff))
                # n_noise_icp.append(float(n_icp_diff))
                # yaw_noise_icp.append(float(yaw_icp_diff))
                # e_noise_icp.append(enu_from_config[frame_idx, 0] + t_e_diff + float(e_icp_diff))
                # n_noise_icp.append(enu_from_config[frame_idx, 1] + t_n_diff + float(n_icp_diff))
                # yaw_noise_icp.append(yaw_from_config[frame_idx] + theta + float(yaw_icp_diff))

            if False:
                # clamp values between log_odds(0.05) and log_odds(0.95)
                occupancy_map = np.where(occupancy_map < -2.95, -2.95, occupancy_map)
                occupancy_map = np.where(occupancy_map > 2.95, 2.95, occupancy_map)
                if is_animation or frame_idx == frame_count - 1:
                    occupancy_map_render = np.zeros(occupancy_map.shape, dtype=int)
                    for r in range(occupancy_map_render.shape[0]):
                        for c in range(occupancy_map_render.shape[1]):
                            val = occupancy_map[r, c]
                            if val > occ_threshold_log_odds:
                                occupancy_map_render[r, c] = 1
                            elif val < free_threshold_log_odds:
                                occupancy_map_render[r, c] = 0
                            else:
                                occupancy_map_render[r, c] = 2

                    car_pos_u = int(car_pos_x // MAP_RESOLUTION)
                    car_pos_v = int(car_pos_y // MAP_RESOLUTION)
                    for r in range(-2, 3):
                        for c in range(-2, 3):
                            occupancy_map_render[car_pos_v + r, car_pos_u + c] = 3
                    # to avoid discoloration in the final render, make sure we have at least one pixel of each color
                    occupancy_map_render[0, 0] = 0
                    occupancy_map_render[0, 1] = 1
                    occupancy_map_render[0, 2] = 2
                    occupancy_map_render[0, 3] = 3
                    render_frame(img, velodyne_raw_bev, occupancy_map_render, frame_idx, config['folder'])
            print("Frame index [{}] done in [{}] seconds".format(frame_idx, int(time.time() - ts)))

            if frame_idx > 1 and frame_idx % 5 == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.plot(enu[:, 0], enu[:, 1], color='b', linestyle='solid', markersize=1)
                ax1.plot(enu_from_config[:, 0], enu_from_config[:, 1], color='r', linestyle='solid', markersize=1)
                ax1.plot(e_noise_icp, n_noise_icp, color='g', linestyle='solid', markersize=1)
                ax1.set_aspect('equal', adjustable='box')
                ax1.set_xlabel('Longitude [degrees]', fontsize=20)
                ax1.set_ylabel('Latitude [degrees]', fontsize=20)
                ax1.legend(["Raw lon/lat samples", "Samples with Gaussian noise sigma=0.000005 deg"], prop={"size": 20},
                           loc="best")

                ax2.plot(range(0, frame_count), yaw, color='b', linestyle='solid', markersize=1)
                ax2.plot(range(0, frame_count), yaw_noise, color='r', linestyle='solid', markersize=1)
                ax2.plot(range(0, len(yaw_noise_icp)), yaw_noise_icp, color='g', linestyle='solid', markersize=1)
                ax2.set_xlabel('Frame Index', fontsize=20)
                ax2.set_ylabel('Yaw [rad]', fontsize=20)
                ax2.legend(["Raw yaw samples", "yaw with Gaussian noise, sigma=0.01 rad"], prop={"size": 20}, loc="best")
                plt.show()
    print("Iterative Closest Point (ICP)")
    print("Done")

if __name__ == "__main__":
    main()
