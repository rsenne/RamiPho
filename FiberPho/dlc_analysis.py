import pykalman
import numpy as np
import pandas as pd

class dlcProcessor:
    def __init__(self, dlc_file):
        self.dlc_file = pd.read_csv(dlc_file)

    def __processfile__(self, bparts=None):
        if bparts is None:
            bparts = ["snout", "l_ear", "r_ear", "back_l_paw", "back_r_paw", "tip_of_tail", "base_of_tail"]
        


    def kalman_filter(self, x_data, y_data, dt=1):
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0 , dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        B = np.array([
            [0.5*(dt**2), 0],
            [0, 0.5*(dt**2)],
            [dt, 0],
            [0, dt]
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.eye(2) * [[np.var(x_data), np.var(y_data)]]
        Q = np.array([
            [(dt**4)/4, 0, (dt**3)/2, 0],
            [0, (dt**4)/4, 0, (dt**3)/2],
            [(dt**3)/2, 0, dt**2, 0],
            [0, (dt**3)/2, 0, dt**2]
        ])

        filter = pykalman.KalmanFilter(transition_matrices=A, observation_matrices=H, transition_covariance=Q, observation_covariance=R, initial_state_covariance=np.eye(4)*0.1, initial_state_mean=(x_data[0], y_data[0],0,0))
        kalman_means = np.zeros((A.shape[0], len(x_data))).T
        kalman_covs = np.zeros((A.shape[0], A.shape[0], len(x_data))).T
        kalman_accel = np.zeros((2, len(x_data))).T
        kalman_means[0, :] = (x_data[0], y_data[0], 0, 0)
        kalman_covs[0, :, :] = 0.1
        kalman_accel[0, :] = 0


        for t in range(1, len(x_data)):
            kalman_means[t, :], kalman_covs[t, :, :] = filter.filter_update(filtered_state_mean=kalman_means[t-1, :], observation=[x_data[t], y_data[t]], filtered_state_covariance=kalman_covs[t-1, :, :], transition_offset=B @ kalman_accel[t-1, :].T)
            kalman_accel[t, 0] = (kalman_means[t, 2] - kalman_means[t-1, 2])
            kalman_accel[t, 1] = (kalman_means[t, 3] - kalman_means[t-1, 3])

        return kalman_means, kalman_covs, kalman_accel
