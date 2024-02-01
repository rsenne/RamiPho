import pykalman
import numpy as np
import pandas as pd

__all__ = ['dlcResults']


class dlcResults:
    def __init__(self, dlc_file):
        self.dlc_df = pd.read_csv(dlc_file, header=[1, 2], index_col=[0])
        self.filtered_df = None

    @staticmethod
    def initialize_matrices(dt):
        """Initializes and returns the Kalman filter matrices."""
        # State transition matrix
        A = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Observation matrix
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Process noise covariance
        Q = np.eye(6)
        Q[4, 4], Q[5, 5] = dt ** 2, dt ** 2  # This is an assumption. Adjust based on system knowledge.

        return A, H, Q

    def kalman_filter(self, x_data, y_data, dt=(1 / 30)):

        A, H, Q = self.initialize_matrices(dt)
        R = np.diag([np.var(x_data), np.var(y_data)])

        kf = pykalman.KalmanFilter(transition_matrices=A, observation_matrices=H, transition_covariance=Q,
                                   observation_covariance=R, initial_state_covariance=np.eye(6) * 0.1,
                                   initial_state_mean=(x_data[0], y_data[0], 0, 0, 0, 0))
        kalman_means = np.zeros((A.shape[0], len(x_data))).T
        kalman_covs = np.zeros((A.shape[0], A.shape[0], len(x_data))).T
        kalman_means[0, :] = (x_data[0], y_data[0], 0, 0, 0, 0)
        kalman_covs[0, :, :] = 0.1

        observations = np.vstack((x_data, y_data)).T
        kalman_means, kalman_covs = kf.smooth(observations)

        return kalman_means, kalman_covs

    def calculate_centroids(self):
        self.dlc_df.loc[:, ('centroid', 'x')] = self.dlc_df.xs('x', axis=1, level=1).mean(axis=1)
        self.dlc_df.loc[:, ('centroid', 'y')] = self.dlc_df.xs('y', axis=1, level=1).mean(axis=1)
        return self.dlc_df

    def filter_predictions(self, bparts=None, fps=None):
        if bparts is None:
            bparts = ['snout', 'l_ear', 'r_ear', 'front_l_paw', 'front_r_paw', 'back_l_paw', 'back_r_paw',
                      'base_of_tail', 'tip_of_tail', 'centroid']
        if fps is None:
            fps = 30
        dt = 1 / fps

        kalman_dict = {}
        for bpart in bparts:
            k_means, _ = self.kalman_filter(self.dlc_df.loc[:, (bpart, 'x')], self.dlc_df.loc[:, (bpart, 'y')],
                                            dt=dt)
            kalman_dict[bpart] = {
                'x': k_means[:, 0],
                'y': k_means[:, 1],
                'velocity_x': k_means[:, 2],
                'velocity_y': k_means[:, 3],
                'acceleration_x': k_means[:, 4],
                'acceleration_y': k_means[:, 5]
            }

        reformed_dict = {}
        for outerKey, innerDict in kalman_dict.items():
            for innerKey, values in innerDict.items():
                reformed_dict[(outerKey, innerKey)] = values

        df = pd.DataFrame.from_dict(reformed_dict)
        self.filtered_df = df
        return df

    def process_dlc(self, bparts, fps):
        self.calculate_centroids()
        self.filter_predictions(bparts=bparts, fps=fps)
