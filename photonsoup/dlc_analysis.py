import pykalman
import numpy as np
import pandas as pd

__all__ = ['dlcResults']

class dlcResults:
    def __init__(self, dlc_file):
        self.dlc_df = pd.read_csv(dlc_file, header=[1, 2], index_col=[0])
        self.filtered_df = None
        

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
    
    def calculate_centroids(self, bparts=None):
        if bparts is None:
            bparts = ['snout', 'l_ear', 'r_ear', 'front_l_paw', 'front_r_paw', 'back_l_paw', 'back_r_paw', 'base_of_tail', 'tip_of_tail']
        self.dlc_df.loc[:, ('centroid','x')] = self.dlc_df.xs('x', axis=1, level=1).mean(axis=1)
        self.dlc_df.loc[:, ('centroid','y')] = self.dlc_df.xs('y', axis=1, level=1).mean(axis=1)
        return self.dlc_df
        
        
    def filter_predictions(self, bparts=None, fps=None):
            if bparts is None:
                 bparts = ['snout', 'l_ear', 'r_ear', 'front_l_paw', 'front_r_paw', 'back_l_paw', 'back_r_paw', 'base_of_tail', 'tip_of_tail', 'centroid']
            if fps is None:
                 fps = 30
            dt = 1/ fps

            kalman_dict = {}
            for bpart in bparts:
                 k_means, _, k_accel = self.kalman_filter(self.dlc_df.loc[:, (bpart,'x')], self.dlc_df.loc[:, (bpart,'y')], dt=dt)
                 kalman_dict[bpart] = {
                      'x': k_means[:, 0],
                      'y': k_means[:, 1],
                      'velocity_x': k_means[:, 2],
                      'velocity_y': k_means[:, 3],
                      'acceleration_x': k_accel[:, 0],
                      'acceleration_y': k_accel[:, 1] 
                 }
                 
            reformed_dict = {}
            for outerKey, innerDict in kalman_dict.items():
                for innerKey, values in innerDict.items():
                    reformed_dict[(outerKey,innerKey)] = values
        
            df = pd.DataFrame.from_dict(reformed_dict)
            self.filtered_df = df
            return df
    
    def process_dlc(self, bparts, fps):
         self.calculate_centroids(bparts=bparts)
         self.filter_predictions(bparts=bparts, fps=fps)

