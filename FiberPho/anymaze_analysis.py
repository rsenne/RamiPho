import numpy as np
import pandas as pd

__all__ = ['anymazeResults']

class anymazeResults:
    def __init__(self, filepath:str):
        self.anymaze_df = pd.read_csv(filepath)
        self.freeze_vector = None

    def calculate_binned_freezing(self, bin_duration=120, format='%H:%M:%S.%f', time_col='Time', behavior_col ='Freezing'):
        self.anymaze_df[time_col] = pd.to_datetime(self.anymaze_df.Time, format=format)
        self.anymaze_df['duration'] = self.anymaze_df[time_col].diff().dt.total_seconds()
        self.anymaze_df['bin'] = pd.cut(self.anymaze_df[time_col], pd.date_range(start=self.anymaze_df[time_col].iloc[0],
                                                end=self.anymaze_df[time_col].iloc[-1],
                                                freq=f'{bin_duration}s'))
        result = self.anymaze_df.groupby(['bin', behavior_col])['duration'].sum().reset_index()
        return result[behavior_col] == 1
    
    def create_freeze_vector(self, timestamps, format='%H:%M:%S.%f', time_col='Time', behavior_col ='Freezing'):
        timestamps = pd.to_datetime(timestamps, format=format)
        binary_vector = np.zeros(len(timestamps), dtype=int)
        for i, ts in enumerate(timestamps):
            state = self.anymaze_df.loc[self.anymaze_df[time_col] <= ts, behavior_col].iloc[-1]  # Get the last label before the current timestamp
            binary_vector[i] = state
        self.freeze_vector = binary_vector
        return binary_vector