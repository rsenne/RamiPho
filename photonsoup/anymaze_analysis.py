import pandas as pd
import numpy as np

__all__ = ["anymazeResults"]

class anymazeResults:
    def __init__(self, filepath: str):
        self.anymaze_df = pd.read_csv(filepath)
        # Convert the Time column to seconds immediately upon reading
        self.anymaze_df['Time'] = self.__convert_time_to_seconds__(self.anymaze_df['Time'])
        self.freeze_vector = None

    def __convert_time_to_seconds__(self, time_series):
        # Split the time and then convert to seconds
        time_data = time_series.str.split(':').tolist()
        return [int(x[0]) * 3600 + int(x[1]) * 60 + float(x[2]) for x in time_data]

    def correct_time_warp(self, true_endtime=None):
        # First, calculate the real duration in seconds
        warped_time = self.anymaze_df['Time'].max()

        # Calculate the correction factor
        correction_factor = true_endtime / warped_time

        # Apply the correction factor to the timestamps
        self.anymaze_df['Time'] *= correction_factor

        return

    def calculate_binned_freezing(self,
                                  bin_duration=120,
                                  start=None, end=None,
                                  offset=0,
                                  time_col='Time',
                                  behavior_col='Freezing'):
        # Subtract the offset directly
        self.anymaze_df[time_col] -= offset

        # Calculate the duration between rows
        self.anymaze_df['duration'] = self.anymaze_df[time_col].diff().fillna(0)

        # Set default start and end times if not specified
        start = start if start is not None else self.anymaze_df[time_col].iloc[0]
        end = end if end is not None else self.anymaze_df[time_col].iloc[-1]

        bins = np.arange(start, end + bin_duration, bin_duration)  # create bins
        self.anymaze_df['bin'] = pd.cut(self.anymaze_df[time_col], bins, right=False)

        result = self.anymaze_df.groupby(['bin', behavior_col])['duration'].sum().reset_index()
        return result[result[behavior_col] == 1]

    def create_freeze_vector(self, timestamps, time_col='Time', behavior_col='Freezing'):
        timestamps = [self.__convert_time_to_seconds__([ts])[0] for ts in timestamps]
        binary_vector = np.zeros(len(timestamps), dtype=int)

        for i, ts in enumerate(timestamps):
            state = self.anymaze_df.loc[self.anymaze_df[time_col] <= ts, behavior_col].iloc[-1]
            binary_vector[i] = state

        self.freeze_vector = binary_vector
        return binary_vector

