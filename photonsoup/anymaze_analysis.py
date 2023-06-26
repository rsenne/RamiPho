import numpy as np
import pandas as pd
from datetime import datetime

__all__ = ['anymazeResults']


class anymazeResults:
    def __init__(self, filepath: str):
        self.anymaze_df = pd.read_csv(filepath)
        self.freeze_vector = None

    @staticmethod
    def __format_time__(time_obj):
        # Format time as a string, including microseconds
        return "{:02d}:{:02d}:{:02d}.{:06d}".format(time_obj.hour, time_obj.minute, time_obj.second,
                                                    time_obj.microsecond)

    def correct_time_warp(self, true_endtime=None):
        # First, calculate the real duration in seconds
        warped_time = pd.to_timedelta(self.anymaze_df['Time']).dt.total_seconds().max()

        # Calculate the correction factor
        correction_factor = true_endtime / warped_time

        # Apply the correction factor to the timestamps
        self.anymaze_df['Time'] = pd.to_timedelta(self.anymaze_df['Time']).dt.total_seconds() * correction_factor

        # Convert the corrected time back to the original format
        self.anymaze_df['Time'] = pd.to_datetime(self.anymaze_df['Time'], unit='s').dt.time

        # The first column if it is zero, will be the wrong format i.e. %H:%M:%S when we need %H:%M:%S.%f. Fix this.
        if self.anymaze_df.loc[0, 'Time'] == datetime.strptime('00:00:00', '%H:%M:%S').time():
            self.anymaze_df.loc[0, 'Time'] = self.__format_time__(self.anymaze_df.loc[0, 'Time'])

        return

    def calculate_binned_freezing(self,
                                  bin_duration=120,
                                  start=None, end=None,
                                  offset=0,
                                  time_format='%H:%M:%S.%f',
                                  time_col='Time',
                                  behavior_col='Freezing'):
        # convert to datetimes and subtract any offset
        self.anymaze_df[time_col] = pd.to_datetime(self.anymaze_df.Time, format=time_format) - pd.Timedelta(seconds=offset)
        self.anymaze_df['duration'] = self.anymaze_df[time_col].diff().dt.total_seconds()

        # If custom_start or custom_end is None, use the first or last timestamp respectively.
        start = pd.to_datetime(start, format=time_format) if start is not None else self.anymaze_df[time_col].iloc[0]
        end = pd.to_datetime(end, format=time_format) if end is not None else self.anymaze_df[time_col].iloc[-1]

        self.anymaze_df['bin'] = pd.cut(self.anymaze_df[time_col], pd.date_range(start=start,
                                                                                 end=end,
                                                                                 freq=f'{bin_duration}s'))
        result = self.anymaze_df.groupby(['bin', behavior_col])['duration'].sum().reset_index()
        return result[result[behavior_col] == 1]

    def create_freeze_vector(self, timestamps, time_format='%H:%M:%S.%f', time_col='Time', behavior_col='Freezing'):
        timestamps = pd.to_datetime(timestamps, format=time_format)
        binary_vector = np.zeros(len(timestamps), dtype=int)
        for i, ts in enumerate(timestamps):
            state = self.anymaze_df.loc[self.anymaze_df[time_col] <= ts, behavior_col].iloc[
                -1]  # Get the last label before the current timestamp
            binary_vector[i] = state
        self.freeze_vector = binary_vector
        return binary_vector
