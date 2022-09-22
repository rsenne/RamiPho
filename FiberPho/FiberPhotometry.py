import pickle as pkl
import warnings as warn
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sb
import statsmodels.api as sm
# import FiberPho.b_spline
from scipy import sparse
from scipy.integrate import simpson
from scipy.interpolate import splrep, splev
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve

__all__ = ["fiberPhotometryCurve", "fiberPhotometryExperiment"]


class fiberPhotometryCurve:
    def __init__(self, npm_file: str, **kwargs):
        """
        :param npm_file: str Path to csv fiber photometry file gathered using a neurophotometrics fp3002 rig and bonsai software
        :param behavioral_data: Path(s) to csv files, either deeplabcut or anymaze, for doing behavioral analysis
        :param keystroke_offset: Value corresponding to a keystroke press in bonsai for removing irrelevant data
        :param manual_off_set: Value obtained from calculating offset from expected event i.e. blue light and its theoretical appearance in video
        :param kwargs: dict containing values such as "ID", "task", and/or "treatment" note: task and treatment necessary for use in fiberPhotometryExperiment

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        # these should always be present
        self.npm_file = npm_file
        self.fp_df = pd.read_csv(self.npm_file)
        self.__T0__ = self.fp_df['Timestamp'][1]

        # unpack extra params
        if hasattr(self, 'keystroke_offset') or hasattr(self, 'manual_off_set'):
            if hasattr(self, 'keystroke_offset'):
                self.OffSet = (self.keystroke_offset - self.fp_df.at[0, 'Timestamp'])
            else:
                self.OffSet = self.manual_off_set
        else:
            self.OffSet = 0.0

        # determine sample time
        self._sample_time_ = np.diff(self.fp_df['Timestamp'])[1]

        # this needs to be here for coherent timestamp data between behavioral analysis and signal

        # mb trying to do behavioral dict of dicts
        # if there's a DLC file, apply calc kinematics on that file and then put the velocity and accel column as keys
        # mb trying to do behavioral dict of dicts
        self.behavioral_data = {}  # behavioral data dictionary
        if hasattr(self, 'DLC_file'):  # if there's a DLC file key word arg
            self.behavioral_data['DLC'] = {}  # creates nested dictionary within behavioral data dictionary

            # if parameters  of calc_kinematics change, pass them through calc kinematics
            # passes csv DLC file through calc_kinematics function, stores it in pandas df
            df = self.calc_kinematics(getattr(self, 'DLC_file'), getattr(self, 'bps', None),
                                      getattr(self, 'interpolate', True), getattr(self, 'int_f', 100),
                                      getattr(self, 'threshold', .6))

            # creates a numpy array as a value for the velocity and acceleration, taken from velocity and acceleration columns of df
            self.behavioral_data['DLC']['velocity'] = df['velocity'].to_numpy()
            self.behavioral_data['DLC']['acceleration'] = df['acceleration'].to_numpy()
            # add kwargs
        if hasattr(self, 'anymaze_file'):  # if there's an anymaze file key word arg
            self.behavioral_data['Anymaze'] = {}  # creates nested dictionary within behavioral data dictionary
            # passes csv anymzze file through process_anymaze method, stores it in pandas df
            df = pd.read_csv(getattr(self, 'anymaze_file'))
            anymaze_df, freeze_vector, inds_start, inds_end = self.process_anymaze(df, self.fp_df.Timestamp[
                self.fp_df['LedState'] == 1])
            # puts freeze vector array and start/end freezing inds from the process anymaze function into the Anymaze dictionary
            self.behavioral_data['Anymaze']['freeze_vector'] = freeze_vector

            self.behavioral_data['Anymaze']['start_freezing'] = inds_start  # start of freezing bouts
            self.behavioral_data['Anymaze']['end_freezing'] = inds_end  # ends of freezing bouts
            self.behavioral_data['Anymaze']['anymaze_df'] = anymaze_df

        if hasattr(self, 'manual_off_set'):
            self.fp_df = self.fp_df[int(self.manual_off_set // self._sample_time_):].reset_index()

        if hasattr(self, 'keystroke_offset'):
            ind = self.fp_df[self.fp_df['Timestamp'] == self.keystroke_offset].index[0]
            self.fp_df = self.fp_df[ind - 1:].reset_index()
            self.__T0__ = self.fp_df['Timestamp'][0]

        # check to see if using old files
        if "Flags" in self.fp_df.columns:
            self.fix_npm_flags()
            print("Old NPM format detected, changing Flags to LedState")
        else:
            pass

        # drop last row if timeseries are unequal
        try:

            while self.fp_df['LedState'].value_counts()[1] != self.fp_df['LedState'].value_counts()[2] or \
                    self.fp_df['LedState'].value_counts()[2] != self.fp_df['LedState'].value_counts()[4]:
                #
                self.fp_df.drop(self.fp_df.index[-1], axis=0, inplace=True)

                print(self.fp_df['LedState'].value_counts()[1])
                print(self.fp_df['LedState'].value_counts()[2])
                print(self.fp_df['LedState'].value_counts()[4])
        except KeyError:
            while self.fp_df['LedState'].value_counts()[1] != self.fp_df['LedState'].value_counts()[2]:
                self.fp_df.drop(self.fp_df.index[-1], axis=0, inplace=True)
        except:
            while self.fp_df['LedState'].value_counts()[1] != self.fp_df['LedState'].value_counts()[4]:
                self.fp_df.drop(self.fp_df.index[-1], axis=0, inplace=True)

        if 2 and 4 in self.fp_df.LedState.values:
            self.__DUAL_COLOR = True
        else:
            self.__DUAL_COLOR = False

        # create essentially a dummy variable for ease of typing
        columns = list(self.fp_df.columns)
        if "Region0G" in columns:
            self.__CONF1 = True
        else:
            self.__CONF1 = False

        # slicing file based on flag values present in file
        # this is a goddamned mess...please for the love of god if there's a better way show me
        # literally brute force
        if self.__DUAL_COLOR:
            gcamp = self.fp_df[self.fp_df['LedState'] == 2]
            rcamp = self.fp_df[self.fp_df['LedState'] == 4]
            isobestic = self.fp_df[self.fp_df['LedState'] == 1]
            if self.__CONF1:
                isobestic_gcamp = isobestic.Region0G
                isobestic_rcamp = isobestic.Region1R
                self.Signal = {"GCaMP": np.array(gcamp['Region0G']),
                               "RCaMP": np.array(gcamp['Region1R']),
                               "Isobestic_GCaMP": np.array(isobestic_gcamp),
                               "Isobestic_RCaMP": np.array(isobestic_rcamp)}
                self.Timestamps = {signal: time.reset_index(drop=True).tolist() - self.__T0__
                                   for
                                   signal, time in zip(['Isobestic_GCaMP', 'Isobestic_RCaMP', 'GCaMP', 'RCaMP'],
                                                       [isobestic.Timestamp, isobestic.Timestamp, gcamp.Timestamp,
                                                        rcamp.Timestamp])}

            else:
                isobestic_gcamp = isobestic.Region1G
                isobestic_rcamp = isobestic.Region0R
                self.Signal = {"GCaMP": np.array(gcamp['Region1G']),
                               "RCaMP": np.array(gcamp['Region0R']),
                               "Isobestic_GCaMP": np.array(isobestic_gcamp),
                               "Isobestic_RCaMP": np.array(isobestic_rcamp)}
                self.Timestamps = {signal: time.values.tolist() - self.__T0__ for
                                   signal, time in zip(['Isobestic_GCaMP', 'Isobestic_RCaMP', 'GCaMP', 'RCaMP'],
                                                       [isobestic.Timestamp, isobestic.Timestamp, gcamp.Timestamp,
                                                        rcamp.Timestamp])}

        elif not self.__DUAL_COLOR:
            isobestic = self.fp_df[self.fp_df['LedState'] == 1]

            try:
                gcamp = self.fp_df[self.fp_df['LedState'] == 2]
                self.Timestamps = {signal: time.values.tolist() - self.__T0__ for
                                   signal, time in
                                   zip(['Isobestic_GCaMP', 'GCaMP'], [isobestic.Timestamp, gcamp.Timestamp])}

                if self.__CONF1:
                    self.Signal = {"GCaMP": np.array(gcamp['Region0G']),
                                   "Isobestic_GCaMP": np.array(isobestic["Region0G"])}

                else:
                    self.Signal = {"GCaMP": np.array(gcamp['Region1G']),
                                   "Isobestic_GCaMP": np.array(isobestic["Region1G"])}

            except KeyError:
                rcamp = self.fp_df[self.fp_df['LedState'] == 4]
                self.Timestamps = {signal: time.values.tolist() - self.__T0__ for
                                   signal, time in zip(['Isobestic_RCaMP', 'RCaMP'], [isobestic, rcamp])}

                if self.__CONF1:
                    self.Signal = {"RCaMP": np.array(rcamp['Region1R']),
                                   "Isobestic_RCaMP": np.array(isobestic["Region1R"])}

                else:
                    self.Signal = {"RCaMP": np.array(rcamp['Region0R']),
                                   "Isobestic_RCaMP": np.array(isobestic["Region0R"])}
        else:
            raise ValueError(
                "No experiment type matches your NPM File input. Make sure you've loaded the correct file.")

        self.DF_F_Signals, self.DF_Z_Signals = self.process_data()
        self.peak_properties = self.find_signal()
        self.neg_peak_properties = self.find_signal(neg=True)

    def __iter__(self):
        return iter(list(self.DF_F_Signals.values()))

    def __eq__(self, other):
        if not isinstance(other, fiberPhotometryCurve):
            raise TypeError("You can only compare the identity of a fiber photometry curve to another fiber"
                            "photometry curve!!")
        val1 = self.DF_F_Signals.values()
        val2 = other
        truthiness_array = [self.__arrays_equal__(a, b) for a, b in zip(val1, val2)]
        if any(truthiness_array):
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.DF_F_Signals.values())

    @staticmethod
    @nb.jit(nopython=True)
    def __arrays_equal__(a, b):
        if a.shape != b.shape:
            return False
        for ai, bi in zip(a.flat, b.flat):
            if ai != bi:
                return False
        return True

    @staticmethod
    def _als_detrend(y, lam=10e7, p=0.01, niter=100):  # asymmetric least squares smoothing method
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        z = np.zeros(len(y))
        for i in range(niter):
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return y - z

    @staticmethod
    def smooth(signal, kernel, visual_check=False):
        """

        :param signal: array of signal of interest (GCaMP, Isobestic_GCaMP, etc)
        :param kernel: int length of uniform filter
        :param visual_check: boolean for plotting original vs smoothed signal
        :return: array smoothed by 1D uniform filter
        """
        smooth_signal = uniform_filter1d(signal, kernel)
        if visual_check:
            plt.figure()
            plt.plot(signal)
            plt.plot(smooth_signal)
        return smooth_signal

    @staticmethod
    def b_smooth(signal, timeseries, s=500):
        knot_parms = splrep(timeseries, signal, s=s)
        smoothed_signal = splev(timeseries, knot_parms)
        return smoothed_signal

    @staticmethod
    def _df_f(raw, kind="std"):
        """
        :param raw: a smoothed baseline-corrected array of fluorescence values
        :param kind: whether you want standard df/f or if you would like a z-scored scaling
        :return: df/f standard or z-scored
        Function to calculate DF/F (signal - median / median or standard deviation).
        """
        F0 = np.median(raw)  # median value of time series
        if kind == "standard":
            df_f = (raw - F0) / F0
        else:  # z-scored
            df_f = (raw - F0) / np.std(raw)
        return df_f

    @staticmethod
    def calc_area(l_index, r_index, timeseries):
        areas = np.asarray([simpson(timeseries[i:j]) for i, j in zip(l_index, r_index)])
        return areas

    def process_data(self):  # correct baseline, smoothed with uniform filter
        signals = [signal for signal in self.Signal.values()]  # list of raw signals
        baseline_corr_signal = [self._als_detrend(sig) for sig in signals]  # list of baseline corrected signals
        df_f_signals = [self._df_f(s, kind='standard') for s in baseline_corr_signal]
        df_z_signals = [self._df_f(s, kind='std') for s in baseline_corr_signal]
        for i in range(len(df_f_signals)):
            if abs(np.median(df_f_signals[i])) < 0.05:
                df_f_signals[i] = self._als_detrend(df_f_signals[i])
                df_z_signals[i] = self._als_detrend(df_z_signals[i])
        # smoothed like a baby's bottom
        smoothed_f_signals = [self.smooth(timeseries, kernel=10) for timeseries in
                              df_f_signals]
        smoothed_z_signals = [self.smooth(timeseries, kernel=10) for timeseries in
                              df_z_signals]
        return {identity: signal for identity, signal in zip(self.Signal.keys(), smoothed_f_signals)}, {identity: signal
                                                                                                        for
                                                                                                        identity, signal
                                                                                                        in
                                                                                                        zip(self.Signal.keys(),
                                                                                                            smoothed_z_signals)}

    def fit_general_linear_model(self, curve, ind_vars):
        dep_var = np.reshape(self.DF_F_Signals[curve], (len(self.DF_F_Signals[curve]), 1))
        ind_var = sm.add_constant(pd.DataFrame(ind_vars))
        gaussian_model = sm.GLM(endog=dep_var, exog=ind_var, family=sm.families.Gaussian())
        res_fit = gaussian_model.fit()
        return res_fit

    # def find_signal(self, neg=False):
    #     """
    #
    #     :param neg: ?
    #     :return: a peak property array based on scipy find_peak function
    #     """
    #
    #     # creates empty peak properties dictionary
    #     peak_properties = {}
    #     if not neg:
    #         for GECI, sig in self.DF_F_Signals.items():
    #             # returns array of indices of  peaks, and properties dictionary
    #             peaks, properties = find_peaks(sig, height=1.0, distance=131, width=25,
    #                                            rel_height=0.50)  # height=1.0, distance=130, prominence=0.5, width=25, rel_height=0.90)
    #             # adds array  of peak indices to properties dictionary
    #             properties['peaks'] = peaks
    #             # calculates area under the curve for each peak and adds to properties dictionary
    #             properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
    #                                                              self.DF_F_Signals[GECI])
    #
    #             # populates  the peak properties dictionary with properties nested dictionary for each GECI
    #             properties['widths'] *= self._sample_time_
    #             peak_properties[GECI] = properties
    #             # michelle tries to do iei
    #             # finds the inter event interval between each peak
    #             iei = [(self.Timestamps[GECI][peak_properties[GECI]['peaks'][i + 1]] - self.Timestamps[GECI][
    #                 peak_properties[GECI]['peaks'][i]]) for i in range(len(peak_properties[GECI]['peaks']) - 1)]
    #             # adds iei  to peak properties dictionary
    #             peak_properties[GECI]['inter_event_interval'] = iei
    #
    #             # mich tries to find latency
    #             # if there's gcamp and rcamp, finds latency between peaks
    #
    #             # threshold for time between peaks to be considered
    #         latency = []
    #         if ('RCaMP' in peak_properties and 'GCaMP' in peak_properties):  # checks if RCaMP and GCaMP both exist
    #             ind_G = 0  # starts GCaMP index count at first index 0
    #             for ind_R in range(len(peak_properties['RCaMP']['peaks'])):
    #                 time_RCaMP = peak_properties['RCaMP']['peaks'][
    #                     ind_R]  # index of when peak happens for each peak in RCaMP
    #                 while ind_G < len(
    #                         peak_properties['GCaMP']['peaks']):  # while there's still indeces left in GCaMP array
    #                     time_GCaMP = peak_properties['GCaMP']['peaks'][
    #                         ind_G]  # index of when peak happens for each peak in RCaMP
    #                     if time_GCaMP > time_RCaMP:  # if GCaMP peak index follows RCaMP (which it should for latency)
    #                         if time_GCaMP - time_RCaMP <= 10:  # if the difference is smaller than 10
    #                             # add time to latency
    #                             latency.append(
    #                                 self.Timestamps['GCaMP'][time_GCaMP] - self.Timestamps['RCaMP'][time_RCaMP])
    #                             # augment GCaMP to next index so it's not reused
    #                             ind_G += 1
    #                             # break out of while loop, move to next RCaMP entry
    #                             break
    #                         else:  # if the difference is greater than 10, then move on to next RCaMP entry
    #                             break
    #                     else:  # GCaMP index precedes RCaMP, need to move it up
    #                         ind_G += 1  # increase index to the next one
    #
    #
    #     else:
    #         for GECI, sig in self.DF_F_Signals.items():
    #             peaks, properties = find_peaks(-sig, height=1.0, distance=131, width=25, rel_height=0.50)
    #             properties['peaks'] = peaks
    #             properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
    #                                                              self.DF_F_Signals[GECI])
    #             # michelle tries to do iei
    #             # finds the inter event interval between each peak
    #             properties['widths'] *= self._sample_time_
    #             peak_properties[GECI] = properties
    #             # finds the inter event interval between each peak
    #             iei = [(self.Timestamps[GECI][peak_properties[GECI]['peaks'][i + 1]] - self.Timestamps[GECI][
    #                 peak_properties[GECI]['peaks'][i]]) for i in range(len(peak_properties[GECI]['peaks']) - 1)]
    #             # adds iei  to peak properties dictionary
    #             peak_properties[GECI]['inter_event_interval'] = iei
    #
    #         latency = []
    #         if ('RCaMP' in peak_properties and 'GCaMP' in peak_properties):  # checks if RCaMP and GCaMP both exist
    #             ind_G = 0  # starts GCaMP index count at first index 0
    #             for ind_R in range(len(peak_properties['RCaMP']['peaks'])):
    #                 time_RCaMP = peak_properties['RCaMP']['peaks'][
    #                     ind_R]  # index of when peak happens for each peak in RCaMP
    #                 while ind_G < len(
    #                         peak_properties['GCaMP']['peaks']):  # while there's still indeces left in GCaMP array
    #                     time_GCaMP = peak_properties['GCaMP']['peaks'][
    #                         ind_G]  # index of when peak happens for each peak in RCaMP
    #                     if time_GCaMP > time_RCaMP:  # if GCaMP peak index follows RCaMP (which it should for latency)
    #                         if time_GCaMP - time_RCaMP <= 10:  # if the difference is smaller than 10
    #                             # add time to latency
    #                             latency.append(
    #                                 self.Timestamps['GCaMP'][time_GCaMP] - self.Timestamps['RCaMP'][time_RCaMP])
    #                             # augment GCaMP to next index so it's not reused
    #                             ind_G += 1
    #                             # break out of while loop, move to next RCaMP entry
    #                             break
    #                         else:  # if the difference is greater than 10, then move on to next RCaMP entry
    #                             break
    #                     else:  # GCaMP index precedes RCaMP, need to move it up
    #                         ind_G += 1  # increase index to the next one
    #
    #     peak_properties['latency'] = latency
    #     return peak_properties

    def find_signal(self, neg=False):
        """

        :param neg:
        :return: a peak property array based on scipy find_peak and other functions
        """
        peak_properties = {}
        if not neg:
            for GECI, sig in self.DF_F_Signals.items():
               # returns array of indices of  peaks, and properties dictionary
                peaks, properties = find_peaks(sig, height=1.0, distance=131, width=25,
                                               rel_height=0.95)  # height=1.0, distance=130, prominence=0.5, width=25, rel_height=0.90)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.DF_F_Signals[GECI])
                properties['widths'] *= self._sample_time_
                peak_properties[GECI] = properties
        else:
            for GECI, sig in self.DF_F_Signals.items():
                # returns array of indices of  peaks, and properties dictionary
                peaks, properties = find_peaks(-sig, height=1.0, distance=131, width=25, rel_height=0.95)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.DF_F_Signals[GECI])
                properties['widths'] *= self._sample_time_
                peak_properties[GECI] = properties
        return peak_properties

    def visual_check_peaks(self, signal):
        """
        Plots object's df_f signal overlayed with peaks
        :param signal: string of which signal to check
        :return:
        """
        if hasattr(self, "peak_properties"):
            plt.figure()
            plt.plot(self.DF_F_Signals[signal])
            plt.plot(self.peak_properties[signal]['peaks'],
                     self.DF_F_Signals[signal][self.peak_properties[signal]['peaks']], "x")
            plt.vlines(x=self.peak_properties[signal]['peaks'],
                       ymin=self.DF_F_Signals[signal][self.peak_properties[signal]['peaks']] -
                            self.peak_properties[signal]["prominences"],
                       ymax=self.DF_F_Signals[signal][self.peak_properties[signal]['peaks']], color="C1")
            plt.hlines(y=self.peak_properties[signal]["width_heights"], xmin=self.peak_properties[signal]["left_ips"],
                       xmax=self.peak_properties[signal]["right_ips"], color="C1")
            plt.show()
        else:
            raise KeyError(f'{signal} is not in {self}')
        return

    def process_anymaze(self, anymaze_file, timestamps):
        """
        :param anymaze_file: panda dataframe of anymaze data
        :param timestamps: array of timestamps
        :return: anymaze file, binary freeze vector, inds_start, inds_end (lists of start/end index of freezing bouts)
        """
        timestamps.reset_index(drop=True, inplace=True)
        timestamps = timestamps.to_numpy() - timestamps[0]
        length = len(timestamps)
        times = anymaze_file.Time.str.split(':')
        if len(times[0]) == 3:
            for i in range(len(times)):
                anymaze_file.loc[i, 'seconds'] = (float(times[i][1]) * 60 + float(times[i][2]))
        else:
            for i in range(len(times)):
                anymaze_file.loc[i, 'seconds'] = (float(times[i][0]) * 60 + float(times[i][1]))
        anymaze_file.seconds = anymaze_file['seconds'].apply(
            lambda x: (x / anymaze_file.seconds.iloc[-1] * timestamps[-1]))

        anymaze_file.seconds = anymaze_file.seconds - self.OffSet
        anymaze_file = anymaze_file[anymaze_file['seconds'] > 0].reset_index()
        binary_freeze_vec = np.zeros(shape=length)
        i = 0
        while i < len(anymaze_file):
            if anymaze_file.loc[i, 'Freezing'] == 1:
                t1 = anymaze_file.loc[i, 'seconds']
                try:
                    t2 = anymaze_file.loc[i + 1, 'seconds']
                except KeyError:
                    t2 = anymaze_file.seconds.iloc[-1]
                try:
                    binary_freeze_vec[np.where(timestamps > t1)[0][0]:np.where(timestamps < t2)[0][-1]] = 1
                except IndexError:
                    if t1 == t2:
                        binary_freeze_vec[np.where(timestamps == t1)] = 0
                    else:
                        binary_freeze_vec[np.where(timestamps > t1)[0][0]:np.where(timestamps < t2)[0][-1]] = 1
                i += 1
            else:
                i += 1
        time_val_0 = [anymaze_file.seconds[i] for i in range(1, len(anymaze_file)) if
                      anymaze_file.Freezing[i] == 0]  # time when non-freezing bout starts
        time_val_1 = [anymaze_file.seconds[i] for i in range(1, len(anymaze_file)) if
                      anymaze_file.Freezing[i] == 1]  # time when freezing bout starts
        # inds_end = [np.argmin(np.abs(timestamps - time_val)) for time_val in time_val_0] #vector of end of freezing
        # bouts
        inds_end = [int(np.argwhere(timestamps <= time_val)[-1]) for time_val in
                    time_val_0]  # list of end indices of freezing bouts
        inds_start = [int(np.argwhere(timestamps >= time_val)[0]) for time_val in
                      time_val_1]  # list of start indices of freezing bouts
        return anymaze_file, binary_freeze_vec, inds_start, inds_end

    def calc_binned_freezing(self, bins):
        bin_time = np.diff(bins)[0]
        if hasattr(self, 'OffSet'):
            bins = [x + self.OffSet for x in bins]
        per_freezing = pd.DataFrame()
        self.behavioral_data['Anymaze']['anymaze_df']['bin'] = pd.cut(
            self.behavioral_data['Anymaze']['anymaze_df'].seconds, bins, include_lowest=True)
        anymaze_df = self.behavioral_data['Anymaze']['anymaze_df'].dropna()
        unique_bins = anymaze_df.bin.unique()

        freeze_time = np.zeros(shape=(len(bins) - 1,))
        if len(unique_bins) < len(bins) - 1:
            i = len(bins) - 1 - len(unique_bins)
        else:
            i = 0
        for bin in unique_bins:
            if i > len(bins) - 2:
                freeze_time = np.zeros(shape=(len(bins) - 1,))
            else:
                # print(i)
                df = anymaze_df[anymaze_df['bin'] == bin].reset_index()
                if df.at[0, 'Freezing'] == 0:
                    even = True
                else:
                    even = False
                if even:
                    time_freezing = np.sum([x for x in np.diff(df['seconds'])[1::2]])
                    if df.Freezing.iloc[-1] == 1:
                        delta_t_e = df.at[0, 'bin'].right - df.seconds.iloc[-1]
                    else:
                        delta_t_e = 0
                    if df.Freezing.iloc[0] == 0:
                        delta_t_b = df.seconds.iloc[0] - df.at[0, 'bin'].left
                    else:
                        delta_t_b = 0
                    time_freezing += delta_t_e + delta_t_b
                    freeze_time[i] = time_freezing
                    i += 1
                else:
                    time_freezing = np.sum([x for x in np.diff(df['seconds'])[0::2]])
                    if df.Freezing.iloc[-1] == 1:
                        delta_t_e = df.at[0, 'bin'].right - df.seconds.iloc[-1]
                    else:
                        delta_t_e = 0
                    if df.Freezing.iloc[0] == 0:
                        delta_t_b = df.seconds.iloc[0] - df.at[0, 'bin'].left
                    else:
                        delta_t_b = 0
                    time_freezing += delta_t_e + delta_t_b
                    freeze_time[i] = time_freezing
                    i += 1
        setattr(self, 'binned_freezing', freeze_time / bin_time)
        return

    def calc_kinematics(self, DLC_file, bps=None, interpolate=True, int_f=100, threshold=.6):

        """
        :param DLC_file: DLC-processed csv file
        :param bps: array of labeled body parts
        :param interpolate: boolean expression, interpolates x and y coordinates for a variable number of frames whose probabilities are less than threshold amount
        :param int_f: variable for maximum number of frames to  interpolate, default is 100 if interpolate is true
        :param threshold: variable for minimum probability to threshold,  values higher  than threshold  will be kept, interpolated frames  will be set to threshold + .001
        :return: data frame  with interpolated  x and  y coordinates and probabilities, x and y centroid columns, distance, velocity, and acceleration
        Function to calculate kinematics i.e. distance, velocity, acceleration
        """

        # default  body part  array
        if bps is None:
            bps = ['snout', 'l_ear', 'r_ear', 'front_l_paw', 'front_r_paw', 'back_l_paw', 'back_r_paw',
                   'base_of_tail']

        # cleans up csv file
        df = pd.read_csv(DLC_file, header=[1, 2])
        df = df.dropna(how='all')
        df = df.dropna(1)
        df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)

        # converts to second by dividing number of frames+1 by the frame number
        df["bodyparts_coords"] = df["bodyparts_coords"].apply(lambda x: x / 29)
        # creates seconds column
        df.rename(columns={'bodyparts_coords': 'seconds'}, inplace=True)

        # sets distance, velocity, and acceleration column   time 0  to 0
        df.at[0, 'distance'] = 0
        df.at[0, 'velocity'] = 0
        df.at[0, 'acceleration'] = 0

        # interpolates for 100 frames
        if interpolate:
            # interpolates for next int_f frames
            for bp in bps:  # GOES BODY PART AT A TIME
                row = 0  # counter for row
                while row < len(df) - 1:
                    num = 0  # counter for how many nums to fill in
                    og_x = 0
                    og_y = 0

                    # finds first value where next value has below threshold prob
                    if (df.at[row, bp + '_likelihood'] > threshold) and (row != len(df)) and (
                            df.at[row + 1, bp + '_likelihood'] <= threshold):
                        # og nums, to be used later when calculating how much to add to each number
                        og_x = df.at[row, bp + '_x']
                        og_y = df.at[row, bp + '_y']
                        row += 1
                        # is going to move onto the next row
                        while (num <= int_f):
                            f = num + row  # row value is still og, num is how many frames it moves, so f is current row to check
                            if (num > int_f) or (f > len(
                                    df) - 1):  # if pass int_f frames, give up updating for this cycle, update row, and then reset everything, or if get to end of dataframe
                                row += num
                                num = 0
                                og_y = 0
                                og_x = 0
                                break
                            elif (df.at[
                                      f, bp + '_likelihood'] > threshold):  # next value is greater  than threshold, will be final value, need another loop to reupdate
                                final_x = df.at[f, bp + '_x']
                                final_y = df.at[f, bp + '_y']
                                update_x = (final_x - og_x) / (num + 1)  # number to increment x by
                                update_y = (final_y - og_y) / (num + 1)  # number to increment y by
                                n = 1
                                for ind in range(row, f):  # number to multiply update_val by
                                    df.at[ind, bp + '_x'] = og_x + update_x * n
                                    df.at[ind, bp + '_y'] = og_y + update_y * n
                                    df.at[ind, bp + '_likelihood'] = threshold + .001
                                    n += 1
                                row += num
                                num = 0
                                og_y = 0
                                og_x = 0
                                break  # break out of loop, update row
                            else:  # next value is <.6, keep going and update num
                                num += 1
                    else:
                        row += 1

        # calculates centroid values for x and y
        for i in range(len(df)):
            df.at[i, 'centroid_x'] = np.average(
                [df.at[i, f"{b_part}" + "_x"] for b_part in bps if df.at[i, f"{b_part}" + "_likelihood"] > threshold])
            df.at[i, 'centroid_y'] = np.average(
                [df.at[i, f"{b_part}" + "_y"] for b_part in bps if df.at[i, f"{b_part}" + "_likelihood"] > threshold])
            # calculates distance traveled from  last frame, velocity, and acceleration
            if (i > 0):
                df.at[i, 'distance'] = np.sqrt((df.at[i - 1, 'centroid_x'] - df.at[i, 'centroid_x']) ** 2 + (
                        df.at[i - 1, 'centroid_y'] - df.at[i, 'centroid_y']) ** 2)
                df.at[i, 'velocity'] = df.at[i, 'distance'] / (1 / 29)
                df.at[i, 'acceleration'] = ((df.at[i, 'velocity']) - (df.at[i - 1, 'velocity'])) / (1 / 29)
        # returns data  frame  with added  centroid  (x/y) columns, distance, velocity and acceleration
        return df

    def save_fp(self, filename):
        file = open(filename, 'wb')
        pkl.dump(self, file)
        file.close()
        return

    def fix_npm_flags(self):
        """This takes a preloaded npm_file i.e. you've run pd.read_csv()"""
        # set inplace to True, so that we modify original DF and do not return a virtual copy
        self.fp_df.rename(columns={"Flags": "LedState"}, inplace=True)
        self.fp_df.LedState.replace([16, 17, 18, 20], [0, 1, 2, 4], inplace=True)
        return

    def calc_avg_peak_props(self, props=None):
        """

        :param props: properties within peak properties dictionary
        :return:
        """
        if props is None:
            props = ['widths', 'areas_under_curve', 'peak_heights', 'total_events']
        condensed_props = {}
        for signal in self.peak_properties: #for the different wavelengthers (GCaMP, RCaMP, Isobestic)
            #print(self.peak_properties[signal])
            #calculates total number of events and adds to peak properties dictionary
            self.peak_properties[signal]['total_events'] = len(self.peak_properties[signal]['peak_heights'])
            condensed_props.update(
                {signal: {"average" + "_" + prop: np.average(self.peak_properties[signal][prop]) for prop in props}})
        setattr(self, "condensed_stats", condensed_props)
        return condensed_props

    def reset_peak_params(self, crit_width, curve_type, props=None):
        """

        :param crit_width: minimum width for peak to be considered an event
        :param curve_type: signal/GECI
        :param props: properties of peak_properties, i.e. widths, auc, peak heights
        :return: deletes peaks that are smaller than the critical width
        """
        #finds indexes of peaks that are smaller than the critical width
        deletion_list = [i for i, j in enumerate(self.peak_properties[curve_type]['widths']) if j < crit_width]
        if props is None:
            props = ['widths', 'areas_under_curve', 'peak_heights']
        for prop in props: #for all properties in peak properties
            #deletes peak information if peak is erreneous
            self.peak_properties[curve_type][prop] = np.delete(self.peak_properties[curve_type][prop], deletion_list)
        return

    def within_trial_eta(self, curve, event_times, window, timepoints=True):
        ind_plus = int(window // self._sample_time_)
        if timepoints:
            inds = [np.argmin(np.abs(self.Timestamps[curve] - event_time)) for event_time in event_times]
            event_times = inds
        else:
            inds = event_times
        bound_cases = int(len(list(x for x in event_times if ((x + ind_plus) - (x - int(ind_plus / 2))) < len(
            self.DF_F_Signals[curve][event_times[0] - (int(ind_plus / 2)):event_times[0] + ind_plus]))))
        if bound_cases == 0:
            part_traces = np.array(
                [self.DF_F_Signals[curve][ind - (int(ind_plus / 2)):ind + ind_plus].tolist() for ind in inds])
        else:
            part_traces = np.array(
                [self.DF_F_Signals[curve][ind - (int(ind_plus / 2)):ind + ind_plus].tolist() for ind in
                 inds[:-bound_cases]])
        eta = np.average(part_traces, axis=0)
        ci = 1.96 * np.std(part_traces, axis=0) / np.sqrt(np.shape(part_traces)[0])
        time_int = np.linspace(-(window / 2), window, len(
            self.DF_F_Signals[curve][event_times[0] - (int(ind_plus / 2)):event_times[0] + ind_plus]))
        return eta, ci, time_int

    def eliminate_extreme_values(self, curve, for_i, j):
        rolled_average = [np.average(np.diff(self.Signal[curve])[i:i + j]) for i in range(for_i)]
        indice_extrema = np.where(np.diff(rolled_average) < 0)[0] - 1
        return indice_extrema

    def calc_iei(self):
        """

        :param self: self
        :return: adds inter-event-interval array to peak properties dictionary
        """
        # finds the inter event interval between each peak
        for signal in self.peak_properties:
            if (len(self.peak_properties[signal]['peaks'])) > 2:  # if theres even an iei to calculate
                self.peak_properties[signal]['iei'] = [(self.Timestamps[signal][self.peak_properties[signal]['peaks'][i + 1]] -
                                    self.Timestamps[signal][self.peak_properties[signal]['peaks'][i]]) for i in
                                    range(len(self.peak_properties[signal]['peaks']) - 1)]

        for signal in self.neg_peak_properties:
            if (len(self.neg_peak_properties[signal]['peaks'])) > 2:  # if theres even an iei to calculate
                self.neg_peak_properties[signal]['iei'] = [
                    (self.Timestamps[signal][self.neg_peak_properties[signal]['peaks'][i + 1]] -
                     self.Timestamps[signal][self.neg_peak_properties[signal]['peaks'][i]]) for i in
                        range(len(self.neg_peak_properties[signal]['peaks']) - 1)]

    # def calc_latency(self):
    #     """
    #
    #     :param self: self
    #     :return: calculates latency and adds array to peak properties dictionary
    #     """
    #     latency = []
    #     if ('RCaMP' in self.peak_properties and 'GCaMP' in self.peak_properties):  # checks if RCaMP and GCaMP both exist
    #             ind_G = 0  # starts GCaMP index count at first index 0
    #             for ind_R in range(len(self.peak_properties['RCaMP']['peaks'])):
    #                 time_RCaMP = self.peak_properties['RCaMP']['peaks'][ind_R]  # index of when peak happens for each peak in RCaMP
    #                 while ind_G < len(self.peak_properties['GCaMP']['peaks']):  # while there's still indeces left in GCaMP array
    #                     time_GCaMP = self.peak_properties['GCaMP']['peaks'][ind_G]  # index of when peak happens for each peak in RCaMP
    #                     if time_GCaMP > time_RCaMP:  # if GCaMP peak index follows RCaMP (which it should for latency)
    #                         if time_GCaMP - time_RCaMP <= 10:  # if the difference is smaller than 10
    #                             # add time to latency
    #                             latency.append(self.Timestamps['GCaMP'][time_GCaMP] - self.Timestamps['RCaMP'][time_RCaMP])
    #                             # augment GCaMP to next index so it's not reused
    #                             ind_G += 1
    #                             # break out of while loop, move to next RCaMP entry
    #                             break
    #                         else:  # if the difference is greater than 10, then move on to next RCaMP entry
    #                             break
    #                     else:  # GCaMP index precedes RCaMP, need to move it up
    #                         ind_G += 1  # increase index to the next one
    #     self.peak_properties['latency'] = latency


# %%
class fiberPhotometryExperiment:
    def __init__(self, *args):
        self.treatment = {}
        self.task = {}
        self.curves = [arg for arg in args]
        self.dlc = {}  # csv

        for arg in args:
            if hasattr(arg, 'treatment'):
                if arg.treatment not in self.treatment and not self.treatment:
                    setattr(self, 'treatment', {arg.treatment: [arg]})
                elif arg.treatment not in self.treatment and self.treatment:
                    self.treatment[arg.treatment] = [arg]
                elif arg.treatment in self.treatment:
                    self.__add_to_attribute_dict__('treatment', arg, arg.treatment)
                else:
                    print('No treatment supplied, assuming all animals were given the same treatment.')

            if hasattr(arg, 'task'):
                if arg.task not in self.task and not self.task:
                    setattr(self, 'task', {arg.task: [arg]})
                elif arg.task not in self.task and self.task:
                    self.task[arg.task] = [arg]
                elif arg.task in self.task:
                    self.__add_to_attribute_dict__('task', arg, arg.task)
                else:
                    print('No task supplied, assuming all animals are in the same group.')

        self.__set_permutation_dicts__('task', 'treatment')
        for GECI in self.curves[1].DF_F_Signals.keys():
            self.__set_crit_width__(curve_type=GECI)

    def __add_to_attribute_dict__(self, attr, value, attr_val):
        if hasattr(self, attr):
            val_list = getattr(self, attr)[attr_val]
            val_list.append(value)
            getattr(self, attr)[attr_val] = val_list
        else:
            raise KeyError(f'{attr} not in {self}')

    def __set_permutation_dicts__(self, attr1, attr2):
        attr2_key_1 = next(iter(getattr(self, attr2).keys()))
        p = getattr(self, attr2)[attr2_key_1]
        list_of_dicts = []
        for x, y in getattr(self, attr1).items():
            group_list1 = []
            group_list2 = []
            for value in y:
                if value in p:
                    group_list1.append(value)
                else:
                    group_list2.append(value)
            try:
                list_of_dicts.append({x + "-" + list(getattr(self, attr2).keys())[0]: group_list1})
                list_of_dicts.append({x + "-" + list(getattr(self, attr2).keys())[1]: group_list2})
            except IndexError:
                list_of_dicts.append({x + "-" + list(getattr(self, attr2).keys())[0]: group_list1})
            for d in list_of_dicts:
                key = next(iter(d.keys()))
                setattr(fiberPhotometryExperiment, key, d)
        return

    @staticmethod
    def __combine_all_lists__(*args):
        combined_list = []
        for list_x in args:
            combined_list += list_x
        return combined_list

    def __get_curve_peak_attrs__(self, curve_type='GCaMP'):
        pos_list = [x.peak_properties[curve_type]['widths'].tolist() for x in self.curves]
        neg_list = [x.neg_peak_properties[curve_type]['widths'].tolist() for x in self.curves]
        return self.__combine_all_lists__(*pos_list), self.__combine_all_lists__(*neg_list)

    def find_critical_width(self, pos_wid, neg_wid):
        """
        :param pos_wid:
        :param neg_wid:
        :return:
        """
        wid_list = list(set(pos_wid + neg_wid))
        wid_list.sort()
        if len(pos_wid) > 0 and len(neg_wid) == 0:
            critical_width = 0
            warn.warn('There are no negative going transients in your signal. This is not necessarily a problem, '
                      'but you should confirm that this is truly the case!')
        else:
            i = 0
            try:
                pos = pos_wid
                neg = neg_wid
                while len(pos) / (len(pos) + len(neg)) < 0.99:
                    pos = [pos_i for pos_i in pos if pos_i > wid_list[i]]
                    neg = [neg_i for neg_i in neg if neg_i > wid_list[i]]
                    i += 1
                else:
                    critical_width = wid_list[i]
            except ZeroDivisionError:
                neg_wid.sort()
                neg_wid = neg_wid[:-1]
                return self.find_critical_width(pos_wid, neg_wid)
            # except IndexError:

        return critical_width

    def __set_crit_width__(self, curve_type='GCaMP'):
        pos_list, neg_list = self.__get_curve_peak_attrs__(curve_type)
        crit_wid = self.find_critical_width(pos_list, neg_list)
        setattr(self, "crit_width" + "_" + curve_type, crit_wid)
        for curve in self.curves:
            #calls reset peak params function that deletes erroneous peaks (below the width)
            curve.reset_peak_params(getattr(self, "crit_width" + "_" + curve_type), curve_type)
            curve.calc_avg_peak_props()
        return

    def comparative_statistics(self, group1, group2, metric, curve='GCaMP', test=scipy.stats.ttest_ind):
        s1 = [y for y in next(iter(getattr(self, group1).values()))]
        s2 = [y for y in next(iter(getattr(self, group2).values()))]
        sample1 = [x.condensed_stats[curve]['average' + '_' + metric] for x in s1]
        sample2 = [x.condensed_stats[curve]['average' + '_' + metric] for x in s2]
        stat, pval = test(sample1, sample2)
        return stat, pval, sample1, sample2

    def raster(self, group, curve, a, b, colormap):

        """
        Plots a heatmap of specified signal over time interval [a:b]
        :param group: str, desired group to plot
        :param curve: str, type of signal to plot (GCaMP or Isobestic_GCaMP)
        :param a: int, start index
        :param b: int, end index
        :param colormap: str, matplotlib color map scheme
        :return:
        """
        fig, ax = plt.subplots()
        vector_array = np.array(
            [vec.DF_Z_Signals[curve][a:b].tolist() for vec in next(iter(getattr(self, group).values()))])
        raster = sb.heatmap(vector_array, vmin=0, vmax=8, cmap=colormap, ax=ax,
                            cbar_kws={'label': r'z-scored $\frac{dF}{F}$ (%)', 'location': 'left'})
        raster.xaxis.set_ticks(np.arange(0, 3920, 560), [0, 60, 120, 180, 240, 300, 360])
        raster.yaxis.set_ticks([])
        plt.title('No Shock')
        plt.xlabel('Time (s)')
        plt.show()
        return fig

    # def st_event_triggered_average(self, curve, event_time, window, group, plot=False, timepoint=True):
    #     time = [time.Timestamps[curve].tolist() for time in next(iter(getattr(self, group).values()))]
    #     max_ind = np.min([len(x) for x in time])
    #     time_array = np.array(
    #         [time.Timestamps[curve][0:max_ind].tolist() for time in next(iter(getattr(self, group).values()))])
    #     try:
    #         average_time = np.average(time_array, axis=0)
    #     except ZeroDivisionError:
    #         average_time = np.zeros(shape=(1, len(time_array)))
    #         average_time[0] = 0
    #         average_time[1:] = np.average(time_array[1:], axis=0)
    #     if timepoint:
    #         ind = np.argmin(np.abs(average_time - event_time))
    #     else:
    #         ind = event_time
    #     index_right_bound = np.argmin(np.abs(average_time - (event_time + window)))
    #     index_left_bound = np.argmin(np.abs(average_time - (event_time - (window / 2))))
    #     vector_array = np.array([vec.DF_F_Signals[curve][index_left_bound:index_right_bound].tolist() for vec in
    #                              next(iter(getattr(self, group).values()))])
    #     try:
    #         averaged_trace = np.average(vector_array, axis=0)
    #     except ZeroDivisionError:
    #         averaged_trace = np.zeros(shape=(1, len(vector_array)))
    #         averaged_trace[0] = 0
    #         averaged_trace[1:] = np.average(vector_array[1:], axis=0)
    #     ci = 1.96 * np.std(vector_array, axis=0) / np.sqrt(np.shape(vector_array)[0])
    #     if plot:
    #         fig, ax = plt.subplots()
    #         plt.axvline(average_time[ind], linestyle='--', color='black')
    #         ax.plot(average_time[index_left_bound:index_right_bound], averaged_trace)
    #         ax.fill_between(average_time[index_left_bound: index_right_bound], (averaged_trace - ci),
    #                         (averaged_trace + ci), color='b', alpha=0.1)
    #         plt.show()
    #     return averaged_trace, average_time[index_left_bound:index_right_bound], ci

    def mt_event_triggered_average(self, curve, event_times, window, group, ci_type='t', shuffle=False,
                                   timepoint=False):
        max_ind = np.min(
            [len(x) for x in [t.Timestamps[curve].tolist() for t in next(iter(getattr(self, group).values()))]])
        time_array = np.array(
            [time.Timestamps[curve][0:max_ind].tolist() for time in next(iter(getattr(self, group).values()))])
        # we make an assumption here that all animals were recorded at same fps, ergo, the sample_time should be the
        # same for all animals
        sample_time = np.diff(time_array[0])[1]
        ind_plus = window / sample_time
        vector_array = np.array(
            [trace.DF_F_Signals[curve][0:max_ind].tolist() for trace in next(iter(getattr(self, group).values()))])
        if shuffle:
            vector_array_copy = vector_array.T
            np.random.shuffle(vector_array_copy)
            vector_array = vector_array_copy.T
        inds = []
        if timepoint:
            for i in range(len(event_times)):
                tps = [np.argmin(np.abs(time_array[i] - event_times[i][j])) for j in range(len(event_times[i]))]
                inds.append(tps)
        else:
            inds = event_times
        mt_eta = []
        for animal in range(np.shape(vector_array)[0]):
            if len(event_times) != 1:
                trace_len = int(ind_plus) + int(ind_plus / 2)
                part_traces = [vector_array[animal][indice - (int(ind_plus / 2)):int(indice + ind_plus)].tolist() for
                               indice in inds[animal]]
                part_traces = [trace for trace in part_traces if len(trace) == trace_len]
            else:
                part_traces = np.array(
                    [vector_array[animal][indice - (int(ind_plus / 2)):int(indice + ind_plus)].tolist() for indice in
                     inds[0]])
            eta = np.average(np.array(part_traces), axis=0)
            mt_eta.append(eta)
        mt_eta = np.array(mt_eta)
        av_tr = np.average(mt_eta, axis=0)
        if ci_type == 't':
            ci = 1.96 * np.std(mt_eta, axis=0) / np.sqrt(np.shape(mt_eta)[0])
        elif ci_type == 'bs':
            ci = self.bootstrap_ci(mt_eta, 0.05, niter=1000)
        else:
            ci = 0
        sem = np.std(mt_eta, axis=0) / np.sqrt(np.shape(mt_eta)[0])
        time_int = np.linspace(-window / 2, window, len(av_tr))
        return av_tr, mt_eta, time_int, ci, sem

    def bootstrap_ci(self, vector_array, sig, niter=1000):
        rng = np.random.default_rng()
        random_bst_vec = rng.integers(low=0, high=int(np.size(vector_array, 0)), size=niter)
        bootstrap_mat = vector_array[random_bst_vec]
        ci_lower_upper = np.percentile(bootstrap_mat, [sig / 2, 100 - (sig / 2)], axis=0)
        return ci_lower_upper

    # def plot_st_eta(self, curve, event_time, window, *args):
    #     for arg in args:
    #         av_tr, av_ti, ci = self.st_event_triggered_average(curve, event_time, window, arg)
    #         ti_ind = np.argmin(np.abs(av_ti - event_time))
    #         plt.axvline(av_ti[ti_ind], linestyle='--', color='black')
    #         plt.plot(av_ti, av_tr)
    #         plt.fill_between(av_ti, (av_tr - ci), (av_tr + ci), alpha=0.1)
    #         plt.show()
    #     return

    def plot_mt_eta(self, curve, event_times, window, *args):
        fig, ax = plt.subplots(1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\frac{dF}{F}$ (%)')
        for i in range(len(args)):
            if len(event_times) == 1:
                av_tr, mt_eta, av_ti, ci, sem = self.mt_event_triggered_average(curve, event_times, window, args[i])
                ax.plot(av_ti, av_tr)
                ax.fill_between(av_ti, (av_tr - sem), (av_tr + sem), alpha=0.1, label=None)
            else:
                av_tr, mt_eta, av_ti, ci, sem = self.mt_event_triggered_average(curve, event_times[i], window, args[i])
                ax.plot(av_ti, av_tr)
                ax.fill_between(av_ti, (av_tr - sem), (av_tr + sem), alpha=0.1, label=None)
        ax.axvline(0, linestyle='--', color='black')
        return fig, ax

    def percent_freezing(self, bins, g1, g2):
        freeze_df = pd.DataFrame(columns=['Animal', 'Group'])
        i = 0
        for g in (g1, g2):
            for animal in list(getattr(self, g).values())[0]:
                animal.calc_binned_freezing(bins)
                if hasattr(animal, 'ID'):
                    freeze_df.loc[i, 'Animal'] = animal.ID
                freeze_df.loc[i, 'Group'] = g
                j = 0
                for b in animal.binned_freezing:
                    freeze_df.loc[i, 'bin' + str(j)] = b
                    j += 1
                i += 1
        return freeze_df

    def create_timeseries(self, group, curve='GCaMP', time_length=0):
        """
        Creates array of specified signals for traces in each group, return value used in make_3d_timeseries()
        Maybe incorporate with make_3d_timeseries, or just add to the front of method
        :param group: str, desired group to grab data from
        :param curve: str, type of curve
        :param time_length: int, desired length of time to slice and make timeseries from
        :return: timeseries, matrix of n animals in group with respective curve from time[0:time_length]
        """
        min_time = np.min(
            [len(x) for x in [t.Timestamps[curve].tolist() for t in
                              next(iter(getattr(self, group).values()))]])  # shortest timeseries in group
        if time_length == 0:  # did not pass in desired time_length, default to min_time within group
            time_length = min_time
        elif time_length > min_time:  # desired time_length longer than min_time
            raise Exception(
                "Desired time length is longer than at least one trace within this group, consider lowering desired time length!")
        timeseries = np.array([series for series in [t.DF_F_Signals[curve][0:time_length].tolist() for t in
                                                     next(iter(getattr(self, group).values()))]])
        return timeseries

    def make_3d_timeseries(self, timeseries, timestamps, x_axis='Time (s)', y_axis='Animal ID', z_axis='dF/F',
                           **kwargs):
        """
        Plots a 3d plot of each animal's dF/F GCaMP trace across time
        :param timeseries: array of n animals each with signal of length m, returned from create_timeseries()
        :param timestamps: timestamps array of length m
        :param x_axis: str, x-axis Time (s)
        :param y_axis: str, y-axis Animal ID
        :param z_axis: str, z-axis dF/F GCaMP
        :param kwargs: ??
        :return: empty
        """
        sb.set()
        if not isinstance(timeseries, np.array):
            timeseries = np.asarray(timeseries)
        if not isinstance(timestamps, np.array):
            timestamps = np.asarray(timestamps)
        if np.shape(timeseries)[1] != np.shape(timestamps)[0]:  # check if timeseries and timestamps match
            raise ValueError(
                "Shape of timeseries and timestamp data do not match! Perhaps, try transposing? If not, you may have "
                "concatenated incorrectly.")
        y_coordinate_matrix = np.zeros(shape=(np.shape(timeseries)[0], np.shape(timeseries)[1]))  # n by m zero matrix
        for i in range(len(timeseries)):
            y_coordinate_matrix[
                i] = i + 1  # fill rows with 1s, 2s, 3s, down to n. maybe adjust to scale down so lines are closer
        plt.figure()
        axs = plt.axes(projection="3d")
        for j in range(len(timeseries)):
            axs.plot(timestamps, y_coordinate_matrix[j], timeseries[j])  # add back **kwargs?
        axs.set_xlabel(x_axis)  # timestamps 'Time (s)'
        axs.set_ylabel(y_axis)  # annies 'Animal ID'
        axs.set_zlabel(z_axis)  # timeseries/signal 'dF/F GCaMP'
        return

    # def plot_stats(self):
    # compare stats between groups

    # Practice things
    # engram_exp = fiberPhotometryExperiment(engram_recall_1, engram_recall_2, sham_recall_1, sham_recall_2)
    # z = engram_exp.event_triggered_average('Recall-ChR2', 'GCaMP', 1, 40)

    # fc_2 = fiberPhotometryCurve('/home/ryansenne/Data/Rebecca/Test_Pho_engram_ChR2_m2_FC.csv', None,
    #                             **{'treatment': 'ChR2', 'task': 'FC'})

    # fc_experiment = fiberPhotometryExperiment(fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7, fc_8)

    # z, z1, z2 = fc_experiment.event_triggered_average('GCaMP', 124, 10, 'FC-ChR2')
    # fc_experiment.plot_eta('GCaMP', 124, 10, 'FC-ChR2', 'FC-eYFP')

    # b_test = b_spline.bSpline(120, 3, 9)
    # maps, dicts = b_test.create_spline_map([1100, 1650, 2200, 2800], len(fc_1.DF_F_Signals['GCaMP']))
    # model = fc_1.fit_general_linear_model('GCaMP', dicts)
    # model.predict(sm.add_constant(sm.add_constant(pd.DataFrame(dicts))))

    # time = fc_1.Timestamps['GCaMP']
    # my_behave_file = pd.read_csv('/Users/ryansenne/Desktop/Rebecca_Data/Engram_Round2_FC_Freeze - ChR2_m1.csv')
    # x, y, z = fc_1.process_anymaze(my_behave_file, time)
# %%
    """fc_prac = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_shock_fc/m1.csv', **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 2501.411968})
    fc_prac.visual_check_peaks('GCaMP')"""
    #issue: negative peak properties array populated with GCaMP and RCaMP, not peak_properties
    #other issue: visual check peaks doesn't work

    #opto bois fc shock chr2
    m1 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m1.csv', **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 333.83184})
    # m2 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m2.csv', **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 1048.644864})
    # m3 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m3.csv', **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 1644.654624})
    # m4 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m4.csv', **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 2220.480448})
    # m5 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m5.csv', **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 2804.140576})
    # m6 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m6.csv', **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 3367.9992})
    # m7 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m7.csv',
    #                           **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 3934.82928})
    # m8 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m8.csv',
    #                           **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 4463.946304})
    # m9 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m9.csv',
    #                           **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 4985.380768})
    # m10 = fiberPhotometryCurve('/Users/michellebuzharsky/Downloads/opto_final_fc/m10.csv',
    #                           **{'treatment': 'ChR2', 'task': 'FC', 'keystroke': 5512.40848})




    #cohort = fiberPhotometryExperiment(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10)
