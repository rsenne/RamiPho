import pickle as pkl
import warnings as warn
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sb
import statsmodels.api as sm
from scipy import sparse
from scipy.integrate import simpson
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve
from FiberPho.anymaze_analysis import anymazeResults
from FiberPho.dlc_analysis import dlcResults
import pykalman

__all__ = ["fiberPhotometryCurve", "fiberPhotometryExperiment", "FiberPhotometryCollection"]


class fiberPhotometryCurve:
    def __init__(self, npm_file: str, dlc_file:str=None, offset:float=None, anymaze_file:str=None, regress:bool=True, ID:str=None, task:str=None, treatment:str=None, smoother='kalman'):
        """
        :param npm_file: str Path to csv fiber photometry file gathered using a neurophotometrics fp3002 rig and bonsai software
        :param behavioral_data: Path(s) to csv files, either deeplabcut or anymaze, for doing behavioral analysis
        :param keystroke_offset: Value corresponding to a keystroke press in bonsai for removing irrelevant data
        :param manual_off_set: Value obtained from calculating offset from expected event i.e. blue light and its theoretical appearance in video
        :param kwargs: dict containing values such as "ID", "task", and/or "treatment" note: task and treatment necessary for use in fiberPhotometryExperiment

        """
        # these should always be present
        self.npm_file = npm_file
        self.fp_df = pd.read_csv(self.npm_file)
        self.__T0__= self.fp_df['Timestamp'][1]
        self.behavioral_data = {}
        self.ID = ID
        self.task = task
        self.treatment = treatment
        self.anymaze_file = anymaze_file
        self.DLC_file = dlc_file
        self.regress = regress
        self.smoother = smoother
        self.raw_signal = None
        self.timestamps = None
        self.dff_signals = None
        self.dfz_signals = None
        self.anymaze_results = None
        self.dlc_results = None

        if offset is None:
            self.offset = 0
        else:
            self.offset = offset

        # determine sample time
        self._sample_time_ = np.diff(self.fp_df['Timestamp'])[1]
        self.fps = 1/self._sample_time_

        # check to see if using old files
        if "Flags" in self.fp_df.columns:
            self.fix_npm_flags()
            print("Old NPM format detected, changing Flags to LedState")

        # do preprocessing as part of initilization
        self._process_data()
        self.region_peak_properties = self.find_signal()


    def __iter__(self):
        return iter(list(self.dff_signals.values()))

    def __eq__(self, other):
        if not isinstance(other, fiberPhotometryCurve):
            raise TypeError("You can only compare the identity of a fiber photometry curve to another fiber"
                            "photometry curve!!")
        val1 = self.dff_signals.values()
        val2 = other
        truthiness_array = [self.__arrays_equal__(a, b) for a, b in zip(val1, val2)]
        if any(truthiness_array):
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.dff_signals.values())
    
    def __getitem__(self, idx):
        return self.dff_signals[idx]

    def _extract_data(self):
        # Initialize the data structures
        raw_signal = {}
        isobestic_data = {}
        timestamps = {}

        # led states
        led_state = [1, 2, 4]

        # Get the unique region columns
        region_columns = self.fp_df.columns[self.fp_df.columns.str.contains('Region')].tolist()

        # Exclude data before offset
        self.fp_df = self.fp_df[self.fp_df['Timestamp'] >= self.offset + self.__T0__].reset_index(drop=True)
        
        # creat new initial time variable for later correction
        temp_t0 = self.fp_df.loc[0, 'Timestamp']

        # Find the length of the shortest trace
        shortest_trace_length = float('inf')

        # Process each region column
        for column in region_columns:

            # Filter the data based on LEDState and Region column
            region_data = self.fp_df[(self.fp_df['LedState'] == 2) if column.endswith('G') else (self.fp_df['LedState'] == 4)]
            

            # Store the region-specific information
            if not region_data.empty:
                raw_signal[column] = region_data[column].reset_index(drop=True)
                isobestic_data[column] = self.fp_df[self.fp_df['LedState'] == 1][column].reset_index(drop=True)

                # Update the length of the shortest trace
                shortest_trace_length = min(shortest_trace_length, len(region_data[column]))

        for state in led_state:
            if not self.fp_df[self.fp_df['LedState'] == state].Timestamp.empty:
                # Store the deinterleaved timestamps for the LEDState
                timestamps[str(state)] = self.fp_df[self.fp_df['LedState'] == state].Timestamp.reset_index(drop=True) - temp_t0

        # Trim the signals and timestamps to the length of the shortest trace
        for column, led_state in zip(raw_signal.keys(), led_state):
            raw_signal[column] = raw_signal[column][:shortest_trace_length]
            isobestic_data[column] = isobestic_data[column][:shortest_trace_length]
            timestamps[str(led_state)] = timestamps[str(led_state)][:shortest_trace_length]

        self.raw_signal = raw_signal
        self.isobestic_channel = isobestic_data
        self.Timestamps = timestamps
        return
    
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
    def _als_detrend(y, lam=10e7, p=0.05, niter=100):  # asymmetric least squares smoothing method
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
    def _df_f(raw, method="standard"):
        """
        :param raw: a smoothed baseline-corrected array of fluorescence values
        :param kind: whether you want standard df/f or if you would like a z-scored scaling
        :return: df/f standard or z-scored
        Function to calculate DF/F (signal - median / median or standard deviation).
        """
        if method == "standard":
            F0 = np.median(raw)  # median value of time series
            df_f = (raw - F0) / F0
        elif method == "z_scored":  # z-scored
            F0 = np.mean(raw)
            df_f = (raw - F0) / np.std(raw)
        elif method == "percentile":
            F0 = np.percentile(raw, 0.08)
            df_f = (raw - F0) / F0
        return df_f
    
    @staticmethod
    def _fit_regression(endog, exog, summary=False):
        model = sm.OLS(endog, sm.add_constant(exog)).fit()
        if summary:
            print(model.summary())
        return model.resid
    
    @staticmethod
    def _kalman_filter(signal):
        ar_model = sm.tsa.ARIMA(signal, order=(3, 0, 0), trend='n').fit()
        A = np.zeros((3, 3))
        A[:, 0] = ar_model.params[:-1]
        A[1, 0] = 1
        A[2, 1] = 1
        H = np.array([1, 0, 0])
        kf = pykalman.KalmanFilter(transition_matrices=A, observation_matrices=H, initial_state_covariance=np.eye(3), initial_state_mean=(0, 0, 0), em_vars=['transition_covariance', 'observation_covariance'])
        kf.em(signal, em_vars=['transition_covariance', 'observation_covariance'])
        means, covs = kf.smooth(signal)
        return means[:, 0]

    @staticmethod
    def _smooth(signal, kernel, visual_check=False):
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
    
    def _process_data(self):
        if self.raw_signal is None:
            self._extract_data()
        

        assert set(self.raw_signal.keys()) == set(self.isobestic_channel.keys()) 
        dff_signal = {}
        dfz_signal = {}

        for key in self.raw_signal.keys():
            # get the timeseries
            region_timeseries = self.raw_signal[key]
            isobestic_timeseries = self.isobestic_channel[key]

            # Perform baseline correction
            region_timeseries_corrected = self._als_detrend(region_timeseries)
            isobestic_timeseries_corrected = self._als_detrend(isobestic_timeseries)
            
            # smooth out the data
            if self.smoother == 'kalman':
                region_timeseries_corrected = self._kalman_filter(region_timeseries_corrected)
            else:
                region_timeseries_corrected = self._smooth(region_timeseries_corrected, kernel=15)

            # Calculate df/f for each timeseries
            region_timeseries_corrected = self._df_f(region_timeseries_corrected)
            isobestic_timeseries_corrected = self._df_f(isobestic_timeseries_corrected)


            # Calculate z-scored df/f for each timeseries
            region_timeseries_corrected_z = self._df_f(region_timeseries_corrected, method="z_scored")
            isobestic_timeseries_corrected_z = self._df_f(isobestic_timeseries_corrected, method="z_scored")

            # Regress out the isobestic signal from the region signal
            if self.regress is True:
                region_timeseries_corrected = self._fit_regression(region_timeseries_corrected, isobestic_timeseries_corrected)
                region_timeseries_corrected_z = self._fit_regression(region_timeseries_corrected_z, isobestic_timeseries_corrected_z)

            dff_signal[key] = region_timeseries_corrected
            dfz_signal[key] = region_timeseries_corrected_z

        self.dff_signals = dff_signal
        self.dfz_signals = dfz_signal
    
    def process_behavioral_data(self):
        self.anymaze_results = anymazeResults(self.anymaze_file)
        self.dlc_results = dlcResults(self.dlc_file)
        return

    @staticmethod
    def calc_area(l_index, r_index, timeseries):
        areas = np.asarray([simpson(timeseries[i:j]) for i, j in zip(l_index, r_index)])
        return areas

    def find_signal(self, neg=False):
        peak_properties = {}
        if not neg:
            for region, sig in self.dff_signals.items():
                peaks, properties = find_peaks(sig, height=np.std(sig), distance=131, width=25,
                                               rel_height=0.5)  # height=1.0, distance=130, prominence=0.5, width=25, rel_height=0.90)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.dff_signals[region])
                properties['widths'] *= self._sample_time_
                peak_properties[region] = properties
        else:
            for region, sig in self.dff_signals.items():
                peaks, properties = find_peaks(-sig, height=np.std(sig), distance=131, width=25, rel_height=0.5)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.dff_signals[region])
                properties['widths'] *= self._sample_time_
                peak_properties[region] = properties
        return peak_properties

    def visual_check_peaks(self, signal):
        """
        Plots object's df_f signal overlayed with peaks
        :param signal: string of which signal to check
        :return:
        """
        if hasattr(self, "region_peak_properties"):
            plt.figure()
            plt.plot(self.dff_signals[signal])
            plt.plot(self.region_peak_properties[signal]['peaks'],
                     self.dff_signals[signal][self.region_peak_properties[signal]['peaks']], "x")
            plt.vlines(x=self.region_peak_properties[signal]['peaks'],
                       ymin=self.dff_signals[signal][self.region_peak_properties[signal]['peaks']] -
                            self.region_peak_properties[signal]["prominences"],
                       ymax=self.dff_signals[signal][self.region_peak_properties[signal]['peaks']], color="C1")
            plt.hlines(y=self.region_peak_properties[signal]["width_heights"], xmin=self.region_peak_properties[signal]["left_ips"],
                       xmax=self.region_peak_properties[signal]["right_ips"], color="C1")
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

    def within_trial_eta(self, curve, event_times, window, timepoints=True):
        ind_plus = int(window // self._sample_time_)
        if timepoints:
            inds = [np.argmin(np.abs(self.Timestamps[curve] - event_time)) for event_time in event_times]
            event_times = inds
        else:
            inds = event_times
        bound_cases = int(len(list(x for x in event_times if ((x + ind_plus) - (x - int(ind_plus / 2))) < len(
            self.dff_signals[curve][event_times[0] - (int(ind_plus / 2)):event_times[0] + ind_plus]))))
        if bound_cases == 0:
            part_traces = np.array(
                [self.dff_signals[curve][ind - (int(ind_plus / 2)):ind + ind_plus].tolist() for ind in inds])
        else:
            part_traces = np.array(
                [self.dff_signals[curve][ind - (int(ind_plus / 2)):ind + ind_plus].tolist() for ind in
                 inds[:-bound_cases]])
        eta = np.average(part_traces, axis=0)
        ci = 1.96 * np.std(part_traces, axis=0) / np.sqrt(np.shape(part_traces)[0])
        time_int = np.linspace(-(window / 2), window, len(
            self.dff_signals[curve][event_times[0] - (int(ind_plus / 2)):event_times[0] + ind_plus]))
        return eta, ci, time_int

    def metric_df(self, columns=None):
        if columns is None:
            columns = ['peaks', 'peak_heights', 'areas_under_curve', 'widths']
        if self.__DUAL_COLOR:
            metric_df_gcamp = pd.DataFrame(self.region_peak_properties['GCaMP'], columns=columns)
            metric_df_gcamp.loc[:, 'Timestamps'] = self.Timestamps['GCaMP'][self.region_peak_properties['GCaMP']['peaks']]
            metric_df_rcamp = pd.DataFrame(self.region_peak_properties['RCaMP'], columns=columns)
            metric_df_rcamp.loc[:, 'Timestamps'] = self.Timestamps['RCaMP'][self.region_peak_properties['RCaMP']['peaks']]
            return metric_df_gcamp, metric_df_rcamp
        else:
            metric_df_gcamp = pd.DataFrame(self.region_peak_properties['GCaMP'], columns=columns)
            metric_df_gcamp.loc[:, 'Timestamps'] = self.Timestamps['GCaMP'][self.region_peak_properties['GCaMP']['peaks']]
            return metric_df_gcamp


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
        for region in self.curves[1].dff_signals.keys():
            self.__set_crit_width__(curve_type=region)

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
                while len(pos) / (len(pos) + len(neg)) < 0.90:
                    pos = [pos_i for pos_i in pos if pos_i > wid_list[i]]
                    neg = [neg_i for neg_i in neg if neg_i > wid_list[i]]
                    i += 1
                else:
                    critical_width = wid_list[i]
            except ZeroDivisionError:
                neg_wid.sort()
                neg_wid = neg_wid[:-1]
                return self.find_critical_width(pos_wid, neg_wid)
        return critical_width

    def __set_crit_width__(self, curve_type='GCaMP'):
        pos_list, neg_list = self.__get_curve_peak_attrs__(curve_type)
        crit_wid = self.find_critical_width(pos_list, neg_list)
        setattr(self, "crit_width" + "_" + curve_type, crit_wid)
        for curve in self.curves:
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

    # def create_metric_df():
    #     s1 = [y for y in next(iter(getattr(self, group1).values()))]
    #     for i in s1:
    #     return metric_df

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

    def mt_event_triggered_average(self, curve, event_times, window, group, ci_type='t', shuffle=False,
                                   timepoint=True):
        max_ind = np.min(
            [len(x) for x in [t.Timestamps[curve].tolist() for t in next(iter(getattr(self, group).values()))]])
        time_array = np.array(
            [time.Timestamps[curve][0:max_ind].tolist() for time in next(iter(getattr(self, group).values()))])
        # we make an assumption here that all animals were recorded at same fps, ergo, the sample_time should be the
        # same for all animals
        sample_time = np.diff(time_array[0])[1]
        ind_plus = window / sample_time
        vector_array = np.array(
            [trace.dff_signals[curve][0:max_ind].tolist() for trace in next(iter(getattr(self, group).values()))])
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
                part_traces = [x for x in part_traces if len(x) == trace_len]
                eta = np.average(np.array(part_traces), axis=0)
                mt_eta.append(eta)
            else:
                part_traces = np.array(
                    [vector_array[animal][indice - (int(ind_plus / 2)):int(indice + ind_plus)].tolist() for indice in
                     inds[0]])
                eta = np.average(np.array(part_traces), axis=0)
                mt_eta.append(eta)
                # eta = np.average(np.array(part_traces), axis=0)
            # mt_eta.append(eta)
        mt_eta = np.array(mt_eta) - np.median(mt_eta, keepdims=True, axis=1)
        av_tr = np.average(mt_eta, axis=0)
        av_tr = av_tr - np.median(av_tr)
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
        bootstrap_mat = np.zeros(shape=(niter, np.size(vector_array, 1)))
        for i in range(niter):
            random_bst_vec = rng.integers(low=0, high=int(np.size(vector_array, 0)), size=np.size(vector_array, 0))
            sampled_mat = vector_array[random_bst_vec]
            average_trace = np.average(sampled_mat, axis=0)
            bootstrap_mat[i, :] = average_trace
        ci_lower_upper = np.percentile(bootstrap_mat, [sig / 2, 100 - (sig / 2)], axis=0)
        return ci_lower_upper

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
        timeseries = np.array([series for series in [t.dff_signals[curve][0:time_length].tolist() for t in
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


class FiberPhotometryCollection:
    def __init__(self, name):
        self.name = name
        self.curves = {}

    def __getitem__(self, attributes):
        """Indexing method for class.

        Usage: my_collection["fc", "ChR2"]
        Args:
            attributes (str): Attributes given e.g. task and treatment variables.
        Returns:
            filtered_curves (list(fiberPhotometryCurve)): List of all fp curves that meet the set of criteria set by the user. I.e. animals that wear fear conditioned and had ChR2.
        """
        filtered_curves = []
        for curve in self.curves.values():
            if (curve.task, curve.treatment) == attributes:
                filtered_curves.append(curve)
        return filtered_curves

    def add_curve(self, *args):
        """add_curve: add a fiberPhotometryCUrve object to the collection for analysis.
            Args:
                *args(fiberPhotometryCurve): fiberPhotometryCurves
        """
        for arg in args:
            if arg.ID is None:
                i = len(self.curves.values())
                print(f"No name supplied for this curve. Defaulting to 'Curve {i}' as the name. Consider updating this name.")
                self.curves.update({f"Curve {i}":arg})
            else:
                self.curves.update({arg.ID:arg})
                
    def peak_dict(self, region, pos=True):
        """Used to create a dictionary that maps each individual curve ID to its event attributes. The absic idea being this will be apssed to a later function for a well formatted df.
        Args:
            region (str, optional): Which region peak properties to look at.
        Returns:
            peak_dict (dict): dictionary that maps each curve to its peak properties e.g. AUC, fwhm, etc.
        """
        if pos:
            id = [k for k in self.curves]
            peak_times = [v.Timestamps[region][v.peak_properties[region]['peaks']] for v in self.curves.values()]
            peak_heights = [v.peak_properties[region]['peak_heights'] for v in self.curves.values()]
            auc = [v.peak_properties[region]["areas_under_curve"] for v in self.curves.values()]
            fwhm = [v.peak_properties[region]["widths"] for v in self.curves.values()]
            peak_dict = {"ID": id, "Peak_Times": peak_times, "Amplitudes": peak_heights, "AUC":auc, "FWHM": fwhm}
        else:
            id = [k for k in self.curves]
            peak_times = [v.Timestamps[region][v.neg_peak_properties[region]['peaks']] for v in self.curves.values()]
            peak_heights = [-v.neg_peak_properties[region]['peak_heights'] for v in self.curves.values()]
            auc = [v.neg_peak_properties[region]["areas_under_curve"] for v in self.curves.values()]
            fwhm = [v.neg_peak_properties[region]["widths"] for v in self.curves.values()]
            peak_dict = {"ID": id, "Peak_Times": peak_times, "Amplitudes": peak_heights, "AUC":auc, "FWHM": fwhm}
        return peak_dict

    def histogram_2d(self, region):
        """Plots a 2D event histogram where Y=Amplitudes and X=FWHM. The idea is to see if there is an easily defined threshold where all events inside that 2D bin are negative.

        Args:
            region (str): Which indicator curve to plot the 2D event histogram for.
        """
        pos_peak_dict = self.peak_dict(region=region, pos=True)
        neg_peak_dict = self.peak_dict(region=region, pos=False)
        event_df = pd.concat((pd.DataFrame(pos_peak_dict).apply(pd.Series.explode).reset_index(drop=True), pd.DataFrame(neg_peak_dict).apply(pd.Series.explode).reset_index(drop=True)))
        event_df.dropna(inplace=True)
        plt.hist2d(x='Amplitudes', y='FWHM', data=event_df)
        plt.show()
        return

    def eliminate_events(self):
        pass

    def event_summaries(self, region):
        """Creates a DataFrame that contains all of the relevant event information e.g. AUC, FWHM, etc. This can be used for further analysis e.g. stats across groups linear mixed models etc.
        Args:
            region (str, optional): String to grab region trace of choice e.g. Region0G, Region1R, etc.
        Returns:
            df_transformed(pd.DataFrame): A DataFrame that contains a conglomeration of all the events across animals. 
        """
        dict = self.peak_dict(region)
        df = pd.DataFrame(dict)
        df_transformed = df.apply(pd.Series.explode).reset_index(drop=True)
        return df_transformed
    

    def raster_plot(self, task, treatment, region, xtick_range=None, xtick_freq=None):
        """Raster plot: Generate a raster plot of Z-scored fiber photometry traces.
        Args:
            task (str): String that represents the task e.g. FC, Recall, Ext etc.Should be identical to what was passed in fiberPhotometryCurve.
            treatment (str): String that represents the treatment e.g. eYFP, ChR2, Shock etc. Should be identical to what was passed in fiberPhotometryCurve.
            region (str): String to grab region trace of choice e.g. Region0G, Region1R, etc.
            xtick_range (int, optional): Length in time of session. Defaults to None.
            xtick_freq (int, optional): How many labels in [0, xtick_range]; end points inclusive. Defaults to None.
        Returns:
            matplotlib figure: a matplotlib figure object
            matplotlib axis: a matplotlib axis object
        """
        curves = self[task, treatment]
        min_len = np.min([len(curve[region]) for curve in curves])
        raster_array = np.zeros(shape=(len(curves), min_len))
        for i, curve in enumerate(curves):
            raster_array[i] = curve.DF_Z_Signals[region][:min_len]

        fig, ax = plt.subplots()
        sb.heatmap(raster_array, cbar=True, cbar_kws={"label":r"$\frac{dF}{F}$"}, center=0, yticklabels=False, ax=ax)
        ax.set_xlabel('Time (s)')
        if xtick_range and xtick_freq is not None:
            ax.set_xticks(np.linspace(0, min_len, xtick_freq), labels=np.linspace(0, xtick_range, xtick_freq, dtype=np.int))
        return fig, ax
