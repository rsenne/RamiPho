import pickle as pkl
import warnings as warn
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sb
import statsmodels.api as sm
import FiberPho.b_spline
from scipy import sparse
from scipy.integrate import simpson
from scipy.interpolate import splrep, splev
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve

__all__ = ["fiberPhotometryCurve", "fiberPhotometryExperiment"]


class fiberPhotometryCurve:
    def __init__(self, npm_file: str, behavioral_data: str = None, keystroke_offset=None, manual_off_set=None,
                 remove_extrema=False,
                 **kwargs):
        """
        :param npm_file: str Path to csv fiber photometry file gathered using a neurophotometrics fp3002 rig and bonsai software
        :param behavioral_data: Path(s) to csv files, either deeplabcut or anymaze, for doing behavioral analysis
        :param keystroke_offset: Value corresponding to a keystroke press in bonsai for removing irrelevant data
        :param manual_off_set: Value obtained from calculating offset from expected event i.e. blue light and its theoretical appearance in video
        :param kwargs: dict containing values such as "ID", "task", and/or "treatment" note: task and treatment necessary for use in fiberPhotometryExperiment

        """

        # these should always be present
        self.npm_file = npm_file
        self.behavioral_data = behavioral_data
        self.fp_df = pd.read_csv(self.npm_file)
        self.__T0__ = self.fp_df['Timestamp'][1]

        # unpack extra params
        for key, value in kwargs.items():
            setattr(self, key, value)

        # determine sample time
        self._sample_time_ = np.diff(self.fp_df['Timestamp'])[1]

        if manual_off_set:
            self.fp_df = self.fp_df[int(manual_off_set // self._sample_time_):].reset_index()

        if keystroke_offset:
            ind = self.fp_df[self.fp_df['Timestamp'] == keystroke_offset]
            self.fp_df = self.fp_df[ind:].reset_index()

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
                self.fp_df.drop(self.fp_df.index[-1], axis=0, inplace=True)
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
                                   zip(['GCaMP_Isobestic', 'GCaMP'], [isobestic.Timestamp, gcamp.Timestamp])}

                if self.__CONF1:
                    self.Signal = {"GCaMP": np.array(gcamp['Region0G']),
                                   "Isobestic_GCaMP": np.array(isobestic["Region0G"])}

                else:
                    self.Signal = {"GCaMP": np.array(gcamp['Region1G']),
                                   "Isobestic_GCaMP": np.array(isobestic["Region1G"])}

            except KeyError:
                rcamp = self.fp_df[self.fp_df['LedState'] == 4]
                self.Timestamps = {signal: time.values.tolist() - self.__T0__ for
                                   signal, time in zip(['RCaMP_Isobestic', 'RCaMP'], [isobestic, rcamp])}

                if self.__CONF1:
                    self.Signal = {"RCaMP": np.array(rcamp['Region1R']),
                                   "Isobestic_RCaMP": np.array(isobestic["Region1R"])}

                else:
                    self.Signal = {"RCaMP": np.array(rcamp['Region0R']),
                                   "Isobestic_RCaMP": np.array(isobestic["Region0R"])}
        else:
            raise ValueError(
                "No experiment type matches your NPM File input. Make sure you've loaded the correct file.")

        self.DF_F_Signals = self.process_data()
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
    def _als_detrend(y, lam=10e7, p=0.01, niter=100):
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
        F0 = np.median(raw)
        if kind == "standard":
            df_f = (raw - F0) / F0
        else:
            df_f = (raw - F0) / np.std(raw)
        return df_f

    @staticmethod
    def calc_area(l_index, r_index, timeseries):
        areas = np.asarray([simpson(timeseries[i:j]) for i, j in zip(l_index, r_index)])
        return areas

    def process_data(self):
        signals = [signal for signal in self.Signal.values()]
        baseline_corr_signal = [self._als_detrend(sig) for sig in signals]
        df_f_signals = [self._df_f(s) for s in baseline_corr_signal]
        for i in range(len(df_f_signals)):
            if abs(np.median(df_f_signals[i])) < 0.05:
                df_f_signals[i] = self._als_detrend(df_f_signals[i])
        # smoothed like a baby's bottom
        smoothed_signals = [self.smooth(timeseries, kernel=10) for timeseries in
                            df_f_signals]
        return {identity: signal for identity, signal in zip(self.Signal.keys(), smoothed_signals)}

    def fit_general_linear_model(self, curve, ind_vars):
        dep_var = np.reshape(self.DF_F_Signals[curve], (len(self.DF_F_Signals[curve]), 1))
        ind_var = sm.add_constant(pd.DataFrame(ind_vars))
        gaussian_model = sm.GLM(endog=dep_var, exog=ind_var, family=sm.families.Gaussian())
        res_fit = gaussian_model.fit()
        return res_fit

    def find_signal(self, neg=False):
        peak_properties = {}
        if not neg:
            for GECI, sig in self.DF_F_Signals.items():
                peaks, properties = find_peaks(sig, height=1.0, distance=131, width=25,
                                               rel_height=0.95)  # height=1.0, distance=130, prominence=0.5, width=25, rel_height=0.90)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.DF_F_Signals[GECI])
                peak_properties[GECI] = properties
        else:
            for GECI, sig in self.DF_F_Signals.items():
                peaks, properties = find_peaks(-sig, height=1.0, distance=131, width=25, rel_height=0.95)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.DF_F_Signals[GECI])
                peak_properties[GECI] = properties
        return peak_properties

    def visual_check_peaks(self, signal):
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
        length = len(timestamps)
        times = anymaze_file.Time.str.split(':')
        if len(times[1]) == 3:
            for i in range(len(times)):
                anymaze_file.loc[i, 'seconds'] = (float(times[i][1]) * 60 + float(times[i][2]))
        else:
            for i in range(len(times)):
                anymaze_file.loc[i, 'seconds'] = (float(times[i][0]) * 60 + float(times[i][1]))
        anymaze_file.seconds = anymaze_file['seconds'].apply(
            lambda x: (x / anymaze_file.seconds.iloc[-1] * timestamps[-1]))
        binary_freeze_vec = np.zeros(shape=(length))
        i = 0
        while i < len(times):
            if anymaze_file.loc[i, 'Freezing'] == 1:
                t1 = anymaze_file.loc[i, 'seconds']
                t2 = anymaze_file.loc[i + 1, 'seconds']
                try:
                    binary_freeze_vec[np.where(timestamps > t1)[0][0]:np.where(timestamps < t2)[0][-1]] = 1
                    # print(np.where(timestamps > t1)[0][0], np.where(timestamps < t2)[0][-1])
                except IndexError:
                    binary_freeze_vec[np.where(timestamps > t1)[0][0]:np.where(timestamps < t2)[0][-1]] = 1
                i += 1
            else:
                i += 1
        time_val_0 = [anymaze_file.seconds[i] for i in range(1, len(anymaze_file)) if anymaze_file.Freezing[i] == 0]
        inds = [np.argmin(np.abs(timestamps - time_val)) for time_val in time_val_0]
        return anymaze_file, binary_freeze_vec, inds

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
        if props is None:
            props = ['widths', 'areas_under_curve', 'peak_heights']
        condensed_props = {}
        for signal in self.peak_properties:
            condensed_props.update(
                {signal: {"average" + "_" + prop: np.average(self.peak_properties[signal][prop]) for prop in props}})
        setattr(self, "condensed_stats", condensed_props)
        return condensed_props

    def reset_peak_params(self, crit_width, curve_type):
        deletion_list = [i for i, j in enumerate(self.peak_properties[curve_type]['widths']) if j < crit_width]
        for prop in self.peak_properties[curve_type]:
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


class fiberPhotometryExperiment:
    def __init__(self, *args):
        self.treatment = {}
        self.task = {}
        self.curves = [arg for arg in args]

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
            curve.reset_peak_params(getattr(self, "crit_width" + "_" + curve_type), curve_type)
            curve.calc_avg_peak_props()
        return

    def comparative_statistics(self, group1, group2, metric, curve='GCaMP', test=scipy.stats.ttest_ind):
        s1 = [y for y in next(iter(getattr(self, group1).values()))]
        s2 = [y for y in next(iter(getattr(self, group2).values()))]
        sample1 = [x.condensed_stats[curve]['average' + '_' + metric] for x in s1]
        sample2 = [x.condensed_stats[curve]['average' + '_' + metric] for x in s2]
        stat, pval = test(sample1, sample2)
        return stat, pval

    def raster(self, group, curve, a, b, colormap):
        sb.set()
        vector_array = np.array(
            [vec.DF_F_Signals[curve][a:b].tolist() for vec in next(iter(getattr(self, group).values()))])
        sb.heatmap(vector_array, cmap=colormap)
        plt.show()
        return

    def st_event_triggered_average(self, curve, event_time, window, group, plot=False, timepoint=True):
        time = [time.Timestamps[curve].tolist() for time in next(iter(getattr(self, group).values()))]
        max_ind = np.min([len(x) for x in time])
        time_array = np.array(
            [time.Timestamps[curve][0:max_ind].tolist() for time in next(iter(getattr(self, group).values()))])
        try:
            average_time = np.average(time_array, axis=0)
        except ZeroDivisionError:
            average_time = np.zeros(shape=(1, len(time_array)))
            average_time[0] = 0
            average_time[1:] = np.average(time_array[1:], axis=0)
        if timepoint:
            ind = np.argmin(np.abs(average_time - event_time))
        else:
            ind = event_time
        index_right_bound = np.argmin(np.abs(average_time - (event_time + window)))
        index_left_bound = np.argmin(np.abs(average_time - (event_time - (window / 2))))
        vector_array = np.array([vec.DF_F_Signals[curve][index_left_bound:index_right_bound].tolist() for vec in
                                 next(iter(getattr(self, group).values()))])
        try:
            averaged_trace = np.average(vector_array, axis=0)
        except ZeroDivisionError:
            averaged_trace = np.zeros(shape=(1, len(vector_array)))
            averaged_trace[0] = 0
            averaged_trace[1:] = np.average(vector_array[1:], axis=0)
        ci = 1.96 * np.std(vector_array, axis=0) / np.sqrt(np.shape(vector_array)[0])
        if plot:
            fig, ax = plt.subplots()
            plt.axvline(average_time[ind], linestyle='--', color='black')
            ax.plot(average_time[index_left_bound:index_right_bound], averaged_trace)
            ax.fill_between(average_time[index_left_bound: index_right_bound], (averaged_trace - ci),
                            (averaged_trace + ci), color='b', alpha=0.1)
            plt.show()
        return averaged_trace, average_time[index_left_bound:index_right_bound], ci

    def mt_event_triggered_average(self, curve, event_times, window, group, plot=False, timepoint=True):
        max_ind = np.min(
            [len(x) for x in [t.Timestamps[curve].tolist() for t in next(iter(getattr(self, group).values()))]])
        time_array = np.array(
            [time.Timestamps[curve][0:max_ind].tolist() for time in next(iter(getattr(self, group).values()))])
        vector_array = np.array(
            [trace.DF_F_Signals[curve][0:max_ind].tolist() for trace in next(iter(getattr(self, group).values()))])
        inds = []
        if timepoint:
            for i in range(len(event_times)):
                tps = [np.argmin(np.abs(time_array[i] - event_times[i][j])) for j in range(len(event_times[i]))]
                inds.append(tps)
        else:
            inds = event_times
        mt_eta = []
        for animal in range(len(vector_array)):
            part_traces = [[vector_array[animal][indice - (int( window/ 2)):indice + window].tolist() for indice in inds[animal]]]
            eta = np.average(part_traces, axis=0)
            mt_eta.append(eta)
        av_tr = np.average(mt_eta, axis=0)
        return av_tr

    # test this function
    def bootstrap(self, inds, average_trace, window, niter):
        average_trace_copy = average_trace
        actual_values = [max(average_trace_copy[i:i + window] for i in inds)]
        avg_max = []
        for i in range(niter):
            np.random.shuffle(average_trace)
            max_average = np.average([max(average_trace[i:i + window] for i in inds)])
            avg_max.append(max_average)
        return avg_max, actual_values

    def plot_eta(self, curve, event_time, window, *args):
        for arg in args:
            av_tr, av_ti, ci = self.st_event_triggered_average(curve, event_time, window, arg)
            ti_ind = np.argmin(np.abs(av_ti - event_time))
            plt.axvline(av_ti[ti_ind], linestyle='--', color='black')
            plt.plot(av_ti, av_tr)
            plt.fill_between(av_ti, (av_tr - ci), (av_tr + ci), alpha=0.1)
        plt.show()
        return


def make_3d_timeseries(timeseries, timestamps, x_axis, y_axis, z_axis, **kwargs):
    sb.set()
    if type(timeseries) != np.array:
        timeseries = np.asarray(timeseries)
    if type(timestamps) != np.array:
        timestamps = np.asarray(timestamps)
    if np.shape(timeseries) != np.shape(timestamps):
        raise ValueError(
            "Shape of timeseries and timestamp data do not match! Perhaps, try transposing? If not, you may have "
            "concatenated incorrectly.")
    y_coordinate_matrix = np.zeros(shape=(np.shape(timeseries)[0], np.shape(timeseries)[1]))
    for i in range(len(timeseries)):
        y_coordinate_matrix[i, :np.shape(timeseries)[1]] = i + 1
    plt.figure()
    axs = plt.axes(projection="3d")
    for i in reversed(range(len(timeseries))):
        axs.plot(timestamps[i], y_coordinate_matrix[i], timeseries[i], **kwargs)
    axs.set_xlabel(x_axis)
    axs.set_ylabel(y_axis)
    axs.set_zlabel(z_axis)
    return
