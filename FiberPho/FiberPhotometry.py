import pickle as pkl
import warnings as warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy import sparse
from scipy.interpolate import splrep, splev
from scipy.integrate import simpson
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve


class fiberPhotometryCurve:
    def __init__(self, npm_file, behavioral_data=None):

        # these should always be present
        self.npm_file = npm_file
        self.behavioral_data = behavioral_data

        # read the file
        fp_df = pd.read_csv(self.npm_file)

        # check to see if using old files
        if "Flags" in fp_df.columns:
            self.fix_npm_flags()
            print("Old NPM format detected, changing Flags to LedState")
        else:
            pass

        # drop last row if timeseries are unequal
        try:
            while fp_df['LedState'].value_counts()[1] != fp_df['LedState'].value_counts()[2] or \
                    fp_df['LedState'].value_counts()[2] != fp_df['LedState'].value_counts()[4]:
                fp_df.drop(fp_df.index[-1], axis=0, inplace=True)
        except KeyError:
            while fp_df['LedState'].value_counts()[1] != fp_df['LedState'].value_counts()[2]:
                fp_df.drop(fp_df.index[-1], axis=0, inplace=True)
        except:
            while fp_df['LedState'].value_counts()[1] != fp_df['LedState'].value_counts()[4]:
                fp_df.drop(fp_df.index[-1], axis=0, inplace=True)

        if 2 and 4 in fp_df.LedState.values:
            self.__DUAL_COLOR = True
        else:
            self.__DUAL_COLOR = False

        # create essentially a dummy variable for ease of typing
        columns = list(fp_df.columns)
        if "Region0G" in columns:
            self.__CONF1 = True
        else:
            self.__CONF1 = False

        # slicing file based on flag values present in file
        # this is a goddamned mess...please for the love of god if there's a better way show me
        # literally brute force
        if self.__DUAL_COLOR:
            gcamp = fp_df[fp_df['LedState'] == 2]
            rcamp = fp_df[fp_df['LedState'] == 4]

            if self.__CONF1:
                isobestic_gcamp = fp_df[fp_df.Region0G['LedState'] == 1]
                isobestic_rcamp = fp_df[fp_df.Region1R['LedState'] == 1]
                self.Signal = {"GCaMP": np.array(gcamp['Region0G']),
                               "RCaMP": np.array(gcamp['Region1R']),
                               "Isobestic_GCaMP": np.array(isobestic_gcamp),
                               "Isobestic_RCaMP": np.array(isobestic_rcamp)}
                self.isobestic = [isobestic_gcamp, isobestic_rcamp]
                self.Timestamps = {signal: time.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for
                                   signal, time in zip(['Isobestic_GCMP', 'Isobestic_RCaMP', 'GCaMP', 'RCaMP'],
                                                       [isobestic_gcamp, isobestic_rcamp, gcamp, rcamp])}

            else:
                isobestic_gcamp = fp_df[fp_df.Region1G['LedState'] == 1]
                isobestic_rcamp = fp_df[fp_df.Region0R['LedState'] == 1]
                self.Signal = {"GCaMP": np.array(gcamp['Region1G']),
                               "RCaMP": np.array(gcamp['Region0R']),
                               "Isobestic_GCaMP": np.array(isobestic_gcamp),
                               "Isobestic_RCaMP": np.array(isobestic_rcamp)}
                self.Timestamps = {signal: time.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for
                                   signal, time in zip(['Isobestic_GCMP', 'Isobestic_RCaMP', 'GCaMP', 'RCaMP'],
                                                       [isobestic_gcamp, isobestic_rcamp, gcamp, rcamp])}

        elif not self.__DUAL_COLOR:
            isobestic = fp_df[fp_df['LedState'] == 1]

            try:
                gcamp = fp_df[fp_df['LedState'] == 2]
                self.Timestamps = {signal: time.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for
                                   signal, time in zip(['GCaMP_ISOBESTIC', 'GCaMP'], [isobestic, gcamp])}

                if self.__CONF1:
                    self.Signal = {"GCaMP": np.array(gcamp['Region0G']),
                                   "Isobestic_GCaMP": np.array(isobestic["Region0G"])}

                else:
                    self.Signal = {"GCaMP": np.array(gcamp['Region1G']),
                                   "Isobestic_GCaMP": np.array(isobestic["Region1G"])}


            except KeyError:
                rcamp = fp_df[fp_df['LedState'] == 3]
                self.Timestamps = {signal: time.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for
                                   signal, time in zip(['RCaMP_ISOBESTIC', 'RCaMP'], [isobestic, rcamp])}
                self.timestamps = [x.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for x in
                                   [isobestic, rcamp]]

                if self.__CONF1:
                    self.Signal = {"RCaMP": np.array(rcamp['Region1R']),
                                   "Isobestic_RCaMP": np.array(isobestic["Region1R"])}

                else:
                    self.Signal = {"RCaMP": np.array(rcamp['Region0R']),
                                   "Isobestic_RCaMP": np.array(isobestic["Region0R"])}
        else:
            raise ValueError("No experiment type matches your NPM File input. Make sure you've loaded the correct file.")

        self.process_data()

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
    def smooth(signal, kernel, visual_check=True):
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
    def _df_f(raw, type="standard"):
        F0 = np.median(raw)
        if type == "standard":
            df_f = (raw - F0) / F0
        else:
            df_f = (raw - F0) / np.std(raw)
        return df_f


    def process_data(self):
        signals = [signal for signal in self.Signal.values()]
        baseline_corr_signal = [self._als_detrend(sig) for sig in signals]
        df_f_signals = [self._df_f((s)) for s in baseline_corr_signal]
        smoothed_signals = [self.b_smooth(timeseries, self.smooth(sigs, 10)) for timeseries, sigs in zip (df_f_signals, self.Timestamps.values())]
        self.DF_F_Signals = {identity: signal for identity, signal in zip(self.Signal.keys(), smoothed_signals)}
        return

    def calculate_event_metrics_gcamp(self, type='standard'):
        event_map = find_events(self.final_dff_gcamp)
        event_boundsL, event_boundR = find_event_bounds(event_map)
        area = calc_area(event_boundsL, event_boundR, self.final_dff_gcamp)
        width = calc_widths(event_boundsL, event_boundR, self.timestamps[0])
        amps = calc_amps(event_boundsL, event_boundR, self.final_dff_gcamp)
        self.event_metrics = pd.DataFrame({"Area": area, "Width": width, "Amplitude": amps})
        return

    def find_beh(self):
        pass

    def final_plot(self):
        pass

    def save_fp(self, filename):
        file = open(filename, 'wb')
        pkl.dump(self, file)
        file.close()
        return

    def fix_npm_flags(self):
        """This takes a preloaded npm_file i.e. you've run pd.read_csv()"""
        # set inplace to True, so that we modify original DF and do not return a virtual copy
        self.npm_file.rename(columns={"Flags": "LedState"}, inplace=True)
        self.npm_file.LedState.replace([16, 17, 18, 20], [0, 1, 2, 4], inplace=True)
        return self.npm_file


def find_event_bounds(event_map):
    if type(event_map) != np.array:
        event_map = np.asarray(event_map)
    else:
        pass
    event_map_shift = event_map[1:]
    left_index = []
    right_index = []
    index = 0
    for i, j in zip(event_map[:-1], event_map_shift):
        if i and j:
            index += 1
        if not i and not j:
            index += 1
        if i and not j:
            right_index.append(index)
            index += 1
        if not i and j:
            left_index.append(index + 1)
            index += 1
        # else:
        #     raise ValueError("Whoopsies, you've encountered a boundary condition please contact your nearest "
        #                      "pythonista if this lasts four or more hours! In all seriousness please talk to Ryan, "
        #                      "unless you are Ryan")
        for i, j in zip(left_index, right_index):
            if i == j:
                left_index.remove(i)
                right_index.remove(j)
            else:
                pass
    return left_index, right_index


def calc_widths(l_index, r_index, timestamps):
    widths = np.asarray([timestamps[j] - timestamps[i] for i, j in zip(l_index, r_index)])
    return widths


def calc_amps(l_index, r_index, timeseries):
    amps = np.asarray([np.max(timeseries[i:j]) for i, j in zip(l_index, r_index)])
    return amps


def calc_area(l_index, r_index, timeseries):
    areas = np.asarray([simpson(timeseries[i:j]) for i, j in zip(l_index, r_index)])
    return areas


def find_events(signal, type="standard"):
    """
    Use this to determine where the df/f is greater than 3 times the standard deviation.
        Since the std can be lower than one, we need an if/else statement because if the std
        is lower than one, then we actually need where the signal is lower than that value otherwise
        we won't find any events
        """
    # create a map of the trace
    events = np.zeros(len(signal))
    # standard deviation variable
    std = np.std(signal)
    if type == "standard":
        events = np.where(signal < 3 * std, events, 1)
    elif type == "std":
        events = np.where(signal < 3, events, 1)
    # return a 1 where there is an event, a 0 where there is not
    return events


def event_triggered_average(signal_array):
    pass


def raster(raster_array, cmap="coolwarm", event_or_heat='event'):
    sb.set()
    if event_or_heat == 'event':
        fig, ax = plt.subplots()
        ax.eventplot(raster_array)
    else:
        sb.heatmap(raster_array)
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
            "Shape of timeseries and timestamp data do not match! Perhaps, try transposing? If not, you may have concaenated incorrectly.")
    y_coordinate_matrix = np.zeros(shape=(np.shape(timeseries)[0], np.shape(timeseries)[1]))
    for i in range(len(timeseries)):
        y_coordinate_matrix[i, :np.shape(timeseries)[1]] = i + 1
    fig = plt.figure()
    axs = plt.axes(projection="3d")
    for i in reversed(range(len(timeseries))):
        axs.plot(timestamps[i], y_coordinate_matrix[i], timeseries[i], **kwargs)
    axs.set_xlabel(x_axis)
    axs.set_ylabel(y_axis)
    axs.set_zlabel(z_axis)
    return


def find_signal(signal):
    peaks, properties = find_peaks(signal, height=1.0, distance=75, width=25, rel_height=0.95)
    plt.plot(signal)
    plt.plot(peaks, signal[peaks], "x")
    plt.vlines(x=peaks, ymin=signal[peaks] - properties["prominences"],
               ymax=signal[peaks], color="C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
               xmax=properties["right_ips"], color="C1")
    plt.show()
    return peaks, properties


def find_alpha_omega(signal_indices, signal):
    offsets = []
    for i in signal_indices:
        j = i
        while signal[j] > np.median(signal):
            j += 1
        else:
            offsets.append(j)
    onsets = []
    for i in reversed(signal_indices):
        j = i
        while signal[j] > np.median(signal):
            j -= 1
        else:
            onsets.append(j)
    return [element for element in reversed(onsets)], offsets


def find_critical_width(pos_wid, neg_wid):
    wid_list = list(set(pos_wid + neg_wid))
    wid_list.sort()
    if len(pos_wid) > 0 and len(neg_wid) == 0:
        critical_width = 0
        warn.warn('There are no negative going transients in your signal. This is not necessarily a problem, '
                  'but you should confirm that this is truly the case!')
    else:
        i = 0
        try:
            while len(pos_wid) / (len(pos_wid) + len(neg_wid)) < 0.99:
                pos_wid = [pos for pos in pos_wid if pos > wid_list[i]]
                neg_wid = [neg for neg in neg_wid if neg > wid_list[i]]
                i += 1
            else:
                critical_width = wid_list[i]
        except ZeroDivisionError:
            critical_width = wid_list[i]
    return critical_width


# rebecca1 = fiberPhotometryCurve('1', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse1.csv')
# rebecca2 = fiberPhotometryCurve('2', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse2.csv')
# rebecca1.process_data()
# rebecca2.process_data()[

rebecca1 = fiberPhotometryCurve('/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse1.csv')
# rebecca2 = fiberPhotometryCurve('/home/ryansenne/Data/Rebecca/Test_Pho_engram_ChR2_m1_Recall.csv')
# rebecca1.process_data(exp_type='gcamp')
# rebecca1.process_data_2()
# rebecca2.process_data(exp_type='gcamp')
# plt.figure(1)
# x, y = find_signal(rebecca1.dff_gcamp)
# plt.figure(0)
# x1, y1 = find_signal(rebecca2.dff_gcamp)

# alpha, omega = find_alpha_omega(x, rebecca1.dff_gcamp)
# alpha1, omega1 = find_alpha_omega(x1, rebecca2.dff_gcamp)
# w = calc_widths(alpha, omega, rebecca1.timestamps[0])
# amp = calc_amps(alpha, omega, rebecca1.dff_gcamp)
# ar = calc_area(alpha, omega, rebecca1.dff_gcamp)
#
# w1 = calc_widths(alpha1, omega1, rebecca2.timestamps[0])
# amp1 = calc_amps(alpha1, omega1, rebecca2.dff_gcamp)
# ar1 = calc_area(alpha1, omega1, rebecca2.dff_gcamp)

# fit = fast.estimate_spikes(rebecca1.dff_gcamp, 0.1, 0.1, True, True)
# plt.plot(rebecca1.dff_gcamp)
# for i in fit['spikes']:
#     plt.axvline(i)
