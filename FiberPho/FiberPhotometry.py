import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy import sparse
from scipy.integrate import simpson
from scipy.ndimage import uniform_filter1d
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, find_peaks_cwt
from sklearn.linear_model import Lasso, LinearRegression
import pickle as pkl


class fiberPhotometryCurve:
    def __init__(self, npm_file, behavioral_data=None):

        # these should always be present
        self.npm_file = npm_file
        self.behavioral_data = behavioral_data

        # read the file
        fp_df = pd.read_csv(self.npm_file)

        # check to see if using old files
        if "Flags" in fp_df.columns:
            fp_df = fix_npm_flags(fp_df)
            print("Old NPM format detected, changing Flags to LedState")
        else:
            pass

        # drop last row if timeseries are unequal
        try:
            while fp_df['LedState'].value_counts()[1] != fp_df['LedState'].value_counts()[2] or fp_df['LedState'].value_counts()[2] != fp_df['LedState'].value_counts()[4]:
                fp_df.drop(fp_df.index[-1], axis=0, inplace=True)
        except KeyError:
            while fp_df['LedState'].value_counts()[1] != fp_df['LedState'].value_counts()[2]:
                fp_df.drop(fp_df.index[-1], axis=0, inplace=True)
        except:
            while fp_df['LedState'].value_counts()[1] != fp_df['LedState'].value_counts()[4]:
                fp_df.drop(fp_df.index[-1], axis=0, inplace=True)

        # isobestic will always be present so we shouldn't need to check anything
        isobestic = fp_df[fp_df['LedState'] == 1]

        # create essentially a dummy variable for ease of typing
        columns = list(isobestic.columns)
        if "Region0G" in columns:
            conf1 = True
        else:
            conf1 = False

        # slicing file based on flag values present in file
        # this is a goddamned mess...please for the love of god if theres a better way show me
        # literally brute force
        if 2 and 4 in fp_df.LedState.values:
            gcamp = fp_df[fp_df['LedState'] == 2]
            rcamp = fp_df[fp_df['LedState'] == 4]
            self.timestamps = [x.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for x in
                               [isobestic, gcamp, rcamp]]
            if conf1:
                self.gcamp = gcamp['Region0G']
                self.rcamp = rcamp['Region1R']
                self.isobestic = isobestic[['Region0G', 'Region1R']]
            else:
                self.gcamp = gcamp['Region1G']
                self.rcamp = rcamp['Region0R']
                self.isobestic = isobestic[['Region1G', 'Region0R']]
        elif 3 not in fp_df.LedState.values:
            gcamp = fp_df[fp_df['LedState'] == 2]
            self.timestamps = [x.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for x in
                               [isobestic, gcamp]]
            if conf1:
                self.gcamp = gcamp['Region0G']
                self.isobestic = isobestic['Region0G']
            else:
                self.gcamp = gcamp['Region1G']
                self.isobestic = isobestic['Region1G']
        elif 2 not in fp_df.LedState.values:
            rcamp = fp_df[fp_df['LedState'] == 3]
            self.timestamps = [x.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for x in
                               [isobestic, rcamp]]
            if conf1:
                self.rcamp = rcamp['Region1R']
                self.isobestic = isobestic['Region1R']
            else:
                self.rcamp = rcamp['Region0R']
                self.isobestic = isobestic['Region0R']

    @staticmethod
    def _als_detrend(y, lam=10e7, p=0.01, niter=100):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    @staticmethod
    def smooth(signal, kernel, visual_check=True):
        smooth_signal = uniform_filter1d(signal, kernel)
        if visual_check:
            plt.figure()
            plt.plot(signal)
            plt.plot(smooth_signal)
        return smooth_signal

    @staticmethod
    def _df_f(raw, type="standard"):
        F0 = np.median(raw)
        if type == "standard":
            df_f = (raw - F0) / F0
        else:
            df_f = (raw - F0) / np.std(raw)
        return df_f

    def process_data(self, exp_type='dual_color', type='std', **kwargs):
        # this also seems like a terrible way to do it
        if exp_type == "dual_color":
            signal = [self.gcamp, self.rcamp, np.asarray(self.isobestic.iloc[:, 0]),
                      np.asarray(self.isobestic.iloc[:, 1])]
            smooth_signals = [self.smooth(raw, 10, visual_check=False).flatten() for raw in signal]
            baselines = [self._als_detrend(raw) for raw in smooth_signals]
            df_f_signals = [self._df_f((smooth_signals[i]) - (baselines[i]), type=type) for i in
                            range(len(smooth_signals))]
            # add dff_properties
            self.dff_gcamp = df_f_signals[0]
            self.dff_rcamp = df_f_signals[1]
            self.dff_isobestic = np.vstack((np.array(df_f_signals[2]), np.array(df_f_signals[3])))

            lin = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                         positive=True, random_state=9999, selection='random')
            n = np.shape(self.dff_isobestic)[1]
            ref1 = lin.fit(self.dff_isobestic[0].reshape(n, 1), self.dff_gcamp.reshape(n, 1))
            ref2 = lin.fit(self.dff_isobestic[1].reshape(n, 1), self.dff_gcamp.reshape(n, 1))
            self.reference = lin.predict(self.dff_isobestic[0].reshape(n, 1)).reshape(n, )
            self.reference2 = lin.predict(self.dff_isobestic[1].reshape(n, 1)).reshape(n, )
            # remove motion, add property
            self.final_dff_gcamp = np.asarray(self.dff_gcamp - pd.Series(self.reference)).flatten()
            self.final_dff_rcamp = np.asarray(self.dff_rcamp - pd.Series(self.reference2)).flatten()
        elif exp_type == "gcamp":
            signal = [self.gcamp, pd.Series(self.isobestic)]
            smooth_signals = [self.smooth(raw, 10, visual_check=False).flatten() for raw in signal]
            baselines = [self._als_detrend(raw) for raw in smooth_signals]
            df_f_signals = [self._df_f((smooth_signals[i]) - (baselines[i]), type=type) for i in
                            range(len(smooth_signals))]
            # add dff_properties
            self.dff_gcamp = df_f_signals[0]
            self.dff_isobestic = np.array(df_f_signals[1])
            # stolen give credit later
            lin = LinearRegression()
            n = len(self.dff_isobestic)
            x = lin.fit(self.dff_isobestic.reshape(n, 1), self.dff_gcamp.reshape(n, 1))
            self.reference = lin.predict(self.dff_isobestic.reshape(n, 1)).reshape(n, )
            # remove motion, add property
            self.final_dff_gcamp = np.asarray(pd.Series(self.dff_gcamp) - pd.Series(self.reference)).flatten()

        elif exp_type == "rcamp":
            signal = [self.gcamp, pd.Series(self.isobestic)]
            smooth_signals = [self.smooth(raw, 10, visual_check=False).flatten() for raw in signal]
            baselines = [self._als_detrend(raw) for raw in smooth_signals]
            df_f_signals = [self._df_f((smooth_signals[i]) - (baselines[i]), type=type) for i in
                            range(len(smooth_signals))]
            # add dff_properties
            self.dff_rcamp = df_f_signals[0]
            self.dff_isobestic = np.array(df_f_signals[1])
            # remove motion, add property
            lin = LinearRegression()
            n = len(self.dff_isobestic)
            x = lin.fit(self.dff_isobestic.reshape(n, 1), self.dff_rcamp.reshape(n, 1))
            self.reference = lin.predict(self.dff_isobestic.reshape(n, 1)).reshape(n, )
            self.final_dff_rcamp = pd.DataFame(self.dff_rcamp) - pd.DataFrame(self.dff_isobestic)
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



def event_triggered_average(self):
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


def fix_npm_flags(npm_df):
    """This takes a preloaded npm_file i.e. you've ran pd.read_csv()"""
    # set inplace to True, so that we modify original DF and do not return a virtual copy
    npm_df.rename(columns={"Flags": "LedState"}, inplace=True)
    npm_df.LedState.replace([16, 17, 18, 20], [0, 1, 2, 4], inplace=True)
    return npm_df

def find_signal(signal):
    peaks, properties = find_peaks(signal, height=0.0, distance=75, width=25)
    plt.plot(signal)
    plt.plot(peaks, signal[peaks], "x")
    plt.vlines(x=peaks, ymin=signal[peaks] - properties["prominences"],
               ymax=signal[peaks], color="C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
               xmax=properties["right_ips"], color="C1")
    plt.show()
    return peaks, properties

def find_critical_width(signal, event_metrics_list):
    return

def calculate_event_ratio(trans_list):
    negative_ele = sum(num for num in trans_list if num < 0)
    positive_ele = sum(num for num in trans_list if num > 0)
    return positive_ele/(positive_ele + negative_ele)


# rebecca1 = fiberPhotometryCurve('1', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse1.csv')
# rebecca2 = fiberPhotometryCurve('2', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse2.csv')
# rebecca1.process_data()
# rebecca2.process_data()

rebecca1 = fiberPhotometryCurve('/home/ryansenne/Data/Rebecca/Test_Pho_FP_engram_day2_recall_mouse1.csv')
rebecca1.process_data(exp_type='gcamp')
