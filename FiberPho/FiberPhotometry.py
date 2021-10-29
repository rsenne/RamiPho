import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter1d



def from_npm_csv(npm_file, behavioral_data=None):
    # read npm file
    npm_file = pd.read_csv(npm_file)
    # split up NPM files based on their flags
    # i will kill the first born child of everyone at NPM if the flags differ between exps
    gcamp = npm_file[npm_file['Flags'] == 18]
    isobestic = npm_file[npm_file['Flags'] == 17]
    # fix this later
    rcamp = npm_file[npm_file['Flags'] == 20]
    # create a list that has each signal so we can check
    signal = [gcamp, rcamp, isobestic]
    # check to see if all columns are of equivalent length, is this socialism fox news?
    min_len = min(gcamp.shape[0], isobestic.shape[0])
    for i, j in enumerate(signal):
        if j.shape[0] != min_len:
            signal[i] = j[:-1]
    # new list of variables
    # this seems like a stupid way to do this, even if its readable...eh who cares
    gcamp = pd.DataFrame(signal[0])
    rcamp = pd.DataFrame(signal[1])
    isobestic = pd.DataFrame(signal[2])
    # create a list of timestamps in order, gcamp, isobestic, rcamp
    # i'm literally attracted to list comprehensions
    timestamps = [x.iloc[:, 1] for x in signal]
    # grab relevant column names
    columns = list(gcamp.columns)
    # create fiber photometry object based on column names
    # we have to do this just because NPM doesn't have a great UI
    if "Region0G" in columns:
        FP = fiberPhotometryCurve(gcamp[['Region0G']], rcamp[['Region1R']],
                                  isobestic[['Region0G', 'Region1R']], timestamps=timestamps,
                                  behavioral_data=behavioral_data)
    else:
        FP = fiberPhotometryCurve(gcamp[['Region1G']], rcamp[['Region0R']],
                                  isobestic[['Region1G', 'Region0R']], timestamps=timestamps,
                                  behavioral_data=behavioral_data)
    return  FP

class fiberPhotometryCurve:
    def __init__(self, gcamp, rcamp, isobestic, timestamps, behavioral_data):
        # initialize a fiber photometry object with gcamp, rcamp, isobestic, and behavioral data properties
        self.gcamp = gcamp
        self.rcamp = rcamp
        self.isobestic = isobestic
        self.timestamps = timestamps
        self.behavior_data = behavioral_data

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
            df_f = abs((raw - F0) / F0)
        else:
            df_f = abs((raw - F0) / np.std(raw))
        return df_f

    def find_events(self, signal):
        """
        Use this to determine where the df/f is greater than 3 times the standard deviation.
        Since the std can be lower than one, we need an if/else statement because if the std
        is lower than one, then we actually need where the signal is lower than that value otherwise
        we won't find any events
        """
        #  create a map of the trace
        events = np.zeros(len(signal))
        # standard deviation variable
        std = np.std(signal)
        # if/else to find significant events
        if std < 1:
            events = np.where(signal < 3 * std, events, 1)
        else:
            events = np.where(signal > 3 * std, events, 1)
        # return a 1 where there is an event, a 0 where there is not
        return events

    def process_data(self):
        signal = [self.gcamp, pd.DataFrame(self.isobestic.iloc[:, 0]), pd.DataFrame(self.isobestic.iloc[:, 1])]
        smooth_signals = [self.smooth(raw,10, visual_check=False).flatten() for raw in signal]
        baselines = [self._als_detrend(raw) for raw in smooth_signals]
        df_f_signals = [self._df_f((smooth_signals[i]) - (baselines[i]), type='std') for i in range(len(smooth_signals))]
        # add dff_properties
        self.dff_gcamp = df_f_signals[0]
        self.dff_isobestic = np.vstack((np.array(df_f_signals[1]), np.array(df_f_signals[2])))
        # self.dff_rcamp = df_f_signals.iloc[3, :, :]
        # remove motion, add property
        self.final_df_gcamp = self.dff_gcamp - pd.DataFrame(self.dff_isobestic).iloc[0, :]
        # self.final_df_rcamp = self.dff_rcamp - self.dff_isobestic[1, :]


    def find_beh(self):
        pass

    def final_plot(self):
        pass

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


