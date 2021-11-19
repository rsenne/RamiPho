import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter1d


class fiberPhotometryCurve:
    def __init__(self, name, npm_file, behavioral_data=None):

        # these should always be present
        self.name = name
        self.npm_file = npm_file
        self.behavioral_data = behavioral_data

        # read the file
        fp_df = pd.read_csv(self.npm_file)

        # drop last row if timeseries are unequal
        if fp_df['Flags'][1] == fp_df['Flags'].iloc[-1]:
            fp_df.drop(fp_df.index[-1], axis=0, inplace=True)


        #isobestic will always be present so we shouldn't need to check anything
        isobestic = fp_df[fp_df['Flags'] == 17]

        # create essentially a dummy variable for ease of typing
        columns = list(isobestic.columns)
        if "Region0G" in columns:
            conf1 = True
        else:
            conf1 = False

        # slicing file based on flag values present in file
        # this is a goddamned mess...please for the love of god if theres a better way show me
        # literally brute force
        if 18 and 20 in fp_df.Flags.values:
            gcamp = fp_df[fp_df['Flags'] == 18]
            rcamp = fp_df[fp_df['Flags'] == 20]
            self.timestamps = [x.iloc[:, 1] - fp_df['Timestamp'][1] for x in [isobestic, gcamp, rcamp]]
            if conf1:
                self.gcamp = gcamp['Region0G']
                self.rcamp = rcamp['Region1R']
                self.isobestic = isobestic[['Region0G', 'Region1R']]
            else:
                self.gcamp = gcamp['Region1G']
                self.rcamp = rcamp['Region0R']
                self.isobestic = isobestic[['Region1G', 'Region0R']]
        elif 20 not in fp_df.Flags.values:
            gcamp = fp_df[fp_df['Flags'] == 18]
            self.timestamps = [x.iloc[:, 1] - fp_df['Timestamp'][1] for x in [isobestic, gcamp]]
            if conf1:
                self.gcamp = gcamp['Region0G']
                self.isobestic = isobestic['Region0G']
            else:
                self.gcamp = gcamp['Region1G']
                self.isobestic = isobestic['Region1G']
        elif 18 not in fp_df.Flags.values:
            rcamp = fp_df[fp_df['Flags'] == 20]
            self.timestamps = [x.iloc[:, 1] - fp_df['Timestamp'][1] for x in [isobestic, rcamp]]
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

    def process_data(self, exp_type='gcamp'):
    # this also seems like a terrible way to do it
        if exp_type == "dual_color":
            signal = [self.gcamp, self.rcamp, pd.DataFrame(self.isobestic.iloc[:, 0]),
                      pd.DataFrame(self.isobestic.iloc[:, 1])]
            smooth_signals = [self.smooth(raw,10, visual_check=False).flatten() for raw in signal]
            baselines = [self._als_detrend(raw) for raw in smooth_signals]
            df_f_signals = [self._df_f((smooth_signals[i]) - (baselines[i]), type='std') for i in range(len(smooth_signals))]
            # add dff_properties
            self.dff_gcamp = df_f_signals[0]
            self.dff_rcamp = df_f_signals[1]
            self.dff_isobestic = np.vstack((np.array(df_f_signals[2]), np.array(df_f_signals[3])))
            # remove motion, add property
            self.final_df_gcamp = self.dff_gcamp - pd.DataFrame(self.dff_isobestic).iloc[0, :]
            self.final_df_rcamp = self.dff_rcamp - pd.DataFrame(self.dff_isobestic)[1, :]
        elif exp_type == "gcamp":
            signal = [self.gcamp, pd.DataFrame(self.isobestic)]
            smooth_signals = [self.smooth(raw, 10, visual_check=False).flatten() for raw in signal]
            baselines = [self._als_detrend(raw) for raw in smooth_signals]
            df_f_signals = [self._df_f((smooth_signals[i]) - (baselines[i]), type='std') for i in
                            range(len(smooth_signals))]
            # add dff_properties
            self.dff_gcamp = df_f_signals[0]
            self.dff_isobestic = np.array(df_f_signals[1])
            # remove motion, add property
            self.final_df_gcamp = pd.DataFrame(self.dff_gcamp) - pd.DataFrame(self.dff_isobestic)
        elif exp_type == "rcamp":
            signal = [self.gcamp, pd.DataFrame(self.isobestic)]
            smooth_signals = [self.smooth(raw, 10, visual_check=False).flatten() for raw in signal]
            baselines = [self._als_detrend(raw) for raw in smooth_signals]
            df_f_signals = [self._df_f((smooth_signals[i]) - (baselines[i]), type='std') for i in
                            range(len(smooth_signals))]
            # add dff_properties
            self.dff_rcamp = df_f_signals[0]
            self.dff_isobestic = np.array(df_f_signals[1])
            # remove motion, add property
            self.final_df_gcamp = pd.DataFame(self.dff_gcamp) - pd.DataFrame(self.dff_isobestic)



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

mia1 = fiberPhotometryCurve('ee', '/Users/ryansenne/Downloads/Test_Pho_potato1_avoidance1.csv')
