'''Literally Fuck Neurophotometrics'''


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

        # check to see if using old files
        if "Flags" in fp_df.columns:
            fp_df = fix_npm_flags(fp_df)
            print("Old NPM format detected, changing Flags to LedState")
        else:
            pass

        # drop last row if timeseries are unequal
        if fp_df['LedState'][1] == fp_df['LedState'].iloc[-1]:
            fp_df.drop(fp_df.index[-1], axis=0, inplace=True)


        #isobestic will always be present so we shouldn't need to check anything
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
        if 2 and 3 in fp_df.LedState.values:
            gcamp = fp_df[fp_df['LedState'] == 2]
            rcamp = fp_df[fp_df['LedState'] == 3]
            self.timestamps = [x.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for x in [isobestic, gcamp, rcamp]]
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
            self.timestamps = [x.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for x in [isobestic, gcamp]]
            if conf1:
                self.gcamp = gcamp['Region0G']
                self.isobestic = isobestic['Region0G']
            else:
                self.gcamp = gcamp['Region1G']
                self.isobestic = isobestic['Region1G']
        elif 2 not in fp_df.LedState.values:
            rcamp = fp_df[fp_df['LedState'] == 3]
            self.timestamps = [x.iloc[:, 1].reset_index(drop=True).tolist() - fp_df['Timestamp'][1] for x in [isobestic, rcamp]]
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

    def find_event_indices(self, event_map):
        if type(event_map) != pd.Series:
            event_map = pd.Series(event_map)
        else:
            pass
        bool_index_df = event_map.isin([1])
        list_of_indices = [[bool_index_df[index] == True].index  for index in bool_index_df]
        return list_of_indices


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
            self.final_df_gcamp = np.asarray(pd.DataFrame(self.dff_gcamp) - pd.DataFrame(self.dff_isobestic))
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

def fix_npm_flags(npm_df):
    """This takes a preloaded npm_file i.e. you've ran pd.read_csv()"""
    # set inplace to True, so that we modify original DF and do not return a virtual copy
    npm_df.rename(columns={"Flags": "LedState"}, inplace=True)
    npm_df.LedState.replace([16, 17, 18, 20], [0, 1, 2, 3], inplace=True)
    return npm_df

###TESTING BITCHES###

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', '')


# rebecca1 = fiberPhotometryCurve('1', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse1.csv')
# rebecca2 = fiberPhotometryCurve('2', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse2.csv')
# rebecca3 = fiberPhotometryCurve('3', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day2_recall_mouse3.csv')
# rebecca4 = fiberPhotometryCurve('4', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day3_opto_mouse1.csv')
# rebecca5 = fiberPhotometryCurve('5', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day3_opto_mouse2.csv')
# rebecca6 = fiberPhotometryCurve('6', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day3_opto_mouse3.csv')

rebecca_shock1 = fiberPhotometryCurve('1', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_astro1_day1_FC_60s.csv')
rebecca_shock2 = fiberPhotometryCurve('2', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_astro2_day1_FC_60s.csv')
rebecca_shock3 = fiberPhotometryCurve('3', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_astro11_day1_FC_60s.csv')
rebecca_shock4 = fiberPhotometryCurve('4', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_astro12_day1_FC_60s.csv')

rebecca_shock5 = fiberPhotometryCurve('5', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day1_FC_mouse1.csv')
rebecca_shock6 = fiberPhotometryCurve('6', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day1_FC_mouse2.csv')
rebecca_shock7 = fiberPhotometryCurve('7', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_engram_day1_FC_mouse3.csv')
rebecca_shock8 = fiberPhotometryCurve('8', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_shock_1_1.5mA.csv')
rebecca_shock9 = fiberPhotometryCurve('9', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_shock_3_1.5mA.csv')
rebecca_shock10 = fiberPhotometryCurve('10', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_shock_4_1.0mA.csv')
rebecca_shock11 = fiberPhotometryCurve('11', '/Users/ryansenne/Desktop/Rebecca_Data/Test_Pho_FP_shock_5_1.0mA.csv')

rebecca_shock1.process_data()
rebecca_shock2.process_data()
rebecca_shock3.process_data()
rebecca_shock4.process_data()
rebecca_shock5.process_data()
rebecca_shock6.process_data()
rebecca_shock7.process_data()
rebecca_shock8.process_data()
rebecca_shock9.process_data()
rebecca_shock10.process_data()
rebecca_shock11.process_data()

# rebecca1.process_data()
# rebecca2.process_data()
# rebecca3.process_data()
# rebecca4.process_data()
# rebecca5.process_data()
# rebecca6.process_data()
#
# plt.figure("Recall")
# rebecca_df_recall = np.hstack([rebecca1.final_df_gcamp[100:6600], rebecca2.final_df_gcamp[100:6600], rebecca3.final_df_gcamp[100:6600]])
# sb.heatmap(rebecca_df_recall.T)
# plt.show()
#
# plt.figure("Opto")
# rebecca_df_opto = np.hstack([rebecca4.final_df_gcamp[100:10000], rebecca5.final_df_gcamp[100:10000], rebecca6.final_df_gcamp[100:10000]])
# sb.heatmap(rebecca_df_opto.T)
# plt.show()
#
# plt.figure('your mom bitch')
# plt.plot(rebecca1.timestamps[0], rebecca1.final_df_gcamp)
#
# plt.show()


plt.figure('Ryan is Great')
rebecca_df_shock = np.hstack([rebecca_shock1.final_df_gcamp[1900:6800], rebecca_shock2.final_df_gcamp[1900:6800], rebecca_shock3.final_df_gcamp[1900:6800], rebecca_shock4.final_df_gcamp[1900:6800], rebecca_shock5.final_df_gcamp[1900:6800],
                              rebecca_shock6.final_df_gcamp[1900:6800], rebecca_shock7.final_df_gcamp[1900:6800], rebecca_shock8.final_df_gcamp[100:5000], rebecca_shock9.final_df_gcamp[100:5000], rebecca_shock10.final_df_gcamp[100:5000],
                              rebecca_shock11.final_ddf_gcamp[100:5000]])
sb.heatmap(rebecca_df_shock.T)
plt.show()


def find_event_indices(event_map):
    if type(event_map) != pd.Series:
        event_map = pd.Series(event_map)
    else:
        pass
    bool_index_df = event_map.isin([1])
    list_of_indices = bool_index_df[bool_index_df == True].index
    return list_of_indices