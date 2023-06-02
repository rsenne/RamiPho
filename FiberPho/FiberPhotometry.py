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
    def __init__(self, npm_file: str, dlc_file: str = None, offset: float = None, anymaze_file: str = None,
                 regress: bool = True, ID: str = None, task: str = None, treatment: str = None, smoother='kalman'):
        """
        """
        # these should always be present
        self.npm_file = npm_file
        self.fp_df = pd.read_csv(self.npm_file)
        self.__T0__ = self.fp_df.loc[1, 'Timestamp']
        self.__TN__ = self.fp_df.Timestamp.iloc[-1]
        self.behavioral_data = {}
        self.ID = ID
        self.task = task
        self.treatment = treatment
        self.anymaze_file = anymaze_file
        self.dlc_file = dlc_file
        self.regress = regress
        self.smoother = smoother
        self.raw_signal = None
        self.timestamps = None
        self.dff_signals = None
        self.dfz_signals = None
        self.anymaze_results = None
        self.dlc_results = None
        self.interp = None

        if offset is None:
            self.offset = 0
        else:
            self.offset = offset

        # determine sample time
        self._sample_time_ = np.diff(self.fp_df['Timestamp'])[1]
        self.fps = 1 / self._sample_time_

        # check to see if using old files
        if "Flags" in self.fp_df.columns:
            self.fix_npm_flags()
            print("Old NPM format detected, changing Flags to LedState")

        # do preprocessing as part of initilization
        self._process_data()
        self.region_region_peak_properties = self.find_signal()

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
        """_summary_
        """
        # Initialize the data structures
        raw_signal = {}
        isobestic_data = {}
        timestamps = {}

        # led states
        led_state = [1, 2, 4]

        # Get the unique region columns
        region_columns = self.fp_df.columns[self.fp_df.columns.str.contains('Region')].tolist()

        # Exclude data before offset
        self.fp_df = self.fp_df[self.fp_df['Timestamp'] >= self.offset].reset_index(drop=True)
        self.fp_df.head(n=3)

        # creat new initial time variable for later correction
        temp_t0 = self.fp_df.loc[0, 'Timestamp']

        # Find the length of the shortest trace
        shortest_trace_length = float('inf')

        # Process each region column
        for column in region_columns:

            # Filter the data based on LEDState and Region column
            region_data = self.fp_df[
                (self.fp_df['LedState'] == 2) if column.endswith('G') else (self.fp_df['LedState'] == 4)]

            # Store the region-specific information
            if not region_data.empty:
                raw_signal[column] = region_data[column].reset_index(drop=True)
                isobestic_data[column] = self.fp_df[self.fp_df['LedState'] == 1][column].reset_index(drop=True)

                # Update the length of the shortest trace
                shortest_trace_length = min(shortest_trace_length, len(region_data[column]), len(self.fp_df[self.fp_df['LedState'] == 1][column]))

        for state in led_state:
            if not self.fp_df[self.fp_df['LedState'] == state].Timestamp.empty:
                # Store the deinterleaved timestamps for the LEDState
                timestamps[str(state)] = self.fp_df[self.fp_df['LedState'] == state].Timestamp.reset_index(
                    drop=True) - temp_t0

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
        """_summary_

        Args:
            y (_type_): _description_
            lam (_type_, optional): _description_. Defaults to 10e7.
            p (float, optional): _description_. Defaults to 0.05.
            niter (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
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
        :param method: whether you want standard df/f or if you would like a z-scored scaling
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
        """_summary_

        Args:
            endog (_type_): _description_
            exog (_type_): _description_
            summary (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        model = sm.OLS(endog, sm.add_constant(exog)).fit()
        if summary:
            print(model.summary())
        return model.resid

    @staticmethod
    def _kalman_filter(signal):
        """_summary_

        Args:
            signal (_type_): _description_

        Returns:
            _type_: _description_
        """
        ar_model = sm.tsa.ARIMA(signal, order=(3, 0, 0), trend='n').fit()
        A = np.zeros((3, 3))
        A[:, 0] = ar_model.params[:-1]
        A[1, 0] = 1
        A[2, 1] = 1
        H = np.array([1, 0, 0])
        kf = pykalman.KalmanFilter(transition_matrices=A, observation_matrices=H, initial_state_covariance=np.eye(3),
                                   initial_state_mean=(0, 0, 0),
                                   em_vars=['transition_covariance', 'observation_covariance'])
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
        """_summary_
        """
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
                print(len(region_timeseries_corrected))
                print(len(isobestic_timeseries_corrected))
                region_timeseries_corrected = self._fit_regression(region_timeseries_corrected,
                                                                   isobestic_timeseries_corrected)
                region_timeseries_corrected_z = self._fit_regression(region_timeseries_corrected_z,
                                                                     isobestic_timeseries_corrected_z)

            dff_signal[key] = region_timeseries_corrected
            dfz_signal[key] = region_timeseries_corrected_z

        self.dff_signals = dff_signal
        self.dfz_signals = dfz_signal

    def process_behavioral_data(self):
        """_summary_
        """
        self.anymaze_results = anymazeResults(self.anymaze_file)
        self.dlc_results = dlcResults(self.dlc_file)
        return

    @staticmethod
    def calc_area(l_index, r_index, timeseries):
        """_summary_

        Args:
            l_index (_type_): _description_
            r_index (_type_): _description_
            timeseries (_type_): _description_

        Returns:
            _type_: _description_
        """
        areas = np.asarray([simpson(timeseries[i:j]) for i, j in zip(l_index, r_index)])
        return areas

    def find_signal(self, neg=False):
        """_summary_

        Args:
            neg (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        region_peak_properties = {}
        if not neg:
            for region, sig in self.dff_signals.items():
                peaks, properties = find_peaks(sig, height=np.std(sig), distance=131, width=25,
                                               rel_height=0.5)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.dff_signals[region])
                properties['widths'] *= self._sample_time_
                region_peak_properties[region] = properties
        else:
            for region, sig in self.dff_signals.items():
                peaks, properties = find_peaks(-sig, height=np.std(sig), distance=131, width=25, rel_height=0.5)
                properties['peaks'] = peaks
                properties['areas_under_curve'] = self.calc_area(properties['left_bases'], properties['right_bases'],
                                                                 self.dff_signals[region])
                properties['widths'] *= self._sample_time_
                region_peak_properties[region] = properties
        return region_peak_properties

    def visual_check_peaks(self, signal):
        """
        Plots object's df_f signal overlayed with peaks
        :param signal: string of which signal to check
        :return:
        """
        if hasattr(self, "region_region_peak_properties"):
            plt.figure()
            plt.plot(self.dff_signals[signal])
            plt.plot(self.region_region_peak_properties[signal]['peaks'],
                     self.dff_signals[signal][self.region_region_peak_properties[signal]['peaks']], "x")
            plt.vlines(x=self.region_region_peak_properties[signal]['peaks'],
                       ymin=self.dff_signals[signal][self.region_region_peak_properties[signal]['peaks']] -
                            self.region_region_peak_properties[signal]["prominences"],
                       ymax=self.dff_signals[signal][self.region_region_peak_properties[signal]['peaks']], color="C1")
            plt.hlines(y=self.region_region_peak_properties[signal]["width_heights"],
                       xmin=self.region_region_peak_properties[signal]["left_ips"],
                       xmax=self.region_region_peak_properties[signal]["right_ips"], color="C1")
            plt.show()
        else:
            raise KeyError(f'{signal} is not in {self}')
        return


class fiberPhotometryExperiment:
    def __init__(self, *args):
        self.treatment = {}
        self.task = {}
        self.curves = [arg for arg in args]
        self.dlc = {}  # csv

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


class FiberPhotometryCollection:
    def __init__(self, name=None):
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
                print(
                    f"No name supplied for this curve. Defaulting to 'Curve {i}' as the name. Consider updating this"
                    f"name.")
                self.curves.update({f"Curve {i}": arg})
            else:
                self.curves.update({arg.ID: arg})

    def curve_array(self, task, treatment, region):
        curves = self[task, treatment]
        min_len = np.min([len(curve[region]) for curve in curves])
        curve_array = np.zeros(shape=(len(curves), min_len))
        for i, curve in enumerate(curves):
            curve_array[i] = curve.dfz_signals[region][:min_len]
        return curve_array

    def peak_dict(self, region, pos=True):
        """Used to create a dictionary that maps each individual curve ID to its event attributes. The absic idea being this will be apssed to a later function for a well formatted df.
        Args:
            region (str, optional): Which region peak properties to look at.
        Returns:
            peak_dict (dict): dictionary that maps each curve to its peak properties e.g. AUC, fwhm, etc.
        """
        id = [k for k in self.curves]
        if pos:
            peak_times = [v.Timestamps[region][v.region_peak_properties[region]['peaks']] for v in self.curves.values()]
            peak_heights = [v.region_peak_properties[region]['peak_heights'] for v in self.curves.values()]
            auc = [v.region_peak_properties[region]["areas_under_curve"] for v in self.curves.values()]
            fwhm = [v.region_peak_properties[region]["widths"] for v in self.curves.values()]
        else:
            peak_times = [v.Timestamps[region][v.neg_region_peak_properties[region]['peaks']] for v in
                          self.curves.values()]
            peak_heights = [-v.neg_region_peak_properties[region]['peak_heights'] for v in self.curves.values()]
            auc = [v.neg_region_peak_properties[region]["areas_under_curve"] for v in self.curves.values()]
            fwhm = [v.neg_region_peak_properties[region]["widths"] for v in self.curves.values()]
        peak_dict = {"ID": id, "Peak_Times": peak_times, "Amplitudes": peak_heights, "AUC": auc, "FWHM": fwhm}
        return peak_dict

    def histogram_2d(self, region):
        """Plots a 2D event histogram where Y=Amplitudes and X=FWHM. The idea is to see if there is an easily defined
        threshold where all events inside that 2D bin are negative.

        Args:
            region (str): Which indicator curve to plot the 2D event histogram for.
        """
        pos_peak_dict = self.peak_dict(region=region, pos=True)
        neg_peak_dict = self.peak_dict(region=region, pos=False)
        event_df = pd.concat((pd.DataFrame(pos_peak_dict).apply(pd.Series.explode).reset_index(drop=True),
                              pd.DataFrame(neg_peak_dict).apply(pd.Series.explode).reset_index(drop=True)))
        event_df.dropna(inplace=True)
        plt.hist2d(x='Amplitudes', y='FWHM', data=event_df)
        plt.show()
        return

    def eliminate_events(self):
        pass

    @staticmethod
    def bci(data, num_samples, cl=0.95):
        """
        Calculate the confidence interval for each timepoint in the data using bootstrap method.

        data: 2D numpy array, rows represent samples and columns represent time points.
        num_samples: Number of bootstrap samples.
        cl: Confidence level.

        Returns a 2D numpy array containing the lower and upper bound of the confidence interval for each timepoint.
        """
        n, t = data.shape
        ci = np.zeros((2, t))
        rng = np.random.default_rng()

        bootstrap_array = np.zeros(shape=(num_samples, t))

        for i in range(t):
            # Generate bootstrap samples for the current time point
            rints = rng.integers(0, n, num_samples)
            bootstrap_array[:, i] = data[rints, i]
        print(bootstrap_array)

        sig = 1 - cl
        ci[0, :] = np.percentile(bootstrap_array, q=(sig / 2)*100, axis=0)
        ci[1, :] = np.percentile(bootstrap_array, q=(1-(sig / 2))*100, axis=0)
        return ci

    @staticmethod
    def tci(array, cl=0.95):
        """_summary_

        Args:
            array (_type_): _description_
            cl (float, optional): _description_. Defaults to 0.95.

        Returns:
            _type_: _description_
        """
        n = array.shape[0]
        crit_t = scipy.stats.t.ppf(cl, df=n - 1)
        c_int = np.zeros(shape=(2, array.shape[1]))
        c_int[0, :] = np.mean(array, axis=0) - (crit_t * scipy.stats.sem(array, axis=0))
        c_int[1, :] = np.mean(array, axis=0) + (crit_t * scipy.stats.sem(array, axis=0))
        return c_int

    def plot_whole_eta(self, task, treatment, region, ci='tci', sig_duration=8):

        # create array of curves
        curves = self.curve_array(task, treatment, region)

        # choose t-confidence interval or bootstrapped
        if ci == 'tci':
            c_int = self.tci(curves)
        elif ci == 'bci':
            c_int = self.bci(curves, num_samples=1000)
        else:
            raise ValueError("Confidence interval options are 'tci' or 'bci'")
        
        # find where there is deflections from baseline
        deflections = c_int[0, :] > 0

        # create a sliding windor for convolution, used here to detect the sliding threshold
        window = np.ones(sig_duration)

        # convolve with boolean array
        true_deflects = np.convolve(deflections, window, 'valid')

        # find start indices where convolution equals to significant_duration
        start_indices = np.where(true_deflects == sig_duration)[0]


        # create figure
        fig, axs = plt.subplots()
        axs.plot(np.average(curves, axis=0))
        axs.fill_between(range(c_int.shape[1]), c_int[0, :], c_int[1, :], alpha=0.3)
        # plot the significant deflections
        y_height = axs.get_ylim()[1] * 0.97
        for start_index in start_indices:
            end_index = start_index + sig_duration
            axs.hlines(y=y_height, xmin=start_index, xmax=end_index, colors='r')
        return fig, axs
    
    def multi_event_eta(self, task, treatment, region,  events=None, window=3, ci='tci', sig_duration=8):
        
        # window in seconds times 30 indices per second and half the window period to visualize before
        number_of_indices = window*1.5*30

        # determine if we need timestamps for green or red indicator
        if "G" in region:
            time_idx = "2"
        else:
            time_idx = "4"

        ###TODO: Gonna need some if-else logic here or something to determind what the length of the events list is ###
        curves = self[task, treatment]
        across_eta_ = np.zeros((len(curves), int(number_of_indices)))
        for j, curve in enumerate(curves):
            within_eta_ = np.zeros((len(events[j]), int(number_of_indices)))
            interp = scipy.interpolate.interp1d(curve.Timestamps[time_idx], curve[region], kind='cubic')
            for i, event in enumerate(events[j]):
                time_period = np.linspace(event-(window/2), event+window, int(number_of_indices))
                within_eta_[i] = interp(time_period)
            across_eta_[j] = np.average(within_eta_, axis=0)

        average_trace = np.average(across_eta_, axis=0)
        # choose t-confidence interval or bootstrapped
        if ci == 'tci':
            c_int = self.tci(across_eta_)
        elif ci == 'bci':
            c_int = self.bci(across_eta_, num_samples=1000)
        else:
            raise ValueError("Confidence interval options are 'tci' or 'bci'")
        
        # make a figure
        fig, axs = plt.subplots()
        time = np.linspace(window/2, window, number_of_indices)
        axs.plot(time, average_trace)
        axs.fill_between(time, c_int[0, :], c_int[0, :], alpha=0.3)

        return fig, axs

    def event_summaries(self, region):
        """Creates a DataFrame that contains all the relevant event information e.g. AUC, FWHM, etc. This can be used
        for further analysis e.g. stats across groups linear mixed models etc.
        Args:
            region (str, optional): String to grab region trace of choice e.g. Region0G, Region1R, etc.
        Returns:
            df_transformed(pd.DataFrame): A DataFrame that contains a conglomeration of all the events across animals. 
        """
        summary_dict = self.peak_dict(region)
        df = pd.DataFrame(summary_dict)
        df_transformed = df.apply(pd.Series.explode).reset_index(drop=True)
        return df_transformed

    def raster_plot(self, task, treatment, region, xtick_range=None, xtick_freq=None):
        """Raster plot: Generate a raster plot of Z-scored fiber photometry traces.
        Args: task (str): String that represents the task e.g. FC, Recall, Ext etc.Should be identical to what was
        passed in fiberPhotometryCurve.
        treatment (str): String that represents the treatment e.g. eYFP, ChR2, Shock etc. Should be identical to what
        was passed in fiberPhotometryCurve.
        region (str): String to grab region trace of choice e.g. Region0G, Region1R, etc.
        xtick_range (int, optional): Length in time of session. Defaults to None.
        xtick_freq (int, optional): How many labels in [0, xtick_range]; end points inclusive. Defaults to None.
        Returns: matplotlib figure: a matplotlib figure object matplotlib axis: a matplotlib axis object
        """
        raster_array = self.curve_array(task, treatment, region)

        fig, ax = plt.subplots()
        sb.heatmap(raster_array, cbar=True, cbar_kws={"label": r"$\frac{dF}{F}$"}, center=0, yticklabels=False, ax=ax)
        ax.set_xlabel('Time (s)')
        if xtick_range and xtick_freq is not None:
            ax.set_xticks(np.linspace(0, raster_array.shape[1], xtick_freq),
                          labels=np.linspace(0, xtick_range, xtick_freq, dtype=np.int))
        return fig, ax
