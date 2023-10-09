import pickle
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sb
import statsmodels.api as sm
import pykalman
from scipy import sparse
from scipy.integrate import simpson
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, resample
from scipy.sparse.linalg import spsolve
from photonsoup.anymaze_analysis import anymazeResults
from photonsoup.dlc_analysis import dlcResults
from photonsoup.b_spline import BSpline
from joblib import Parallel, delayed

__all__ = ["FiberPhotometryCurve", "FiberPhotometryCollection"]


class FiberPhotometryCurve:
    def __init__(self,
                 npm_file: str,
                 dlc_file: str = None,
                 offset: float = None,
                 anymaze_file: str = None,
                 regress: bool = True,
                 ID: str = None,
                 task: str = None,
                 treatment: str = None,
                 smoother='kalman',
                 batch=True):
        """
        """
        # these should always be present
        self.npm_file = npm_file
        self.fp_df = pd.read_csv(self.npm_file)
        self.__T0__ = self.fp_df.loc[0, 'Timestamp']
        self.__TN__ = self.fp_df.Timestamp.iloc[-1] - self.__T0__
        self.behavioral_data = {}
        self.ID = ID
        self.task = task
        self.treatment = treatment
        self.anymaze_file = anymaze_file
        self.dlc_file = dlc_file
        self.regress = regress
        self.smoother = smoother
        self.batch = batch
        self.dff_signals = None
        self.dfz_signals = None
        self.raw_signal = None
        self.timestamps = None
        self.anymaze_results = None
        self.dlc_results = None
        self.interp = None
        self.splines = None

        if offset is None:
            self.offset = 0
        else:
            self.offset = offset

        self.offset_seconds = self.offset - self.__T0__

        # determine sample time
        self._sample_time_ = np.diff(self.fp_df['Timestamp'])[1]
        self.fps = 1 / self._sample_time_

        # extract data, needs to be done so that batching can be done
        self._extract_data()

        # do preprocessing as part of initilization
        if not batch:
            self.dff_signals, self.dfz_signals = self._process_data()
            self.region_peak_properties = self.find_signal()
            self.neg_region_peak_properties = self.find_signal(neg=True)

    def __iter__(self):
        return iter(list(self.dff_signals.values()))

    def __eq__(self, other):
        if not isinstance(other, FiberPhotometryCurve):
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

        # Get the unique values in the "LedState" column
        led_states_unique = self.fp_df['LedState'].unique()

        # Filter out any invalid values (if present)
        led_state = [state for state in led_states_unique if state in [1, 2, 4]]

        # Get the unique region columns
        region_columns = self.fp_df.columns[self.fp_df.columns.str.contains('Region')].tolist()

        # Exclude data before offset
        self.fp_df = self.fp_df[self.fp_df['Timestamp'] >= self.offset].reset_index(drop=True)

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
                shortest_trace_length = min(shortest_trace_length, len(region_data[column]),
                                            len(self.fp_df[self.fp_df['LedState'] == 1][column]))

        for state in led_state:
            if not self.fp_df[self.fp_df['LedState'] == state].Timestamp.empty:
                # Store the de-interleaved timestamps for the LEDState
                timestamps[str(state)] = self.fp_df[self.fp_df['LedState'] == state].Timestamp.reset_index(
                    drop=True) - temp_t0

        # Trim the signals and timestamps to the length of the shortest trace
        for column in raw_signal.keys():
            raw_signal[column] = raw_signal[column][:shortest_trace_length]

        for column in isobestic_data.keys():
            isobestic_data[column] = isobestic_data[column][:shortest_trace_length]

        for state in led_state:
            timestamps[str(state)] = timestamps[str(state)][:shortest_trace_length]

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
                region_timeseries_corrected = self._fit_regression(region_timeseries_corrected,
                                                                   isobestic_timeseries_corrected)
                region_timeseries_corrected_z = self._fit_regression(region_timeseries_corrected_z,
                                                                     isobestic_timeseries_corrected_z)

            dff_signal[key] = region_timeseries_corrected
            dfz_signal[key] = region_timeseries_corrected_z

        return dff_signal, dfz_signal

    def _process_behavioral_data_batch(self, bin_duration=60, start=0, end=None):
        anymaze_results = anymazeResults(self.anymaze_file)
        anymaze_results.correct_time_warp(self.__TN__)
        percent_freezing = anymaze_results.calculate_binned_freezing(bin_duration=bin_duration, start=start, end=end,
                                                                     offset=self.offset_seconds)
        if "2" in self.Timestamps.keys():
            freeze_vector = anymaze_results.create_freeze_vector(self.Timestamps["2"])
        else:
            freeze_vector = anymaze_results.create_freeze_vector(self.Timestamps["4"])
            print(freeze_vector)

        freeze_onset, freeze_offset = anymaze_results.find_onset_offset()

        # process dlc results for getting kalman filter predictions
        dlc_results = dlcResults(self.dlc_file)
        dlc_results.process_dlc(bparts=None, fps=self.fps)

        return anymaze_results, percent_freezing, freeze_vector, freeze_onset, freeze_offset, dlc_results

    def calc_area(self, l_index, r_index, timeseries):
        """Calculates the area between the timeseries curve and the x-axis.

        Args:
            l_index (List[int]): List of left indices for the areas to calculate.
            r_index (List[int]): List of right indices for the areas to calculate.
            timeseries (List[float]): List of y values (height of the curve at each point).

        Returns:
            List[float]: List of the areas calculated between each pair of indices.
        """
        areas = []
        dt = self._sample_time_
        for i, j in zip(l_index, r_index):
            if (timeseries[i:j + 1] < 0).any():
                min_val = min(timeseries[i:j + 1])
                area = simpson([val - min_val for val in timeseries[i:j + 1]], dx=dt)
                areas.append(area)
            else:
                area = simpson([val for val in timeseries[i:j + 1]], dx=dt)
                areas.append(area)
        return areas

    def find_signal(self, neg=False):
        """_summary_

        Args:
            neg (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        region_peak_properties = {}

        for region, sig in self.dff_signals.items():
            signal = -sig if neg else sig
            peaks, properties = find_peaks(signal, height=np.std(signal), prominence=2 * np.std(signal), width=(7, 150),
                                           rel_height=0.5)
            properties['peaks'] = peaks
            properties['areas_under_curve'] = self.calc_area([int(x) for x in properties['left_ips']],
                                                             [int(x) for x in properties['right_ips']],
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
        if hasattr(self, "region_peak_properties"):
            fig = plt.figure()
            plt.plot(self.dff_signals[signal])
            plt.plot(self.region_peak_properties[signal]['peaks'],
                     self.dff_signals[signal][self.region_peak_properties[signal]['peaks']], "x")
            plt.vlines(x=self.region_peak_properties[signal]['peaks'],
                       ymin=self.dff_signals[signal][self.region_peak_properties[signal]['peaks']] -
                            self.region_peak_properties[signal]["prominences"],
                       ymax=self.dff_signals[signal][self.region_peak_properties[signal]['peaks']], color="C1")
            plt.hlines(y=self.region_peak_properties[signal]["width_heights"],
                       xmin=self.region_peak_properties[signal]["left_ips"],
                       xmax=self.region_peak_properties[signal]["right_ips"], color="C1")
            plt.show()
        else:
            raise KeyError(f'{signal} is not in {self}')
        return fig

    def generate_splines(self, n_knots=12, spline_indices=None, spline_length=30):
        b_set = BSpline(spline_length, 3, n_knots)
        try:
            spline_set = b_set.create_spline_map(spline_indices, len(self.dff_signals["Region0G"]))
        except KeyError:
            spline_set = b_set.create_spline_map(spline_indices, len(self.dff_signals["Region1G"]))
        return spline_set

    def regression_splines(self):
        def find_nearest_index(array, value):
            array = np.asarray(array)
            idx = np.argmin(np.abs(array - value))
            return int(idx)
        shock_idxes = [find_nearest_index(self.Timestamps["2"], idx) for idx in [120, 180, 240, 300]]
        print(shock_idxes)
        shock1_spline_set = self.generate_splines(spline_indices=shock_idxes)
        freeze_onset_set = self.generate_splines(spline_indices=self.behavioral_data["freeze_onsets"])
        freeze_offset_set = self.generate_splines(spline_indices=self.behavioral_data["freeze_onsets"])
        return shock1_spline_set, freeze_onset_set, freeze_offset_set


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
        """add_curve: add a fiberPhotometryCurve object to the collection for analysis.
            Args:
                *args(FiberPhotometryCurve): fiberPhotometryCurves
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

    def curve_array(self, task, treatment, region, kind="f"):
        curves = self[task, treatment]
        min_len = np.min([len(curve[region]) for curve in curves])
        curve_array = np.zeros(shape=(len(curves), min_len))
        for i, curve in enumerate(curves):
            if kind == "f":
                curve_array[i] = curve.dff_signals[region][:min_len]
            else:
                curve_array[i] = curve.dfz_signals[region][:min_len]
        return curve_array

    def batch_data(self):
        # results is now a list of tuples (dff_signal, dfz_signal)
        results = Parallel(n_jobs=-1)(delayed(curve._process_data)() for curve in self.curves.values())

        # Update each curve object with the returned results
        for curve, (dff_signal, dfz_signal) in zip(self.curves.values(), results):
            curve.dff_signals = dff_signal
            curve.dfz_signals = dfz_signal

        # now process signals
        signal_props = Parallel(n_jobs=-1)(delayed(curve.find_signal)() for curve in self.curves.values())
        neg_signal_props = Parallel(n_jobs=-1)(delayed(curve.find_signal)(neg=True) for curve in self.curves.values())

        # update attr
        for curve, props, neg_props in zip(self.curves.values(), signal_props, neg_signal_props):
            curve.region_peak_properties, curve.neg_region_peak_properties = props, neg_props

    def batch_behavior(self, end):
        if not end:
            raise ValueError("End-time for session must be specified.")
        # results in a complex tuple, refer to function in curve class for reference
        results = Parallel(n_jobs=-1)(
            delayed(curve._process_behavioral_data_batch)(end=end) for curve in self.curves.values())

        # update attributes
        for curve, (anymaze_results,
                    percent_freezing,
                    freeze_vector,
                    freeze_onset,
                    freeze_offset,
                    dlc_results) in zip(self.curves.values(), results):
            curve.anymaze_results = anymaze_results
            curve.behavioral_data["percent_freezing"] = percent_freezing
            curve.behavioral_data["freeze_vector"] = freeze_vector
            curve.behavioral_data["freeze_onsets"] = freeze_onset
            curve.behavioral_data["freeze_offsets"] = freeze_offset
            curve.dlc_results = dlc_results

    def peak_dict(self, region, pos=True):
        """Used to create a dictionary that maps each individual curve ID to its event attributes.
        Args:
            region (str, optional): Which region peak properties to look at.
            pos (bool, optional): Indicate if we are looking for positive peaks. Defaults to True.
        Returns:
            peak_dict (dict): dictionary that maps each curve to its peak properties e.g. AUC, fwhm, etc.
        """
        ids = list(self.curves.keys())
        ts_id = "2" if "G" in region else "4"
        peak_properties_key = 'region_peak_properties' if pos else 'neg_region_peak_properties'

        peak_dicts = []
        for v in self.curves.values():
            peak_properties = getattr(v, peak_properties_key)[region]
            peak_time = v.Timestamps[ts_id][peak_properties['peaks']].to_numpy()
            peak_height = peak_properties['peak_heights'] if pos else -peak_properties['peak_heights']
            auc = peak_properties["areas_under_curve"]
            fwhm = peak_properties["widths"]

            peak_dicts.append({
                "Peak_Times": peak_time,
                "Amplitudes": peak_height,
                "AUC": auc,
                "FWHM": fwhm
            })

            # Check if peak_dicts is populated
            if not peak_dicts:
                print(f"No peak properties found for region {region} and pos {pos}")
                return {}

            # Check if keys in peak_dicts are as expected
            expected_keys = {"Peak_Times", "Amplitudes", "AUC", "FWHM"}
            if not expected_keys.issubset(set(peak_dicts[0].keys())):
                print(f"Unexpected keys in peak properties: {peak_dicts[0].keys()}")
                return {}

        combined_dict = {k: [dic[k] for dic in peak_dicts] for k in peak_dicts[0]}
        return {"ID": ids, **combined_dict}

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

        sig = 1 - cl
        ci[0, :] = np.percentile(bootstrap_array, q=(sig / 2) * 100, axis=0)
        ci[1, :] = np.percentile(bootstrap_array, q=(1 - (sig / 2)) * 100, axis=0)
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

    @staticmethod
    def eta_significance(ci, sig_duration):

        # Find deflections from baseline
        # True for intervals entirely above zero or entirely below zero
        deflections = (ci[0, :] > 0) | (ci[1, :] < 0)

        # Create a sliding window for convolution
        window = np.ones(sig_duration)

        # Convolve with boolean array
        true_deflects = np.convolve(deflections, window, 'valid')

        # Find start indices where convolution equals to sig_duration
        start_indices = np.where(true_deflects == sig_duration)[0]
        return start_indices

    def plot_whole_eta(self, task, treatment, region, trial_len=360, ci='tci', sig_duration=8, ax=None, a=0.05,
                       **kwargs):

        # create array of curves
        curves = self.curve_array(task, treatment, region)

        # choose t-confidence interval or bootstrapped
        confidence_level = 1 - a
        if ci == 'tci':
            c_int = self.tci(curves, confidence_level)
        elif ci == 'bci':
            c_int = self.bci(curves, cl=confidence_level, num_samples=1000)
        else:
            raise ValueError("Confidence interval options are 'tci' or 'bci'")

        # find significant indices
        start_indices = self.eta_significance(c_int, sig_duration=8)  # change to be dependent on the sampling rate

        # create figure
        if ax is None:
            fig, ax = plt.subplots()

        # create time
        time = np.linspace(0, trial_len, len(curves.T))

        ax.plot(time, np.average(curves, axis=0), **kwargs)
        ax.fill_between(time, c_int[0, :], c_int[1, :], alpha=0.3)
        # plot the significant deflections
        y_height = ax.get_ylim()[1] * 0.97
        for start_index in start_indices:
            end_index = start_index + sig_duration
            try:
                ax.hlines(y=y_height, xmin=time[start_index], xmax=time[end_index], colors='r')
            except IndexError:
                end_index = len(time) - 1  # set end_index to the last valid index
                print(f"Warning: Adjusted end index to {end_index} due to out-of-bounds error.")
                ax.hlines(y=y_height, xmin=time[start_index], xmax=time[end_index], colors='r')

        if ax is None:
            return fig, ax
        else:
            return ax

    def multi_event_eta(self, task, treatment, region, events=None, window=3, ci='tci', sig_duration=8, a=0.05, ax=None,
                        **kwargs):
        # window in seconds times 30 indices per second and half the window period to visualize before
        number_of_indices = int(window * 1.5 * 15)

        # determine if we need timestamps for green or red indicator
        time_idx = "2" if "G" in region else "4"

        def event_interpolation(curve, events_):
            within_eta_ = np.zeros((len(events_), number_of_indices))
            interp = scipy.interpolate.interp1d(curve.Timestamps[time_idx], curve.dfz_signals[region], kind='cubic',
                                                bounds_error=False, fill_value=np.nan)
            for i, event in enumerate(events_):
                time_period = np.linspace(event - (window / 2), event + window, number_of_indices)
                within_eta_[i] = interp(time_period)
            return np.nanmean(within_eta_, axis=0)

        curves = self[task, treatment]
        # If there's only one event, repeat it for each curve
        if len(events) == 1:
            events = events * len(curves)
        else:
            # Identify valid indices where events is not an empty list
            valid_indices = [i for i, e in enumerate(events) if len(e) > 0]
            # Filter curves and events to only include valid entries
            curves = [curves[i] for i in valid_indices]
            events = [events[i] for i in valid_indices]
        # Pre-allocate eta_array
        eta_array = np.zeros((len(events), number_of_indices))
        for j, curve in enumerate(curves):
            eta_array[j] = event_interpolation(curve, events[j])

        average_trace = np.average(eta_array, axis=0)
        # choose t-confidence interval or bootstrapped
        confidence_level = 1 - a
        if ci == 'tci':
            c_int = self.tci(eta_array, cl=confidence_level)
        elif ci == 'bci':
            c_int = self.bci(eta_array, cl=confidence_level, num_samples=1000)
        else:
            raise ValueError("Confidence interval options are 'tci' or 'bci'")

        # find significant indices
        start_indices = self.eta_significance(c_int, sig_duration=8)  # change to be dependent on the sampling rate

        # make a figure
        if ax is None:
            fig, ax = plt.subplots()
        time = np.linspace(-window / 2, window, number_of_indices)
        ax.plot(time, average_trace, **kwargs)
        ax.fill_between(time, c_int[0, :], c_int[1, :], alpha=0.3)

        # plot the significant deflections
        y_height = ax.get_ylim()[1] * 0.97
        for start_index in start_indices:
            end_index = start_index + sig_duration
            try:
                ax.hlines(y=y_height, xmin=time[start_index], xmax=time[end_index], colors='r')
            except IndexError:
                end_index = len(time) - 1  # set end_index to the last valid index
                print(f"Warning: Adjusted end index to {end_index} due to out-of-bounds error.")
                ax.hlines(y=y_height, xmin=time[start_index], xmax=time[end_index], colors='r')

        if ax is None:
            return fig, ax, eta_array
        else:
            return ax, eta_array

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
        df_transformed = df.set_index('ID').apply(pd.Series.explode).reset_index()

        # a set of nasty one-liners that maps the curve task and treatment values to a new column in the dataframe
        df_transformed.loc[:, 'task'] = df_transformed['ID'].map(self.curves).apply(
            lambda x: x.task if x is not None else 'Unknown')
        df_transformed.loc[:, 'treatment'] = df_transformed['ID'].map(self.curves).apply(
            lambda x: x.treatment if x is not None else 'Unknown')

        return df_transformed.infer_objects()

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

    def design_matrix(self, task, treatment, region_g, region_r):
        curves = self[task, treatment]
        list_of_dfs = []
        for curve in curves:
            length_curve = len(curve.Timestamps["2"])
            shock_splines, freeze_onset_splines, freeze_offset_splines = curve.regression_splines()
            df = pd.DataFrame({"G": curve[region_g],
                               "R": curve[region_r],
                               "X": resample(curve.dlc_results.filtered_df["centroid"].x, length_curve),
                               "Y": resample(curve.dlc_results.filtered_df["centroid"].y, length_curve),
                               "X_velocity": resample(curve.dlc_results.filtered_df["centroid"].velocity_x, length_curve),
                               "Y_velocity": resample(curve.dlc_results.filtered_df["centroid"].velocity_y, length_curve),
                               "X_acceleration": resample(curve.dlc_results.filtered_df["centroid"].acceleration_x, length_curve),
                               "Y_acceleration": resample(curve.dlc_results.filtered_df["centroid"].acceleration_y, length_curve),
                               "Freezing": curve.behavioral_data["freeze_vector"],
                               "Mouse_ID": [curve.ID for i in range(len(curve[region_g]))],
                               })
            for i in range(shock_splines.shape[0]):
                df[f"Shock_Spline{i}"] = shock_splines[i, :]
            for i in range(freeze_onset_splines.shape[0]):
                df[f"Freeze_Onset_Spline{i}"] = freeze_onset_splines[i, :]
            for i in range(freeze_offset_splines.shape[0]):
                df[f"Freeze_Offset_Spline{i}"] = freeze_offset_splines[i, :]
            list_of_dfs.append(df)
        mega_df = pd.concat(list_of_dfs, ignore_index=True)
        return mega_df

    def save(self, filename):
        """Save the FiberPhotometryCollection object to a file.

        Args:
            filename (str): The name of the file to save the object to. Should include the .pkl extension.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a FiberPhotometryCollection object from a file.

        Args:
            filename (str): The name of the file to load the object from. Should include the .pkl extension.

        Returns:
            FiberPhotometryCollection: The loaded object.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
