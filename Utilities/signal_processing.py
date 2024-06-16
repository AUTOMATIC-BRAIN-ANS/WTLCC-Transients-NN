import math
import numpy as np
import scipy.signal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import blackmanharris, correlate
import scipy.ndimage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score

def find_pulses(sig):
    """
    Retrieves onset locations & pulses

    Parameters
    ----------
        
    sig: Signal to find pulses & locations

    Returns
    -------
        
    (pulses, pulse_locations): tuple of (segmented_pulses, pulse_onset_indices)

    """


    sig[:, 1] = remove_phase_shift(sig[:, [1]], sig[:, [0]]) # Remove phase shift between ABP & FVR
    sig[:, 2] = remove_phase_shift(sig[:, [2]], sig[:, [0]]) # Remove phase shift between ICP & FVR



    window_len = 6000
    abp_sig = sig[:, 0].flatten() # Isolate ABP signal for pulse extraction

    abp_sections = split_array(abp_sig, 6000) # Split ABP signal into equal lengths of 1 minute (assuming 100hz)
    
    pulse_locations = [find_pulses_window(section) + (i * window_len) for i, section in enumerate(abp_sections)] # Get pulse locations for each section
    pulse_locations = [p for p in pulse_locations if len(p) != 1] # Concatenate pulse locations for entire ABP signal
    pulse_locations = np.concatenate(pulse_locations, 0)
    pulses = segment_pulses(sig, pulse_locations) # Segment all signals into pulses based on pulse locations
    pulses = np.array([cluster_pulses(p) for p in pulses]) # Cluster pulses into dominant cluster 
    #pulses = np.concatenate(pulses, 0) # Concatenate pulses together
 
    
    return (pulses, pulse_locations) # Return pulses and pulse locations


def segment_pulses(sig, pulse_locations, pulse_len=100, pulse_window=60):
    """
    Segments a signal into pulses of length pulse_len. Every pulse_window pulses are averaged into one pulse.

    Parameters
    ----------

        sig: Signal to segment
        pulse_locations: indices of pulse onsets in scipy.signal
        pulse_len: length to reslice each extracted pulse [DEFAULT: 50]
        pulse_window: amount of pulses to average for each averaged pulse [DEFAULT: 5]

    Returns
    -------

        ndarray of shape (num_pulses, pulse_len, num_features)
    
    """

    pulses = [scipy.signal.resample(sig[pulse_locations[i]:pulse_locations[i + 1]], pulse_len) for i in range(len(pulse_locations) - 1)]
    pulses = np.array([p for p in pulses])
    pulses = np.array(split_array(pulses, pulse_window))

    return pulses





##### NEEDS UPDATING #####
def pulse_checker(pulse):

    if(np.min(pulse[0:10]) < 30 or np.max(pulse) > 300):
        print("False pulse found")
        return False

    return True


def find_pulses_window(sig):
    """
    Find pulse onset locations within a window of a signal

    Parameters
    ----------

        sig: The signal to pass in, preferable a smaller window of
             a bigger signal, as a full signal can be hard to detrend.

    Returns
    -------

        locations (indices) of pulse onsets within window

    """

    sig = sig.flatten()
    fund_freq = get_fundamental_frequency(sig)

    zero_crossings = np.where(np.diff(np.sign(fund_freq)))[0]
    zero_crossings = zero_crossings[::2]
    zero_crossing_distance = np.average(np.diff(zero_crossings))
    if(math.isnan(zero_crossing_distance)):
        return np.arange(1)

    peaks = scipy.signal.find_peaks(sig, distance=(
        int(zero_crossing_distance * 0.92)))[0]
    pulse_locations = []

    


    for peak in peaks:

        new_window = int(zero_crossing_distance / 4)
        start = peak - new_window if peak - new_window > 0 else 0
        window = sig[start:peak]
        pulse_locations.append(start + np.argmin(window))

    onset_pressures = sig[pulse_locations]
    if(not onset_pressures.size):
        return np.arange(1)

    pulse_locations = np.array(pulse_locations)

    
    return pulse_locations


def cluster_pulses(pulses):

    """
    Clusters pulses using hierarchical clustering & returns dominant cluster

    Parameters
    ----------

        pulses: list or array of standardised pulses

    Returns
    -------

        Dominant cluster of pulses

    """

    pulses = np.array(pulses)
    new_pulses = np.zeros((100, 3))

    for i in range(0, 3):

        sig_pulses = np.array([pulse[:, [i]] for pulse in pulses]).reshape(len(pulses), 100)
        sig_pulses_clustered = cluster(sig_pulses)  # sil(sig_pulses))
        new_pulses[:, [i]] = sig_pulses_clustered

    return new_pulses


def cluster(pulses, num_clusters=3):
    """
    Performs local clustering algorithm
    """
    cluster_alg = AgglomerativeClustering(n_clusters=num_clusters, linkage='average')
    cluster_alg.fit(pulses)
    labels = cluster_alg.labels_
    largest_cluster = np.bincount(labels).argmax()
    pulses = pulses[np.argwhere(labels == largest_cluster)]
    pulses = np.average(pulses, 0).reshape(100, 1)

    return pulses


def sil(X):

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    s_scores = []

    for n_clusters in range_n_clusters:

        s_scores.append(silhouette_score(X, labels))

    return range_n_clusters[np.argmax(s_scores)]


def remove_phase_shift(sig_a, sig_b):
    '''

    Usage: Eliminates phase shift between two signals at the fundamental frequency

    Parameters:

        sig_a: Signal in which to eliminate phase shift

        sig_b: Signal for comparison

    Returns:

        sig_a shifted

    '''

    sig_b = np.ravel(sig_b)
    sig_a = np.ravel(sig_a)
    xcorr = correlate(sig_a, sig_b)
    dt = np.arange(1 - len(sig_a), len(sig_a))
    rts = dt[xcorr.argmax()]
    sig_a = scipy.ndimage.shift(sig_a, -rts)
    sig_a = sig_a.reshape(len(sig_a))

    return sig_a


def split_array(arr, N):

    if (len(arr) == 0):
        return 0

    if arr.shape[0] % N == 0:
        split_idx = arr.shape[0] // N
    else:
        split_idx = np.arange(N, arr.shape[0], N)[:-1]

    return np.array_split(arr, split_idx, 0)


def get_fundamental_frequency(sig):

    sig_copy = sig
    sample_rate = 100
    N = len(sig)
    sig = bandpass_filter(scipy.signal.detrend(sig, type='linear'), 0.5, 3) # Detrend and filter signal
    fs = sample_rate
    windowed = sig * scipy.signal.blackmanharris(len(sig))

    f = scipy.fft.rfft(windowed)
    xf = scipy.fft.rfftfreq(N, 1 / sample_rate)
    i = np.argmax(np.abs(f))  # Just use this for less-accurate, naive version
    if(i==0):
        
        plot.plot_waveform(sig_copy)

    true_i = parabolic(np.log(np.abs(f)), i)[0]
    fund = (fs * (true_i) / len(windowed))

    points = len(xf) / (sample_rate / 2)
    
    if(math.isnan(fund)):
        return np.zeros((6000))

    target = int(points * fund)
    f = np.abs(f)
    f_cop = np.zeros(len(f))
    f_cop[target] = f[target]
    f = f_cop
    new_data = scipy.fft.irfft(f)
    return new_data


def filter_data(data, cutoff_freq):

    order = 3
    sampling_freq = 50
    N = len(data)
    normalized_cutoff_freq = (2 * cutoff_freq) / sampling_freq
    num, denom = scipy.signal.butter(order, normalized_cutoff_freq)
    filtered_signal = scipy.signal.lfilter(num, denom, data)

    return filtered_signal


def bandpass_filter(data, low, high):

    order = 3
    sampling_freq = 100
    N = len(data)
    normal_low = (2 * low) / sampling_freq
    normal_high = (2 * high) / sampling_freq
    num, denom = scipy.signal.butter(
        order, [normal_low, normal_high], btype='bandpass')
    filtered_signal = scipy.signal.lfilter(num, denom, data)

    return filtered_signal


def parabolic(f, x):
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


def parabolic_polyfit(f, x, n):

    a, b, c = np.polyfit(np.arange(x - n // 2, x + n // 2 + 1),
                         f[x - n // 2:x + n // 2 + 1], 2)
    xv = -0.5 * b / a
    yv = a * xv**2 + b * xv + c
    return (xv, yv)


