import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.signal import detrend,butter,lfilter,filtfilt,savgol_filter,medfilt2d
import librosa
import librosa.display
import noisereduce as nr
from tqdm.notebook import tqdm
import numba as nb
from numba import cuda
import math
from PIL import Image
from io import BytesIO  
import urllib.request
import matplotlib
import pyproj
import geopandas as gpd
from shapely.geometry import Point,Polygon,box,MultiPoint,LineString
from os import path
import xml.dom.minidom
import acoular
import h5py
import tables
import contextily as ctx

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

def bandpass_filter(data, fs, lower_cut, upper_cut, order, _axis):
    """
    Apply a bandpass filter to the input data.

    Parameters:
        - data (ndarray): Input signal data.
        - fs (int): Sampling frequency in Hz.
        - lower_cut (float): Lower cutoff frequency for the bandpass filter in Hz.
        - upper_cut (float): Upper cutoff frequency for the bandpass filter in Hz.
        - order (int): Filter order.
        - _axis (int): Axis along which to apply the filter.

    Returns:
        - filtered_data (ndarray): Filtered signal data.
    """
    nyquist = 0.5 * fs
    low = lower_cut / nyquist
    high = upper_cut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data, axis=_axis)
    return filtered_data

def highpass_filter(data, cutoff_freq, fs, order=4, _axis=1):
    """
    Apply a high-pass filter to the input data.

    Parameters:
    - data: 2D array containing the data to be filtered.
    - cutoff_freq: The cutoff frequency of the high-pass filter in Hz.
    - fs: The sampling frequency of the data in Hz.
    - order: The order of the high-pass filter (default is 4).

    Returns:
    - filtered_data: The filtered data.
    """
    nyquist = 0.5 * fs
    high = cutoff_freq / nyquist
    b, a = butter(order, high, btype='high')
    filtered_data = filtfilt(b, a, data, axis=_axis)    
    return filtered_data

def extract_fourth_part(filename):    
    """
    Extract the fourth part of a filename, assuming the filename is divided by hyphens.

    Parameters:
        - filename (str): Input filename string.

    Returns:
        - fourth_part (int or None): The fourth part as an integer, or None if conversion fails.
    """
    parts = filename.split('-')
    if len(parts) == 5:
        try:
            fourth_part = int(parts[3])
            return fourth_part
        except ValueError:
            return None
    else:
        return None

def extract_start_time(filename):
    """
    Extract the starting time from a filename, assuming a specific format.

    Parameters:
        - filename (str): Input filename string.

    Returns:
        - start_time (datetime or None): Parsed starting time as a datetime object, or None if parsing fails.
    """
    parts = filename.split('-')
    if len(parts) == 5:
        try:
            return datetime.strptime(parts[2][3:], "%H%M%S")
        except ValueError:
            return None
    else:
        return None
    
def extract_start_channel(filename):
    """
    Extract the starting channel number from a filename.

    Parameters:
        - filename (str): Input filename string.

    Returns:
        - start_channel (int or None): The starting channel number as an integer, or None if conversion fails.
    """
    parts = filename.split('-')
    if len(parts) == 5:
        try:
            return int(parts[3])
        except ValueError:
            return None
    else:
        return None

def extract_end_channel(filename):
    """
    Extract the ending channel number from a filename.

    Parameters:
        - filename (str): Input filename string.

    Returns:
        - end_channel (int or None): The ending channel number as an integer, or None if conversion fails.
    """
    parts = filename.split('-')
    if len(parts) == 5:
        try:
            return int(parts[4][:-4])
        except ValueError:
            return None
    else:
        return None

def plot_das_data(t_axis, x_axis, data, 
                  lb_perc = 1, up_perc = 99, lower_bound=False, upper_bound=False,
                  start_channel=None, end_channel=None, start_time=None, end_time=None,
                  save_fig=False):
    """
    Plot DAS (Distributed Acoustic Sensing) data as an image.

    Parameters:
        - t_axis (ndarray): Time axis values.
        - x_axis (ndarray): Spatial axis (channel) values.
        - data (ndarray): 2D array containing DAS data.
        - lb_perc (float): Lower percentile for plotting.
        - up_perc (float): Upper percentile for plotting.
        - lower_bound (bool or float): Lower bound for data visualization.
        - upper_bound (bool or float): Upper bound for data visualization.
        - start_channel (int or None): Starting channel to include in the plot.
        - end_channel (int or None): Ending channel to include in the plot.
        - start_time (datetime or None): Starting time to include in the plot.
        - end_time (datetime or None): Ending time to include in the plot.
        - save_fig (bool or str): File path to save the figure, or `False` to display directly.

    Returns:
        None
    """
    if start_channel is not None and end_channel is not None:
        x_indices = np.where((x_axis >= start_channel) & (x_axis <= end_channel))[0]
        x_axis = x_axis[x_indices]
        data = data[:, x_indices]

    if start_time is not None and end_time is not None:
        start_index = np.searchsorted(t_axis, start_time)
        end_index = np.searchsorted(t_axis, end_time)
        t_axis = t_axis[start_index:end_index]
        data = data[start_index:end_index, :]

    y_lims = mdates.date2num(t_axis)
    if not lower_bound:
        lower_bound = np.percentile(data, lb_perc)
    if not upper_bound:
        upper_bound = np.percentile(data, up_perc)

    # Create the imshow plot
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed
    plt.imshow(data[::-1,:], aspect='auto', vmin=lower_bound, vmax=upper_bound,
               extent=[min(x_axis), max(x_axis), y_lims[0], y_lims[-1]],
               cmap='seismic')

    # Convert numeric y-axis to datetime format
    date_format = mdates.DateFormatter("%H:%M:%S")
    ax.yaxis.set_major_formatter(date_format)

    colorbar = plt.colorbar(label='DAS')

    plt.xlabel('Channel #')
    plt.ylabel('Time')
    plt.gca().invert_yaxis()  # Invert the y-axis to have time increasing upwards
    if save_fig:
        plt.savefig(save_fig)
        plt.close()
    else:
        plt.show()
    
def plot_psds(data, fs, x_axis, nfft, freq_range=[0,50]):    
    """
    Plot the Power Spectral Density (PSD) of each channel of the input data.

    Parameters:
        - data (ndarray): 2D array containing time-series data (Time x Channels).
        - fs (int): Sampling frequency of the data in Hz.
        - x_axis (ndarray): Array of spatial channel indices.
        - nfft (int): Length of the FFT window.
        - freq_range (list): List with two elements specifying the frequency range to plot [min_freq, max_freq].

    Returns:
        None
    """
    num_channels = data.shape[1]

    # Compute the PSD for each channel
    psd_values = np.zeros((num_channels, nfft // 2 + 1))
    for i in range(num_channels):
        psd_values[i, :], f = plt.psd(data[:, i], NFFT=nfft, detrend='linear',
                                     Fs=fs, noverlap=nfft // 2,
                                     window=np.hanning(nfft))

    # Create a 2D plot of PSD values
    plt.figure(figsize=(10, 6))   
    freq = np.where((f  >= freq_range[0]) & (f  <= freq_range[1]))[0]
    plt.imshow(10 * np.log10(psd_values[:,freq]), aspect='auto', cmap='jet', origin='lower',interpolation='gaussian',
               extent=[freq_range[0], freq_range[1], min(x_axis), max(x_axis)])
    plt.colorbar(label='PSD (dB/Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channel #')
#     plt.title('Power Spectral Density (PSD) of Each Channel')
    plt.show()
    
def remove_percentile_outliers(data, window_size, lower_percentile=10, upper_percentile=90):
    """
    Remove outliers from the input data using percentile-based thresholds.

    Parameters:
        - data (ndarray): 2D array representing time-series data (Time x Channels).
        - window_size (int): Size of the moving window used for outlier removal.
        - lower_percentile (float): Lower percentile threshold for outlier detection.
        - upper_percentile (float): Upper percentile threshold for outlier detection.

    Returns:
        - denoised_signal (ndarray): Data array with outliers removed based on percentile thresholds.
    """
    denoised_signal = data.copy()
    
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    is_outlier = np.where((data < lower_bound) | (data > upper_bound))

    for row, col in zip(*is_outlier):
        # Calculate the moving mean over the specified window
        start = max(0, col - window_size // 2)
        end = min(data.shape[1], col + window_size // 2 + 1)
        moving_median = np.median(data[row, start:end])

        # Replace the outlier with the moving mean
        denoised_signal[row, col] = moving_median

    return denoised_signal
  
def remove_lower_percentile(signal, percentile=5, window_size=50):
    """
    Replace data values below a given percentile with the percentile value itself.

    Parameters:
        - signal (ndarray): 1D array containing time-series signal data.
        - percentile (float): The percentile value below which data points will be replaced.
        - window_size (int): Size of the moving window used for the percentile calculation.

    Returns:
        - filtered_signal (ndarray): Modified signal with values below the percentile replaced.
    """
    filtered_signal = np.copy(signal)
    half_window = window_size // 2

    for i in range(len(signal)):
        window_start = max(0, i - half_window)
        window_end = min(len(signal), i + half_window + 1)
        window = signal[window_start:window_end]
        lower_percentile = np.percentile(window, percentile)
        if signal[i] < lower_percentile:
            filtered_signal[i] = lower_percentile

    return filtered_signal

def post_processing(data, x_axis, t_axis, data_ori=None, fs=200, data_detrend=False, spatial_highpass=False, 
                    temp_highpass = False, norm_x = False, norm_t = False,
                    median2d = False, bandpass = False, taper_t=False, taper_x=False, 
                    spec_gating=False, spatial_sg=False,rm_out=False,
                    start_channel=None, end_channel=None, start_time=None, end_time=None):
    """
    Perform various signal post-processing operations on input data.

    Parameters:
        - data (ndarray): 2D array containing time-series data (Time x Channels).
        - x_axis (ndarray): Array of spatial channel indices.
        - t_axis (ndarray): Array of time indices.
        - data_ori (ndarray or None): Original unprocessed data for certain normalization steps.
        - fs (int): Sampling frequency in Hz.
        - data_detrend (bool): Whether to detrend the data.
        - spatial_highpass (dict or bool): Parameters for a high-pass filter along the spatial axis.
        - temp_highpass (dict or bool): Parameters for a high-pass filter along the temporal axis.
        - norm_x (bool): Whether to normalize along the spatial axis.
        - norm_t (int or bool): Window size for normalization along the temporal axis.
        - median2d (tuple or bool): Kernel size for a 2D median filter.
        - bandpass (dict or bool): Parameters for a bandpass filter.
        - taper_t (int or bool): Taper length for tapering along the temporal axis.
        - taper_x (int or bool): Taper length for tapering along the spatial axis.
        - spec_gating (dict or bool): Parameters for spectral gating noise reduction.
        - spatial_sg (dict or bool): Parameters for Savitzky-Golay filtering along the spatial axis.
        - rm_out (dict or bool): Parameters for outlier removal using percentile thresholds.
        - start_channel (int or None): Starting channel index for selection.
        - end_channel (int or None): Ending channel index
        - start_time (datetime or None): Starting time index for selection.
        - end_time (datetime or None): Ending time index for selection.
    Returns:
        - x_axis (ndarray): Selected spatial channel indices after filtering.
        - t_axis (ndarray): Selected time indices after filtering.
        - data (ndarray): Post-processed data array (Time x Channels).
    """
    # Filter channels by the specified range
    if start_channel is not None and end_channel is not None:
        x_indices = np.where((x_axis >= start_channel) & (x_axis <= end_channel))[0]
        x_axis = x_axis[x_indices]
        data = data[:, x_indices]

    if start_time is not None and end_time is not None:
        start_index = np.searchsorted(t_axis, start_time)
        end_index = np.searchsorted(t_axis, end_time)
        t_axis = t_axis[start_index:end_index]
        data = data[start_index:end_index, :]
        
    # Post-process the concatenated data
    data_ori = data.copy()
    if data_detrend:
        data = detrend(data, axis=0, type='constant')
    if spatial_highpass:
        data = highpass_filter(data, spatial_highpass['hp'], 
                               spatial_highpass['fs'], spatial_highpass['order'])
    if temp_highpass:
        data = highpass_filter(data, temp_highpass['hp'], temp_highpass['fs'], 
                               temp_highpass['order'], temp_highpass['axis'])
    if bandpass:
        data = bandpass_filter(data, bandpass['fs'], bandpass['lp'], 
                               bandpass['hp'], bandpass['order'], 0)
    if spec_gating:
        for i in range(np.shape(data)[1]):
            data[:,i] = nr.reduce_noise(y=data[:,i], sr=spec_gating['fs'], stationary=False,
                                   time_constant_s=spec_gating['time_constant_s'], 
                                   chunk_size=spec_gating['chunk_size'], 
                                   time_mask_smooth_ms=spec_gating['time_mask_smooth_ms'])
    if norm_x:
        data = data - np.mean(data, axis=1)[:, np.newaxis]
    if norm_t:  
        data_bp = bandpass_filter(data_ori, fs, 2.5, 5, 2, 0)   
        energy = np.mean(np.sqrt((data_bp)**2),axis=0)
        filted_energy = remove_lower_percentile(energy, percentile=50, window_size=norm_t)
        plt.plot(x_axis,np.convolve(filted_energy,np.ones(int(norm_t/2))/int(int(norm_t/2)),mode='same'))
        data = data / filted_energy
    if rm_out:
        data = remove_percentile_outliers(data, rm_out['win_size'], 
                                           lower_percentile=rm_out['low'], 
                                           upper_percentile=rm_out['up'])
    if median2d:
        data = medfilt2d(data, kernel_size=median2d)
    if taper_t:
        data = signal_taper_2d(data, taper_length=taper_t, taper_shape="hann", axis=0)
    if taper_x:
        data = signal_taper_2d(data, taper_length=taper_x, taper_shape="hann", axis=1)
    if spatial_sg:
        data = data-savgol_filter(data, spatial_sg['win_len'], spatial_sg['order'], axis=1)

    return x_axis,t_axis,data

def signal_taper_2d(signal, taper_length=10, taper_shape="hann", axis=0):
    """
    Apply a signal taper to a 2D signal along the specified axis using a window function.

    Parameters:
    - signal: 2D numpy array
        The input signal to which the taper will be applied.
    - taper_length: int
        The length of the taper region at the beginning and end along the specified axis.
    - taper_shape: str
        The shape of the taper (e.g., "hann" for a Hanning window, "hamming" for a Hamming window).
    - axis: int
        The axis along which the taper will be applied (0 for rows, 1 for columns).

    Returns:
    - tapered_signal: 2D numpy array
        The signal with the taper applied along the specified axis.
    """
    signal_shape = signal.shape
    signal_length = signal_shape[axis]

    if taper_length >= signal_length:
        raise ValueError("Taper length cannot be greater than or equal to the signal length along the specified axis.")

    if taper_shape == "hann":
        taper = np.hanning(2 * taper_length)
    elif taper_shape == "hamming":
        taper = np.hamming(2 * taper_length)
    else:
        raise ValueError("Unsupported taper shape. Use 'hann' or 'hamming'.")

    # Create a taper along the specified axis
    if axis == 0:
        taper = taper[:, np.newaxis]
    else:
        taper = taper[np.newaxis, :]

    # Apply the taper to the signal along the specified axis
    tapered_signal = np.copy(signal)
    if axis == 0:
        tapered_signal[:taper_length, :] *= taper[:taper_length, :]
        tapered_signal[-taper_length:, :] *= taper[taper_length:, :]
    else:
        tapered_signal[:, :taper_length] *= taper[:, :taper_length]
        tapered_signal[:, -taper_length:] *= taper[:, taper_length:]

    return tapered_signal

def calculate_edge_coordinates(start_coords, end_coords, start_cha, end_cha):
    """
    Calculate geographical coordinates for each edge channel between the start and end channels.

    Parameters:
        - start_coords (tuple): GPS coordinates (latitude, longitude) of the start point.
        - end_coords (tuple): GPS coordinates (latitude, longitude) of the end point.
        - start_cha (int): Starting channel number.
        - end_cha (int): Ending channel number.

    Returns:
        - edge_coordinates (list): List of tuples containing the channel number and corresponding (latitude, longitude) coordinates.
    """
    edge_coordinates = []
    for i in range(start_cha,end_cha+1):
        fraction = (i-start_cha) / (end_cha-start_cha)
        lat = start_coords[0] + fraction * (end_coords[0] - start_coords[0])
        lon = start_coords[1] + fraction * (end_coords[1] - start_coords[1])
        edge_coordinates.append((i, lat, lon))

    return edge_coordinates

def delays(grid2sensor_dist, c):
    """
    Compute delays for each grid point to each sensor based on distance and propagation speed.

    Parameters:
        - grid2sensor_dist (ndarray): 2D array containing distances from each grid point to each sensor (Grid Points x Sensors).
        - c (float): Wave propagation speed in meters per second.

    Returns:
        - delays (ndarray): 2D array containing the calculated delays in number of samples (Grid Points x Sensors).
    """
    gridsize, numchannels = grid2sensor_dist.shape
    delays = np.empty((gridsize,numchannels),dtype=np.float64)
    for gi in range(gridsize):
        for mi in range(numchannels):
            delays[gi,mi] = (1/c) * grid2sensor_dist[gi,mi]
    return delays

def freq_beamformer_dyn(freq_lb,freq_up,vel_func,plane_thr,grid,mics,env,source,ps,st,r_diag=False,Q=10):
    """
    Apply a frequency domain dynamic beamforming algorithm to a given source grid.

    Parameters:
        - freq_lb (float): Lower bound of the frequency range.
        - freq_up (float): Upper bound of the frequency range.
        - vel_func (function): Function to calculate the velocity at each frequency.
        - plane_thr (float): Plane threshold for considering microphones in the steering vector.
        - grid (Grid): Grid object representing the source grid.
        - mics (MicrophoneArray): Microphone array object.
        - env (Environment): Environmental parameters.
        - source (Source): Source parameters.
        - ps (PowerSpectrum): Power spectrum object.
        - st (Steering): Steering vector.
        - r_diag (bool): Whether to apply diagonal removal in the cross-spectral matrix.
        - Q (int): Quality factor for attenuation.

    Returns:
        - h (ndarray): Beamforming output as a reshaped grid.
        - ac (ndarray): Accumulated beamforming outputs for each frequency.
    """
    freq = ps.fftfreq()
    ind1 = np.searchsorted(freq, freq_lb)
    ind2 = np.searchsorted(freq, freq_up)
    num_freq = ind2-ind1
    ac = np.zeros((num_freq, grid.size), dtype=np.float64)
    distGridToAllMic = env._r(grid.pos(),mics.mpos)
    gd_size,nMics = distGridToAllMic.shape
    csm = np.array(ps.csm[ind1:ind2].copy())
    for i in tqdm(range(ind1,ind2)):
        for gi in tqdm(range(gd_size), leave=False):
            steerVec = np.empty((nMics), np.complex128)
            idx = i - ind1
            if vel_func(freq[i])/freq[i]>plane_thr:
                p_thr = vel_func(freq[i])/freq[i]
            else:
                p_thr = plane_thr
            mic_idx = distGridToAllMic[gi,:]>=p_thr
            distGridToMic = distGridToAllMic[:,mic_idx]
            waveNumber = 2*np.pi*freq[i]/vel_func(freq[i])

            _beamformer(distGridToMic,waveNumber,csm,mic_idx,steerVec,gi,idx,ac,st,r_diag,Q)

    h = np.sum(ac,0).reshape(grid.shape)
    return h,ac

@nb.njit(
    cache=True,
    parallel=True,
    error_model="numpy",
)
def _beamformer(distGridToMic,waveNumber,csm,mic_idx,steerVec,gi,idx,ac,st,r_diag=False,Q=10):
    """
    Beamforming computation for a specific frequency using steering vectors.

    Parameters:
        - distGridToMic (ndarray): Array of distances between grid points and microphones.
        - waveNumber (float): Wavenumber based on the current frequency.
        - csm (ndarray): Cross-Spectral Matrix (CSM).
        - mic_idx (ndarray): Indices of active microphones.
        - steerVec (ndarray): Steering vector.
        - gi (int): Grid index.
        - idx (int): Frequency index.
        - ac (ndarray): Accumulated beamforming results.
        - st (Steering): Steering parameters.
        - r_diag (bool): Whether to remove diagonal components from the CSM.
        - Q (int): Quality factor for attenuation.

    Returns:
        None
    """
    nMics = distGridToMic.shape[-1]
    for cntMics in nb.prange(nMics):
        expArg = np.float32(waveNumber * (distGridToMic[gi, cntMics]))
        steerVec[cntMics] = np.cos(expArg) - 1j * np.sin(expArg)
    
    helpNormalize = 0.0
    for cntMics in nb.prange(nMics):
        expArg_att = distGridToMic[gi,cntMics]*np.exp(np.float32(waveNumber * distGridToMic[gi,cntMics]/2/Q))
        helpNormalize += 1.0 / (expArg_att * expArg_att)
        steerVec[cntMics] /= expArg_att
    
    csm_sub = csm[idx]
    csm_sub = csm_sub[mic_idx][:, mic_idx]

    scalarProd = 0.0
    for cntMics in nb.prange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in nb.prange(cntMics):
            leftVecMatrixProd += (csm_sub[cntMics2, cntMics] * steerVec[cntMics2].conjugate())
        scalarProd += (2 * (leftVecMatrixProd * steerVec[cntMics]).real)
    if not r_diag:
        for cntMics in nb.prange(nMics):
            scalarProd += (csm_sub[cntMics, cntMics]* steerVec[cntMics].conjugate()* steerVec[cntMics]).real
    
    normalizeFactor = nMics * helpNormalize
    ac[idx,gi] = scalarProd / normalizeFactor
        
def getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom):
    """
    Retrieve and combine OpenStreetMap tiles to create a high-resolution map image for the specified region.

    Parameters:
        - lat_deg (float): Latitude of the bottom-left corner of the region.
        - lon_deg (float): Longitude of the bottom-left corner of the region.
        - delta_lat (float): Height of the region in degrees of latitude.
        - delta_long (float): Width of the region in degrees of longitude.
        - zoom (int): OpenStreetMap zoom level (0 to 19).

    Returns:
        - cropped_image (PIL.Image): High-resolution image of the specified map region.
    """
    smurl = r"http://tile.openstreetmap.org/{0}/{1}/{2}.png"
    xmin, ymax =deg2num(lat_deg, lon_deg, zoom)
    xmax, ymin =deg2num(lat_deg + delta_lat, lon_deg + delta_long, zoom)
    
    # Calculate the size of the cropped image
    width = (xmax - xmin + 1) * 256
    height = (ymax - ymin + 1) * 256

    # Create a blank image with the calculated size
    Cluster = Image.new('RGB', (width, height))
    
    for xtile in range(xmin, xmax+1):
        for ytile in range(ymin,  ymax+1):
            imgurl=smurl.format(zoom, xtile, ytile)
#             print("Opening: " + imgurl)
            request=urllib.request.Request(imgurl,None,hdr)
            imgstr = urllib.request.urlopen(request).read()
            tile = Image.open(BytesIO(imgstr))
            Cluster.paste(tile, box=((xtile-xmin)*256 ,  (ytile-ymin)*255))

    left = int((lon_deg - num2deg(xmin, ymin, zoom)[1]) * width / delta_long)
    upper = int((num2deg(xmin, ymin, zoom)[0] - (lat_deg + delta_lat)) * height / delta_lat)
    right = int((lon_deg - num2deg(xmax+1, ymin, zoom)[1] + delta_long) * width / delta_long)
    lower = int((-num2deg(xmin, ymax+1, zoom)[0] + lat_deg) * height / delta_lat)
    cropped_image = Cluster.crop((left, upper, width+right, height-lower))

    return cropped_image

def deg2num(lat_deg, lon_deg, zoom):
    """
    Convert latitude/longitude coordinates to OpenStreetMap tile numbers at a specific zoom level.

    Parameters:
        - lat_deg (float): Latitude in degrees.
        - lon_deg (float): Longitude in degrees.
        - zoom (int): Zoom level (0 to 19).

    Returns:
        - (int, int): Tuple of x and y tile numbers.
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """
    Convert OpenStreetMap tile numbers to latitude/longitude coordinates at a specific zoom level.

    Parameters:
        - xtile (int): X tile number.
        - ytile (int): Y tile number.
        - zoom (int): Zoom level (0 to 19).

    Returns:
        - (float, float): Tuple of latitude and longitude in degrees.
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def load_das_data(das, start_time, duration, start_channel, end_channel, downsample_x, downsample_t, bp_args, channels):
    """
    Load and preprocess DAS data for a specified time and channel range.

    Parameters:
        - das (DAS): DAS object containing the data.
        - start_time (datetime): Starting time for data extraction.
        - duration (int): Duration of the data in minutes.
        - start_channel (int): Starting channel number for data extraction.
        - end_channel (int): Ending channel number for data extraction.
        - downsample_x (int): Downsampling factor in the spatial dimension.
        - downsample_t (int): Downsampling factor in the temporal dimension.
        - bp_args (dict): Bandpass filter arguments.
        - channels (list): Specific channels to include.

    Returns:
        - concatenated_data (dict): Dictionary with the preprocessed data and axes.
    """
    
    end_time = start_time + timedelta(minutes=duration)
    return das.load_das_data(downsample_x=downsample_x, downsample_t=downsample_t,
                             start_time=start_time, end_time=end_time,
                             start_channel=start_channel, end_channel=end_channel,
                             time_len=10, extend_channel=False, bandpass=bp_args, channels=channels)

def plot_time_series_w_delay(matrix, x_axis, t_axis, delays, channel_interval=1, save_fig=None,
                     start_channel=None, end_channel=None, start_time=None, end_time=None):
    """
    Plot time series DAS data with delays.

    Parameters:
        - matrix (ndarray): 2D array containing the time-series data (Time x Channels).
        - x_axis (ndarray): Array containing spatial channel indices.
        - t_axis (ndarray): Array containing temporal indices.
        - delays (ndarray): Array containing the delays per channel.
        - channel_interval (int): Interval between plotted channels.
        - save_fig (str or None): Path to save the figure if specified.
        - start_channel (int or None): Starting channel index for plotting.
        - end_channel (int or None): Ending channel index for plotting.
        - start_time (datetime or None): Starting time index for plotting.
        - end_time (datetime or None): Ending time index for plotting.

    Returns:
        None
    """
    
    if start_channel is not None and end_channel is not None:
        x_indices = np.where((x_axis >= start_channel) & (x_axis <= end_channel))[0]
        x_axis = x_axis[x_indices]
        matrix = matrix[:, x_indices]
        delays = delays[x_indices]
        delays = delays-np.min(delays)

    if start_time is not None and end_time is not None:
        start_index = np.searchsorted(t_axis, start_time)
        end_index = np.searchsorted(t_axis, end_time)
        t_axis = t_axis[start_index:end_index]
        matrix = matrix[start_index:end_index, :]

    num_channels = matrix.shape[1]

    for i in range(0, num_channels, channel_interval):
        data = matrix[:, i]
        _x = x_axis[i]
        normalized_data = 10*channel_interval*((data - np.min(data)) / (np.max(data) - np.min(data)) - 0.5)
        
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_das_data_w_delay(ax, t_axis,x_axis,matrix,lb_perc = 2, up_perc = 98, delays=delays,save_fig=save_fig)
    
def plot_das_data_w_delay(ax, t_axis, x_axis, data, 
                  lb_perc = 1, up_perc = 99, lower_bound=False, upper_bound=False,
                  start_channel=None, end_channel=None, start_time=None, end_time=None,
                  save_fig=False, delays=None):
    """
    Plot DAS data with optional delay lines.

    Parameters:
        - ax (matplotlib.axes.Axes): Axis object for plotting.
        - t_axis (ndarray): Array of time indices.
        - x_axis (ndarray): Array of spatial channel indices.
        - data (ndarray): 2D array containing the DAS data (Time x Channels).
        - lb_perc (int): Lower bound percentile for color mapping.
        - up_perc (int): Upper bound percentile for color mapping.
        - lower_bound (float or False): Custom lower color bound.
        - upper_bound (float or False): Custom upper color bound.
        - start_channel (int or None): Starting channel index for plotting.
        - end_channel (int or None): Ending channel index for plotting.
        - start_time (datetime or None): Starting time index for plotting.
        - end_time (datetime or None): Ending time index for plotting.
        - save_fig (str or False): Path to save the figure if specified.
        - delays (ndarray or None): Array of delay values to plot delay lines.

    Returns:
        None
    """
    if start_channel is not None and end_channel is not None:
        x_indices = np.where((x_axis >= start_channel) & (x_axis <= end_channel))[0]
        x_axis = x_axis[x_indices]
        data = data[:, x_indices]

    if start_time is not None and end_time is not None:
        start_index = np.searchsorted(t_axis, start_time)
        end_index = np.searchsorted(t_axis, end_time)
        t_axis = t_axis[start_index:end_index]
        data = data[start_index:end_index, :]

    y_lims = mdates.date2num(t_axis)
    if not lower_bound:
        lower_bound = np.percentile(data, lb_perc)
    if not upper_bound:
        upper_bound = np.percentile(data, up_perc)

    # Create the imshow plot
    plt.imshow(data[::-1,:], aspect='auto', vmin=lower_bound, vmax=upper_bound,
               extent=[min(x_axis)-11310, max(x_axis)-11310, y_lims[0], y_lims[-1]],
               cmap='seismic')

    # Convert numeric y-axis to datetime format
    date_format = mdates.DateFormatter("%H:%M:%S")
    ax.yaxis.set_major_formatter(date_format)
    
    if delays is not None:        
        for i in range(0,len(t_axis)-np.int64(np.max(delays)),200):
            plt.plot(x_axis-11310,t_axis[np.int64(delays)+i],'k--',linewidth=3)
    
    plt.ylim([t_axis[30*200],t_axis[36*200]])
    plt.xlabel('DAS Channel Number',fontsize=30)
    plt.ylabel('Time',fontsize=30)    
    plt.tick_params(axis='both', which='major', labelsize=28)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig)
        plt.close()
    else:
        plt.show()
    
def create_sensor_mask(cha_save_file, xml_file, increment, radius, buffer, bounds_gps, shrk_buffer):
    """
    Generate sensor masks and a GeoDataFrame containing sensor positions.

    Parameters:
        - cha_save_file (str): File name containing channel coordinates.
        - xml_file (str): XML file containing sensor positions.
        - increment (float): Grid increment value for `acoular.RectGrid`.
        - radius (float): Radius used for proximity calculations.
        - buffer (float): Buffer distance to expand the convex hull.
        - bounds_gps (tuple): Tuple containing the (left, bottom, right, top) GPS coordinates.
        - shrk_buffer (float): Shrink distance for a secondary mask.

    Returns:
        - mask (ndarray): Boolean 2D array marking active regions.
        - mask_shrk (ndarray): Boolean 2D array with a smaller active region.
        - gdf (GeoDataFrame): GeoDataFrame containing sensor positions.
    """
    
    if cha_save_file[-1] == 'L':
        cha_save_file = cha_save_file[:-2]
    bbox_width = np.load(f'data/array_pos/{cha_save_file}.npz')['bbox_width']
    bbox_height = np.load(f'data/array_pos/{cha_save_file}.npz')['bbox_height']
    
    rg = acoular.RectGrid( x_min=-bbox_width/2, x_max=bbox_width/2,
                          y_min=-bbox_height/2, y_max=bbox_height/2,
                          z=0, increment=increment )
    doc = xml.dom.minidom.parse(f'data/array_pos/{xml_file}.xml')
    names = []
    xyz = []
    for el in doc.getElementsByTagName('pos'):
        names.append(el.getAttribute('Name'))
        xyz.append(list(map(lambda a : float(el.getAttribute(a)), 'xyz')))
    xyz = np.array(xyz)   
    
    bottom_left = (bounds_gps[1], bounds_gps[0])
    top_right = (bounds_gps[3], bounds_gps[2])
    lat_scale = (top_right[0] - bottom_left[0]) / bbox_height
    lon_scale = (top_right[1] - bottom_left[1]) / bbox_width
    xyz_coords = [(bottom_left[0] + (x[1]+bbox_height/2) * lat_scale, 
                   bottom_left[1] + (x[0]+bbox_width/2) * lon_scale) for x in xyz]
    
    cvx_coords, shrk_coords = buffered_convex_hull(xyz[:,0], xyz[:,1], buffer, shrk_buffer, 10)
    cvx_coords = cvx_coords.flatten()
    shrk_coords = shrk_coords.flatten()
    sensor_coords = xyz[:,0:2]

    mask = np.ones((rg.nxsteps, rg.nysteps), dtype=bool)
    mask_hull_idx = rg.indices(*cvx_coords)
    for x,y in zip(mask_hull_idx[0],mask_hull_idx[1]):
        mask[x,y] = 0           
    
    mask_shrk = np.ones((rg.nxsteps, rg.nysteps), dtype=bool)
    mask_shrk_hull_idx = rg.indices(*shrk_coords)
    for x,y in zip(mask_shrk_hull_idx[0],mask_shrk_hull_idx[1]):
        mask_shrk[x,y] = 0
                
    gdf = gpd.GeoDataFrame({'geometry': [Point(lon, lat) for lat, lon in xyz_coords]}, 
                           crs="EPSG:4326")
    
    return mask, mask_shrk, gdf

def buffered_convex_hull(x_coords, y_coords, buffer_distance, shrink_distance, simplification_tolerance=0.0):
    """
    Calculate a simplified and buffered convex hull of a set of points.
    
    Args:
    - x_coords (list or np.ndarray): X coordinates of the points.
    - y_coords (list or np.ndarray): Y coordinates of the points.
    - buffer_distance (float): Buffer distance.
    - simplification_tolerance (float): Tolerance for simplification.
    
    Returns:
    - list of tuple: List of coordinates that form the simplified buffered convex hull.
    """
    # Create a MultiPoint object from the x and y coordinates
    points = MultiPoint(list(zip(x_coords, y_coords)))
    
    # Calculate the convex hull
    hull = points.convex_hull
    
    # Buffer the convex hull
    buffered_hull = hull.buffer(buffer_distance)    
    shrinked_hull = hull.buffer(-shrink_distance)

    # Simplify the buffered hull
    simplified_hull = buffered_hull.simplify(simplification_tolerance, preserve_topology=False)
    shrinked_hull = shrinked_hull.simplify(simplification_tolerance, preserve_topology=False)

    # Extract the coordinates of the simplified hull
    hull_coords = np.array(simplified_hull.exterior.coords)
    shrinked_hull_coords = np.array(shrinked_hull.exterior.coords)
    
    return hull_coords, shrinked_hull_coords

def create_geodataframe(maps, mask, bounds_gps, crs='EPSG:4326'):
    """
    Create a GeoDataFrame from a matrix and bounding box coordinates.

    Args:
    - maps (np.array): Multi-dimensional array from which the matrix is derived.
    - mask (np.array): Mask to apply to the matrix.
    - bounds_gps (tuple): Tuple of GPS coordinates in the order (left, bottom, right, top).
    - crs (str, optional): Coordinate Reference System of the GeoDataFrame. Default is 'EPSG:4326'.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame representing the matrix data.
    """
    # Process matrix
    matrix = np.ma.masked_where(mask, maps)
    matrix = matrix.T

    # Extract bottom left and top right coordinates
    bottom_left = (bounds_gps[1], bounds_gps[0])
    top_right = (bounds_gps[3], bounds_gps[2])

    # Generate latitude and longitude arrays
    rows, cols = matrix.shape
    latitudes = np.linspace(bottom_left[0], top_right[0], rows + 1)
    longitudes = np.linspace(bottom_left[1], top_right[1], cols + 1)

    # Create polygons for each grid cell
    polygons = []
    zoning = []
    color_code = []
    for i in range(len(latitudes) - 1):
        for j in range(len(longitudes) - 1):
            poly = Polygon([(longitudes[j], latitudes[i]),
                            (longitudes[j + 1], latitudes[i]),
                            (longitudes[j + 1], latitudes[i + 1]),
                            (longitudes[j], latitudes[i + 1])])
            polygons.append(poly)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'intensity': matrix.flatten(), 'geometry': polygons})
    gdf.set_crs(crs, inplace=True)

    return gdf

def plot_heatmap(gdf,vmin_,panel_params=None, object_params=None, boundary_params=None, 
                 figsize=(12,10), save_fig=None,show_ava=True):    
    """
    Plot a heatmap with overlays for sensor positions and other objects.

    Parameters:
        - gdf (GeoDataFrame): GeoDataFrame containing intensity values and polygons.
        - vmin_ (float): Minimum intensity value for the heatmap.
        - panel_params (tuple or None): Parameters for the inset map panel.
        - object_params (tuple or None): Coordinates and label of an object to overlay.
        - boundary_params (tuple or None): Coordinates and label of a boundary polygon.
        - figsize (tuple): Size of the figure.
        - save_fig (str or None): Path to save the figure.
        - show_ava (bool): Whether to show available DAS channels.

    Returns:
        None
    """
    cha_save = np.load(f'data/array_pos/cha_save.npz')['cha_save']
    alpha=0.7
    cmap = 'Reds'
    # Normalize the intensity column
    vmin = gdf['intensity'].min()
    vmax = gdf['intensity'].max()
    gdf['intensity'] = (gdf['intensity'] - vmin) / (vmax - vmin)
    fig, ax = plt.subplots(figsize=figsize)
    gdf.set_crs(epsg=4326, inplace=True)
    gdf.to_crs(epsg=3857).plot(ax=ax, column='intensity', legend=False, 
                               cmap=cmap, vmin=vmin_, vmax=1., alpha=alpha)
        
    if object_params is not None:
        line = LineString(object_params[0])    
        gdf_object = gpd.GeoDataFrame({'geometry': [line]})
        gdf_object.set_crs(epsg=4326, inplace=True)
        gdf_object.iloc[[0]].to_crs(epsg=3857).plot(ax=ax, color='Blue',alpha=0.7,
                                                    lw=10,linestyle='-',label=object_params[1])
        
    if boundary_params is not None:
        boundary_params[0].set_crs(epsg=4326, inplace=True)
        boundary_params[0].iloc[[0]].to_crs(epsg=3857).plot(ax=ax, aspect=1, edgecolor='blue', label=boundary_params[1],
                                                             facecolor='none', linestyle='-', lw=3)
        
    # Add an inset panel if provided
    if panel_params is not None:
        inset_ax = ax.inset_axes([0.575, 0.225, 0.4, 1])
        minx, miny, maxx, maxy = [panel_params[0][0], panel_params[0][1], 
                                  panel_params[0][2], panel_params[0][3]]
        bbox = Polygon([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])
        star_point = Point(panel_params[1][0], panel_params[1][1])
        gdf_small = gpd.GeoDataFrame({'geometry': [bbox, star_point]})
        gdf_small.set_crs(epsg=4326, inplace=True)
        gdf_small.iloc[[0]].to_crs(epsg=3857).plot(ax=inset_ax, aspect=1, edgecolor='k', facecolor='none')
        gdf_small.iloc[[1]].to_crs(epsg=3857).plot(ax=inset_ax, marker='*', color='red', markersize=20)   
        ctx.add_basemap(inset_ax, source=ctx.providers.OpenStreetMap.Mapnik)
        current_ylim = ax.get_ylim()        
        inset_ax.set_xticklabels([])
        inset_ax.set_yticklabels([])
        inset_ax.set_xlabel('')
        inset_ax.set_ylabel('')
        inset_ax.set_ylim(current_ylim[0]-2000,current_ylim[1]+10000)
        
    # Add color bar for intensity
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_, vmax=1))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, alpha=alpha)
    cbar.set_label('Normalized SPI',size=20)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(20)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.legend(fontsize=20)
    add_lat_lon_ticks(ax)    
    plt.xlabel('Longitude',fontsize=30)
    plt.ylabel('Latitude',fontsize=30)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=30,fontsize=20)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig,bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        
def add_lat_lon_ticks(ax):    
    """
    Add latitude and longitude ticks to a map plot.

    Parameters:
        - ax (matplotlib.Axes): The axes object to update.

    Returns:
        None
    """
    proj_transform = pyproj.Proj('epsg:3857')
    lonlat_transform = pyproj.Proj(init='epsg:4326')

    # Define the function for transforming x & y to lon & lat
    def x_to_lon(x):
        lon, lat = pyproj.transform(proj_transform, lonlat_transform, x, 0)
        return lon

    def y_to_lat(y):
        lon, lat = pyproj.transform(proj_transform, lonlat_transform, 0, y)
        return lat

    # Update the tick labels
    ax.set_xticklabels([round(x_to_lon(x), 2) for x in ax.get_xticks()])
    ax.set_yticklabels([round(y_to_lat(y), 2) for y in ax.get_yticks()])
