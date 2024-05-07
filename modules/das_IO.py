import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os
from datetime import datetime, timedelta
from scipy.signal import detrend, medfilt2d
from modules.utils import *

class DAS:
    """
    A class to handle distributed acoustic sensing (DAS) data.

    Attributes:
        directory_path (str): The path of the directory where .pk1 data files are stored.
        file_list (list): A list of .pk1 files found in the directory.

    Methods:
        get_pk1_files(): Retrieves the list of .pk1 files in the directory.
        load_das_data(downsample_x=1, downsample_t=1, ...): Loads and processes DAS data from the files.
    """
    
    def __init__(self, directory_path):
        """
        Initialize the DAS object and get the list of .pk1 files.

        Args:
            directory_path (str): The path to the directory containing the .pk1 data files.
        """
        self.directory_path = directory_path
        self.file_list = self.get_pk1_files()

    def get_pk1_files(self):
        """
        Retrieve all .pk1 files from the specified directory.

        Returns:
            file_list (list): A list of .pk1 filenames in the specified directory.
        """
        file_list = [filename for filename in os.listdir(self.directory_path) if filename.endswith(".pk1")]
        return file_list

    def load_das_data(self, downsample_x=1, downsample_t=1, start_time=None, end_time=None, 
                      start_channel=None, end_channel=None, bandpass=False, time_len=30, 
                      extend_channel=True, channels=None):
        """
        Load and process DAS data based on specified parameters.

        Args:
            downsample_x (int): Factor to downsample the spatial dimension (channels). Default is 1 (no downsampling).
            downsample_t (int): Factor to downsample the temporal dimension (time). Default is 1 (no downsampling).
            start_time (datetime): The start time for filtering data. Default is None.
            end_time (datetime): The end time for filtering data. Default is None.
            start_channel (int): The starting channel index to include in the data. Default is None.
            end_channel (int): The ending channel index to include in the data. Default is None.
            bandpass (bool or dict): Whether to apply a bandpass filter. If dict, provide filter parameters.
            time_len (int): The duration in minutes to consider when filtering files by time. Default is 30.
            extend_channel (bool): Whether to extend the list of channels sequentially. Default is True.
            channels (list): Specific list of channels to include. If provided, overrides start/end channels. Default is None.

        Returns:
            concatenated_data (dict): A dictionary containing the loaded and processed data:
                - 't_axis': An array of time points (after downsampling).
                - 'x_axis': An array of spatial channels (after downsampling).
                - 'data': A 2D array containing the signal data (Time x Channel).
                - 'fs': The sampling frequency after downsampling.
        """
        
        concatenated_data = {'t_axis': [], 'x_axis': [], 'data': [], 'fs': None}

        # Filter the sorted_file_list based on the specified channel range
        start_channel = start_channel if start_channel is not None else 0
        end_channel = end_channel if end_channel is not None else float('inf')
        channel_set = set(range(start_channel,end_channel))
        
        filtered_file_list = [filename for filename in self.file_list if
                              extract_start_channel(filename) is not None and
                              extract_end_channel(filename) is not None and
                              channel_set.intersection(set(range(extract_start_channel(filename),
                                                                 extract_end_channel(filename))))]
        
        current_date = datetime.now().date()
        time_set = set(range(int(datetime.combine(current_date, start_time.time()).timestamp()),
                             int(datetime.combine(current_date, end_time.time()).timestamp())))
        
        
        filtered_file_list = [filename for filename in filtered_file_list if
                              extract_start_time(filename) is not None and
                              time_set.intersection(set(range(int(datetime.combine(current_date, 
                                                    extract_start_time(filename).time()).timestamp()),
                                                    int(datetime.combine(current_date, 
                                                    (extract_start_time(filename)
                                                     + timedelta(minutes=time_len)).time()).timestamp()))))]
        print(filtered_file_list)

        # Sort the file list based on the number of channels
        sorted_file_list = sorted(filtered_file_list, key=extract_fourth_part)

        for filename in sorted_file_list:
            if filename.endswith(".pk1"):
                file_path = os.path.join(self.directory_path, filename)
                with open(file_path, "rb") as file:
                    data_dict = pickle.load(file)

                    if concatenated_data['fs'] is None:
                        concatenated_data['fs'] = data_dict['fs']
                        self.fs = concatenated_data['fs'] # fs before downsampling

                    if start_time is not None and end_time is not None:
                        # Calculate the indices for the selected time range
                        start_index = np.searchsorted(data_dict['t_axis'][0], start_time)
                        end_index = np.searchsorted(data_dict['t_axis'][0], end_time)
                        if extend_channel:
                            concatenated_data['t_axis'] = data_dict['t_axis'][0][start_index:end_index][::downsample_t]
                        else:
                            concatenated_data['t_axis'].extend(data_dict['t_axis'][0][start_index:end_index][::downsample_t])

                    # Find the indices for the selected channel range
                    if start_channel is not None and end_channel is not None:
                        x_indices = np.where((data_dict['x_axis'] >= start_channel) & (data_dict['x_axis'] <= end_channel))[0]
                        if channels is not None:
                            x_indices = channels

                        # Concatenate x_axis for the selected channel range
                        if extend_channel:
                            concatenated_data['x_axis'].extend(data_dict['x_axis'][x_indices][::downsample_x])
                        else:
                            concatenated_data['x_axis'] = data_dict['x_axis'][x_indices][::downsample_x]

                        # Downsample and concatenate the first dimension of data for the selected time and channel range
                        if start_time is not None and end_time is not None:
                            data_to_concatenate = data_dict['data'][x_indices, start_index:end_index]
                            data_to_concatenate = data_to_concatenate[::downsample_x,:]

                            if bandpass:
                                data_to_concatenate = bandpass_filter(data_to_concatenate, self.fs,
                                                                      bandpass['lp'], bandpass['hp'], bandpass['order'],1)
                            
                            if extend_channel:
                                concatenated_data['data'].extend(data_to_concatenate[:, ::downsample_t])
                            else:
                                if len(concatenated_data['data'])==0:
                                    concatenated_data['data']=data_to_concatenate[:, ::downsample_t]
                                else:
                                    concatenated_data['data']=np.hstack((concatenated_data['data'],
                                                                        data_to_concatenate[:, ::downsample_t]))

        # Reshape the data to match the time and spatial dimensions
        concatenated_data['t_axis'] = np.array(concatenated_data['t_axis'])
        concatenated_data['x_axis'] = np.array(concatenated_data['x_axis'])
        concatenated_data['data'] = np.array(concatenated_data['data']).transpose()  # Time X Channel
        concatenated_data['fs'] = self.fs/downsample_t
        self.fs = self.fs/downsample_t

        return concatenated_data
        