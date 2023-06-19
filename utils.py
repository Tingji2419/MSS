import numpy as np
import pandas as pd
import random
import os
import torch
import torch.backends.cudnn as cudnn
import numpy.fft as nf
import copy
def load_data(data_path="/data/huangtj/human_activity_segmentation_challenge/datasets/has2023.csv.zip", debug=False):
    """
    Load the given CSV file containing the sensor data for the challenge.
    Returns a pandas DataFrame where each column is a sensor measurement and
    each row corresponds to a single time series of sensor data.

    Parameters
    ----------
    data_path : str, default: "../datasets/has_challenge_no_labels.csv.zip".
        Path to the csv file to be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sensor data for the challenge.

    Examples
    --------
    >>> data = load_data()
    >>> data.head()
    """
    np_cols = ["x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    if debug:
        return pd.read_csv(data_path, converters=converters, compression="zip", nrows=10)
    else:
        return pd.read_csv(data_path, converters=converters, compression="zip")


def to_submission(df, change_points):
    """
    Convert the change points predicted by an algorithm into the format required for
    submission.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the 250 time series data to be segmented.
    change_points : dict
        A list containing the change points as numpy arrays for each time series
        of the in the DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns: 'ts_id' and 'segment'. The 'Id' column should contain
        the row indices of the original DataFrame, and the 'Offsets' column contain
        CPs and segment lengths as a string in the format
        '<change point> <segment length>'.
    """
    prediction = []

    for ID, row in df.iterrows():
        ts_len = row["x-acc"].shape[0]   # length of the time series
        segments = np.concatenate(([1], np.sort(change_points[ID]) + 1, [ts_len]))

        for idx in range(segments.shape[0] - 1):
            cp = segments[idx]
            seg_len = segments[idx + 1] - segments[idx]

            prediction.append((ID, f"{int(cp)} {int(seg_len)}"))

    return pd.DataFrame.from_records(prediction, columns=["ts_id", "segment"])


def set_seed(seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def merge_point(main_axis, sub_axises, main_channel, gap=50, cut=True, type='main'):
    '''
    main_axis: 2D-array = [ [1, 5, 20],
                            [3, 8, 11, 120],
                            ...]

    sub_axises: dict = {'x-acc': [[1, 5, 10], [3, 8, 11, 110]], 
                        'y-acc': [[2, 5, 10], [5, 8, 11, 120]], 
                        ...}
    gap: merge condition

    Return: 2D-array=[[...]], the merged points of main axis and sub axises
    '''
    def merge(main_points, sub_points):
        merged_points = [i for i in main_points]
        for add_point in sub_points:
            if len(merged_points) >= 15 and cut:    # each containing most 15 actions 
                break
            if len(merged_points) == 0 or min(abs(np.array(merged_points) - add_point)) > gap:
                merged_points.append(add_point)
        return merged_points


    change_points = []
    sub_channels = sub_axises.keys()
    if type == 'main':
        for cur_idx, cur_points in enumerate(main_axis):
            sub_points = [sub_axises[cur_sub_channel][cur_idx] for cur_sub_channel in sub_channels if cur_sub_channel != main_channel]
            merged_points = cur_points
            if len(cur_points) == 0:
                for cur_sub_points in sub_points:
                    merged_points = merge(merged_points, cur_sub_points)
            change_points.append(np.sort(merged_points))
    # elif type == 'vote':
    #     for cur_idx, cur_points in enumerate(main_axis):
    #         if len(cur_points) > 0:
    #             continue
    #         sub_points = [sub_axises[cur_sub_channel][cur_idx] for cur_sub_channel in sub_channels if cur_sub_channel != main_channel]
    #         merged_points = cur_points
    #         for cur_sub_points in sub_points:
    #             merged_points = merge(merged_points, cur_sub_points)
    #         change_points.append(np.sort( merged_points))
    return change_points






# def fft_tran(T,sr):
#     complex_ary = nf.fft(sr)
#     y_ = nf.ifft(complex_ary).real
#     fft_freq = nf.fftfreq(y_.size, T[1] - T[0])
#     fft_pow = np.abs(complex_ary)  # 复数的摸-Y轴
#     return fft_freq, fft_pow

from pykalman import KalmanFilter
def Kalman_1D(observations , damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state



def preprocess_data(data, method='normal', channel='All'):
    df = data

    if channel == 'All':
        channel_choices = ["x-acc", "y-acc", "z-acc", 'x-mag', 'y-mag', 'z-mag', "x-gyro", "y-gyro", "z-gyro", 'lat', 'lon', 'speed']
    else:
        channel_choices = [channel] if type(channel) == str else channel

    if method == 'normal':
        for idx, row in df.iterrows():
            for channel in channel_choices:
                if len(row[channel]) != 0:
                    _range = np.max(row[channel]) - np.min(row[channel])
                    _range = _range if _range != 0 else 1
                    df.loc[idx][channel][:] = (row[channel] - np.min(row[channel])) / _range
                    # for cur_index, i in enumerate((row[channel] - np.min(row[channel])) / _range):
                    #     df.loc[idx][channel][cur_index] = i
           
    elif method == 'standard':
        for idx, row in df.iterrows():
            for channel in channel_choices:
                if len(row[channel]) != 0:
                    mu = np.mean(row[channel], axis=0)
                    sigma = np.std(row[channel], axis=0)
                    sigma = sigma if sigma != 0 else 1
                    df.loc[idx][channel][:] = (row[channel] - mu) / sigma
                    # for cur_index, i in enumerate((row[channel] - mu) / sigma):
                    #     df.loc[idx][channel][cur_index] = i
    
    elif method == 'diff':
        for idx, row in df.iterrows():
            for channel in channel_choices:
                if len(row[channel]) != 0:
                    
                    df.loc[idx][channel][1:] = df.loc[idx][channel][1:] - df.loc[idx][channel][:-1]
                    df.loc[idx][channel][0] = 0
    elif method == 'fft':
        for idx, row in df.iterrows():
            for channel in channel_choices:
                if len(row[channel]) != 0:
                    df.loc[idx][channel][:] = np.abs(nf.fft(df.loc[idx][channel]))
        
    elif method == 'kalman':
        for idx, row in df.iterrows():
            for channel in channel_choices:
                if len(row[channel]) != 0:
                    df.loc[idx][channel][:] = Kalman_1D(df.loc[idx][channel], damping=1)[:, 0]

    elif method == 'absnormal':
        for idx, row in df.iterrows():
            for channel in channel_choices:
                if len(row[channel]) != 0:
                    df.loc[idx][channel][:] = abs(df.loc[idx][channel][:])
                    _range = np.max(row[channel]) - np.min(row[channel])
                    _range = _range if _range != 0 else 1
                    df.loc[idx][channel][:] = (row[channel] - np.min(row[channel])) / _range
        
    elif method == None:
        pass
    else:
        raise NotImplementedError
    
    return df

def test():
    with open('/data/huangtj/human_activity_segmentation_challenge/datasets/label.csv', 'r') as f:
        points_labels = f.readline()
    points_labels = [np.array(i.strip().split(',')) for i in points_labels]
    

    pass