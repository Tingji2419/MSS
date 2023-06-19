import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import pandas as pd
from claspy.segmentation import BinaryClaSPSegmentation
# from aeon.annotation.plotting.utils import plot_time_series_with_change_points
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse


from utils import load_data, to_submission, set_seed, merge_point, preprocess_data

from measure import parse_csv_file, evaluate_segmentation

set_seed(42)


def main(args):
    cur_window_size = ["suss"] # window_size:['suss', 'fft', 'acf']
    cur_score = ["roc_auc"] # score:["roc_auc", "f1"]

    preprocess_method = args.preprocess
    main_channel = args.maxis
    channel_choices = [main_channel]
    if len(args.saxis) > 0:
        channel_choices.extend(args.saxis)

    # channel_choices = ["x-acc", "y-acc", "z-acc", 'x-mag', 'y-mag', 'z-mag', "x-gyro", "y-gyro", "z-gyro", 'lat', 'lon', 'speed']
    subset = 250
    gap = args.gap
    cut = True

    df = load_data()
    df = preprocess_data(df, method=preprocess_method, channel=channel_choices)

    main_points = []
    sub_points = {i:[] for i in channel_choices}

    if args.n_segments == 'probe':
        from probe import pre_train
        segments_classfier = pre_train(num=100000, windows_size=args.probe_ws)

    for channel in channel_choices:
        for idx, row in df.iloc[:subset,:].iterrows():
            ts = row[channel]
            if len(ts) == 0:
                sub_points[channel].append(np.array([]))   # only sub_channels will be len()==0
                continue

            ws = args.ws
            if args.ws == 'auto':
                from autoperiod import autoperiod
                ws = autoperiod(ts)
                ws = ws if ws != -1 else args.ws
            
            if args.n_segments == 'probe':
                from probe import probe
                n_segments = probe(segments_classfier, ts, ws=args.probe_ws)
                # print('Idx:', idx)
                # print('Probe n_segments:', n_segments)
                n_segments = n_segments if n_segments != 0 else 'learn'
            else:
                n_segments = args.n_segments

            clasp = BinaryClaSPSegmentation(n_estimators=args.estimators, window_size=ws, n_segments=n_segments, validation=args.validation, threshold=args.threhold)
            # clasp = BinaryClaSPSegmentation(n_estimators=args.estimators)

            try:
                found_cps = clasp.fit_predict(ts)
            except Exception as e:
                print(e)
                found_cps = np.array([])

            if channel == main_channel:
                main_points.append(found_cps)
            else:
                sub_points[channel].append(found_cps)


    change_points = merge_point(main_points, sub_points, main_channel=main_channel, gap=gap, cut=cut)


    submission = to_submission(df.iloc[:subset,:], change_points)
    submission.head()
    
    file_path = f"./result.csv"
    submission.to_csv(file_path, index=False)

    # ground_truth = parse_csv_file('/data/huangtj/human_activity_segmentation_challenge/datasets/label.csv')
    # predictions = parse_csv_file(file_path)
    # avg_scores = evaluate_segmentation(predictions, ground_truth)
    # print(avg_scores)
    # with open('./result.csv', 'a') as f:
    #     f.write(f'{args.time_str}, {avg_scores}, {args.maxis}, ' + ' '.join(args.saxis) + f', {args.gap}, {args.estimators}, {args.preprocess}, {args.ws}\n')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--maxis', default='y-acc')
    parse.add_argument('--saxis', nargs='*', type=str, default=['z-mag'])

    parse.add_argument('--gap', type=int, default=120)
    parse.add_argument('--estimators', type=int, default=10)
    parse.add_argument('--preprocess', type=str, default=None)
    parse.add_argument('--ws', default='suss')
    parse.add_argument('--probe_ws', type=int, default=200)
    parse.add_argument('--n_segments', default='learn')

    parse.add_argument('--validation', type=str, default='significance_test')
    parse.add_argument('--threhold', type=float, default=1e-15)
    # parse.add_argument('--cut', action='store_false', default=True)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--time_str', type=str, default='')
    args = parse.parse_args()
    if args.time_str == '':
        from datetime import datetime
        args.time_str = datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    print(args)
    main(args)

