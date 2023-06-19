import numpy as np
import csv

def calculate_iou(segment1, segment2):
    start_intersection = max(segment1[0], segment2[0])
    end_intersection = min(segment1[0]+segment1[1], segment2[0]+segment2[1])
    intersection = max(0, end_intersection - start_intersection)
    
    union = max(segment1[0]+segment1[1], segment2[0]+segment2[1]) - min(segment1[0], segment2[0])
    iou = intersection / union

    return iou

def calculate_f1_score(tp, fp, fn):
    if tp == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def evaluate_segmentation(predictions, ground_truth):
    thresholds = np.arange(0.5, 1.0, 0.05)
    num_thresholds = len(thresholds)
    num_ts = len(predictions)
    
    normed_scores = np.zeros((num_ts, num_thresholds))
    
    for i, ts_id in enumerate(predictions.keys()):
        pred_segments = predictions[ts_id]
        gt_segments = ground_truth[ts_id]
        
        num_pred_segments = len(pred_segments)
        num_gt_segments = len(gt_segments)
        
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        
        confusion_matrix  = np.zeros((num_pred_segments,num_gt_segments))
        
        for j, pred_segment in enumerate(pred_segments):
            for k, gt_segment in enumerate(gt_segments):
                iou = calculate_iou(pred_segment, gt_segment)
                confusion_matrix[j][k] = iou
        #print(confusion_matrix)
        max_matrix = np.max(confusion_matrix,axis = 1)

        for  j in range(len(pred_segments)):
            for l, threshold in enumerate(thresholds):
                if max_matrix[j] >= threshold:                    
                    tp[l] += 1
                else:                    
                    fp[l] += 1
        
        fn = num_gt_segments - tp

        # print(tp,fp,fn)

        for l in range(num_thresholds):
            
            normed_scores[i, l] = calculate_f1_score(tp[l], fp[l], fn[l])
            

    avg_normed_scores = np.mean(normed_scores, axis=0)
    avg_normed_scores = np.mean(avg_normed_scores)
    return avg_normed_scores

def parse_segment(segment_str):
    start, end = segment_str.split()
    return int(start), int(end)

def parse_csv_file(filename):
    ground_truth = {}
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        
        for row in reader:
            ts_id = int(row[0])
            segment = parse_segment(row[1])
            
            if ts_id not in ground_truth:
                ground_truth[ts_id] = []
            
            ground_truth[ts_id].append(segment)
    
    return ground_truth


# for file_path in ['submission.csv','sub423.csv','sub480.csv','sub485.csv','sub492.csv']:
#     print(file_path)

#     ground_truth = parse_csv_file('label.csv')

#     predictions = parse_csv_file(file_path)

#     avg_scores = evaluate_segmentation(predictions, ground_truth)

#     print(avg_scores)