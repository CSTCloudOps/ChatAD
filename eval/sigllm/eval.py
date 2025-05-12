import sys

sys.path.append("../")
# from vllm import LLM, SamplingParams
import json
import os
import re
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

from sigllm import SigLLM
import pandas as pd
from typing import List, Tuple

def point_adjust_f1(
    pred_intervals: List[List[int]], gt_intervals: List[List[int]], ts_length: int
) -> Tuple[float, float, float, int, int, int]:
    """
    计算point-adjust F1分数

    Args:
        pred_intervals: 预测的异常区间列表 [[start1, end1], [start2, end2], ...]
        gt_intervals: 真实的异常区间列表 [[start1, end1], [start2, end2], ...]
        ts_length: 时间序列总长度

    Returns:
        f1_score: Point-Adjust F1分数
        precision: 精确率
        recall: 召回率
        tp: 真正例数量
        fp: 假正例数量
        fn: 假负例数量
    """
    # 创建预测和真实的点标签序列
    pred_labels = np.zeros(ts_length, dtype=bool)
    gt_labels = np.zeros(ts_length, dtype=bool)

    # 填充预测标签
    for start, end in pred_intervals:
        pred_labels[start : end + 1] = True

    # 填充真实标签
    gt_intervals_after_enlarge = []
    for start, end in gt_intervals:
        start = max(start - 5, 0)
        end = min(end + 5, ts_length - 1)
        gt_labels[start : end + 1] = True
        gt_intervals_after_enlarge.append([start, end])

    # 执行point adjustment
    adjusted_pred_labels = np.copy(pred_labels)
    for start, end in gt_intervals_after_enlarge:
        # 如果在真实异常区间内有任何一个点被预测为异常
        if np.any(pred_labels[start : end + 1]):
            # 则将整个区间都标记为预测正确
            adjusted_pred_labels[start : end + 1] = True
        # else:
        #     # 否则整个区间都标记为预测错误
        #     # adjusted_pred_labels[start:end+1] = False
        #     pass

    # 计算TP、FP、FN
    tp = np.sum(np.logical_and(adjusted_pred_labels, gt_labels))
    fp = np.sum(np.logical_and(adjusted_pred_labels, ~gt_labels))  # 真实为0 预测为1
    fn = np.sum(np.logical_and(~adjusted_pred_labels, gt_labels))  # 真实为1 预测为0

    # 计算precision和recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算F1
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return f1_score, precision, recall, int(tp), int(fp), int(fn)

def generate_eval_dataset(window, test_portion, dataset_dir, COT, moving_average, FEW_SHOT=True, FEW_SHOT_NUM=1):
    if not dataset_dir.endswith("/"):
        dataset_dir += "/"
    result = []
    file_list = os.listdir(dataset_dir)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for file in file_list:
        df = pd.read_csv(dataset_dir + file)
        if moving_average:
            df["value"] = df["value"].rolling(window=4).mean()
        df["value"] = df["value"].ffill()
        df["label"] = df["label"].ffill()

        # 获取真实标签
        labels = df["label"].to_numpy()[-int(len(df) * test_portion):]
        df = df[-int(len(df) * test_portion):]
        df = df[['timestamp', 'value']].reset_index(drop=True)
        df['timestamp'] = df['timestamp'].astype(int)
        timestamp = [1222840800 + 22400 * i for i in range(len(df))]
        df['timestamp'] = timestamp

        # df = df[:200]
        hyperparameters = {
                "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
                "time_column": "timestamp",
                "interval": 22400,
                "method": "mean"
            },
            # "sigllm.primitives.prompting.timeseries_preprocessing.rolling_window_sequences#1": {
            #     "window_size": 500,
            #     "step_size": 300
            # },
        }

        sigllm = SigLLM(
            pipeline='gpt_prompter',
            decimal=2,
            window_size=500,
            # interval=1,
            hyperparameters=hyperparameters
        )
        anomalies = sigllm.detect(df)
        # print(anomalies)
        # anomalies = pd.DataFrame({'start': [1226626400], 'end': [1228866400]})
        
        # 找到真实的异常区间
        true_anomaly_segments = []
        start = None
        for i in range(len(labels)):
            if labels[i] == 1:
                if start is None:
                    start = i
            elif start is not None:
                true_anomaly_segments.append([start, i-1])  # 改为列表而不是元组
                start = None
        if start is not None:
            true_anomaly_segments.append([start, len(labels)-1])
        
        # 获取预测的异常区间
        pred_anomaly_segments = []
        for i in range(len(anomalies)):
            anomaly = anomalies.iloc[i]
            start_time = anomaly['start']
            end_time = anomaly['end']
            start_idx = df[df['timestamp'] >= start_time].index[0]
            end_idx = df[df['timestamp'] <= end_time].index[-1]
            pred_anomaly_segments.append([start_idx, end_idx])
        
        # 调用point_adjust_f1函数计算指标
        f1, precision, recall, tp, fp, fn = point_adjust_f1(
            pred_intervals=pred_anomaly_segments,
            gt_intervals=true_anomaly_segments,
            ts_length=len(labels)
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        result.append({
            'file': file,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # 使用总体的tp、fp、fn计算最终的F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Overall Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
    result.append({
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1': f1,
        'overall_tp': total_tp,
        'overall_fp': total_fp,
        'overall_fn': total_fn
    })
    with open(f'./result_{dataset_name}.json', 'w') as f:
        json.dump(result, f)
    return result

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    window = int(sys.argv[2])
    model = sys.argv[3]
    moving_average = sys.argv[4]
    few_shot = sys.argv[5]
    few_shot_num = int(sys.argv[6])
    if moving_average == "t":
        moving_average = True
    else:
        moving_average = False
    if few_shot == "t":
        few_shot = True
    else:
        few_shot = False
    dataset_dir = "/home/wzx/ChatAD/eval/public_ts_data/data/" + dataset_name
    COT = False
    result_dir = f"/home/wzx/ChatAD/eval/eval_result_{dataset_name}_COT{str(COT)}.txt"
    if dataset_name == "TODS":
        test_portion = 1
    elif dataset_name == "Yahoo":
        test_portion = 0.5
    elif dataset_name == "WSD":
        test_portion = 0.1
    elif dataset_name == "UCR" or dataset_name == "AIOPS":
        test_portion = 0.2

    # eval public dataset by dataset
    qa_file = generate_eval_dataset(
        window, test_portion, dataset_dir, COT, moving_average, few_shot, few_shot_num
    )

# python anollm.py Yahoo 120 qwen2_5_vl t