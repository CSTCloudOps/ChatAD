import json
import numpy as np

def cal_diff_class(file = './eval/image_stft_sft.json'):
    with open(file) as f:
        all_results = json.load(f)

    f1 = {'level_shift': [], 'frequency':[], 'trend':[], 'spike':[]}
    pre = {'level_shift': [], 'frequency':[], 'trend':[], 'spike':[]}
    recall = {'level_shift': [], 'frequency':[], 'trend':[], 'spike':[]}
    for sample in all_results['samples']:
        for anomaly_type in sample['anomaly_type']:
            f1[anomaly_type].append(sample['f1'])
            pre[anomaly_type].append(sample['precision'])
            recall[anomaly_type].append(sample['recall'])

    f1_mean = {key:np.mean(f1[key]) for key in f1}
    pre_mean = {key:np.mean(pre[key]) for key in pre}
    recall_mean = {key:np.mean(recall[key]) for key in recall}

    print("f1: ", f1_mean)
    print("precision: ",pre_mean)
    print("recall: ",recall_mean)



cal_diff_class()
cal_diff_class('./eval/image_stft.json')