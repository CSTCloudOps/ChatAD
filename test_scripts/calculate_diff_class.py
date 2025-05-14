import json
import numpy as np

def cal_diff_class(file = './eval/image_stft_sft.json'):
    with open(file) as f:
        all_results = json.load(f)

    with open('./eval/vlm-local-baseline/14B_results.json') as f:
        all_results_label = json.load(f)

    f1 = {'level_shift': [], 'frequency':[], 'trend':[], 'spike':[]}
    pre = {'level_shift': [], 'frequency':[], 'trend':[], 'spike':[]}
    recall = {'level_shift': [], 'frequency':[], 'trend':[], 'spike':[]}
    for sample, label in zip(all_results['samples'], all_results_label['samples']):
        if  'anomaly_type' not in sample:
            sample['anomaly_type'] = label['anomaly_type']
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



# cal_diff_class()
# cal_diff_class('./eval/vlm-local-baseline/38B_results.json')
# cal_diff_class('./eval/RL_normal.json')
# cal_diff_class('./eval/RL_normal_and_basic.json')
# cal_diff_class('./eval/RL_normal_and_shaplet.json')
cal_diff_class('./eval/vlm-local-baseline/Qwen72B_results.json')
