import random
from tqdm import tqdm
import json
import os
from typing import *
from ts_generator import generate_controlled_attributes, generate_time_series, attribute_to_text
from encoding_utils import timeseries_encoding, timeseries_to_list
from attribute_utils import metric_to_controlled_attributes
import numpy as np


# CONFIG
ENCODING_METHOD = 'z-score'
SEQ_LEN = 1680  # Set to none for random seq_len
TOTAL_CNT = 10000
ANOMALY = True
MODE = 'SFT' # DPO
OUTPUT_DATASET = f'/home/wzx/LLaMA-Factory/ts_ano_dataset/result/sft-zscore-syn.json'
OUTPUT_LABEL = f'/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-syn.json'

# All Config for TS Attributes (type & probability)
metric_config = json.load(open('/home/wzx/LLaMA-Factory/ts_ano_dataset/config/metric_set.json', 'rt'))


def univariate_seed_qa(COT: bool=False):
    if SEQ_LEN is None:
        current_seq_len = random.randint(64, 192)
    else:
        current_seq_len = SEQ_LEN

    # Randomly choose a type and metric name
    sample = random.choice(list(metric_config))
    category = sample['category']
    metric = random.choice(sample['metrics'])

    position_start = []
    while len(position_start) < 3:
        num = random.randint(int(current_seq_len*0.5), int(current_seq_len*0.95))
        if all([abs(num - existing) > current_seq_len/20 for existing in position_start]):
            position_start.append(num)

    attribute_set = metric_to_controlled_attributes(metric)
    attribute_set['change']['position_start'] = position_start
    attribute_pool = generate_controlled_attributes(attribute_set)
    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # print(timeseries)
    # print(attribute_pool['local'])
    ano_index = set()
    for local in attribute_pool['local']:
        for index in local['index']:
            for i in range(max(0, index - 4), min(index + 4, len(timeseries))):
                ano_index.add(i)
    ano_index = list(ano_index)

    label = [0 for i in range(len(timeseries))]
    for i in ano_index:
        label[i] = 1
    # print(timeseries, label)
    return timeseries, label        


def generate_dataset(num):
    dataset_dir = '/home/wzx/LLaMA-Factory/ts_ano_dataset/public_ts_data/data/SYN/'
    for i in tqdm(range(num)):
        while True:
            try:
                value, label = univariate_seed_qa()
                break
            except Exception as e:
                print(e)

        try:
            train_value = value[:int(0.5* len(value))] 
            test_value = value[int(0.5* len(value)):] 
            train_label = label[:int(0.5* len(value))] 
            test_label = label[int(0.5* len(value)):]
            if not os.path.exists(dataset_dir + f'{str(i)}'):
                os.mkdir(dataset_dir + f'{str(i)}')
            np.save(dataset_dir + f'{str(i)}/' + 'train.npy' , train_value)
            np.save(dataset_dir + f'{str(i)}/' + 'test.npy' , test_value)
            np.save(dataset_dir + f'{str(i)}/' + 'train_label.npy' , train_label)
            np.save(dataset_dir + f'{str(i)}/' + 'test_label.npy' , test_label)
            info = {'intervals': 1, 'training set anomaly ratio': sum(train_label)/len(train_label), 'testing set anomaly ratio':  sum(test_label)/len(test_label), 'total anomaly ratio': sum(label)/len(label)}
            with open(dataset_dir + f'{str(i)}/info.json', 'w') as f:
                json.dump(info, f)
        except Exception as e:
            print(e)
            continue
generate_dataset(20000)
    
