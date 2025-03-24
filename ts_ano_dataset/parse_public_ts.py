import numpy as np
import pandas as pd
import os
datasets = ['UCR', 'TODS']

data_dir = '/home/wzx/TSAD/datasets/UTS/'
save_dir = '/home/wzx/LLaMA-Factory/ts_ano_dataset/public_ts_data/data/'

for dataset in datasets:
    data_path = data_dir + dataset
    file_lists = os.listdir(data_path)
    for file in file_lists:
        file_path = data_path + '/' + file + '/'
        label = np.load(file_path + 'test_label.npy').astype(int)
        value = np.load(file_path + 'test.npy')
        df = pd.DataFrame()
        df['value'] = value
        df['label'] = label
        save_path = save_dir + dataset + '/' + file + '.csv'
        df.to_csv(save_path, index=False)

