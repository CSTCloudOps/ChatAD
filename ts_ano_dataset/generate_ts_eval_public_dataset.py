import numpy as np
import os
import pandas as pd
from encoding_utils import timeseries_encoding, timeseries_to_list
import json

ENCODING_METHOD = 'z-score'
# ENCODING_METHOD = None
MOVING_AVERAGE = False

def generate_eval_dataset(window: int, test_size: float, dataset_dir: str, COT: bool=False, MOVING_AVERAGE: bool=False):
    if not dataset_dir.endswith('/'):
        dataset_dir += '/'
    result = []
    questions, answers, fields = [], [], []
    instruction = f"You are a time series analysis expert. "
    file_list = os.listdir(dataset_dir)
    for file in file_list:
        df = pd.read_csv(dataset_dir + file)
        if MOVING_AVERAGE:
            df['value_new'] = df['value'].rolling(window=4).mean()
        else:
            df['value_new'] = df['value']
        df['value_new'] = df['value_new'].ffill()
        df['label'] = df['label'].ffill()
        size = len(df)
        df = df[-int(test_size*size):]
        values = df['value_new'].to_numpy().astype(float)
        labels = df['label'].to_numpy()
        if np.sum(labels) == 0:
            continue
        for i in range(len(values) // window):
            values_now = values[i*window: (i+1)*window]
            labels_now = labels[i*window: (i+1)*window]
            fields.append(file)
            if ENCODING_METHOD != None:
                scaled_timeseries, cur_ts_prompt, feature = timeseries_encoding(values_now, ENCODING_METHOD)
                scaled_timeseries = [round(num, 3) for num in scaled_timeseries]
                if COT:
                    questions.append(f"There is a metric, mean of it is {feature['mean']}, standard deviation of it is {feature['std']}. After z-score standardization, the time series is: {str(list(scaled_timeseries))}. To get the original value, you need to use the following formula:\n\n$origin_x = x*std + mean$\n\nIs there any data anomaly?")
                else:
                    questions.append(f"There is a metric, mean of it is {feature['mean']}, standard deviation of it is {feature['std']}. After z-score standardization, the time series is: {str(list(scaled_timeseries))}. To get the original value, you need to use the following formula:\n\n$origin_x = x*std + mean$\n\nIs there any data anomaly? Just give me the result.")
            else:
                scaled_timeseries = [round(num, 3) for num in values_now]
                if COT:
                    questions.append(f"There is a metric: {str(list(scaled_timeseries))}. Is there any data anomaly?")
                else:
                    questions.append(f"There is a metric: {str(list(scaled_timeseries))}. Is there any data anomaly? Just give me the result.")

            if sum(labels_now) == 0:
                answers.append("There are no data anomalies.")
            else:
                answers.append(f"Anomaly Index is {str(np.where(labels_now == 1)[0].tolist())}.")

    label = []
    for q, a, f in zip(questions, answers, fields):
        result.append({
            'instruction': instruction + q,
            'input': '',
            'output': a,
            'file': f
        })
        label.append({'fields': {'anomaly': 'RESULT'}})
        

    # print(result)
    data_name = dataset_dir.split('/')[-2]
    json.dump(result, open(f'/home/wzx/LLaMA-Factory/ts_ano_dataset/result/{data_name}.json', 'wt'), ensure_ascii=False, indent=4)
    json.dump(label, open(f'/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/{data_name}.json', 'wt'), ensure_ascii=False, indent=4)

    return f'/home/wzx/LLaMA-Factory/ts_ano_dataset/result/{data_name}.json', f'/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/{data_name}.json'


def generate_eval_dataset_by_file(window: int, test_size: float, dataset_dir: str, COT: bool=False):
    if not dataset_dir.endswith('/'):
        dataset_dir += '/'
    file_list = os.listdir(dataset_dir)
    qa_files = []
    qa_label_files = []
    for file in file_list:
        result = []
        questions, answers, fields = [], [], []
        instruction = f"You are a time series analysis expert. "
        df = pd.read_csv(dataset_dir + file)
        if MOVING_AVERAGE:
            df['value_new'] = df['value'].rolling(window=5).mean()
        else:
            df['value_new'] = df['value']
        df['value_new'] = df['value_new'].ffill()
        size = len(df)
        df = df[-int(test_size*size):]
        values = df['value_new'].to_numpy().astype(float)
        labels = df['label'].to_numpy()
        # skip none anomaly time series
        if np.sum(labels) == 0:
            continue
        for i in range(len(values) // window):
            values_now = values[i*window: (i+1)*window]
            labels_now = labels[i*window: (i+1)*window]
            if ENCODING_METHOD != None:
                scaled_timeseries, cur_ts_prompt, feature = timeseries_encoding(values_now, ENCODING_METHOD)
                scaled_timeseries = [round(num, 3) for num in scaled_timeseries]
                if COT:
                    questions.append(f"There is a metric, mean of it is {feature['mean']}, standard deviation of it is {feature['std']}. After z-score standardization, the time series is: {str(list(scaled_timeseries))}. To get the original value, you need to use the following formula:\n\n$origin_x = x*std + mean$\n\nIs there any data anomaly?")
                else:
                    questions.append(f"There is a metric, mean of it is {feature['mean']}, standard deviation of it is {feature['std']}. After z-score standardization, the time series is: {str(list(scaled_timeseries))}. To get the original value, you need to use the following formula:\n\n$origin_x = x*std + mean$\n\nIs there any data anomaly? Just give me the result.")
            else:
                scaled_timeseries = [round(num, 3) for num in values_now]
                if COT:
                    questions.append(f"There is a metric: {str(list(scaled_timeseries))}. Is there any data anomaly?")
                else:
                    questions.append(f"There is a metric: {str(list(scaled_timeseries))}. Is there any data anomaly? Just give me the result.")

            if sum(labels_now) == 0:
                answers.append("There are no data anomalies.")
            else:
                answers.append(f"Anomaly Index is {str(np.where(labels_now == 1)[0].tolist())}.")

        label = []
        for q, a in zip(questions, answers):
            result.append({
                'instruction': instruction + q,
                'input': '',
                'output': a,
            })
            label.append({'fields': {'anomaly': 'RESULT'}})
        

        # print(result)
        data_name = dataset_dir.split('/')[-2]
        json.dump(result, open(f'/home/wzx/LLaMA-Factory/ts_ano_dataset/result/{data_name}/{file}.json', 'wt'), ensure_ascii=False, indent=4)
        json.dump(label, open(f'/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/{data_name}/{file}.json', 'wt'), ensure_ascii=False, indent=4)
        qa_files.append(f'/home/wzx/LLaMA-Factory/ts_ano_dataset/result/{data_name}/{file}.json')
        qa_label_files.append(f'/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/{data_name}/{file}.json')
    return qa_files, qa_label_files


if __name__ == '__main__':
    generate_eval_dataset(120, 0.5, './public_ts_data/data/Yahoo')
    generate_eval_dataset(196, 0.5, './public_ts_data/data/AIOPS')
