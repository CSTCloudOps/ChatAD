import sys
sys.path.append('../')
# from vllm import LLM, SamplingParams
import json
import os
import re
import numpy as np
from src.llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from src.llamafactory.model import load_config, load_tokenizer
from src.llamafactory.chat.vllm_engine import VllmEngine
from src.llamafactory.data import get_template_and_fix_tokenizer
from src.llamafactory.hparams import get_infer_args
import asyncio
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from matplotlib import pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


def generate_eval_dataset(window, test_portion, dataset_dir, COT, moving_average):
    if not dataset_dir.endswith('/'):
        dataset_dir += '/'
    result = []
    file_list = os.listdir(dataset_dir)
    for file in file_list:
        df = pd.read_csv(dataset_dir + file)
        if moving_average:
            df['value_new'] = df['value'].rolling(window=4).mean()
        else:
            df['value_new'] = df['value']
        df['value_new'] = df['value_new'].ffill()
        df['label'] = df['label'].ffill()
        
        # Get test data based on test_portion
        values = df['value_new'].to_numpy()[-int(len(df)*test_portion):]
        labels = df['label'].to_numpy()[-int(len(df)*test_portion):]
        
        os.makedirs('/home/wzx/ChatAD/eval/public_ts_data/images', exist_ok=True)
        
        question = f"<image>Given time series visualization, analyze the time series and detect anomalies.\n\n"
        background_info = (
            "Background Information:\n"
            "In time series analysis, we focus on four main types of anomalies:\n"
            "1. Spike Anomaly: Sudden significant increase or decrease in values\n"
            "2. Trend Anomaly: Sudden change in trend direction or slope\n"
            "3. Frequency Anomaly: Changes in periodicity or amplitude\n"
            "4. Level Shift: Sudden overall increase or decrease in baseline level\n\n"
        )
        for i in range(len(values) // window):
            values_now = values[i*window: (i+1)*window]
            labels_now = labels[i*window: (i+1)*window]
            
            # Min-max normalization for better visualization
            values_now = (values_now - np.min(values_now)) / (np.max(values_now) - np.min(values_now))
            values_now = [round(num, 3) for num in values_now]
            plt.figure(figsize=(12, 6))
            plt.plot(values_now, label='Time Series', color='blue')
            plt.title('Time Series')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.savefig(f'/home/wzx/ChatAD/eval/public_ts_data/images/{file}_{i}.png')
            plt.close()
            # Find continuous anomaly segments
            anomaly_segments = []
            start = None
            for j in range(len(labels_now)):
                if labels_now[j] == 1 and start is None:
                    start = j
                elif (labels_now[j] == 0 or j == len(labels_now)-1) and start is not None:
                    end = j-1 if labels_now[j] == 0 else j
                    anomaly_segments.append([start, end])
                    start = None
            if COT:
                cot_prompt = (
                    "Let's think step by step.\n"
                )
                problem = (
                        f"{question}\n"
                        f"{background_info}\n"
                        "Requirements:\n"
                        "1. Write analysis process within <think> </think> tags. Try to analyze directly, don't use python code.\n"
                        "2. Write anomalous intervals (final answer) using python list format in \\boxed{}, for example: \\boxed{[[start1, end1], [start2, end2], ...]}\n"
                        "3. If no anomalies detected, just output \\boxed{[]}\n"
                        "4. Do not overlap anomalous intervals\n\n"
                        f"{cot_prompt}\n"
                    )
            else:
                cot_prompt = "Just output the final answer.\n"
                problem = (
                        f"{question}\n"
                        f"{background_info}\n"
                        "Requirements:\n"
                        "1. Write anomalous intervals (final answer) using python list format in \\boxed{}, for example: \\boxed{[[start1, end1], [start2, end2], ...]}\n"
                        "2. If no anomalies detected, just output \\boxed{[]}\n"
                        "3. Do not overlap anomalous intervals\n\n"
                        f"{cot_prompt}\n"
                    )
                
            if len(anomaly_segments) == 0:
                answers = "Final Answer\\boxed{[]}"
            else:
                answers = f"Final Answer\\boxed{{{str(anomaly_segments)}}}"

            result.append({
                "messages": [
                    {
                        "role": "user",
                        "content": problem
                    },
                    {
                        "role": "assistant",
                        "content": answers
                    }
                ],
                "images": [f'/home/wzx/ChatAD/eval/public_ts_data/images/{file}_{i}.png']
            })


    # Save results
    data_name = dataset_dir.split('/')[-2]
    os.makedirs('/home/wzx/ChatAD/eval/public_ts_data/qa_files', exist_ok=True)
    result_path = f'/home/wzx/ChatAD/eval/public_ts_data/qa_files/{data_name}.json'
    json.dump(result, open(result_path, 'wt'), ensure_ascii=False, indent=4)

    return result_path

def parse_num(s):
    numbers = re.findall(r'-?\d+\.?\d*', s[:-1])
    return numbers

def parse_int_list(s):
    matches = re.findall(r'\[\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\]', s[:-1])
    lists = [list(map(int, match.split(','))) for match in matches]
    if len(lists)>0:
        return lists[0]
    else:
        return []

def parse_float_list(s):
    # 匹配浮点数的正则表达式
    matches = re.findall(r'\[\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)*)\s*\]', s[:-1])
    lists = [list(map(float, match.split(','))) for match in matches]
    return lists[0]

def parse_noboundary_list(s):
    # matches = re.findall(r'\b(?:-?\d+(?:,-?\d+)*)\b', s[:-1])
    # matches = re.findall(r'\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*', s[:-1])
    # lists = [list(map(int, match.split(','))) for match in matches]
    match = re.search(r'index.*?is\s+(\d+(?:,\d+)*)', s[:-1])
    if match:
        int_list = [int(num) for num in match.group(1).split(',')]
    else:
        int_list = []
    return int_list

def parse_prompt(flag, qa_file='./ts_ano_dataset/result/template_qa_10000_minmax_SFT_eval.json', qa_label_file='./ts_ano_dataset/labels/template_qa_10000_minmax_SFT_eval.json'):
    with open(qa_file, 'r') as f:
        qa = json.load(f)
    with open(qa_label_file, 'r') as f:
        qa_label = json.load(f)
    answer_format = {}
    answer_format['length'] = "The following is an example answer format (refer to the format to answer the question):\nThe length of the time series is xxx."
    answer_format['max'] = "The following is an example answer format (refer to the format to answer the question):\nThe maximum value of the time series is 90.054."
    answer_format['min']= "The following is an example answer format (refer to the format to answer the question):\nThe minimum value of the time series is 90.054."
    answer_format['index'] ="The following is an example answer format (refer to the format to answer the question):\nThe index of 79.539 is 109,110."
    answer_format['trend'] = "The following some example answer format (refer to the format to answer the question):\nFrom the perspective of the slope, the overall trend is steady. Local phase changes were observed, including: sudden decrease, decrease after upward spike.\nFrom the perspective of the slope, the overall trend contains multiple different segments: From point 0 to point 68, there is a decreasing trend with some variation in slope. From point 68 to point 98, there is an increasing trend. From point 98 to point 129, there is a decreasing trend."
    answer_format['local'] = "The following some example answer format (refer to the format to answer the question):\nThere is The differences between 38th and 39th point are much smaller than others, which indicatesa sudden decrease with an amplitude of 9.98 occurred between point 38 and point 39, with the time series value falling from around 86.73 to around 76.78."
    answer_format['anomaly'] = "The following some example answer format (refer to the format to answer the question):\nIn order to identify whether there are data anomalies in the time series, I will to calculate the difference between two adjacent points, identify the local pattern of the time series and finally determine the anomaly.\nAfter finishing these steps, result are as follows:\n\nThere are data anomalies in the time series.\n1. Index: 38~39. Local Change: Sudden decrease between 38th and 39th point.\n2. Index: 90~98. Local Change: Decrease after upward spike between 90th and 98th points.\nAll anomaly index are: [38, 39, 90, 91, 92, 93, 94, 95, 96, 97, 98]."
    answer_format['noise'] = "The following some example answer format (refer to the format to answer the question):\nThe overall noise standard deviation is around 0.00, very small compared the overall change of the curve. The curve is overall smooth with almost no noise."

    prompts = []
    labels = []
    files = []
    for prompt, label in zip(qa, qa_label):
        if 'diff' in label['fields']:
            continue
        if flag == 'naive_qwen':
            q_type = [i for i in ['length', 'max', 'min', 'index', 'trend', 'local', 'anomaly', 'noise'] if i in label['fields']][0]
            prompts.append(prompt['instruction'] + '\n\n' + answer_format[q_type])
            labels.append(prompt['output'])
            files.append(prompt['file'])

        else:
            prompts.append(prompt['instruction'])
            labels.append(prompt['output'])
            files.append(prompt['file'])

            
    return prompts, labels, files

async def generate(model='', flag='naive_qwen', qa_file='', qa_label_file='', dataset_name=''):
    args = {'model_name_or_path': model, 'finetuning_type': 'full', 'quantization_bit': None, 'quantization_method': 'bitsandbytes', 'template': 'qwen', 'flash_attn': 'fa2', 'use_unsloth': False, 'rope_scaling': None, 'infer_backend': 'vllm', 'infer_dtype': 'auto', 'trust_remote_code': True}
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
    vllm = VllmEngine(model_args, data_args, finetuning_args, generating_args)

    prompts, labels, files = parse_prompt(flag, qa_file, qa_label_file)

    # prompts = prompts[:10]

    async def chat_with_prompt(prompt):
        messages = [{'role': 'user', 'content': prompt}]
        return await vllm.chat(messages)
    results = []
    for i in tqdm(range(0, len(prompts), 10)):
        batch = prompts[i:i + 10]  # 切分批次
        tasks = [chat_with_prompt(prompt) for prompt in batch]
        batch_results = await asyncio.gather(*tasks)  # 并发处理一个批次
        # print('bababa',batch_results)
        batch_results = [item[0].response_text for item in batch_results]
        results.extend(batch_results)

    qa_naive_qwen = []
    for prompt, result, label, file in zip(prompts, results, labels, files):
        qa_naive_qwen.append({'q':prompt, 'a':result, 'gt':label, 'file': file})

    with open(f'/home/wzx/LLaMA-Factory/eval_result/{flag}.json', 'w') as f:
        json.dump(qa_naive_qwen, f)

    precision, recall, f1 = eval(f'/home/wzx/LLaMA-Factory/eval_result/{flag}.json', dataset_name)
    return precision, recall, f1


def point_adjust(predict, label, margin):
    label = add_margin(label, margin)
    anomaly_state = False
    for i in range(len(predict)):
        if label[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                else:
                    predict[j] = 1
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = 1
    return predict, label

def point_adjust_new(predict, label):
    anomaly_state = False
    for i in range(len(predict)):
        if label[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                else:
                    predict[j] = 1
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = 1
    predict = refine_prediction(label, predict)
    # label_indices = []
    # i = 0
    # while i < len(label):
    #     if label[i] == 1:
    #         start = i
    #         while i < len(label) and label[i] == 1:
    #             i += 1
    #         end = i  # i now points to the first non-1 after the abnormal segment
    #         label_indices.append((start, end))
    #     else:
    #         i += 1
    
    # for i in range(1, len(predict)):
    #     if label[i] and predict[i] and not label[i-1]:
    #         for j in range(i-1, min(0, i-6), -1):
    #             if pre
    return predict, label

def add_margin(arr, k):
    # 确保数组长度大于2
    n = len(arr)
    if n < 2:
        return arr
    result = np.array(arr)  # 创建一个副本避免修改原数组

    label_indices = []
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            start = i
            while i < len(arr) and arr[i] == 1:
                i += 1
            end = i  # i now points to the first non-1 after the abnormal segment
            label_indices.append((start, end))
        else:
            i += 1

    for start, end in label_indices:
        result[max(0, start - k): min(end + k, len(arr))] = 1
    
    result = list(result)
    return result

def refine_prediction(label, predict):
    n = len(label)
    refined_predict = predict[:]

    def find_intervals(arr):
        intervals = []
        start = None
        for i in range(len(arr)):
            if arr[i] == 1 and start is None:
                start = i
            elif arr[i] == 0 and start is not None:
                intervals.append((start, i - 1))
                start = None
        if start is not None:  # 处理最后一个区间
            intervals.append((start, len(arr) - 1))
        return intervals
    
    label_intervals = find_intervals(label)
    predict_intervals = find_intervals(predict)
    
    # 对每个 label 区间进行匹配调整
    for a, b in label_intervals:
        for c, d in predict_intervals:
            # 如果 predict 区间完全包含 label 区间
            if c <= a and d >= b:
                # 将 predict 区间缩小为 label 区间
                for i in range(c, d + 1):
                    refined_predict[i] = 0  # 先清零
                for i in range(a, b + 1):
                    refined_predict[i] = 1  # 再赋值为 1

    return refined_predict
        

def eval(path, dataset_name):
    if dataset_name == 'WSD':
        margin = 3
    elif dataset_name == 'UCR':
        margin = 50
    else:
        margin = 5

    with open(path, 'r') as f:
        all_qas = json.load(f)
    
    all_label_length = []
    all_label_length2 = []
    all_label_min = []
    all_label_max = []
    all_label_index = []
    all_label_index2 = []

    anomaly_mode = 'SYN'
    anomaly_predict_series = {}
    anomaly_gt_series = {}
    for qa in tqdm(all_qas, desc="Eval QAs"):
        q = qa['q']
        a = qa['a']
        gt = qa['gt']
        file = qa['file']
        if "what's the length" in q :
            a = int(parse_num(a)[0])
            gt = int(parse_num(gt)[0])
            all_label_length.append(a == gt)
            all_label_length2.append(1 - abs(a - gt)/max(a, gt))
        if 'minimum' in q :
            a = float(parse_num(a)[0])
            gt = float(parse_num(gt)[0])
            all_label_min.append(a == gt)
        if 'maximum' in q :
            a = float(parse_num(a)[0])
            gt = float(parse_num(gt)[0])
            all_label_max.append(a == gt)
        if 'anomaly' in q:
            original_ts = parse_float_list(q)
            a = parse_int_list(a)
            gt = parse_int_list(gt)
            print(a, gt)
            gt_series = [0 for i in range(len(original_ts))]
            predict_series = [0 for i in range(len(original_ts))]
            for i in a:
                if i >= len(original_ts):
                    continue
                predict_series[i] = 1
            for i in gt:
                if i >= len(original_ts):
                    continue
                gt_series[i] = 1
            if file in anomaly_predict_series:
                anomaly_gt_series[file].extend(gt_series)
                anomaly_predict_series[file].extend(predict_series)
            else:
                anomaly_gt_series[file] = gt_series
                anomaly_predict_series[file] = predict_series
        if 'the index of ' in q:
            a = parse_noboundary_list(a)
            gt = parse_noboundary_list(gt)
            # print(a, gt)
            all_label_index.append(set(a) == set(gt))
    scores = {}
    for file in anomaly_gt_series:
        anomaly_predict_series[file], anomaly_gt_series[file] = point_adjust(anomaly_predict_series[file], anomaly_gt_series[file], margin)
        # anomaly_predict_series, anomaly_gt_series = point_adjust(anomaly_predict_series, anomaly_gt_series, margin)
        precision = precision_score(anomaly_gt_series[file], anomaly_predict_series[file], average='binary')
        recall = recall_score(anomaly_gt_series[file], anomaly_predict_series[file], average='binary')
        f1 = f1_score(anomaly_gt_series[file], anomaly_predict_series[file], average='binary')
        print(f"file:{file} Precision:{precision:2f} Recall:{recall:2f} F1:{f1:2f}")
        scores[file] = {'precision':precision, 'recall':recall, 'f1':f1}
    
    avg_precision = sum([scores[item]['precision'] for item in scores])/len([scores[item]['precision'] for item in scores])
    avg_recall = sum([scores[item]['recall'] for item in scores])/len([scores[item]['recall'] for item in scores])
    avg_f1 = sum([scores[item]['f1'] for item in scores])/len([scores[item]['f1'] for item in scores])
    return avg_precision, avg_recall, avg_f1
    # print('Length Acc:', np.sum(all_label_length)/len(all_label_length),'Length Acc 2:', np.sum(all_label_length2)/len(all_label_length), 'Min Acc:', np.sum(all_label_min)/len(all_label_min), 'Max Acc:', np.sum(all_label_max)/len(all_label_max), 'Index Acc:', np.sum(all_label_index)/len(all_label_index))


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    window = int(sys.argv[2])
    model = sys.argv[3]
    moving_average = sys.argv[4]
    if moving_average == 't':
        moving_average = True
    else:
        moving_average = False
    dataset_dir = '/home/wzx/ChatAD/eval/public_ts_data/data/' + dataset_name
    COT = False
    result_dir = f'/home/wzx/ChatAD/eval/eval_result_{dataset_name}_COT{str(COT)}.txt'
    if dataset_name == 'TODS':
        test_portion = 1
    elif dataset_name == 'Yahoo':
        test_portion = 0.5
    elif dataset_name == 'WSD':
        test_portion = 0.1
    elif dataset_name == 'UCR' or dataset_name == 'AIOPS':
        test_portion = 0.2

    # eval public dataset by dataset
    qa_file = generate_eval_dataset(window, test_portion, dataset_dir, COT, moving_average)


    # avg_precision, avg_recall, avg_f1 = asyncio.run(generate(model, f'ts_eval_public_{dataset_name}_COT{str(COT)}_WS{str(window)}', qa_file, qa_label_file, dataset_name))
    # print(f"AVG Precision {avg_precision} AVG Recall: {avg_recall} AVG F1: {avg_f1}")
    # with open(result_dir, 'a') as f:
    #     f.write('\n' +str(window) + '\n')
    #     f.write(f"AVG Precision {avg_precision} AVG Recall: {avg_recall} AVG F1: {avg_f1}")
