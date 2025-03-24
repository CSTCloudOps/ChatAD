# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    This script is used to generate training QA data with a set of prompts, which can be further used as seed QAs for TSEvol.
    We only show some simple templates here for demonstration. You can modify the code to generate your own QA dataset.
    Usage:
        python3 -m chatts.generate_template_qa
"""

import random
from tqdm import tqdm
import json
import os
from typing import *
from ts_generator import generate_controlled_attributes, generate_time_series, attribute_to_text
from encoding_utils import timeseries_encoding, timeseries_to_list
from attribute_utils import metric_to_controlled_attributes


# CONFIG
ENCODING_METHOD = 'z-score'
# ENCODING_METHOD = None
SEQ_LEN = None  # Set to none for random seq_len
TOTAL_CNT = 100000
ANOMALY = True
MODE = 'SFT' # DPO
OUTPUT_DATASET = f'../data/sft-zscore-new5.json'
OUTPUT_LABEL = f'./labels/sft-zscore-new5.json'

# All Config for TS Attributes (type & probability)
metric_config = json.load(open('config/metric_set.json', 'rt'))


def univariate_seed_qa():
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
        num = random.randint(1, int(current_seq_len*0.85))
        if all([abs(num - existing) > current_seq_len/8 for existing in position_start]):
            position_start.append(num)

    # Choose a metric and generate
    # attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric))
    # timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)
    attribute_set = metric_to_controlled_attributes(metric)
    attribute_set['change']['position_start'] = position_start
    attribute_pool = generate_controlled_attributes(attribute_set)
    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # print(attribute_pool)
    # return
    attribute_pool['metric name'] = metric
    attribute_pool['anomaly'] = []

    # parse trend anomaly
    if len(attribute_pool['trend']['trend_list']) == 1:
        if attribute_pool['trend']['trend_list'][0][0] == 'keep steady':
            pass
        elif attribute_pool['trend']['trend_list'][0][0] == 'increase' or attribute_pool['trend']['trend_list'][0][0] == 'decrease':
            attribute_pool['anomaly'].append(
                {
                    'index':(attribute_pool['trend']['trend_list'][0][1], attribute_pool['trend']['trend_list'][0][1] + 1),
                    'description':f"there might be data anomalies because the metric {metric} continuously {str(attribute_pool['trend']['trend_list'][0][0])} from index {str(attribute_pool['trend']['trend_list'][0][1])}. Whether the business is anomalous needs to be judged based on metric and business information",
                    'level':'middle'
                }
            )
            pass
    else:
        for i in range(1, len(attribute_pool['trend']['trend_list'])):
            attribute_pool['anomaly'].append(
                {
                    'index':(attribute_pool['trend']['trend_list'][i][1], attribute_pool['trend']['trend_list'][i][1] + 1),
                    'description':f"there are data anomalies because the trend of the metric {metric} changes at index {attribute_pool['trend']['trend_list'][i][1]}",
                    'level':'high'
                }
            )
    # parse local anomaly
    if len(attribute_pool['local']) !=0:
        for i in range(len(attribute_pool['local'])):
            attribute_pool['anomaly'].append(
                {
                    'index': (attribute_pool['local'][i]['position_start'], attribute_pool['local'][i]['position_end']),
                    'description':f"there are data anomalies because the metric {metric} has {attribute_pool['local'][i]['type']} from index {str(attribute_pool['local'][i]['position_start'])} to index {str(attribute_pool['local'][i]['position_end'])}",
                    'level':'high'
                }
            )
    
    # parse business anomaly

    # Scalar
    if ENCODING_METHOD != None:
        scaled_timeseries, cur_ts_prompt, feature = timeseries_encoding(timeseries, ENCODING_METHOD)
        scaled_timeseries = [round(num, 3) for num in scaled_timeseries]
        instruction = f"You are a time series analysis expert. There is a metric called {metric} collected from {category}, mean of it is {feature['mean']}, standard deviation of it is {feature['std']}. After z-score standardization, the time series is: {str(list(scaled_timeseries))}. To get the original value, you need to use the following formula:\n\n$origin_x = x*std + mean$\n\n"
    else:
        scaled_timeseries = [round(num, 3) for num in timeseries]
        instruction = f"You are a time series analysis expert. This is a metric called {metric} collected from {category}: {str(list(scaled_timeseries))}. "

    # Generate QA
    questions, answers, fields, reject_answers = [], [], [], []
    
    # step 1 length
    if random.random()< 0.1:
        questions.append("What is the length of the time series?")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['length'] ,cot=True))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['length'] ,cot=True, wrong_answer=True))
        cur_fields = {"length": "COT"}
        fields.append(cur_fields)

        questions.append("What is the length of the time series? Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['length'] ,cot=False))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['length'] ,cot=False, wrong_answer=True))
        cur_fields = {"length": "RESULT"}
        fields.append(cur_fields)

    # step 2 min max
    if random.random()< 0.1:
        questions.append("What is the maximum value of the time series?")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['max'] ,cot=True))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['max'] ,cot=True, wrong_answer=True))
        cur_fields = {"max": "COT"}
        fields.append(cur_fields)

        questions.append("What is the maximum value of the time series? Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['max'] ,cot=False))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['max'] ,cot=False, wrong_answer='True'))
        cur_fields = {"max": "RESULT"}
        fields.append(cur_fields)

        questions.append("What is the minimum value of the time series?")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['min'] ,cot=True))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['min'] ,cot=True, wrong_answer=True))
        cur_fields = {"min": "COT"}
        fields.append(cur_fields)

        questions.append("What is the minimum value of the time series? Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['min'] ,cot=False))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['min'] ,cot=False, wrong_answer=True))
        cur_fields = {"min": "RESULT"}
        fields.append(cur_fields)

    # step3 trend
    if random.random()< 0.1:
        questions.append("What is the trend of the time series?")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['trend'] ,cot=True))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['trend'] ,cot=True, wrong_answer=True))
        cur_fields = {"trend": "COT"}
        fields.append(cur_fields)

        questions.append("What is the trend of the time series? Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['trend'] ,cot=False))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['trend'] ,cot=False, wrong_answer=True))
        cur_fields = {"trend": "RESULT"}
        fields.append(cur_fields)

        # step3 period
        questions.append("What is the periodicity of the time series?")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['period'] ,cot=True))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['period'] ,cot=True, wrong_answer=True))
        cur_fields = {"period": "COT"}
        fields.append(cur_fields)

        questions.append("What is the periodicity of the time series? Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['period'] ,cot=False))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['period'] ,cot=False, wrong_answer=True))
        cur_fields = {"period": "RESULT"}
        fields.append(cur_fields)

    # step 4 local
    questions.append("Is there any local change in the time series?")
    answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['local'] ,cot=True))
    reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['local'] ,cot=True, wrong_answer=True))
    cur_fields = {"local": "COT"}
    fields.append(cur_fields)

    questions.append("Is there any local change in the time series? Just give me the result.")
    answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['local'] ,cot=False))
    reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['local'] ,cot=False, wrong_answer=True))
    cur_fields = {"local": "RESULT"}
    fields.append(cur_fields)

    # step 5 index
    if random.random() < 0.1:
        num = random.choice(list(scaled_timeseries))
        questions.append(f"What's the index of {str(num)} in the time series?")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['index'] ,cot=True, index_number=num))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['index'] ,cot=True, wrong_answer=True, index_number=num))
        cur_fields = {"index": "COT"}
        fields.append(cur_fields)

        questions.append(f"What's the index of {str(num)} in the time series? Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['index'] ,cot=False, index_number=num))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['index'] ,cot=False, wrong_answer=True, index_number=num))
        cur_fields = {"index": "RESULT"}
        fields.append(cur_fields)
    
    # print('original series', scaled_timeseries)
    # step 6 noise
        questions.append("Please analyze the noise intensity in the time series.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['noise'] ,cot=True))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['noise'] ,cot=True, wrong_answer=True))
        cur_fields = {"noise": "COT"}
        fields.append(cur_fields)

        questions.append("Please analyze the noise intensity in the time series. Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['noise'] ,cot=False))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['noise'] ,cot=False, wrong_answer=True))
        cur_fields = {"noise": "RESULT"}
        fields.append(cur_fields)

    # mean and std
    # questions.append("What's the mean and std of the time series?")
    # answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['mean'] ,cot=True))
    # reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['mean'] ,cot=True, wrong_answer=True))
    # cur_fields = {"mean": "COT"}
    # fields.append(cur_fields)

    # questions.append("What's the mean and std of the time series? Just give me the result.")
    # answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['mean'] ,cot=False))
    # reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['mean'] ,cot=False, wrong_answer=True))
    # cur_fields = {"mean": "RESULT"}
    # fields.append(cur_fields)

    # step 7 anomaly
    questions.append("Is there any data anomaly in the time series? If so, give me the anomalous index.")
    answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['anomaly'] ,cot=True))
    reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['anomaly'] ,cot=True, wrong_answer=True))
    cur_fields = {"anomaly": "COT"}
    fields.append(cur_fields)

    questions.append("Is there any data anomaly in the time series? Just give me the result.")
    answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['anomaly'] ,cot=False))
    reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['anomaly'] ,cot=False, wrong_answer=True))
    cur_fields = {"anomaly": "RESULT"}
    fields.append(cur_fields)

    # step 8 diff
    if random.random() < 0.1:
        questions.append("Calculate the difference sequence between two adjacent points.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['diff'] ,cot=True))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['diff'] ,cot=True, wrong_answer=True))
        cur_fields = {"diff": "COT"}
        fields.append(cur_fields)

        questions.append("Calculate the difference sequence between two adjacent points. Just give me the result.")
        answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['diff'] ,cot=False))
        reject_answers.append(attribute_to_text(scaled_timeseries, attribute_pool, generate_values=False, include_attributes=['diff'] ,cot=False, wrong_answer=True))
        cur_fields = {"diff": "RESULT"}
        fields.append(cur_fields)


    # print(questions)
    # for q,a in zip(questions, answers):
    #     # if 'local' in q:
    #         print('### Qestion\n')
    #         print(q)
    #         print('### Answer\n')
    #         print(a)
    # # (Step 1) general shape
    # questions.append("Now, please analyze the characteristics of this time series from the perspectives of periodicity, trend, local characteristics, frequency characteristics, and noise.")
    # answers.append(attribute_to_text(timeseries, attribute_pool, generate_values=False))
    # cur_fields = {"trend": [0], "seasonal": [0], "noise": [0], "local": [0]}
    # fields.append(cur_fields)

    # # (Step 2) shape and values
    # questions.append("Now, please analyze the characteristics of this time series from the perspectives of periodicity, trend, local characteristics, frequency characteristics, and noise. Also include the approximate mean values for every 16 points, as well as the maximum and minimum values of the time series (rounded to 2 decimal places).")
    # answers.append(attribute_to_text(timeseries, attribute_pool, generate_values=True))
    # cur_fields = {"trend": [0], "seasonal": [0], "noise": [0], "local": [0], "statistic": [0]}
    # fields.append(cur_fields)

    # # (Step 3) local change
    # for local_char in attribute_pool['local']:
    #     question_position = local_char['position_start'] + random.randint(-5, 5)
    #     questions.append(f"Is there a local characteristic fluctuation starting around point {question_position} in this time series?")
    #     answers.append(f"Yes, this time series " + local_char['detail'])
    #     cur_fields = {"local": [0]}
    #     fields.append(cur_fields)
    
    # # (Step 4) data anomaly detection
    # if len(attribute_pool['anomaly']) == 0:
    #     questions.append(f"Is there data anomaly in the time series?")
    #     answers.append(f"No. Based on trend and local characteristics of the time series, there are no data anomalies in the time series.")
    #     cur_fields = {"anomaly": ['data']}
    #     fields.append(cur_fields)
    # else:
    #     for anomaly in attribute_pool['anomaly']:
    #         question_position = anomaly['index'][0] + random.randint(-2, 2)
    #         questions.append(f"Is there any data anomaly around index {str(question_position)} in this time series?")
    #         answers.append("Yes. Based on trend and local characteristics of the time series, " + anomaly['description'] + '.')
    #         cur_fields = {"anomaly": ['data']}
    #         fields.append(cur_fields)
    #     questions.append(f"Is there data anomaly in the time series?")
    #     answers.append("Yes. Based on trend and local characteristics of the time series, " + ', '.join([anomaly['description'] for anomaly in attribute_pool['anomaly']]) + '.')
    #     cur_fields = {"anomaly": ['data']}
    #     fields.append(cur_fields)

    # (Step 5) business anomaly detection
    # questions.append(f"Is there business anomaly in the time series?")
    # local_description = attribute_to_text(timeseries, attribute_pool, generate_values=False, include_attributes=['trend', 'local'])
    # answers.append(local_description)
    # cur_fields = {"anomaly": ['business']}
    # fields.append(cur_fields)

    # Generate final result
    result = []
    for q, a, f,ra in zip(questions, answers, fields, reject_answers):
        result.append({
            'instruction': instruction,
            'question': q,
            'answer': a,
            'reject_answer': ra,
            'fields': f,
            'metrics': [metric],
            'attribute_pool': [attribute_pool],
            'timeseries': [scaled_timeseries],
            'original_timeseries': [timeseries],
            'corr_pool': []
        })

    return result


def multivariate_seed_qa():
    if SEQ_LEN is None:
        current_seq_len = random.randint(64, 192)
    else:
        current_seq_len = SEQ_LEN

    # Randomly choose the number of time series (variables)
    num_series = random.randint(2, 5)

    # Randomly select a category from metric_config
    sample = random.choice(metric_config)
    category = sample['category']

    # Get all metrics from the selected category
    metrics_in_category = []
    for s in metric_config:
        if s['category'] == category:
            metrics_in_category.extend(s['metrics'])

    # Ensure we have enough metrics to sample
    if len(metrics_in_category) < num_series:
        num_series = len(metrics_in_category)

    # Randomly select metrics for the time series
    metrics = random.sample(metrics_in_category, num_series)

    timeseries_list = []
    attribute_pool_list = []

    # Ensure some metrics shape are similar
    all_attribute_set = {}
    for metric in metrics:
        # print(metric)
        # all_attribute_set.append(metric_to_controlled_attributes(metric))
        attribute = metric_to_controlled_attributes(metric)
        attribute_str =  ' '.join(attribute['trend']['attributes']) +  ' '.join(attribute['change']['attributes'])
        if attribute_str not in all_attribute_set:
            all_attribute_set[attribute_str] = [metric]
        else:
            all_attribute_set[attribute_str].append(metric)

    # print(all_attribute_set)
    local_change_class = random.sample(["shake", "upward spike", "downward spike", "continuous upward spike", "continuous downward spike", "upward convex", "downward convex", "sudden increase",
      "sudden decrease", "rapid rise followed by slow decline", "slow rise followed by rapid decline",
      "rapid decline followed by slow rise", "slow decline followed by rapid rise", "decrease after upward spike",
      "increase after downward spike", "increase after upward spike", "decrease after downward spike",
      "wide upward spike", "wide downward spike"], 4)
    
    position_start = []
    while len(position_start) < 3:
        num = random.randint(1, int(current_seq_len*0.85))
        if all([abs(num - existing) > current_seq_len/8 for existing in position_start]):
            position_start.append(num)

    for metric in metrics:
        # Generate attribute_pool and time series for each metric
        attribute_set = metric_to_controlled_attributes(metric)
        attribute_set['change']['position_start'] = position_start
        # attribute_set['change']['position_start'] = [random.randint(0, current_seq_len) for i in range(4)]
        attribute_pool = generate_controlled_attributes(attribute_set)
        timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

        attribute_pool['metric name'] = metric
        attribute_pool['anomaly'] = []

        # print(attribute_pool)
        # parse trend anomaly
        if len(attribute_pool['trend']['trend_list']) == 1:
            if attribute_pool['trend']['trend_list'][0][0] == 'keep steady':
                pass
            elif attribute_pool['trend']['trend_list'][0][0] == 'increase' or attribute_pool['trend']['trend_list'][0][0] == 'decrease':
                attribute_pool['anomaly'].append(
                    {
                        'index':(attribute_pool['trend']['trend_list'][0][1], attribute_pool['trend']['trend_list'][0][1] + 1),
                        'description':f"there might be data anomalies because the metric {metric} continuously {str(attribute_pool['trend']['trend_list'][0][0])} from index {str(attribute_pool['trend']['trend_list'][0][1])}",
                        'level':'middle'
                    }
                )
                # pass
        else:
            for i in range(1, len(attribute_pool['trend']['trend_list'])):
                attribute_pool['anomaly'].append(
                    {
                        'index':(attribute_pool['trend']['trend_list'][i][1], attribute_pool['trend']['trend_list'][i][1] + 1),
                        'description':f"there are data anomalies because the trend of the metric {metric} changes at index {attribute_pool['trend']['trend_list'][i][1]}",
                        'level':'high'
                    }
                )
        # parse local anomaly
        if len(attribute_pool['local']) !=0:
            for i in range(len(attribute_pool['local'])):
                attribute_pool['anomaly'].append(
                    {
                        'index': (attribute_pool['local'][i]['position_start'], attribute_pool['local'][i]['position_end']),
                        'description':f"there are data anomalies because the metric {metric} has {attribute_pool['local'][i]['type']} from index {str(attribute_pool['local'][i]['position_start'])} to index {str(attribute_pool['local'][i]['position_end'])}",
                        'level':'high'
                    }
                )

        # Append to lists
        timeseries_list.append(timeseries)
        attribute_pool_list.append(attribute_pool)

    # Scale and encode the time series
    scaled_timeseries_list = []
    cur_ts_prompts = []

    for timeseries in timeseries_list:
        scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)
        scaled_timeseries_list.append(scaled_timeseries)
        cur_ts_prompts.append(cur_ts_prompt)

    # Generate instruction
    instruction = f"You are a time series analysis expert. There are {num_series} metrics collected from {category} with length of {current_seq_len}."

    # List the metrics and their data
    for i in range(num_series):
        instruction += f"\nMetric {i+1}: {metrics[i]}. Time series data: {cur_ts_prompts[i] + str(list(scaled_timeseries_list[i]))}"

    questions, answers, fields = [], [], []

    # (Task 1) Detailed analysis of a specific time series
    selected_index = random.randint(0, num_series - 1)
    questions.append(f"Please analyze the characteristics of time series {selected_index+1} ({metrics[selected_index]}), including its periodicity, trend, local characteristics, frequency characteristics, and noise.")

    desc_text = attribute_to_text(
        timeseries_list[selected_index],
        attribute_pool_list[selected_index],
        generate_values=False
    )
    answers.append(desc_text)
    cur_fields = {"trend": [selected_index], "seasonal": [selected_index], "noise": [selected_index], "local": [selected_index]}
    fields.append(cur_fields)

    # (Task 2) Describe a specific attribute for all time series
    available_attributes = ['trend', 'periodicity', 'frequency', 'noise', 'local']
    selected_attribute = random.choice(available_attributes)
    questions.append(f"Please describe the {selected_attribute} characteristics of all the time series.")            

    # Build the answer by extracting the selected attribute from each time series
    attribute_text = ""
    for i in range(num_series):
        desc_text = attribute_to_text(
            timeseries_list[i],
            attribute_pool_list[i],
            generate_values=False,
            include_attributes=[selected_attribute]
        )
        attribute_text += f"Time series {i+1} ({metrics[i]}):\n{desc_text}\n"
    attribute_fields_map = {
        'trend': 'trend',
        'periodicity': 'seasonal',
        'frequency': 'seasonal',
        'noise': 'noise',
        'local': 'local'
    }
    cur_fields = {
        attribute_fields_map[selected_attribute]: list(range(num_series))
    }
    fields.append(cur_fields)
    answers.append(attribute_text.strip())

    # (Task 3) Searching for time series with similar trend
    visited = set()
    corr_pool = []
    for i in range(num_series):
        if i in visited:
            continue
        visited.add(i)
        cur_result = {i}
        for j in range(i + 1, num_series):
            if j in visited:
                continue
            if attribute_pool_list[i]['trend']['type'] == attribute_pool_list[j]['trend']['type'] and attribute_pool_list[i]['trend']['type'] != 'multiple':
                # Similar trend found
                visited.add(j)
                cur_result.add(j)

        if len(cur_result) > 1:
            # Add a question
            question = f"Please find the time series with similar trend characteristics with {metrics[random.choice(list(cur_result))]}."
            questions.append(question)
            cur_answer = f"Time series with similar trend characteristics: {', '.join([metrics[i] for i in cur_result])}, because their trend are all {attribute_pool_list[i]['trend']['type']}."
            answers.append(cur_answer)
            cur_fields = {
                "trend": list(cur_result),
                "correlation": [len(corr_pool)]
            }
            corr_pool.append((sorted(cur_result), cur_answer))

    # (Task 4) Searching for time series with similar local fluctuation
    visited = set()
    for i in range(num_series):
        if i in visited:
            continue
        visited.add(i)
        cur_result = {i}
        for j in range(i + 1, num_series):
            if j in visited:
                continue
            if len(attribute_pool_list[i]['local']) != len(attribute_pool_list[j]['local']):
                continue
            for l1, l2 in zip(attribute_pool_list[i]['local'], attribute_pool_list[j]['local']):
                if abs(l1['position_start'] - l2['position_start']) > 15:
                    break
            else:
                # Similar local found
                visited.add(j)
                cur_result.add(j)

        if len(cur_result) > 1:
            # Add a question
            question = f"Please find the time series with similar local fluctuations characteristics with {metrics[random.choice(list(cur_result))]} in terms of their fluctuation positions."
            questions.append(question)
            cur_answer = f"Time series with similar local fluctuations characteristics: {', '.join([metrics[i] for i in cur_result])}, because they all have local fluctuations near points: {', '.join([str(l['position_start']) for l in attribute_pool_list[i]['local']])}."
            answers.append(cur_answer)
            cur_fields = {
                "local": list(cur_result),
                "correlation": [len(corr_pool)]
            }
            corr_pool.append((sorted(cur_result), cur_answer))
    
    # (Task 5) data anomaly detection
    # questions.append(f"Is there any data anomalies in these time series?")
    # all_anomalies = {attri['metric']:attri['anomaly'] for attri in attribute_pool_list}
    # all_anomalies_number = sum([len(all_anomalies[anomalies]) for anomalies in all_anomalies])

    selected_index = random.randint(0, num_series - 1)
    questions.append(f"Is there data anomaly in time series {selected_index+1} ({metrics[selected_index]})?")
    all_anomalies = attribute_pool_list[selected_index]['anomaly']

    if len(all_anomalies) == 0:
        answers.append(f"No. Based on the trend and local characteristics, there are no data anomaly in time series {selected_index+1} ({metrics[selected_index]}).")
        cur_fields = {"anomaly": ['data']}
        fields.append(cur_fields)
    else:
        cur_answer = "Yes. Based on the trend and local characteristics"
        for anomaly in all_anomalies:
            # question_position = anomaly['index'][0] + random.randint(-2, 2)
            # questions.append(f"Is there any data anomalies around index {str(question_position)} in this time series?")
            cur_answer += ( ', ' + anomaly['description'])
        answers.append(cur_answer + '.')
        cur_fields = {"anomaly": ['data']}
        fields.append(cur_fields)

    # business anomaly
    # questions.append('Is there business anomaly in these time series?')
    # answers.append("")
    # cur_fields = {"anomaly": ['business']}
    # fields.append(cur_fields)

    # Generate final result
    result = []
    for q, a, f in zip(questions, answers, fields):
        result.append({
            'instruction': instruction,
            'question': q,
            'answer': a,
            'fields': f,
            'metrics': metrics,
            'attribute_pool': attribute_pool_list,
            'timeseries': scaled_timeseries_list,
            'original_timeseries': timeseries_list,
            'corr_pool': corr_pool
        })

    return result

def generate_seed_qa_dataset():
    generated_cnt = 0
    ts_idx = 0
    labels = []
    datas = []
    # Create output directory
    os.makedirs("result", exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    with tqdm(total=TOTAL_CNT, desc='Generating seed qa') as t, open(OUTPUT_DATASET, 'wt') as f:
        while True:
            # Generate seed qa
            try:
                if random.random() > -1:
                    seed_qa = univariate_seed_qa()
                else:
                    seed_qa = multivariate_seed_qa()
            except Exception as err:
                print(err)
                # Error when generating random time series, just try again
                continue
            
            # Process seed qa
            for item in seed_qa:
                cur_label = {
                    'fields': item['fields'],
                    'metrics': item['metrics'],
                    'corr_pool': item['corr_pool'],
                    'attribute_pool': item['attribute_pool'],
                    'instruction': item['instruction'],
                    'question': item['question'],
                    'ts_idx': ts_idx
                }
                if MODE == 'SFT':
                    cur_data = {
                        'instruction': item['instruction'] + item['question'],
                        'input': '',
                        'output': item['answer'],
                        # 'timeseries': timeseries_to_list(item['timeseries'])
                    }
                else:
                    cur_data = {
                        'conversations':[
                            {
                                'from': 'human',
                                'value': item['instruction'] + item['question']
                            }
                        ],
                        'chosen':{
                            'from': 'gpt',
                            'value': item['answer']
                        },
                        'rejected':{
                            'from': 'gpt',
                            'value': item['reject_answer']
                        }

                    }
                # if ANOMALY:
                #     # print(item['fields'])
                #     if 'anomaly' not in item['fields']:
                #         continue
                # else:
                #     # print(item['fields'])
                #     if 'anomaly' in item['fields']:
                #         continue
                labels.append(cur_label)
                datas.append(cur_data)
                # f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
                generated_cnt += 1
                t.update()
            ts_idx += 1

            if generated_cnt >= TOTAL_CNT:
                break
    json.dump(labels, open(OUTPUT_LABEL, 'wt'), ensure_ascii=False, indent=4)
    json.dump(datas, open(OUTPUT_DATASET, 'wt'), ensure_ascii=False, indent=4)

    print(f"Finished! Total {generated_cnt} samples generated. Saved to {OUTPUT_DATASET} and {OUTPUT_LABEL}.")

if __name__ == '__main__':
    generate_seed_qa_dataset()
    # multivariate_seed_qa()
    # print(univariate_seed_qa())
    # univariate_seed_qa()
    # while True:
    #     univariate_seed_qa()