import sys

sys.path.append("../")
# from vllm import LLM, SamplingParams
import json
import os
import re
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

FEW_SHOT_IMAGES = {'/home/wzx/ChatAD/eval/llm_baseline/ts_012995d7.png': "Final answer: \\boxed{[[130, 131]]}", '/home/wzx/ChatAD/eval/llm_baseline/ts_85bfe7a2.png': "Final answer: \\boxed{[[56, 62]]}", '/home/wzx/ChatAD/eval/llm_baseline/ts_9848a2c7.png':"Final answer: \\boxed{[[32, 38], [95, 100]]}"}

PROMPT = """Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
List one by one, in python list format. 
If there are no anomalies, answer with an empty list [].

Write your answer in \\boxed{}.

Output template:
\\boxed{[[start1, end1], [start2, end2], ...]}
"""
COT_PROMPT = (
    PROMPT.replace(
        "Output template:",
        "Your output should include step-by-step explanation and final answer in the following \\boxed{} format: ",
    )
    + "Let's think step by step. "
)


COT_ANSWER_TEMPLATE = """To detect anomalies in the provided time series data, we can look for sudden changes or outliers in the time series pattern.
Based on the general pattern, <|normal|>.
The following ranges of anomalies can be identified: \n```<|answer_json|>```
During those periods, <|abnormal|>.
"""

COT_NORMAL_ANSWER_TEMPLATE = """To detect anomalies in the provided time series data, we can look for sudden changes or outliers in the time series pattern.
Based on the general pattern, <|normal|>.
The anomalies are: \n```[]```
The values appear to follow a consistent pattern without sudden <|abnormal_summary|> that would indicate an anomaly.
"""


def dataset_descriptions():
    description = {
        "trend": {
            "normal": "the normal data follows a steady but slowly increasing trend from -1 to 1",
            "abnormal": "the data appears to either increase much faster or decrease, deviating from the normal trend",
            "abnormal_summary": "trend or speed changes",
        },
        "point": {
            "normal": "the normal data is a periodic sine wave between -1 and 1",
            "abnormal": "the data appears to become noisy and unpredictable, deviating from the normal periodic pattern",
            "abnormal_summary": "noises",
        },
        "freq": {
            "normal": "the normal data is a periodic sine wave between -1 and 1",
            "abnormal": "the data suddenly changes frequency, with very different periods between peaks",
            "abnormal_summary": "frequency changes",
        },
        "range": {
            "normal": "the normal data appears to be Gaussian noise with mean 0",
            "abnormal": "the data suddenly encounter spikes, with values much further from 0 than the normal noise",
            "abnormal_summary": "amplitude changes",
        },
        "flat-trend": {
            "normal": "the normal data follows a steady but slowly increasing trend from -1 to 1",
            "abnormal": "the data appears to either increase much faster, deviating from the normal trend",
            "abnormal_summary": "trend or speed changes",
        },
    }

    full_description = description.copy()
    for key, value in description.items():
        full_description["noisy-" + key] = value

    return full_description


def generate_eval_dataset(window, test_portion, dataset_dir, COT, moving_average, FEW_SHOT=True, FEW_SHOT_NUM=1):
    if not dataset_dir.endswith("/"):
        dataset_dir += "/"
    result = []
    file_list = os.listdir(dataset_dir)
    for file in file_list:
        df = pd.read_csv(dataset_dir + file)
        if moving_average:
            df["value_new"] = df["value"].rolling(window=4).mean()
        else:
            df["value_new"] = df["value"]
        df["value_new"] = df["value_new"].ffill()
        df["label"] = df["label"].ffill()

        # Get test data based on test_portion
        values = df["value_new"].to_numpy()[-int(len(df) * test_portion) :]
        labels = df["label"].to_numpy()[-int(len(df) * test_portion) :]

        os.makedirs("/home/wzx/ChatAD/eval/public_ts_data/images", exist_ok=True)

        if COT:
            question = f"{COT_PROMPT}"
        else:
            question = f"{PROMPT}"

        for i in range(len(values) // window):
            messages = []
            values_now = values[i * window : (i + 1) * window]
            labels_now = labels[i * window : (i + 1) * window]

            # Min-max normalization for better visualization
            values_now = (values_now - np.min(values_now)) / (
                np.max(values_now) - np.min(values_now)
            ) if np.max(values_now) - np.min(values_now) != 0 else values_now
            values_now = [round(num, 3) for num in values_now]
            plt.figure(figsize=(12, 6))
            plt.plot(values_now, label="Time Series", color="blue")
            plt.title("Time Series")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.savefig(f"/home/wzx/ChatAD/eval/public_ts_data/images/{file}_{i}.png")
            plt.close()

            if FEW_SHOT:
                for i in range(FEW_SHOT_NUM):
                    messages += [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": list(FEW_SHOT_IMAGES.keys())[i],
                                },
                                {"type": "text", "text": question},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": list(FEW_SHOT_IMAGES.values())[i]}]
                        }
                    ]
            else:
                pass

            # Find continuous anomaly segments
            anomaly_segments = []
            start = None
            for j in range(len(labels_now)):
                if labels_now[j] == 1 and start is None:
                    start = j
                elif (
                    labels_now[j] == 0 or j == len(labels_now) - 1
                ) and start is not None:
                    end = j - 1 if labels_now[j] == 0 else j
                    anomaly_segments.append([start, end])
                    start = None
            if len(anomaly_segments) == 0:
                answers = "Final Answer\\boxed{[]}"
            else:
                answers = f"Final Answer\\boxed{{{str(anomaly_segments)}}}"
            messages += [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f"/home/wzx/ChatAD/eval/public_ts_data/images/{file}_{i}.png",
                                },
                                {"type": "text", "text": question},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answers}],
                        },
                    ]
            result.append(
                {
                    "messages": messages
                }
            )
    # Save results
    data_name = dataset_dir.split("/")[-2]
    result_path = (
        f"/home/wzx/ChatAD/eval/llm_baseline/anollm_{data_name}.json"
    )
    json.dump(result, open(result_path, "wt"), ensure_ascii=False, indent=4)
    return result_path


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