import numpy as np
import json
from typing import List, Dict, Any, Union, Tuple
import sys
import re
from openai import OpenAI, AzureOpenAI
import base64
import os


def extract_boxed_content(text: str) -> Union[str, None]:
    """从文本中提取最后一个 \boxed{} 中的内容"""
    try:
        pattern = r"\\boxed\{(.*?)\}"
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            # 获取最后一个匹配项
            last_match = matches[-1]
            return last_match.group(1)
        else:
            return None
    except:
        return None


def extract_anomaly_intervals(text: str) -> Union[List[List[int]], None]:
    """从 \boxed{} 标签中提取异常区间"""
    content = extract_boxed_content(text)
    if not content:
        return None
    try:
        intervals = eval(content)
        if not isinstance(intervals, list):
            return None

        processed_intervals = []
        for x in intervals:
            if not isinstance(x, list):
                return None

            # 处理单点情况
            if len(x) == 1:
                if isinstance(x[0], (int, float)):
                    processed_intervals.append([int(x[0]), int(x[0])])
            # 处理区间情况
            elif len(x) == 2:
                if (
                    isinstance(x[0], (int, float))
                    and isinstance(x[1], (int, float))
                    and x[0] <= x[1]
                ):
                    processed_intervals.append([int(x[0]), int(x[1])])
            else:
                return None

        # 确保区间是有序的
        processed_intervals.sort(key=lambda x: x[0])
        return processed_intervals
    except Exception as e:
        print(e)
        return None


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


# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def evaluation(model_name: str = "gpt-4o-mini", output_path: str = "results.json"):
    output_path = sys.argv[1]

    with open("/home/wzx/ChatAD/data/ts_eval_image_mixed_plot1.json", "r") as f:
        data = json.load(f)
    with open("/home/wzx/ChatAD/data/eval_label.json", "r") as f:
        label = json.load(f)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    results = []
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_version = os.getenv("OPENAI_API_VERSION")
    client = AzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version=api_version)
    for item, item_label in zip(data, label):
        image_path = item["images"][0]
        question = item["messages"][0]["content"]
        groundtruth = item["messages"][1]["content"]

        if (
            "Write anomalous intervals (final answer) using python list format"
            not in question
        ):
            continue
        base64_image = encode_image(image_path)
        if "gpt" in model_name.lower():
            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": question},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
            )
            response = response.output_text
        elif "gemini" in model_name.lower():
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                # messages=[
                #     {"role": "system", "content": "You are a helpful assistant."},
                #     {"role": "user", "content": "Explain to me how AI works"},
                # ],
                model=model_name,
            )

            # print(response)
            response = response.choices[0].message.content

        print(f"Assistant: {response}")

        pred_intervals = extract_anomaly_intervals(response)
        if pred_intervals is None:
            pred_intervals = []

        # 从solution中提取真实的异常区间
        gt_intervals = extract_anomaly_intervals(groundtruth)
        if gt_intervals is None:
            gt_intervals = []

        # 计算point-adjust F1分数
        f1, precision, recall, tp, fp, fn = point_adjust_f1(
            pred_intervals, gt_intervals, ts_length=1000  # 使用实际的时间序列长度
        )
        print("pred_intervals, gt_intervals", pred_intervals, gt_intervals)
        # 累积统计量
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 保存结果
        results.append(
            {
                "prediction": pred_intervals,
                "ground_truth": gt_intervals,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "anomaly_type": item_label["type"],
                "image_path": image_path,
            }
        )

        print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    total_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0
    )

    print("\nOverall Results:")
    print(f"Total F1: {total_f1:.4f}")
    print(f"Total Precision: {total_precision:.4f}")
    print(f"Total Recall: {total_recall:.4f}")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")

    # 保存详细结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "overall": {
                    "f1": total_f1,
                    "precision": total_precision,
                    "recall": total_recall,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                },
                "samples": results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    # evaluation(model_name="gemini-2.0-flash-thinking-exp-01-21")
    evaluation(model_name="gpt-4o-2024-11-20", output_path="./eval/vlm-api-baseline/results_gpt-4o-mini.json")
