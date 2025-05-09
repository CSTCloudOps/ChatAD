import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import matplotlib.pyplot as plt
import re
import io
import base64
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2ForCausalLM
from qwen_vl_utils import process_vision_info

ONE_SHOT = False
FEW_SHOT = False
COT = True


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
    except:
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


def load_model(model_path: str) -> Tuple[Any, Any]:
    """
    加载模型和处理器

    Args:
        model_path: 模型路径
        model_type: 模型类型，可选值：
            - "auto": 自动检测模型类型
            - "qwen-vl": Qwen-VL模型
            - "qwen": Qwen模型
            - "other": 其他类型模型
    """

    # 读取config.json获取模型架构
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", None)
    else:
        model_type = None
    print(f"Loading model from {model_path} with type {model_type}")

    if model_type == "qwen2_5_vl":
        # 设置图像处理的参数
        min_pixels = 256 * 28 * 28  # 最小像素数
        max_pixels = 1280 * 28 * 28  # 最大像素数

        # 加载处理器和模型
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )

        # 根据config.json中的架构类型加载对应的模型类
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
        )
    return model, model_type, processor


def image_to_base64(image_bytes: bytes) -> str:
    """
    将图像字节数据转换为base64格式

    Args:
        image_bytes: 图像的字节数据

    Returns:
        base64编码的图像数据字符串
    """
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def process_input(
    model_type: str, processor: Any, messages: List[Dict]
) -> Dict:
    """
    处理输入，根据模型类型返回适当的输入格式

    Args:
        model_type: 模型类型
        processor: 处理器（可能是tokenizer或processor）
        messages: 消息列表，每个消息包含role和content
    """
    if model_type == "qwen2_5_vl":
        # 使用处理器处理输入
        print(messages)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
    else:
        # 对于非VL模型，我们只处理文本内容
        text_only_messages = []
        for msg in messages:
            if msg["role"] == "user":
                text_content = ""
                for content in msg["content"]:
                    if content["type"] == "text":
                        text_content += content["text"]
                text_only_messages.append({"role": "user", "content": text_content})
            else:
                text_only_messages.append(msg)
        
        inputs = processor.apply_chat_template(text_only_messages, return_tensors="pt")
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen-VL-Chat",
        help="Path to the model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/eval_data.json",
        help="Path to the evaluation data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results_syn.json",
        help="Path to save the results",
    )
    args = parser.parse_args()

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, model_type, processor = load_model(args.model_path)

    # 加载评估数据
    print(f"Loading evaluation data from {args.data_path}...")
    with open(args.data_path, "r") as f:
        df = json.load(f)

    # 初始化指标统计
    total_tp = 0
    total_fp = 0
    total_fn = 0
    results = []
    # with open("./data/eval_label.json", "r") as f:
    #     label = json.load(f)

    for item in df:
        # 处理输入
        inputs = process_input(model_type, processor, item["messages"][:-1])
        inputs = inputs.to(model.device)
        
        # 生成回答
        generated_ids = model.generate(
            **inputs, max_new_tokens=512, repetition_penalty=1.1
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print("Model Response:", response)
        # print("Ground Truth:", item["messages"][-1]["content"])

        # 提取预测的异常区间
        pred_intervals = extract_anomaly_intervals(response[0])
        if pred_intervals is None:
            pred_intervals = []

        # 从ground truth中提取真实的异常区间
        gt_text = item["messages"][-1]["content"][0]["text"] if isinstance(item["messages"][1]["content"], list) else item["messages"][1]["content"]
        gt_intervals = extract_anomaly_intervals(gt_text)
        if gt_intervals is None:
            gt_intervals = []
        print("pred_intervals:", pred_intervals, "gt_intervals:", gt_intervals)
        # 计算point-adjust F1分数
        f1, precision, recall, tp, fp, fn = point_adjust_f1(
            pred_intervals, gt_intervals, ts_length=1000  # 使用实际的时间序列长度
        )

        # 累积统计量
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 获取图像路径（如果存在）
        image_path = None
        for content in item["messages"][-1]["content"]:
            if content["type"] == "image":
                image_path = content["image"]
                break

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
                # "anomaly_type": item_label["type"],
                "image_path": image_path,
            }
        )

        print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # 计算总体指标
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
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
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
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

    print(f"\nDetailed results saved to {args.output_path}")


if __name__ == "__main__":
    main()
    # print(point_adjust_f1([[46,55]], [[49,55]], 200))


#