import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Any, Union, Tuple
import sys
import re

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.


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

def eval():
    path = sys.argv[1]
    output_path = sys.argv[2]

    with open('/home/wzx/ChatAD/data/ts_eval_image_mixed_plot1.json', 'r') as f:
        data = json.load(f)
    with open('/home/wzx/ChatAD/data/eval_label.json', 'r') as f:
        label = json.load(f)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    results = []

    for item, item_label in zip(data,label):
        image_path = item['images'][0]
        question = item['messages'][0]['content']
        groundtruth = item['messages'][1]['content']

        
        # path = 'OpenGVLab/InternVL3-8B'
        device_map = split_model(path)
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        # set the max number of tiles in `max_num`
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        # single-image single-round conversation (单图单轮对话)
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        print(f'Assistant: {response}')

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
    import os
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
    eval()
