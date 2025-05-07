import json
from collections import defaultdict
import numpy as np

def calculate_metrics_by_type(json_file):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 用于存储每种异常类型的指标
    metrics_by_type = defaultdict(lambda: {'f1': [], 'precision': [], 'recall': []})
    
    # 遍历所有样本
    for sample in data['samples']:
        # 获取异常类型列表
        anomaly_types = sample['anomaly_type']
        
        # 获取样本的指标
        f1 = sample['f1']
        precision = sample['precision']
        recall = sample['recall']
        
        # 将指标添加到对应的异常类型中
        for anomaly_type in anomaly_types:
            metrics_by_type[anomaly_type]['f1'].append(f1)
            metrics_by_type[anomaly_type]['precision'].append(precision)
            metrics_by_type[anomaly_type]['recall'].append(recall)
    
    # 计算每种类型的平均指标
    results = {}
    for anomaly_type, metrics in metrics_by_type.items():
        results[anomaly_type] = {
            'avg_f1': np.mean(metrics['f1']),
            'avg_precision': np.mean(metrics['precision']),
            'avg_recall': np.mean(metrics['recall']),
            'sample_count': len(metrics['f1'])
        }
    
    return results

def main():
    json_file = './eval/image_stft_sft.json'
    results = calculate_metrics_by_type(json_file)
    
    # 打印结果
    print("\n不同异常类型的评估指标:")
    print("-" * 80)
    print(f"{'异常类型':<20} {'样本数':<10} {'平均F1':<12} {'平均Precision':<12} {'平均Recall':<12}")
    print("-" * 80)
    
    for anomaly_type, metrics in results.items():
        print(f"{anomaly_type:<20} {metrics['sample_count']:<10d} "
              f"{metrics['avg_f1']:.4f}      {metrics['avg_precision']:.4f}      "
              f"{metrics['avg_recall']:.4f}")

if __name__ == '__main__':
    main() 