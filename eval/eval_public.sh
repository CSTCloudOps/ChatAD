#!/bin/bash

# 定义数据集数组
datasets=("Yahoo" "AIOPS" "WSD")
lengths=(120 500 1000)

# 定义一个函数来运行单个数据集的所有实验
run_dataset_experiments() {
    local dataset=$1
    local gpu_id=$2
    
    echo "Starting experiments for $dataset on GPU $gpu_id"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 运行不同长度的实验
    for length in "${lengths[@]}"; do
        echo "Running $dataset with length $length"
        python eval_public_dataset.py $dataset $length m t
        # python eval_normal.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ \
        #     --data_path ./public_ts_data/qa_files/${dataset}.json \
        #     --output_path ./public_ts_data/result/${dataset}_${length}_m_f_normal.json
        python eval_normal.py --model_path /data/wangzexin/saves/qwen_25_vl_sft_all_after_sft_vision_tower \
            --data_path ./public_ts_data/qa_files/${dataset}.json \
            --output_path ./public_ts_data/result/${dataset}_${length}_m_f_normal.json
    done
    
    echo "Completed all experiments for $dataset on GPU $gpu_id"
}

# 创建一个临时目录来存储锁文件
mkdir -p /tmp/dataset_locks

# 为每个数据集创建一个锁文件
for dataset in "${datasets[@]}"; do
    touch "/tmp/dataset_locks/${dataset}.lock"
done

# 启动不同的进程处理每个数据集
for i in "${!datasets[@]}"; do
    gpu_id=$i  # 每个数据集使用一个独立的GPU
    dataset=${datasets[$i]}
    
    # 使用flock确保同一数据集的实验串行执行
    (
        flock -x 200
        run_dataset_experiments "$dataset" "$gpu_id"
    ) 200>"/tmp/dataset_locks/${dataset}.lock" &
done

# 等待所有后台进程完成
wait
echo "All experiments completed!"

# 清理锁文件
rm -rf /tmp/dataset_locks