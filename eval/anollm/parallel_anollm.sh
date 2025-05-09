#!/bin/bash

# 定义数据集数组
datasets=("Yahoo" "AIOPS" "WSD")

# 定义一个函数来运行单个数据集的所有实验
run_dataset_experiments() {
    local dataset=$1
    local gpu_id=$2
    
    echo "Starting experiments for $dataset on GPU $gpu_id"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 按顺序执行所有few shot实验
    for shot_num in 1 2 3; do
        echo "Running $dataset with $shot_num few shots"
        python anollm.py $dataset 200 qwen2_5_vl f t $shot_num
        python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ \
            --data_path ./anollm_${dataset}.json \
            --output_path ./${dataset,,}_result_${shot_num}.json
    done
    
    # 运行无few shot的实验
    echo "Running $dataset without few shot"
    python anollm.py $dataset 200 qwen2_5_vl f f 3
    python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ \
        --data_path ./anollm_${dataset}.json \
        --output_path ./${dataset,,}_result_0.json
        
    echo "Completed all experiments for $dataset on GPU $gpu_id"
}

# 创建一个目录来存储锁文件
mkdir -p ./.locks/anollm

# 为每个数据集创建一个锁文件
for dataset in "${datasets[@]}"; do
    touch "./.locks/anollm/${dataset}.lock"
done

# 启动不同的进程处理每个数据集
for i in "${!datasets[@]}"; do
    gpu_id=$((i % 8))  # 使用8个GPU循环分配
    dataset=${datasets[$i]}
    
    # 使用flock确保同一数据集的实验串行执行
    (
        flock -x 200
        run_dataset_experiments "$dataset" "$gpu_id"
    ) 200>"./.locks/anollm/${dataset}.lock" &
done

# 等待所有后台进程完成
wait
echo "All experiments completed!"

# 清理锁文件
rm -rf ./.locks/anollm





# # cot false, few shot true, few shot num 1
# export CUDA_VISIBLE_DEVICES=0
# python anollm.py Yahoo 200 qwen2_5_vl f t 1
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_Yahoo.json --output_path ./yahoo_result_1.json

# # cot false, few shot true, few shot num 2
# python anollm.py Yahoo 200 qwen2_5_vl f t 2
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_Yahoo.json --output_path ./yahoo_result_2.json

# # cot false, few shot true, few shot num 3
# python anollm.py Yahoo 200 qwen2_5_vl f t 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_Yahoo.json --output_path ./yahoo_result_3.json

# python anollm.py Yahoo 200 qwen2_5_vl f f 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_Yahoo.json --output_path ./yahoo_result_0.json


# # cot false, few shot true, few shot num 1
# python anollm.py AIOPS 200 qwen2_5_vl f t 1
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_AIOPS.json --output_path ./aiops_result_1.json

# # cot false, few shot true, few shot num 2
# python anollm.py AIOPS 200 qwen2_5_vl f t 2
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_AIOPS.json --output_path ./aiops_result_2.json

# # cot false, few shot true, few shot num 3
# python anollm.py AIOPS 200 qwen2_5_vl f t 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_AIOPS.json --output_path ./aiops_result_3.json

# python anollm.py AIOPS 200 qwen2_5_vl f f 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_AIOPS.json --output_path ./aiops_result_0.json


# # cot false, few shot true, few shot num 1
# python anollm.py WSD 200 qwen2_5_vl f t 1
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_WSD.json --output_path ./wsd_result_1.json

# # cot false, few shot true, few shot num 2
# python anollm.py WSD 200 qwen2_5_vl f t 2
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_WSD.json --output_path ./wsd_result_2.json

# # cot false, few shot true, few shot num 3
# python anollm.py WSD 200 qwen2_5_vl f t 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_WSD.json --output_path ./wsd_result_3.json

# python anollm.py WSD 200 qwen2_5_vl f f 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_WSD.json --output_path ./wsd_result_0.json


# # cot false, few shot true, few shot num 1
# python anollm.py NAB 200 qwen2_5_vl f t 1
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_NAB.json --output_path ./nab_result_1.json

# # cot false, few shot true, few shot num 2
# python anollm.py NAB 200 qwen2_5_vl f t 2
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_NAB.json --output_path ./nab_result_2.json

# # cot false, few shot true, few shot num 3
# python anollm.py NAB 200 qwen2_5_vl f t 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_NAB.json --output_path ./nab_result_3.json

# python anollm.py NAB 200 qwen2_5_vl f f 3
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_NAB.json --output_path ./nab_result_0.json


# # cot false, few shot true, few shot num 1
# python anollm.py TODS 200 qwen2_5_vl f t 1
# python eval_anollm.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./anollm_TODS.json --output_path ./tods_result_1.json