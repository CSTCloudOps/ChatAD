# Getting Started

## Environment

git clone ...  
pip install -e ".[torch,metrics]"  
pip install deepspeed==0.16.4  
pip install flash-attention.whl # download from website  

Download Qwen2.5-VL-7B-Instruct from huggingface  

## Train Model
sh train_mllm.sh



## How to Eval Different Plot Methods
1. Implement plot methods in plot_time_series (generate_sft_data.py)
```
# After Implementation
def plot_time_series(ts: np.ndarray, features: TimeSeriesFeatures) -> List[Dict[str, Any]]:
    """将时间序列绘制成图像并返回字节字典格式列表"""
    plt.figure(figsize=(12, 6))
    if PLOT_TYPE == "PLOT1":
        plt.plot(ts, label='Time Series', color='blue')
    elif PLOT_TYPE == "PLOT2":
        plt.bar([i for i in range(len(ts))], ts, label='Time Series', color='blue')
    elif PLOT_TYPE == "PLOT3":
        Your Implementation
    else:
        raise Exception("Not Implented")
```

2. Change the PLOT_TYPE in generate_sft_data.py
3. generate training and eval data
```
python generate_sft_data.py
```
4. If training, modify the dataset_info.json in /data and train_mllm.sh. 
5. Eval
```
python eval/eval_normal.py --model_path [model_path] --data_path [data_path] --output_path [output_path]
```