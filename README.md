# CoT-SFT Training for TSAD

## Getting Started

### Synthetic Training Dataset
Refer to LLama Factory README.md and install packages.
Install Flash Attention && vllm && DeepSpeed

### Synthetic Training Dataset
```
cd ts_ano_dataset
python generate_ts_alignment.py
```
### Training LLM
```
sh train_deepspeed.sh
```
### Eval Dataset
```
cd ts_ano_dataset
python generate_ts_eval_public_dataset.py
```
Result are saved in public_dataset_eval_result
### Eval
```
cd ts_ano_dataset
sh eval_all.sh
```