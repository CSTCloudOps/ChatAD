# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch  src/train.py \
#     --deepspeed ds_config/ds_config.json \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path /data/wangzexin/qwen2.5-14B-addtoken \
#     --trust_remote_code True \
#     --flash_attn fa2 \
#     --finetuning_type lora \
#     --template qwen \
#     --dataset_dir data \
#     --dataset template_qa_50000_minmax_SFT \
#     --cutoff_len 10000 \
#     --learning_rate 5e-04 \
#     --num_train_epochs 2 \
#     --max_samples 50000 \
#     --overwrite_cache True \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 10 \
#     --save_steps 300 \
#     --warmup_ratio 0.1 \
#     --neftune_noise_alpha 0 \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --lora_target  all \
#     --output_dir /data/wangzexin/saves/qwen-14B-addtoken/lora/sft \
#     --fp16 True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --ddp_timeout 180000000


# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 accelerate launch  src/train.py \
#     --deepspeed ds_config/ds_config.json \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path /data/wangzexin/ \
#     --trust_remote_code True \
#     --flash_attn fa2 \
#     --finetuning_type full \
#     --template qwen \
#     --dataset_dir data \
#     --dataset sft-zscore-new3-woCOTDATA \
#     --cutoff_len 10000 \
#     --learning_rate 1e-05 \
#     --num_train_epochs 1 \
#     --max_samples 100000 \
#     --overwrite_cache True \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 10 \
#     --save_steps 300 \
#     --warmup_ratio 0.1 \
#     --neftune_noise_alpha 0 \
#     --output_dir /data/wangzexin/saves/qwen-14B/full/sft-zscore-new3-wocotdata \
#     --fp16 True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --ddp_timeout 180000000


# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 accelerate launch  src/train.py \
#     --deepspeed ds_config/ds_config.json \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path /data/wangzexin/ \
#     --trust_remote_code True \
#     --flash_attn fa2 \
#     --finetuning_type full \
#     --template qwen \
#     --dataset_dir data \
#     --dataset sft-zscore-new3-woINDEXLENGTH \
#     --cutoff_len 10000 \
#     --learning_rate 1e-05 \
#     --num_train_epochs 1 \
#     --max_samples 100000 \
#     --overwrite_cache True \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 10 \
#     --save_steps 300 \
#     --warmup_ratio 0.1 \
#     --neftune_noise_alpha 0 \
#     --output_dir /data/wangzexin/saves/qwen-14B/full/sft-zscore-new3-woindexlength \
#     --fp16 True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --ddp_timeout 180000000


# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 accelerate launch  src/train.py \
#     --deepspeed ds_config/ds_config.json \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path /data/wangzexin/ \
#     --trust_remote_code True \
#     --flash_attn fa2 \
#     --finetuning_type full \
#     --template qwen \
#     --dataset_dir data \
#     --dataset sft-zscore-new3-wolocal \
#     --cutoff_len 10000 \
#     --learning_rate 1e-05 \
#     --num_train_epochs 1 \
#     --max_samples 100000 \
#     --overwrite_cache True \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 10 \
#     --save_steps 300 \
#     --max_steps 601 \
#     --warmup_ratio 0.1 \
#     --neftune_noise_alpha 0 \
#     --output_dir /data/wangzexin/saves/qwen-14B/full/sft-zscore-new3-wolocal \
#     --fp16 True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --ddp_timeout 180000000

# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 accelerate launch  src/train.py \
#     --deepspeed ds_config/ds_config.json \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path /data/wangzexin/ \
#     --trust_remote_code True \
#     --flash_attn fa2 \
#     --finetuning_type full \
#     --template qwen \
#     --dataset_dir data \
#     --dataset sft-zscore-new3-wominmax \
#     --cutoff_len 10000 \
#     --learning_rate 1e-05 \
#     --num_train_epochs 1 \
#     --max_samples 100000 \
#     --overwrite_cache True \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 10 \
#     --save_steps 300 \
#     --max_steps 601 \
#     --warmup_ratio 0.1 \
#     --neftune_noise_alpha 0 \
#     --output_dir /data/wangzexin/saves/qwen-14B/full/sft-zscore-new3-wominmax \
#     --fp16 True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --ddp_timeout 180000000

# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 accelerate launch  src/train.py \
#     --deepspeed ds_config/ds_config.json \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path /data/wangzexin/ \
#     --trust_remote_code True \
#     --flash_attn fa2 \
#     --finetuning_type full \
#     --template qwen \
#     --dataset_dir data \
#     --dataset sft-zscore-new3-wotrendperiodnoise \
#     --cutoff_len 10000 \
#     --learning_rate 1e-05 \
#     --num_train_epochs 1 \
#     --max_samples 100000 \
#     --overwrite_cache True \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 10 \
#     --save_steps 300 \
#     --max_steps 601 \
#     --warmup_ratio 0.1 \
#     --neftune_noise_alpha 0 \
#     --output_dir /data/wangzexin/saves/qwen-14B/full/sft-zscore-new3-wotrendperiodnoise \
#     --fp16 True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --ddp_timeout 180000000



CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 accelerate launch  src/train.py \
    --deepspeed ds_config/ds_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/wangzexin/ \
    --trust_remote_code True \
    --flash_attn fa2 \
    --finetuning_type full \
    --template qwen \
    --dataset_dir data \
    --dataset sft-zscore-new5 \
    --cutoff_len 10000 \
    --learning_rate 1e-05 \
    --num_train_epochs 1 \
    --max_samples 100000 \
    --overwrite_cache True \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 200 \
    --warmup_ratio 0.1 \
    --neftune_noise_alpha 0 \
    --output_dir /data/wangzexin/saves/qwen-14B/full/sft-zscore-new5 \
    --fp16 True \
    --plot_loss True \
    --overwrite_output_dir True \
    --ddp_timeout 180000000