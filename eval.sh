export CUDA_VISIBLE_DEVICES=0


# python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_image_mixed_plot1.json --output_path=./eval/image_plot1.json



# python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_image_mixed_plot2.json --output_path=./eval/image_plot2.json


# python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_text_mixed.json --output_path=./eval/text.json




# python eval/eval_normal.py --model_path=/home/wzx/saves/qwen_25_vl_sft_image_plot1 --data_path=./data/ts_eval_image_mixed_plot1.json --output_path=./eval/image_plot1.json



# python eval/eval_normal.py --model_path=/home/wzx/saves/qwen_25_vl_sft_image_plot2 --data_path=./data/ts_eval_image_mixed_plot2.json --output_path=./eval/image_plot2.json



# python eval/eval_normal.py --model_path=/home/wzx/saves/qwen_25_vl_sft_text --data_path=./data/ts_eval_text_mixed.json --output_path=./eval/text.json


# python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_image_mixed_frequency.json --output_path=./eval/image_frequency.json

python eval/eval_normal.py --model_path=/home/wzx/saves/qwen_25_vl_sft_image_stft --data_path=./data/ts_eval_image_mixed_stft.json --output_path=./eval/image_stft_sft.json

python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_image_mixed_stft.json --output_path=./eval/image_stft.json
