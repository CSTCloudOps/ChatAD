export CUDA_VISIBLE_DEVICES=0

mv SFT_IMAGE_PLOT1 SFT_MAGE
python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_image_mixed_plot1.json --output_path=./eval/image_plot1.json
mv SFT_IMAGE SFT_MAGE_PLOT1

mv SFT_IMAGE_PLOT2 SFT_MAGE
python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_image_mixed_plot2.json --output_path=./eval/image_plot2.json
mv SFT_IMAGE SFT_MAGE_PLOT2

python eval/eval_normal.py --model_path=/data/wangzexin/qwen2.5_7b_vlm --data_path=./data/ts_eval_text_mixed.json --output_path=./eval/text.json
