export CUDA_VISIBLE_DEVICES=0,1,2,3




python eval/eval_normal.py --model_path=/data/wangzexin/saves/qwenvl_7b_rl_vision_only_normal/global_step_710/actor/huggingface/ --data_path=./data/ts_eval_image_mixed_PLOT1.json --output_path=./eval/RL_normal.json


python eval/eval_normal.py --model_path=/data/wangzexin/saves/rl_after_sft_vision_tower/global_step_65/actor/huggingface/ --data_path=./data/ts_eval_image_mixed_PLOT1.json --output_path=./eval/SFT_Vison_RL_TSAD.json


python eval/eval_normal.py --model_path=/data/wangzexin/saves/qwenvl_7b_rl_vision_normal_and_basic/global_step_3520/actor/huggingface/ --data_path=./data/ts_eval_image_mixed_PLOT1.json --output_path=./eval/RL_normal_and_basic.json

python eval/eval_normal.py --model_path=/data/wangzexin/saves/qwenvl_7b_rl_vision_normal_and_shaplet/global_step_880/actor/huggingface/ --data_path=./data/ts_eval_image_mixed_PLOT1.json --output_path=./eval/RL_normal_and_shaplet.json