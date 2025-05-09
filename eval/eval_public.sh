export CUDA_VISIBLE_DEVICES=0
python eval_public_dataset.py Yahoo 120 m f
python eval_normal.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./public_ts_data/qa_files/Yahoo.json --output_path ./public_ts_data/result/Yahoo_120_m_f_normal.json

python eval_public_dataset.py Yahoo 500 m f
python eval_normal.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./public_ts_data/qa_files/Yahoo.json --output_path ./public_ts_data/result/Yahoo_500_m_f_normal.json


python eval_public_dataset.py AIOPS 120 m f
python eval_normal.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./public_ts_data/qa_files/AIOPS.json --output_path ./public_ts_data/result/AIOPS_120_m_f_normal.json


python eval_public_dataset.py AIOPS 500 m f
python eval_normal.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./public_ts_data/qa_files/AIOPS.json --output_path ./public_ts_data/result/AIOPS_500_m_f_normal.json


python eval_public_dataset.py WSD 120 m f
python eval_normal.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./public_ts_data/qa_files/WSD.json --output_path ./public_ts_data/result/WSD_120_m_f_normal.json


python eval_public_dataset.py WSD 500 m f
python eval_normal.py --model_path /data/wangzexin/qwen2.5_7b_vlm/ --data_path ./public_ts_data/qa_files/WSD.json --output_path ./public_ts_data/result/WSD_500_m_f_normal.json