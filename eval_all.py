import threading
import subprocess

def run_eval(gpu_id, model_path, data_path, output_path):
    cmd = [
        'bash', '-c',
        f'CUDA_VISIBLE_DEVICES={gpu_id} python eval/eval_normal.py '
        f'--model_path={model_path} '
        f'--data_path={data_path} '
        f'--output_path={output_path}'
    ]
    subprocess.run(cmd)

# 任务配置（每项对应一个线程）
tasks = [
    (0, '/data/wangzexin/qwen2.5_7b_vlm', './data/ts_eval_image_mixed_plot1.json', './eval/image_plot1.json'),
    (1, '/data/wangzexin/qwen2.5_7b_vlm', './data/ts_eval_image_mixed_plot2.json', './eval/image_plot2.json'),
    (2, '/data/wangzexin/qwen2.5_7b_vlm', './data/ts_eval_text_mixed.json', './eval/text.json'),
    (3, '/home/wzx/saves/qwen_25_vl_sft_image_plot1', './data/ts_eval_image_mixed_plot1.json', './eval/image_plot1_sft.json'),
    (4, '/home/wzx/saves/qwen_25_vl_sft_image_plot2', './data/ts_eval_image_mixed_plot2.json', './eval/image_plot2_sft.json'),
    (5, '/home/wzx/saves/qwen_25_vl_sft_text', './data/ts_eval_text_mixed.json', './eval/text_sft.json'),
]

# 启动所有线程
threads = []
for gpu_id, model_path, data_path, output_path in tasks:
    t = threading.Thread(target=run_eval, args=(gpu_id, model_path, data_path, output_path))
    t.start()
    threads.append(t)

# 等待所有线程完成
for t in threads:
    t.join()

print("✅ 所有评估任务已完成。")
