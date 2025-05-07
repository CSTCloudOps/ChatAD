# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import spectrogram

# # 你的数据
# ts = np.array([
#     0.24, 0.099, 0.026, 0.218, 0.237, 0.258, 0.169, 0.0, 0.018, 0.021, 0.044, 0.214, 0.117,
#     0.264, 0.462, 0.369, 0.388, 0.401, 0.443, 0.47, 0.449, 0.5, 0.517, 0.506, 0.629, 0.524,
#     0.512, 0.368, 0.248, 0.349, 0.577, 0.583, 0.468, 0.39, 0.284, 0.335, 0.406, 0.423, 0.531,
#     0.333, 0.388, 0.467, 0.436, 0.4, 0.256, 0.08, 0.102, 0.159, 0.247, 0.191, 0.153, 0.07,
#     0.034, 0.075, 0.068, 0.105, 0.201, 0.158, 0.447, 0.306, 0.393, 0.374, 0.477, 0.496, 0.523,
#     0.488, 0.434, 0.522, 0.536, 0.563, 0.489, 0.359, 0.259, 0.306, 0.505, 0.576, 0.531, 0.412,
#     0.283, 0.292, 0.481, 0.488, 0.509, 0.347, 0.376, 0.293, 0.401, 0.358, 0.263, 0.033, 0.368,
#     0.532, 0.547, 0.574, 0.475, 0.435, 0.339, 0.351, 0.478, 0.479, 0.545, 0.644, 0.719, 0.685,
#     0.693, 0.675, 0.727, 0.751, 0.906, 0.871, 0.825, 0.878, 0.879, 0.904, 0.915, 0.645, 0.63,
#     0.616, 0.804, 1.0
# ])

# fs = 100  # 任意采样率，只影响频率刻度，不影响 index 横坐标

# # index 轴
# index = np.arange(len(ts))

# # 计算频谱图（设置滑窗参数避免空图）
# f, t_spec, Sxx = spectrogram(ts, fs, nperseg=64, noverlap=8)

# # 将 t_spec 从秒转换为索引（点位置）
# t_index = (t_spec * fs).astype(int)

# # 画图
# # fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
# fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# # 1️⃣ 时序图（x轴为 index）
# axs[0].plot(index, ts, color='teal')
# # 设置x轴刻度，每10个点显示一个刻度
# xticks = np.arange(0, len(ts) + 1, 10)
# axs[0].set_xticks(xticks)
# axs[0].set_xticklabels([str(x) for x in xticks], fontsize=10)
# axs[0].tick_params(axis='both', which='major', labelsize=10)  # 设置刻度标签大小
# axs[0].set_ylabel('Amplitude', fontsize=10)
# axs[0].set_xlabel('Index', fontsize=10)
# axs[0].set_title('Time Series', fontsize=12)
# axs[0].grid(True)

# # 2️⃣ 频谱图（x轴为 index）
# pcm = axs[1].pcolormesh(t_index, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
# axs[1].set_ylabel('Frequency [Hz]')
# axs[1].set_xlabel('Index')
# axs[1].set_title('Spectrogram')
# # fig.colorbar(pcm, ax=axs[1], label='Power [dB]')

# plt.tight_layout()
# plt.savefig('./test.png')


from transformers import Qwen2_5_VLForConditionalGeneration