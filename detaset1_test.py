import dataset1
import numpy as np
import matplotlib.pyplot as plt

# データセットの生成
data_df = dataset1.generate_observed_points_and_true_values(n=30, seed=42)

# ノイズの付与
noisy_data_df = dataset1.generate_noisy_observations(data_df.copy(), noise_std=0.3)

# TSVファイルへの出力
dataset1.output_dataset_to_tsv(noisy_data_df, output_filename='my_dataset.tsv')

# TSVファイルからの読み込み
loaded_data = dataset1.load_dataset_from_tsv(tsv_file_path='my_dataset.tsv')

# 真の関数のプロット
x_smooth = np.linspace(-1, 1, 100)
y_smooth = dataset1.true_function(x_smooth)
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.figure(figsize=(8, 6))
plt.plot(x_smooth, y_smooth, label='真の関数', color='blue')
plt.scatter(loaded_data['観測点'], loaded_data['観測値'], label='観測値 (読み込み)', color='purple', marker='o')
plt.xlabel('観測点 (x)')
plt.ylabel('値 (y)')
plt.title('読み込まれたデータセットと真の関数')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()