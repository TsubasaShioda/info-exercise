import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

def generate_observed_points_and_true_values(n=20, seed=0):
    np.random.seed(seed)
    x_observed = np.random.uniform(low=-1, high=1, size=n)
    y_true = true_function(x_observed)
    df = pd.DataFrame({'観測点': x_observed, '真値': y_true})
    return df

def generate_noisy_observations(df, noise_std=np.sqrt(2.0) / 2.0, seed=0):
    np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=len(df))
    df['観測値'] = df['真値'] + noise
    return df

def output_dataset_to_tsv(df, output_filename='sample_data.tsv'):
    df.to_csv(output_filename, sep='\t', index=False)
    print(f"データセットを {output_filename} に出力しました。")

def load_dataset_from_tsv(tsv_file_path='sample_data.tsv'):
    loaded_df = pd.read_csv(tsv_file_path, sep='\t')
    print("読み込んだDataFrame:")
    print(loaded_df)
    return loaded_df

if __name__ == '__main__':
    # 演習1.1：真の関数の準備

    # 演習1.2：観測点と真値の準備
    sample_df_no_noise = generate_observed_points_and_true_values()
    print("真値のみのサンプルデータ:")
    print(sample_df_no_noise.head())

    # 演習1.3：ノイズを付与した観測値の準備
    sample_df = generate_noisy_observations(sample_df_no_noise.copy())
    print("\nノイズを付与したサンプルデータ:")
    print(sample_df.head())

    # 演習1.4：データセットのファイル出力
    output_dataset_to_tsv(sample_df)

    # 演習1.5：データセットのファイル読み込み
    loaded_df = load_dataset_from_tsv()

    # グラフの描画 (演習1.2と1.3の内容を統合)
    x_smooth = np.linspace(-1, 1, 100)
    y_smooth = true_function(x_smooth)
    plt.rcParams['font.family'] = 'Hiragino Sans'
    plt.figure(figsize=(8, 6))
    plt.plot(x_smooth, y_smooth, label='真の関数', color='blue')
    plt.scatter(sample_df['観測点'], sample_df['真値'], label='真値 (ノイズなし)', color='red', marker='o')
    plt.scatter(sample_df['観測点'], sample_df['観測値'], label='観測値 (ノイズあり)', color='green', marker='x')
    plt.xlabel('観測点 (x)')
    plt.ylabel('値 (y)')
    plt.title('真の関数、真値、観測値')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ex1.3.png')
    plt.show()