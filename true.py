import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

def generate_sample_data(n=20, seed=0):
    np.random.seed(seed)
    x_observed = np.random.uniform(low=-1, high=1, size=n)
    y_true = true_function(x_observed)
    df = pd.DataFrame({'観測点': x_observed, '真値': y_true})
    return df

if __name__ == '__main__':
    # サンプルデータの生成
    sample_df = generate_sample_data()

    # ノイズの生成
    np.random.seed(0)  # 乱数シードを固定
    noise = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=len(sample_df)) / 2.0

    # 観測値の計算とDataFrameへの追加
    sample_df['観測値'] = sample_df['真値'] + noise
    print("ノイズを付与したサンプルデータ:")
    print(sample_df.head())

    # DataFrameをTSVファイルに出力
    output_filename = 'sample_data.tsv'
    sample_df.to_csv(output_filename, sep='\t', index=False)
    
    # 演習1.2のグラフを再現
    x_smooth = np.linspace(-1, 1, 100)
    y_smooth = true_function(x_smooth)
    plt.rcParams['font.family'] = 'Hiragino Sans' 
    plt.figure(figsize=(8, 6))
    plt.plot(x_smooth, y_smooth, label='真の関数', color='blue')
    plt.scatter(sample_df['観測点'], sample_df['真値'], label='真値 (ノイズなし)', color='red', marker='o')

    # 観測値をプロット
    plt.scatter(sample_df['観測点'], sample_df['観測値'], label='観測値 (ノイズあり)', color='green', marker='x')

    plt.xlabel('観測点 (x)')
    plt.ylabel('値 (y)')
    plt.title('真の関数、真値、観測値')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ex1.3.png')
    plt.show()