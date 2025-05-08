import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

def generate_sample_data(n=20, seed=0):
    np.random.seed(seed)
    # -1 <= x <= 1 の範囲で乱数を生成
    x_observed = np.random.uniform(low=-1, high=1, size=n)
    y_true = true_function(x_observed)
    df = pd.DataFrame({'観測点': x_observed, '真値': y_true})
    return df

if __name__ == '__main__':
    # サンプルデータの生成
    sample_df = generate_sample_data()
    print("生成されたサンプルデータ:")
    print(sample_df.head())

    # 演習1.1の線グラフを再現 
    x_smooth = np.linspace(-1, 1, 100)
    y_smooth = true_function(x_smooth)
    plt.rcParams['font.family'] = 'Hiragino Sans' 
    plt.figure(figsize=(8, 6))
    plt.plot(x_smooth, y_smooth, label='真の関数', color='blue')

    # サンプル集合をプロット
    plt.scatter(sample_df['観測点'], sample_df['真値'], label='サンプル集合', color='red', marker='o')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('真の関数とサンプル集合')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ex1.2.png')
    plt.show()