import numpy as np

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

if __name__ == '__main__':
    # 動作確認
    x_values = np.array([-1, 0, 1, 2])
    y_values = true_function(x_values)
    print(f"入力 x: {x_values}")
    print(f"出力 y: {y_values}")