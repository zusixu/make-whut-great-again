import pandas as pd
import numpy as np

def Remove_outliers(data):
    mean = data['性能'].mean()
    std_dev = data['性能'].std()

    # 定义阈值，例如，超过均值加减2倍标准差的点视为离群点
    threshold = 2 * std_dev

    # 根据阈值删除离群点
    cleaned_data = data[(data['性能'] >= mean - threshold) & (data['性能'] <= mean + threshold)]

    return cleaned_data

# Define the model function, for example, a quadratic function
def model_func(x, a, b, c):
    return a * x**2 + b * x + c