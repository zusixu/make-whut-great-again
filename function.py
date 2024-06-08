import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, fsolve
from numpy.polynomial.polynomial import Polynomial

# 删除数据中的离群点
def Remove_outliers(data):
    mean = data['性能'].mean()
    std_dev = data['性能'].std()
    # 定义阈值，例如，超过均值加减2倍标准差的点视为离群点
    threshold = 2 * std_dev
    # 根据阈值删除离群点
    cleaned_data = data[(data['性能'] >= mean - threshold) & (data['性能'] <= mean + threshold)]

    return cleaned_data

# 定义模型函数f，例如，三次函数(可修改)
def f(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# 定义模型函数f的导数derivative_f
def derivative_f(x, a, b, c, d):
    return 3 * a * x**2 + 2 * b * x + c

# 定义目标函数 pso_goal(t)
def pso_goal(t, x_value, delta_y, a, b, c, d):
    g_val = 0
    for i in range(len(x_value)):
        xi = x_value[i]
        delta_y_i = delta_y[i]
        g_val += (f(xi, a, b, c, d) - f(xi - t, a, b, c, d) - delta_y_i)**2
    return g_val

# 定义函数gx
def gx(x, k, x0):
    return k * (x - x0)