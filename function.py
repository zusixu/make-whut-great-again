import pandas as pd
import numpy as np
import function as F
import random
from parameters import Env_parameters
from scipy.optimize import curve_fit, fsolve
from numpy.polynomial.polynomial import Polynomial


# 超参数设置
parameters = Env_parameters()
para = parameters.parse_args()
# 初始化超参数
data_path = para.data_path
figure_path = para.figure_path
interval = para.interval
interval_segment = para.interval_segment
normal_segment = para.normal_segment
abnormal_segment = para.abnormal_segment
SLD = para.SLD
SRW = para.SRW
MMI = para.MMI
MTE = para.MTE
min_dis = para.min_dis


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
def g1x(x, k, x0):
    g1x = k * (x - x0)
    return g1x

def g2x(g2, segment, x, k, x0):
    g2x = k * (x - x0)
    if segment != 0:
        g2x = g2x + g2[segment-1].values[-1] - g2x.values[0]
    return g2x

def hx(segment, backoff_time, h, x, a, b, c, d):
    hx = F.f(x, a, b, c, d) - F.f(x - backoff_time, a, b, c, d) + h[segment - 1]
    return hx

def h2x(T, backoff_time, x, a, b, c, d):
    x_i0, x_i1 = F.find_interval(T, x)
    h2x = 0
    for i, segment in enumerate(T):
        if x > segment:
            if segment > backoff_time:
                h2x = h2x - (F.f(segment, a, b, c, d) - F.f(segment - backoff_time, a, b, c, d))
            else:
                h2x = h2x - F.f(segment, a, b, c, d)
    return h2x
def ex(data_segments, h, segment, x, k_mean, a, b, c, d):
    if segment == 3 or segment == 6 or segment == 9 or segment == 10:
        e = F.f(x, a, b, c, d) + F.g1x(x, k_mean[segment], data_segments[segment].iloc[0, 0]) - h[segment]
    else:
        e = F.f(x, a, b, c, d) - h[segment]

    return e

def derivative_ex(k_mean, backoff_time, x, a, b, c, d):
    k = 0
    for i, (start, end) in enumerate(interval_segment, start=1):
        if start <= x <= end:
            k = k_mean[i-1]
            break
    return k

def Wuyichangx(h, segment, x, a, b, c, d):
    wuyichang = F.f(x, a, b, c, d) - h[segment]

    return wuyichang

def Wuweihux(g, segment, num, x, a, b, c, d):
    wuweihu = F.f(x, a, b, c, d) + g[segment].values[num]

    return wuweihu

def Douwux(x, a, b, c, d):
    douwu = F.f(x, a, b, c, d)
    return douwu

def generate_random_points(n):
    points = [random.uniform(0, 5000) for _ in range(n)]
    return sorted(points)

def find_interval(points, value):
    for i in range(len(points) - 1):
        if points[i] <= value < points[i + 1]:
            return points[i], points[i+1]
    return 0,0
def find_interval2(points, value):
    for i in range(len(points) - 1):
        if points[i][0] <= value <= points[i][1]:
            return points[i][0]
    a=1
    return 0

def g_val2(T, k, backoff_time, t_i0, x_i0, x, a, b, c, d):
    return F.g1x(x, k, max(t_i0, x_i0)) + F.f(x, a, b, c, d) + h2x(T, backoff_time, x, a, b, c, d)

def interval_generate(n):
    # 先生成 1 到 5000 之间的 10 个不同的随机数
    random_numbers = random.sample(range(1, 5001), n)
    # 从小到大排序
    sorted_numbers = sorted(random_numbers)
    # 计算相邻点之间的差值
    differences = [sorted_numbers[i + 1] - sorted_numbers[i] for i in range(len(sorted_numbers) - 1)]

    return sorted_numbers, differences


def e1x(T, x, k_mean, backoff_time, a ,b, c, d):
    t_i0, t_i1 = F.find_interval(T, x)
    x_i0 = F.find_interval2(interval_segment, x)
    k = 0
    if 860<=x_i0<= 1018 or 1724<=x_i0<= 1882 or 2539<=x_i0<=2734 or 2738<=x_i0<=2919:
        if 860<=x_i0<= 1018: k = k_mean[3]
        elif 1724<=x_i0<= 1882: k = k_mean[6]
        elif 2539<=x_i0<=2734: k = k_mean[9]
        else: k = k_mean[10]
    g_val2 = F.g_val2(T, k, backoff_time, t_i0, x_i0, x, a, b, c, d)

    return g_val2

def pso_goal2(T, n, x):
    num = 0
    for i in range(n-1):
        if np.any(T[i+1] - T[i] > min_dis):
            num += 1
    return num * 200 - x