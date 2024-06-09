import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import function as F
from scipy.optimize import curve_fit
from pyswarm import pso
from parameters import Env_parameters
# 设置字体
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 超参数设置
parameters = Env_parameters()
para = parameters.parse_args()
# 初始化超参数
path = para.path
interval_segment = para.interval_segment
normal_segment = para.normal_segment
abnormal_segment = para.abnormal_segment
SLDsize = para.SLDsize


# 读取Excel文件
data = pd.read_excel(path)

# 对每段数据进行数据清洗, 删除离群点
data_segments = []
for segment in interval_segment:
    segment_data = data.iloc[segment[0]:segment[1]].dropna()
    segment_data = F.Remove_outliers(segment_data)
    data_segments.append(segment_data)