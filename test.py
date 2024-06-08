import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import function as F
from scipy.optimize import curve_fit
from pyswarm import pso

# 设置字体
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取Excel文件
data = pd.read_excel('data/C1.xlsx')

# 删除每段的离群点并进行数据清洗
data_segments = []
for segment in [(0, 287), (293, 592), (592, 856), (860, 1018), (1022, 1251), (1279, 1685), (1724, 1882), (1892, 2071), (2078, 2515), (2539, 2734), (2738, 2919)]:
    segment_data = data.iloc[segment[0]:segment[1]].dropna()
    segment_data = F.Remove_outliers(segment_data)
    data_segments.append(segment_data)

# 处理需要拟合的段（1, 2, 3, 5, 6, 9段）的数据
age_x = np.concatenate([data_segments[i].iloc[:, 0] for i in [0, 1, 2, 4, 5, 8]])
age_y = np.concatenate([
    data_segments[0].iloc[:, 1],
    data_segments[1].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1]),
    data_segments[2].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1]),
    data_segments[4].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1] + data_segments[2].iloc[-1, 1] - data_segments[4].iloc[0, 1]),
    data_segments[5].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1] + data_segments[2].iloc[-1, 1] - data_segments[4].iloc[0, 1] + data_segments[4].iloc[-1, 1] - data_segments[5].iloc[0, 1]),
    data_segments[8].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1] + data_segments[2].iloc[-1, 1] - data_segments[4].iloc[0, 1] + data_segments[4].iloc[-1, 1] - data_segments[5].iloc[0, 1] + data_segments[5].iloc[-1, 1] - data_segments[8].iloc[0, 1])
])

# 将数据转换为np.array
age_x = np.array(age_x)
age_y = np.array(age_y)

# 用数据拟合模型
age_params, age_params_covariance = curve_fit(F.model_func, age_x, age_y)
age_x_fit = np.linspace(min(age_x), max(age_x), 100)
age_y_fit = F.model_func(age_x_fit, *age_params)

# 画出数据点和拟合曲线
plt.scatter(age_x, age_y, label='老化数据(y轴为: 累计维护高度+每段数据y值)', color='lightblue')
plt.plot(age_x_fit, age_y_fit, color='black', label='老化拟合曲线')

# 添加标签和图例
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('figure/123569段数据清洗后拟合性能趋势图')
plt.show()

# 计算每次维护后delta_x的位移量
delta_x = [
    -(data_segments[i].iloc[-1, 0] - data_segments[i + 1].iloc[0, 0])
    for i in [0, 1, 2, 4, 5]
]
print('第1~2, 2~3, 3~5, 5~6, 6~9段位移量分别为')
print(delta_x)
print('===============================================')

# 计算回推的delta_t, 即delta_x
delta_y = [
    data_segments[i].iloc[-1, 1] - data_segments[i + 1].iloc[0, 1]
    for i in [0, 1, 2, 4, 5]
]

# 用PSO算法求解最优的delta_t值
x_value = [data_segments[i].iloc[0, 0] for i in [1, 2, 4, 5, 8]]
lb = [0]  # t 的下界
ub = [1000]  # t 的上界（根据实际情况调整）

optimal_t, optimal_g = pso(lambda t: F.g(t, x_value, delta_y, *age_params), lb, ub)
print('使用PSO算法求解最优回退时间与最小误差')
print('最优回退时间', optimal_t, '最优误差', optimal_g)
print('===============================================')

# 拟合第4, 7, 10, 11段的斜率，用局部的点来拟合
# datasize = 30, 即选取拟合局部导数的数据量大小
datasize = 30
local_segments = [data_segments[i].iloc[0:datasize] for i in [3, 6, 9, 10]]

# 使用numpy进行线性拟合
coefficients = [np.polyfit(segment['时间'], segment['性能'], 1) for segment in local_segments]
print(f'清洗后数据集{4}的局部斜率: {coefficients[0][0]}')
print(f'清洗后数据集{7}的局部斜率: {coefficients[1][0]}')
print(f'清洗后数据集{10}的局部斜率: {coefficients[2][0]}')
print(f'清洗后数据集{11}的局部斜率: {coefficients[3][0]}')
polys = [np.poly1d(coef) for coef in coefficients]
print('===============================================')

colors = ['blue', 'cyan']

for i, segment in enumerate([3, 6, 9, 10]):
    plt.scatter(data_segments[segment]['时间'], data_segments[segment]['性能'], label=f'数据集{segment + 1}全部点', color='blue')
    plt.scatter(local_segments[i]['时间'], local_segments[i]['性能'], label=f'数据集{segment + 1}局部点', color='cyan')
    plt.plot(local_segments[i]['时间'], polys[i](local_segments[i]['时间']), label=f'拟合曲线{segment + 1}', color='black')

plt.xlabel('时间'); plt.ylabel('性能')
plt.legend()
plt.savefig('figure/在清洗数据后得到的第4_7_10_11段起始点的斜率')
plt.show()

# 下面计算拟合曲线在3, 6, 9, 10段的末尾点的斜率(直接代入末尾点的x坐标算就行), 前面拟合的是4, 7, 10, 11起始点的局部斜率(取了30个点)
data_3_end_x = data_segments[2].iloc[-1,0]; data_6_end_x = data_segments[5].iloc[-1,0]
data_9_end_x = data_segments[8].iloc[-1,0]; data_10_end_x = data_segments[9].iloc[-1,0]
data_3_end_slope = F.derivative_model_func(data_3_end_x, *age_params); data_6_end_slope = F.derivative_model_func(data_6_end_x, *age_params)
data_9_end_slope = F.derivative_model_func(data_9_end_x, *age_params); data_10_end_slope = F.derivative_model_func(data_10_end_x, *age_params)

# 画出数据点和拟合曲线
plt.scatter(age_x, age_y, label='老化数据(y轴为: 累计维护高度+每段数据y值)', color='lightblue')
plt.plot(age_x_fit, age_y_fit, color='black', label='老化拟合曲线')

# 绘制这四点和它们的导数
points_x = [data_3_end_x, data_6_end_x, data_9_end_x, data_10_end_x]
points_y = [F.model_func(x, *age_params) for x in points_x]
slopes = [data_3_end_slope, data_6_end_slope, data_9_end_slope, data_10_end_slope]

for (px, py, slope) in zip(points_x, points_y, slopes):
    plt.scatter(px, py, color='red')  # 点的位置
    x_range = np.linspace(px - 10, px + 10, 100)
    y_range = slope * (x_range - px) + py
    plt.plot(x_range, y_range, color='red')  # 绘制导数直线

# 添加标签和图例
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('figure/123569段数据清洗后拟合性能趋势图_带导数')
plt.show()

# 输出导数值
print('拟合曲线在第3, 6, 9, 10段的末尾点的x坐标和斜率分别为:')
print(f'data_3_end_x = {data_3_end_x}, 斜率 = {data_3_end_slope}')
print(f'data_6_end_x = {data_6_end_x}, 斜率 = {data_6_end_slope}')
print(f'data_9_end_x = {data_9_end_x}, 斜率 = {data_9_end_slope}')
print(f'data_10_end_x = {data_10_end_x}, 斜率 = {data_10_end_slope}')
print('===============================================')

# 下面计算 k_{mean}, 计算如'figure/部分计算公式所示.png'所示
# 先计算 k原始, 例如: 将第4段的datasize(30)个数的x坐标带入拟合函数中, 计算拟合函数分别在这datasize(30)个座标处的导数值(斜率)
# 首先针对第4(7, 10, 11)段前30个数据，计算每一个数据在拟合曲线(三次函数)上对应点的导数值(斜率)
# 第4(7, 10, 11)段, 注意: 我们在前面已经设置好datasize = 30
slope_memory = [[], [], [], []]
abnormal_data = [3, 6, 9, 10]
for num, segment in enumerate(abnormal_data):
    for i in range(datasize):
        # 首先得到每一个点的x坐标
        x = data_segments[segment].iloc[i,0]
        # 然后利用function函数中的derivative_model_func, 计算这一点在拟合曲线上的导数
        # 注意: 利用前面得到的arg——params作为参数计算即可
        slope = F.derivative_model_func(x, *age_params)
        slope_memory[num].append(slope)

# 下面计算 k_{mean}
k4_mean = 1 / (1018 - 860 + 1 -30) * (coefficients[0][0] * datasize - sum(slope_memory[0]))
k7_mean = 1 / (1882 - 1724 + 1 -30) * (coefficients[1][0] * datasize - sum(slope_memory[1]))
k10_mean = 1 / (2734 - 2539 + 1 -30) * (coefficients[2][0] * datasize - sum(slope_memory[2]))
k11_mean = 1 / (2919 - 2738 + 1 -30) * (coefficients[3][0] * datasize - sum(slope_memory[3]))
# 将四个k_mean存储到k_mean列表中
k_mean = [k4_mean, k7_mean, k10_mean, k11_mean]
print('四个k_mean')
print('k4_mean', k_mean[0], 'k7_mean', k_mean[1], 'k10_mean', k_mean[2], 'k11_mean', k_mean[3])
print('===============================================')

# 下面计算 g(x), 公式如'figure/部分计算公式所示.png'所示
# 先算 g4(x), abnormal_data = [3, 6, 9, 10]
# 建立g列表
g = [[], [], [], []]
for num, segment in enumerate(abnormal_data):
    gx = k_mean[0] * (data_segments[segment].iloc[:,0] - data_segments[segment].iloc[0,0])
    g[num].append(gx)
print(g)