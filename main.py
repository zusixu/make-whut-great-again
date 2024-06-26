import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import function as F
from scipy.optimize import curve_fit
from pyswarm import pso
import pyswarms as ps
from parameters import Env_parameters
from pyswarms.utils.plotters import plot_cost_history
# 设置字体
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# ======================================================================================================================

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

# ======================================================================================================================
# ======================================================================================================================

# 读取Excel文件
data = pd.read_excel(data_path)
# 对每段数据进行数据清洗, 删除离群点
data_segments = []
for segment in interval_segment:
    segment_data = data.iloc[segment[0]:segment[1]].dropna()
    segment_data = F.Remove_outliers(segment_data)
    data_segments.append(segment_data)


# 首先拟合模型f(x), 利用三次函数
age_x = np.concatenate([data_segments[i].iloc[:, 0] for i in [0, 1, 2, 4, 5, 7, 8]])
age_y = np.concatenate([
    data_segments[0].iloc[:, 1],
    data_segments[1].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1]),
    data_segments[2].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1]),
    data_segments[4].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1] + data_segments[2].iloc[-1, 1] - data_segments[4].iloc[0, 1]),
    data_segments[5].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1] + data_segments[2].iloc[-1, 1] - data_segments[4].iloc[0, 1] + data_segments[4].iloc[-1, 1] - data_segments[5].iloc[0, 1]),
    data_segments[7].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1] + data_segments[2].iloc[-1, 1] - data_segments[4].iloc[0, 1] + data_segments[4].iloc[-1, 1] - data_segments[5].iloc[0, 1] + data_segments[5].iloc[-1, 1] - data_segments[7].iloc[0, 1]),
    data_segments[8].iloc[:, 1] + (data_segments[0].iloc[-1, 1] - data_segments[1].iloc[0, 1] + data_segments[1].iloc[-1, 1] - data_segments[2].iloc[0, 1] + data_segments[2].iloc[-1, 1] - data_segments[4].iloc[0, 1] + data_segments[4].iloc[-1, 1] - data_segments[5].iloc[0, 1] + data_segments[5].iloc[-1, 1] - data_segments[7].iloc[0, 1] + data_segments[7].iloc[-1, 1] - data_segments[8].iloc[0, 1])
])
age_x = np.array(age_x); age_y = np.array(age_y)
age_params, age_params_covariance = curve_fit(F.f, age_x, age_y)
age_x_fit = np.linspace(min(age_x), max(age_x), 100); age_y_fit = F.f(age_x_fit, *age_params)
# 画出数据点和拟合曲线
plt.scatter(age_x, age_y, label='老化数据(y轴为: 累计维护高度+每段数据y值)', color='lightblue')
plt.plot(age_x_fit, age_y_fit, color='black', label='老化拟合曲线')
# 添加标签和图例
plt.xlabel('X'); plt.ylabel('Y'); plt.legend()
plt.savefig('figure/123569段数据清洗后拟合性能趋势图')
#plt.show()

# ======================================================================================================================
# ======================================================================================================================

# 利用PSO算法求解回退时间
# t的下界与上界（根据实际情况调整）
lb = [0]; ub = [1000]
delta_x = [data_segments[i].iloc[0, 0] for i in normal_segment[1:6]]
delta_y = [data_segments[i].iloc[-1, 1] - data_segments[i + 1].iloc[0, 1] for i in normal_segment[0:5]]
backoff_time, backoff_error = pso(lambda t: F.pso_goal(t, delta_x, delta_y, *age_params), lb, ub)
print('回退时间', backoff_time, '回退误差', backoff_error)


# 拟合异常数据段的斜率(4, 7, 10, 11)
local_segments = [data_segments[i].iloc[0:SLD] for i in abnormal_segment]
coefficients = [np.polyfit(segment['时间'], segment['性能'], 1) for segment in local_segments]
polys = [np.poly1d(coef) for coef in coefficients]
print('===================================================')
print('===================================================')
# 计算4条线各自斜率之差
threshold_all = []
for i, segment in enumerate(abnormal_segment):
    plt.scatter(data_segments[segment]['时间'], data_segments[segment]['性能'], label=f'数据集{segment + 1}全部点', color='blue')
    plt.scatter(local_segments[i]['时间'], local_segments[i]['性能'], label=f'数据集{segment + 1}局部点',
                color='cyan')
    plt.plot(local_segments[i]['时间'], polys[i](local_segments[i]['时间']), label=f'拟合曲线{segment+1}',
             color='black')
    print(f'清洗后数据集{abnormal_segment[i] + 1}的局部斜率: {coefficients[i][0]}')
    threshold_all.append(coefficients[i][0] - F.derivative_f(data_segments[segment].iloc[0,0], *age_params))
plt.xlabel('时间'); plt.ylabel('性能'); plt.legend()
plt.savefig('figure/在清洗数据后得到的第4_7_10_11段起始点的斜率')
#plt.show()
print('4条线各自的斜率之差:',threshold_all)

# ======================================================================================================================
# ======================================================================================================================

# 下面计算拟合曲线在3, 6, 9, 10段的末尾点的斜率(直接代入末尾点的x坐标算就行), 前面拟合的是4, 7, 10, 11起始点的局部斜率(取了30个点)
# 绘制这四点和它们的导数
points_x = [data_segments[segment-1].iloc[-1,0] for i, segment in enumerate(abnormal_segment)]
points_y = [F.f(x, *age_params) for x in points_x]
points_slope = [F.derivative_f(data_segments[segment-1].iloc[-1,0], *age_params) for i, segment in enumerate(abnormal_segment)]

plt.scatter(age_x, age_y, label='老化数据(y轴为: 累计维护高度+每段数据y值)', color='lightblue')
plt.plot(age_x_fit, age_y_fit, color='black', label='老化拟合曲线')
for (px, py, slope) in zip(points_x, points_y, points_slope):
    plt.scatter(px, py, color='red')  # 点的位置
    x_range = np.linspace(px - 10, px + 10, 100)
    y_range = slope * (x_range - px) + py
    plt.plot(x_range, y_range, color='red')  # 绘制导数直线
# 添加标签和图例
plt.xlabel('X'); plt.ylabel('Y'); plt.legend()
plt.savefig('figure/123569段数据清洗后拟合性能趋势图_带导数')
#plt.show()

# 输出导数值
print('===================================================')
print('===================================================')
print('拟合曲线在第3, 6, 9, 10段的末尾点的x坐标和斜率分别为:')
for i, segment in enumerate(abnormal_segment):
    print(f'第{segment}段末尾x坐标 = {points_x[i]}, 第{segment}段末尾斜率 = {points_slope[i]}')
# 将data_segment中多个dataframe合并为一个dataframe
data_all = pd.concat(data_segments)





# ======================================================================================================================
# ======================================================================================================================

# 存储4, 7, 10, 11段上每个x的斜率
slope_memory = [[] for i in range(len(abnormal_segment))]
for i, segment in enumerate(abnormal_segment):
    for num in range(interval_segment[segment][1] - interval_segment[segment][0] + 1 - SLD):
        x = data_segments[segment].iloc[i, 0]
        slope = F.derivative_f(x, *age_params)
        slope_memory[i].append(slope)
# 下面计算k_mean
k_mean = [0 for i in range(len(interval_segment))]
for i, segment in enumerate(abnormal_segment):
    k_mean[segment] = 1 / (interval_segment[segment][1] - interval_segment[segment][0] + 1 - SLD) * (coefficients[i][0] * (interval_segment[segment][1] - interval_segment[segment][0] + 1 - SLD) - sum(slope_memory[i]))
print('===================================================')
print('===================================================')
print('k_mean: ', k_mean)

# ======================================================================================================================
# ======================================================================================================================

# 计算斜率阈值
# 用平均值作为斜率
#threshold = np.average(threshold_all[:len(abnormal_segment)])
# 用最小值作为斜率阈值
threshold = min(threshold_all)
print('斜率阈值:', threshold)
print('===================================================')
print('===================================================')


# 设置slope_RW用于存放选取时间节点前30个节点的x坐标在拟合曲线上的斜率
slope_fx = []
print('请输入需要判断是否异常的时间节点(整数):')
any_time = int(input())
# 设置slope_ex用于存放选取时间节点前30个节点的x坐标在模型e(x)上的斜率
slope_ex = []
if any_time < SRW:
    slope_ex = [F.derivative_ex(k_mean, backoff_time, x, *age_params) for x in range(0,any_time)]
else:
    slope_ex = [F.derivative_ex(k_mean, backoff_time, x, *age_params) for x in range(any_time-SRW,any_time)]
# 筛选出大于斜率阈值min_slope的数据
slope_ex_1 = [slope_ex[i] for i in range(SRW) if slope_ex[i] > threshold]
if len(slope_ex_1) >= len(slope_ex)/2:
    print(f'{any_time}时的机器需要维修')
else:
    print(f'{any_time}时的机器正常, 无需维修')
print('===================================================')
print('===================================================')

# ======================================================================================================================
# ======================================================================================================================

# 下面计算g1(x)
g1 = []
for i, segment in enumerate(interval):
    g1x = F.g1x(data_segments[segment].iloc[:, 0], k_mean[segment], data_segments[segment].iloc[0, 0])
    g1.append(g1x)
# 下面计算g2(x)
g2 = []
for i, segment in enumerate(interval):
    g2x = F.g2x(g2, segment, data_segments[segment].iloc[:, 0], sum(k_mean[:segment + 1]), data_segments[segment].iloc[0, 0])
    g2.append(g2x)
# 下面计算h(x)
h = []
for i,  segment in enumerate(interval):
    if segment == 0:
        hx = 0
    else:
        x = data_segments[segment].iloc[0, 0]
        hx = F.hx(segment, backoff_time[0], h, x, *age_params)
    h.append(hx)
# 下面计算e(x)
e = []
for num, segment in enumerate(interval):
    x = data_segments[segment].iloc[:, 0]
    ex = F.ex(data_segments, h, segment, x, k_mean, *age_params)
    e.append(ex)
# 画图
for i, segment in enumerate(interval):
    #plt.scatter(data_segments[segment]['时间'], data_segments[segment]['性能'], label=f'数据集{segment + 1}全部点', color='blue')
    plt.plot(e[i].index[:], e[i].values, label=f'第{segment + 1}段的e曲线', color='black')
plt.xlabel('时间'); plt.ylabel('性能'); plt.legend()
plt.savefig('figure/模型e(x).png')
#plt.show()

# ======================================================================================================================
# ======================================================================================================================

# 下面计算无异常
Wuyichang = []
for i, segment in enumerate(interval):
    x_total = data_segments[segment].iloc[:, 0]
    # judge用来判断性能是否达到1
    judge = 0
    for num, x in enumerate(x_total):
        Wuyichangx = F.Wuyichangx(h, segment, x, *age_params)
        Wuyichang.append((x, Wuyichangx))
        if Wuyichangx >= 1:
            judge = 1
            break
    if judge == 1:
        break
    curve_x, curve_y = zip(*Wuyichang)
    plt.plot(curve_x, curve_y, label=f'无异常曲线', color='black')
    Wuyichang = []
# 使用虚线画出对应的 x, y 坐标
plt.plot([curve_x[-1], curve_x[-1]], [0, curve_y[-1]], 'r--', linewidth=1)
plt.plot([0, curve_x[-1]], [curve_y[-1], curve_y[-1]], 'r--', linewidth=1)
plt.annotate(f'({curve_x[-1]}, 0)', (curve_x[-1], 0), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
plt.annotate(f'(0, {curve_y[-1]})', (0, curve_y[-1]), textcoords="offset points", xytext=(-30, 0), ha='center', fontsize=8)
plt.savefig('figure/无异常')
#plt.show()

# 下面计算无维护
Wuweihu = []
for i, segment in enumerate(interval):
    x_total = data_segments[segment].iloc[:, 0]
    # judge用来判断性能是否达到1
    judge = 0
    for num, x in enumerate(x_total):
        Wuweihux = F.Wuweihux(g2, segment, num, x, *age_params)
        Wuweihu.append((x, Wuweihux))
        if Wuweihux >= 1:
            judge = 1
            break
    curve_x, curve_y = zip(*Wuweihu)
    plt.plot(curve_x, curve_y, label=f'无维护曲线', color='black')
    if judge == 1:
        break
    Wuweihu = []
# 使用虚线画出对应的 x, y 坐标
plt.plot([curve_x[-1], curve_x[-1]], [0, curve_y[-1]], 'r--', linewidth=1)
plt.plot([0, curve_x[-1]], [curve_y[-1], curve_y[-1]], 'r--', linewidth=1)
plt.annotate(f'({curve_x[-1]}, 0)', (curve_x[-1], 0), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
plt.annotate(f'(0, {curve_y[-1]})', (0, curve_y[-1]), textcoords="offset points", xytext=(-30, 0), ha='center', fontsize=8)
plt.savefig('figure/无维护')
#plt.show()

# 下面计算都无
Douwu = []
for i, segment in enumerate(interval):
    x_total = data_segments[segment].iloc[:, 0]
    # judge用来判断性能是否达到1
    judge = 0
    for num, x in enumerate(x_total):
        Douwux = F.Douwux(x, *age_params)
        Douwu.append((x, Douwux))
        if Douwux >= 1:
            judge = 1
            break
    curve_x, curve_y = zip(*Douwu)
    plt.plot(curve_x, curve_y, label=f'无维护曲线', color='black')
    if judge == 1:
        break
    Douwu = []
# 使用虚线画出对应的 x, y 坐标
plt.plot([curve_x[-1], curve_x[-1]], [0, curve_y[-1]], 'r--', linewidth=1)
plt.plot([0, curve_x[-1]], [curve_y[-1], curve_y[-1]], 'r--', linewidth=1)
plt.annotate(f'({curve_x[-1]}, 0)', (curve_x[-1], 0), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
plt.annotate(f'(0, {curve_y[-1]})', (0, curve_y[-1]), textcoords="offset points", xytext=(-30, 0), ha='center', fontsize=8)
plt.savefig('figure/都无')
#plt.show()

# 利用pso算法求解最优维护次数
# 给出确定的维修的点的个数
n = 10
# 计算生成的维修时间, 以及各个时间节点之差
T, T_length = F.interval_generate(n)
x = 0
for i in range(5000):
    # e1(x) = g(x,T) + + f(x,T) + h(x,T)与y=1交点的横坐标
    if F.e1x(T, i, k_mean, backoff_time, *age_params) >= 1:
        x = i
        break
print('===================================================')
print('===================================================')
print('e1(x) = g(x,T) + f(x,T) + h(x,T)与y=1交点的横坐标为: ', x)

# =================================================================================
# =================================================================================

# t的下界与上界（根据实际情况调整）
def pso_goal2(T):
    num = 0
    T_sorted = np.sort(T)  # 确保 T 是有序的，以计算相邻点的距离
    for i in range(len(T_sorted) - 1):
        if np.any(T[i+1] - T[i] > min_dis):
            num += 1
    return num * 200 -x
def fitness_function(T):
    return pso_goal2(T),

lb = [0 for i in range(n)]; ub = [x for i in range(n)]
dimensions = n; n_particles = 10
# 设置PSO算法的参数
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, bounds=(lb, ub), options=options)
best_solution, best_T = optimizer.optimize(fitness_function, iters=100)

print('===================================================')
print('===================================================')
print('最佳维护时间:')
best_T = np.sort(best_T)
print(best_T)
# 画出粒子群迭代函数图
plot_cost_history(optimizer.cost_history)
plt.title('Cost History')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.savefig('figure/cost_history.png')
plt.show()

# =================================================================================
# =================================================================================

print('===================================================')
print('===================================================')
# 现在将best_T带入e1x模型中
print('现在将best_T带入e1x模型中:')
x = 0
for i in range(5000):
    # e1(x) = g(x,T) + + f(x,T) + h(x,T)与y=1交点的横坐标
    if F.e1x(best_T, i, k_mean, backoff_time, *age_params) >= 1:
        x = i
        break
print('===================================================')
print('===================================================')
print('best_T时, e1(x) = g(x,T) + f(x,T) + h(x,T)与y=1交点的横坐标为: ', x)

# 下面计算原始数据带入e1x模型中
print('下面计算原始数据带入e1x模型中:')
T_origin = [segment[1] for segment in interval_segment if segment[0] < interval_segment[-1][0]]
print('原始维护点为:', T_origin)
x = 0
for i in range(5000):
    # e1(x) = g(x,T) + + f(x,T) + h(x,T)与y=1交点的横坐标
    if F.e1x(T_origin, i, k_mean, backoff_time, *age_params) >= 1:
        x = i
        break
print('===================================================')
print('===================================================')
print('T_origin时, e1(x) = g(x,T) + f(x,T) + h(x,T)与y=1交点的横坐标为: ', x)