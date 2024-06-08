import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import function as F
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
# 设置字体
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

# 读取Excel文件
data = pd.read_excel('data/C1.xlsx')

# 第1~11个大段 (0~285)，并删除每段的离群点
data_1 = data.iloc[0:287]; data_1 = data_1.dropna(); data_1 = F.Remove_outliers(data_1)
data_2 = data.iloc[295:592]; data_2 = data_2.dropna(); data_2 = F.Remove_outliers(data_2)
data_3 = data.iloc[592:856]; data_3 = data_3.dropna(); data_3 = F.Remove_outliers(data_3)
data_4 = data.iloc[860:1018]; data_4 = data_4.dropna(); data_4 = F.Remove_outliers(data_4)
data_5 = data.iloc[1022:1251];data_5 = data_5.dropna();data_5 = F.Remove_outliers(data_5)
data_6 = data.iloc[1279:1685];data_6 = data_6.dropna(); data_6 = F.Remove_outliers(data_6)
data_7 = data.iloc[1724:1882]; data_7 = data_7.dropna(); data_7 = F.Remove_outliers(data_7)
data_8 = data.iloc[1892:2071];data_8 = data_8.dropna(); data_8 = F.Remove_outliers(data_8)
data_9 = data.iloc[2078:2515]; data_9 = data_9.dropna(); data_9 = F.Remove_outliers(data_9)
data_10 = data.iloc[2539:2734]; data_10 = data_10.dropna(); data_10 = F.Remove_outliers(data_10)
data_11 = data.iloc[2738:2919];data_11 = data_11.dropna(); data_11 = F.Remove_outliers(data_11)

# 拟合计入维护距离的实际性能曲线, 用1, 2, 3, 5, 6, 9段曲线
age_x = np.concatenate([data_1.iloc[:,0], data_2.iloc[:,0], data_3.iloc[:,0], data_5.iloc[:,0], data_6.iloc[:,0], data_9.iloc[:,0]])
age_y = np.concatenate([data_1.iloc[:,1],
         data_2.iloc[:,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1]),
         data_3.iloc[:,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1]),
         data_5.iloc[:,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_5.iloc[0,1]),
         data_6.iloc[:,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1]),
         data_9.iloc[:,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1] + data_6.iloc[-1,1] - data_9.iloc[0,1])])

# 将数据转换为np.array
age_x = np.array(age_x); age_y = np.array(age_y)

# 用数据拟合模型
age_params, age_params_covariance = curve_fit(F.model_func, age_x, age_y)
age_x_fit = np.linspace(min(age_x), max(age_x), 100)
age_y_fit = F.model_func(age_x_fit, *age_params)

# 画出数据点, 画出拟合曲线
plt.scatter(age_x, age_y, label='老化数据(y轴为: 累计维护高度+每段数据y值)')
plt.plot(age_x_fit, age_y_fit, color='blue', label='老化拟合曲线')

# 添加标签和图例, 并做图, 然后保存到figure文件中
plt.xlabel('X'); plt.ylabel('Y'); plt.legend()
plt.savefig('figure/123569段数据清洗后拟合性能趋势图')
plt.show()

# 计算每次维护后delta_x的位移量(因为有6段数据，所以共5个位移量), 为了保证位移量为正，前面加上负号。
# 例如对于3->5段的位移量: 计算方式为: delta_x3_x5 = -(第3段最后一点的x - 第5段第一点的x)
delta_x1_x2 = -(data_1.iloc[-1,0] - data_2.iloc[0,0])
delta_x2_x3 = -(data_2.iloc[-1,0] - data_3.iloc[0,0])
delta_x3_x5 = -(data_3.iloc[-1,0] - data_5.iloc[0,0])
delta_x5_x6 = -(data_5.iloc[-1,0] - data_6.iloc[0,0])
delta_x6_x9 = -(data_6.iloc[-1,0] - data_9.iloc[0,0])
print('第1~2, 2~3, 3~5, 5~6, 6~9段位移量分别为')
print(delta_x1_x2, '$$', delta_x2_x3, '$$',delta_x3_x5, '$$',delta_x5_x6, '$$',delta_x6_x9)