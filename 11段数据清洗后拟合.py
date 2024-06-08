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

# 对每1个大段的数据进行分类
# 第一个大段 (0~285)
data_1 = data.iloc[0:287]
data_1 = data_1.dropna()
data_1 = F.Remove_outliers(data_1)


# 第2个大段 ()
data_2 = data.iloc[295:592]
data_2 = data_2.dropna()
data_2 = F.Remove_outliers(data_2)

# 第3个大段 ()
data_3 = data.iloc[592:856]
data_3 = data_3.dropna()
data_3 = F.Remove_outliers(data_3)

# 第4个大段
data_4 = data.iloc[860:1018]
data_4 = data_4.dropna()
data_4 = F.Remove_outliers(data_4)

# 第5个大段
data_5 = data.iloc[1022:1251]
data_5 = data_5.dropna()
data_5 = F.Remove_outliers(data_5)

# 第6个大段
data_6 = data.iloc[1279:1685]
data_6 = data_6.dropna()
data_6 = F.Remove_outliers(data_6)

# 第7个大段
data_7 = data.iloc[1724:1882]
data_7 = data_7.dropna()
data_7 = F.Remove_outliers(data_7)

# 第8个大段
data_8 = data.iloc[1892:2071]
data_8 = data_8.dropna()
data_8 = F.Remove_outliers(data_8)

# 第9个大段
data_9 = data.iloc[2078:2515]
data_9 = data_9.dropna()
data_9 = F.Remove_outliers(data_9)

# 第10个大段
data_10 = data.iloc[2539:2734]
data_10 = data_10.dropna()
data_10 = F.Remove_outliers(data_10)

# 第11个大段
data_11 = data.iloc[2738:2919]
data_11 = data_11.dropna()
data_11 = F.Remove_outliers(data_11)

# 拟合老化曲线
age_x = [data_1.iloc[0,0], data_2.iloc[0,0], data_3.iloc[0,0], data_4.iloc[0,0], data_5.iloc[0,0], data_6.iloc[0,0], data_7.iloc[0,0],
          data_8.iloc[0,0], data_9.iloc[0,0], data_10.iloc[0,0], data_11.iloc[0,0]]
age_y = [data_1.iloc[0,1],
         data_2.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1]),
         data_3.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1]),
         data_4.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1]),
         data_5.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1] + data_4.iloc[-1,1] - data_5.iloc[0,1]),
         data_6.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1] + data_4.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1]),
         data_7.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1] + data_4.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1] + data_6.iloc[-1,1] - data_7.iloc[0,1]),
         data_8.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1] + data_4.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1] + data_6.iloc[-1,1] - data_7.iloc[0,1] + data_7.iloc[-1,1] - data_8.iloc[0,1]),
         data_9.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1] + data_4.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1] + data_6.iloc[-1,1] - data_7.iloc[0,1] + data_7.iloc[-1,1] - data_8.iloc[0,1] + data_8.iloc[-1,1] - data_9.iloc[0,1]),
         data_10.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1] + data_4.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1] + data_6.iloc[-1,1] - data_7.iloc[0,1] + data_7.iloc[-1,1] - data_8.iloc[0,1] + data_8.iloc[-1,1] - data_9.iloc[0,1] + data_9.iloc[-1,1] - data_10.iloc[0,1]),
         data_11.iloc[0,1] + (data_1.iloc[-1,1] - data_2.iloc[0,1] + data_2.iloc[-1,1] - data_3.iloc[0,1] + data_3.iloc[-1,1] - data_4.iloc[0,1] + data_4.iloc[-1,1] - data_5.iloc[0,1] + data_5.iloc[-1,1] - data_6.iloc[0,1] + data_6.iloc[-1,1] - data_7.iloc[0,1] + data_7.iloc[-1,1] - data_8.iloc[0,1] + data_8.iloc[-1,1] - data_9.iloc[0,1] + data_9.iloc[-1,1] - data_10.iloc[0,1] + data_10.iloc[-1,1] - data_11.iloc[0,1])]



# Convert data to numpy arrays
age_x = np.array(age_x)
age_y = np.array(age_y)

# Fit the model to the data
age_params, age_params_covariance = curve_fit(F.model_func, age_x, age_y)


# Generate x values for the fitted curve
age_x_fit = np.linspace(min(age_x), max(age_x), 100)
age_y_fit = F.model_func(age_x_fit, *age_params)

# Plot the data points
plt.scatter(age_x, age_y, label='老化数据(y轴为: 维护高度+每段第一个点)')

# Plot the fitted curve
plt.plot(age_x_fit, age_y_fit, color='blue', label='老化拟合曲线')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()