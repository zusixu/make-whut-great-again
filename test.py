import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import function as F
from pandas import Series
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller

# 设置字体
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

# 设置参数
from parameters import Env_parameters
parameters = Env_parameters()
para = parameters.parse_args()


# 读取数据
df1 = pd.read_excel("data/C1.xlsx")

# 筛选性能指标为 NaN 的行，并保留索引
# C1_abnormal_data_time = df1.loc[df1['时间'].isna(), ['时间', '性能']]
# C1_abnormal_data_value = df1.loc[df1['性能'].isna(), ['时间', '性能']]
# C1_abnormal_data_time.to_excel('data/C1_abnormal_data_time.xlsx', index=True)
# C1_abnormal_data_value.to_excel('data/C1_abnormal_data_value.xlsx', index=True)

# 使用非平稳时间序列分析
df2 = pd.read_excel("data/C1.xlsx" )
df2.time = df2['时间']; df2.value = df2['性能']
df2.value.index = pd.Index(df2.time) #添加日期

# 填充NaN值, 使用后一个非NaN值填充NaN值, 这里不采用向前填充，因为前三个value值为NaN
df2.value = df2.value.fillna(method='bfill')

F.draw_trend(df2.value,12)
print(F.stationarity(df2.value))

df2value_log = np.log(df2.value)
print(F.stationarity(df2value_log))

plt.subplot(211)
plt.plot(df2.value, label=u'原始数据')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(df2value_log, label=u'取对数后')
plt.legend(loc='best')
plt.show()