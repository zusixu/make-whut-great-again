import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

# 读取数据
df1 = pd.read_excel("data/C1.xlsx")

# 筛选性能指标为 NaN 的行，并保留索引
C1_abnormal_data_time = df1.loc[df1['时间'].isna(), ['时间', '性能']]
C1_abnormal_data_value = df1.loc[df1['性能'].isna(), ['时间', '性能']]
#C1_abnormal_data_time.to_excel('data/C1_abnormal_data_time.xlsx', index=True)
#C1_abnormal_data_value.to_excel('data/C1_abnormal_data_value.xlsx', index=True)
#print(C1_abnormal_data_time)
#print(C1_abnormal_data_value)

# 使用非平稳时间序列分析
df2 = pd.read_excel("data/C1.xlsx" )
x = df2['时间']
y = df2['性能']
y.index = pd.Index(x) #添加日期
print(y)


# 填充NaN值
y = y.fillna(method='bfill')  # 使用后一个非NaN值填充NaN值
y.to_excel("data/y.xlsx")
# 绘制移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()
    timeseries.plot(color='blue', label=u'原始数据')
    rol_mean.plot(color='red', label=u'移动平均')
    rol_std.plot(color='black', label=u'移动平均标准差')
    plt.legend(loc='best')
    plt.title('移动平均值和标准差')
    plt.show()  # 移动这行代码到这里
    return f

#Dickey-Fuller 检验:
def stationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['t统计值','p值','#滞后项','观测点'])
    for key,value in dftest[4].items():
        dfoutput['关键值 (%s)'%key] = value
    return dfoutput

draw_trend(y,12)
print(stationarity(y))
