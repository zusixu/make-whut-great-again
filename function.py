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