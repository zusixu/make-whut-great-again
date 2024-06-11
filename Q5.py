import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("data/feature.csv")

# 提取特征列（除第4列外的所有列）
features = data.iloc[:, [1, 2, 4, 5, 6, 7]]

# 提取目标列（第4列）
target = data.iloc[:, 3]

# 创建RandomForestRegressor模型作为递归特征消除的基本模型
base_model = RandomForestRegressor()

# 使用递归特征消除进行特征选择
selector = RFE(estimator=base_model, n_features_to_select=3, step=1)
selector = selector.fit(features, target)

# 输出选择的特征列索引
selected_features_index = selector.get_support(indices=True)

# 输出选择的特征列名字
selected_features_names = features.columns[selected_features_index]
print("Selected Features:", selected_features_names)

# 输出特征的重要性排名
print("Feature Rankings:", selector.ranking_)

# 画图
plt.figure()
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance Ranking")
plt.plot(range(1, len(selector.ranking_) + 1), selector.ranking_, 'o')
plt.savefig('figure/基于随机森林的递归特征消失模型')
plt.show()
