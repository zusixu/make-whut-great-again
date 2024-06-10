import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# 定义目标函数
def pso_goal(T):
    num = 0
    for i in range(n-1):
        if np.any(T[i+1] - T[i] > min_dis):
            num += 1
    return num * 200

# 定义适应度函数
def fitness_function(T):
    return pso_goal(T),

# 设置参数
n = 10  # 点的数量
min_dis = 100  # 最小距离
lb = np.zeros(n)  # T 的下限
ub = np.full(n, 5000)  # T 的上限
dimensions = n
n_particles = 10

# 设置PSO算法的参数
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# 运行PSO算法
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, bounds=(lb, ub), options=options)
best_T, best_cost = optimizer.optimize(fitness_function, iters=100)

print("最优解:", best_T)
print("最优解的目标函数值:", best_cost)

# 画出粒子群迭代函数图
plot_cost_history(optimizer.cost_history)
plt.title('Cost History')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()