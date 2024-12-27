import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 数据读取
data = np.genfromtxt('multiple_linear_regression_dataset.csv', delimiter=',')
X = np.c_[np.ones(len(data)), data[:, :-1]]
Y = np.c_[data[:, -1]]
X = X[1:, :]  # 删除 X 的第一行
Y = Y[1:, :]  # 删除 Y 的第一行
# print(Y)

# 绘制原始数据散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], Y, c='r', marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Salary')
plt.show()


# 损失函数
def computeCost(x, y, theta=[[0], [0], [0]]):
    m = len(y)
    h = x.dot(theta)
    J = 1.0 / (2 * m) * np.sum(np.square(h - y))
    # print(J)
    return J


# 梯度下降
def gradientDescent(x, y, theta=[[0], [0], [0]], alpha=0.01, num_iters=1500):
    m = len(y)
    # 数据标准化
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)
    # print("x_scaled:", x_scaled)
    mean_y = np.mean(y)
    std_y = np.std(y)
    y_scaled = (y - mean_y) / std_y
    # print("y_scaled:", y_scaled)
    j_history = np.zeros(num_iters)

    for i in range(num_iters):
        h = x_scaled.dot(theta)
        theta = theta - alpha * (1.0 / m) * x_scaled.T.dot(h - y_scaled)
        j_history[i] = computeCost(x_scaled, y_scaled, theta)
    return theta, j_history


# 调用函数计算参数
theta, J_history = gradientDescent(X, Y)
print("theta:", theta.ravel())

# 损失函数可视化na
plt.plot(J_history)
plt.ylabel('J_history')
plt.xlabel('Iterations')
plt.show()
print('theta[2]:', theta[2])
print('theta[1]:', theta[1])
print('theta[0]:', theta[0])

# 绘制拟合曲面
x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x2 = np.linspace(X[:, 2].min(), X[:, 2].max(), 100)
X1, X2 = np.meshgrid(x1, x2)

Y_pred = theta[0] + theta[1] * 10000 * X1 + theta[2] * 10000 * X2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制拟合曲面
ax.plot_surface(X1, X2, Y_pred, cmap='viridis', alpha=0.6)

# 绘制原始数据点
ax.scatter(X[:, 1], X[:, 2], Y, c='r', marker='o')

ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Salary')
plt.show()

