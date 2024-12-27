import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# 数据读取
data = np.genfromtxt('Salary_dataset.csv', delimiter=',')
X = np.c_[np.ones(data[1:, 1:-1].shape), data[1:, 1:-1]]
Y = np.c_[data[1:, -1]]

# 绘制原始数据散点图
plt.scatter(X[:, 1], Y, color='blue', marker='o', s=10, label='Original data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# 损失函数
def computeCost(x, y, theta=[[0], [0]]):
    m = len(y)
    h = x.dot(theta)
    J = 1.0 / (2 * m) * np.sum(np.square(h - y))
    # print(J)
    return J


# 梯度下降
def gradientDescent(x, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    m = len(y)
    j_history = np.zeros(num_iters)
    for i in range(num_iters):
        h = x.dot(theta)
        theta = theta - alpha / m * (x.T.dot(h - y))
        j_history[i] = computeCost(x, y, theta)
    return theta, j_history


# 调用函数计算参数
theta, J_history = gradientDescent(X, Y)
print("theta:", theta.ravel())

plt.plot(J_history)
plt.ylabel('J_history')
plt.xlabel('Iterations')
plt.show()
# print('theta[1]', theta[1])
# print('theta[0]', theta[0])

# 绘制拟合曲线
x = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
f = theta[0] + theta[1] * x
plt.plot(x, f, 'r', label='Prediction')
plt.scatter(X[:, 1], Y, color='blue', marker='o', s=10, label='Original data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'''使用sklearn库进行线性回归'''
# 读取数据
data = pd.read_csv('Salary_dataset.csv')
years_experience = data['YearsExperience'].values.reshape(-1, 1)
salary = data['Salary'].values.reshape(-1, 1)
# print(years_experience)
# 线性回归模型
model = LinearRegression()
model.fit(years_experience, salary)

# 预测
predicted_salary = model.predict(years_experience)

# 绘制原始数据散点图
plt.scatter(years_experience, salary, color='blue', marker='o', s=10, label='Original data')

# 绘制拟合曲线
plt.plot(years_experience, predicted_salary, color='red', label='Linear Regression')

# 设置图表标题和轴标签
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# 显示图例
plt.legend()

# 显示图表
plt.show()

# 将未使用sklearn的线性模型与sklearn的线性模型进行绘图比较
print('None sklearn Prediction:', theta[0], theta[1])
print('sklearn Prediction:', model.intercept_, model.coef_)
plt.scatter(X[:, 1], Y, color='blue', marker='o', s=10, label='Original data')
plt.plot(x, f, 'r', label='Prediction')
plt.plot(years_experience, predicted_salary, color='black', label='Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
