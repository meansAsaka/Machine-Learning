import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 线性回归，构造随机数据点
X = np.random.rand(100, 1) * 2
y = 4 + 3 * X + np.random.randn(100, 1)
plt.plot(X, y, 'b.')
plt.xlabel('X_1')
plt.ylabel('$y$')
plt.axis((0, 2, 0, 15))
plt.show()

# 矩阵多加一列1
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 测试数据点
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis((0, 2, 0, 15))
plt.show()

# 批量梯度下降
lin_reg = LinearRegression()
lin_reg.fit(X, y)
eta = 0.01  # 学习率
iterations = 1000  # 迭代次数
m = 100  # 数据点个数
theta = np.random.randn(2, 1)  # 随机初始化
for i in range(iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

X_new_b.dot(theta)

theta_path_bgd = []


def plot_gradient_descent(theta, eta, theta_path=None):
    n = len(X_b)
    plt.plot(X, y, 'b.')
    n_iterations = 1000
    for iteration in range(n_iterations):
        Y_predict = X_new_b.dot(theta)
        plt.plot(X_new, Y_predict, 'r-')
        gradient = 2 / n * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradient
        theta_path_bgd.append(theta)
        if theta_path is not None:
            theta_path.append(theta)
    plt.plot('X_1', 'y')
    plt.axis((0, 2, 0, 15))
    plt.title('eta={}'.format(eta))


theta = np.random.randn(2, 1)
plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1)
plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)
plt.show()

# 随机梯度下降
theta_path_sgd = []
m = len(X_b)
n_epochs = 50
t0, t1 = 5, 50  # 学习率超参数


# 衰减策略
def learning_schedule(t):
    return t0 / (t + t1)


for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, 'r-')
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, 'b.')
plt.axis((0, 2, 0, 15))
plt.show()

# 小批量梯度下降
theta_path_mgd = []

n_epochs = 50
minibatch_size = 20
theta = np.random.randn(2, 1)
t = 0
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i + minibatch_size]
        yi = y_shuffled[i:i + minibatch_size]
        plt.plot(X_new, X_new_b.dot(theta), 'r-')
        gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

plt.plot(X, y, 'b.')
plt.axis((0, 2, 0, 15))
plt.show()

# 三种方法对比
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(8, 4))
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], 'b-o', linewidth=3, label='BGD')
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], 'r-s', linewidth=1, label='SGD')
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'g-+', linewidth=2, label='MGD')
plt.legend(loc='upper left')
plt.axis((3.5, 4.5, 2.0, 4.0))
plt.show()
