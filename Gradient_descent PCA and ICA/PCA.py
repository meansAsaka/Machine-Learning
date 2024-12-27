import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)
X[:, 0] = np.random.uniform(0, 100, size=100)
X[:, 1] = 0.6 * X[:, 0] + 3 + np.random.normal(0, 10, size=100)
# print("X:", X)
plt.scatter(X[:, 0], X[:, 1])


# 对数据进行中心化处理：减去每个特征的均值
def demean(x):
    return x - np.mean(x, axis=0)


X_demean = demean(X)  # 保存中心化后的数据
# print("X_demean:", X_demean)
plt.figure(2)
plt.scatter(X_demean[:, 0], X_demean[:, 1])


# 定义目标函数：数据在投影到方向 w 上时的方差
def f(w, x):
    return np.sum((x.dot(w) ** 2)) / len(x)


# 目标函数关于参数 w 的梯度
def df_math(w, x):
    return x.T.dot(x.dot(w)) * 2 / len(x)


def direction(w):
    return w / np.linalg.norm(w)


def gradient_ascent(df, x, in_w, Eta, n_iters=1e4, epsilon=1e-8):
    w = direction(in_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, x)
        last_w = w
        w = w + Eta * gradient
        w = direction(w)
        if abs(f(w, x) - f(last_w, x)) < epsilon:
            break

        cur_iter += 1
    return w


initial_w = np.random.random(X.shape[1])
eta = 0.001
w = gradient_ascent(df_math, X_demean, initial_w, eta)
plt.figure(3)
plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.plot([0, w[0] * 50], [0, w[1] * 50], color='r')


# 画出样本在主成分轴上的投影
X_pca = X_demean.dot(w)
plt.figure(4)
plt.scatter(X_pca, np.zeros_like(X_pca))  # 降维后的数据在一维空间上

plt.show()
