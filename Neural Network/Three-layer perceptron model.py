import numpy as np
import matplotlib.pyplot as plt


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 求导sigmoid函数
def derivative_sigmoid(x):
    return x * (1 - x)


# 输入数据
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

# 设置权重和偏置项
np.random.seed(2)
w1 = 2 * np.random.random((2, 3)) - 1
w2 = 2 * np.random.random((3, 2)) - 1
# print(w1)
# print(w2)
b1 = 2 * np.random.random((1, 3)) - 1
b2 = 2 * np.random.random((1, 2)) - 1
# print(b1)
# print(b2)

# 训练感知机
layer1_errors = []  # 存储每一次的误差
layer2_errors = []  # 存储每一次的误差
for j in range(10000):
    # 前向传播
    layer1 = sigmoid(np.dot(X, w1) + b1)
    layer2 = sigmoid(np.dot(layer1, w2) + b2)

    # 计算误差
    layer2_error = y - layer2
    layer1_error = layer2_error.dot(w2.T)

    # 反向传播误差
    layer2_delta = layer2_error * derivative_sigmoid(layer2)
    layer1_delta = layer1_error * derivative_sigmoid(layer1)

    # 更新权重和偏置
    w2 += layer1.T.dot(layer2_delta)
    w1 += X.T.dot(layer1_delta)
    b2 += np.sum(layer2_delta, axis=0, keepdims=True)
    b1 += np.sum(layer1_delta, axis=0, keepdims=True)

    # 存储误差
    layer1_errors.append(np.mean(np.abs(layer1_error)))
    layer2_errors.append(np.mean(np.abs(layer2_error)))

# 预测感知机
layer1 = sigmoid(np.dot(X, w1) + b1)
layer2 = sigmoid(np.dot(layer1, w2) + b2)

# 计算均方误差
mse = np.mean((y - layer2) ** 2)

print('Mse结果: ', mse)
print('layer: ', layer1)
print('output: ', layer2)

# 绘制误差曲线
# print('layer1_errors: ', layer1_errors)
# print('layer2_errors: ', layer2_errors)
plt.plot(layer1_errors, label='Layer 1 Error')
plt.plot(layer2_errors, label='Layer 2 Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()
