import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :1]  # 只取第一个特征（花萼长度）
Y = iris.data[:, 1:2]  # 只取第二个特征（花萼宽度）

# 创建线性回归模型并拟合数据
model = LinearRegression()
model.fit(X, Y)

# 绘制原始数据散点图
plt.scatter(X, Y, c='r', marker='o')

# 绘制拟合直线
x_min, x_max = X.min() - 1, X.max() + 1
x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='blue')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
