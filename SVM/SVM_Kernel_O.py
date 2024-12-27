import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 加载 Iris 数据集
iris = load_iris()
X = iris.data[:, :2]  # 选择前两个特征 (sepal length, sepal width)
y = iris.target
# print(X)

# 将标签转换为二分类问题 (只保留两类数据)
X = X[y != 2]
y = y[y != 2]
y = np.where(y == 0, -1, 1)  # 将类别 0 改为 -1，类别 1 改为 1

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 定义核函数
def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)


def polynomial_kernel(x1, x2, degree=3):
    return (np.dot(x1, x2.T) + 1) ** degree


def gaussian_kernel(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x1[:, None] - x2, axis=2) ** 2)


# 定义 SVM 分类器
class SVM:
    def __init__(self, kernel=linear_kernel, C=1.0):
        self.alpha = None
        self.y = None
        self.X = None
        self.b = None
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X = X
        self.y = y
        K = self.kernel(X, X)

        # 简单实现硬间隔优化
        for _ in range(1000):  # 迭代次数
            for i in range(n_samples):
                condition = y[i] * (np.sum(self.alpha * y * K[:, i]) + self.b) < 1
                if condition:
                    self.alpha[i] += self.C
                    self.b += self.C * y[i]
        return self

    def predict(self, X):
        K = self.kernel(X, self.X)
        return np.sign(np.dot(K, self.alpha * self.y) + self.b)


# 可视化函数
def plot_decision_boundary(X, y, model, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1 (Sepal Length)")
    plt.ylabel("Feature 2 (Sepal Width)")
    plt.show()


# 核函数配置
kernels = {
    "Linear Kernel": linear_kernel,
    "Polynomial Kernel (degree=3)": lambda x1, x2: polynomial_kernel(x1, x2, degree=3),
    "Gaussian Kernel (gamma=0.5)": lambda x1, x2: gaussian_kernel(x1, x2, gamma=0.5)
}

# 训练和可视化
for name, kernel in kernels.items():
    model = SVM(kernel=kernel, C=0.1).fit(X, y)
    plot_decision_boundary(X, y, model, title=name)
