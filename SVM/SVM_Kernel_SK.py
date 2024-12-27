import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 加载 Iris 数据集
iris = load_iris()
X = iris.data[:, :2]  # 选择前两个特征 (sepal length, sepal width)
y = iris.target

# 将标签转换为二分类问题 (只保留两类数据)
X = X[y != 2]
y = y[y != 2]
y = np.where(y == 0, -1, 1)  # 将类别 0 改为 -1，类别 1 改为 1

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)


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
    "Linear Kernel": {"kernel": "linear"},
    "Polynomial Kernel (degree=3)": {"kernel": "poly", "degree": 3},
    "Gaussian Kernel (RBF, gamma=0.5)": {"kernel": "rbf", "gamma": 0.5}
}

# 训练和可视化
for name, params in kernels.items():
    model = SVC(C=1.0, **params).fit(X, y)
    plot_decision_boundary(X, y, model, title=name)
