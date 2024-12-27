import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 绘制学习曲线函数
def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring="accuracy", n_jobs=-1
    )

    # 计算均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 绘制学习曲线
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


# 绘制混淆矩阵函数
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


# 定义最优参数
best_parameters = {
    "Linear Kernel": {"kernel": "linear", "C": 1.0},
    "Polynomial Kernel (Optimized)": {"kernel": "poly", "C": 100, "degree": 3},
    "Gaussian Kernel (Optimized)": {"kernel": "rbf", "C": 1, "gamma": 0.001}
}

# 绘制每种核函数的学习曲线
for name, params in best_parameters.items():
    print(f"======== {name} ========")

    # 初始化模型
    model = SVC(**params)
    model.fit(X_train, y_train)

    # 绘制学习曲线
    plot_learning_curve(model, f"Learning Curve ({name})", X_train, y_train, cv=5)

    # 模型预测和评估
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    # 混淆矩阵可视化
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix ({name})")

# 特征分布可视化
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.xlabel("Feature 1(standardized)")
plt.ylabel("Feature 2(standardized)")
plt.title("Feature Distribution")
plt.show()
