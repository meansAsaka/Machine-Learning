from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

# 加载数据
X, y = load_breast_cancer(return_X_y=True)

# 定义核函数和参数范围
param_grid_poly = {
    'kernel': ['poly'],
    'degree': [2, 3, 4, 5],
    'C': [0.1, 1, 10, 100]
}
param_grid_rbf = {
    'kernel': ['rbf'],
    'gamma': [0.001, 0.01, 0.1, 1],
    'C': [0.1, 1, 10, 100]
}

# 初始化模型
svm_poly = GridSearchCV(SVC(), param_grid_poly, cv=5, scoring='accuracy', n_jobs=-1)
svm_rbf = GridSearchCV(SVC(), param_grid_rbf, cv=5, scoring='accuracy', n_jobs=-1)

# 训练并找出最佳参数
svm_poly.fit(X, y)
svm_rbf.fit(X, y)

print("Best parameters for Polynomial Kernel:", svm_poly.best_params_)
print("Best parameters for Gaussian Kernel:", svm_rbf.best_params_)
