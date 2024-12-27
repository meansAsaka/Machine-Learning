import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pydotplus
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from IPython.display import Image
import pydotplus

'''使用pandas读取数据并进行数据预处理'''
data = pd.read_csv('student_data_DT2.csv', index_col=0)

# 丢弃无用的特征
data.drop(['UserId', 'term'], axis=1, inplace=True)

# 处理缺失值
table = data[data['totalLearningTime'].isnull().values == False]
score_avg = data['totalLearningTime'].mean()
table = data['totalLearningTime'].fillna(value=score_avg)
data['totalLearningTime'] = table

data['gender'] = (data['gender'] == 'male').astype('int')

labels = data['UserClass'].unique().tolist()
data['UserClass'] = data['UserClass'].apply(lambda n: labels.index(n))

labels = data['majorClass'].unique().tolist()
data['majorClass'] = data['majorClass'].apply(lambda n: labels.index(n))

labels = data['isPassExam'].unique().tolist()
data['isPassExam'] = data['isPassExam'].apply(lambda n: labels.index(n))

# 划分数据集
y = data['isPassExam'].values
X = data.drop(['isPassExam'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('train dataset: {0}; test dataset: {1}'.format(X_train.shape, X_test.shape))

'''构建决策树模型'''
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, min_samples_split=10)

# 拟合模型
clf.fit(X_train, y_train)
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predict)
print('train_accuracy', train_accuracy)

train_recall = recall_score(y_train, train_predict)
print('train_recall', train_recall)

test_accuracy = accuracy_score(y_test, test_predict)
print('test_accuracy', test_accuracy)

test_recall = recall_score(y_test, test_predict)
print('test_recall', test_recall)

# 将决策树导出为.dot文件
with open("students_dt_2.dot", 'w') as f:
    f = export_graphviz(clf, out_file=f)

# 或者导出为pdf
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("students_dt_2.pdf")

# 交叉验证
print('X.shape', X.shape)
print('y.shape', y.shape)
scores = cross_val_score(clf, X, y, cv=10)
scores_series = pd.Series(scores)
scores_mean = scores_series.mean()
print('scores_mean', scores_mean)

'''模型调优'''


# 参数选择 -max_depth
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return tr_score, cv_score


depths = range(2, 15)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))

# 画出决策树最大深度与评分之间的关系
plt.figure(figsize=(10, 6), dpi=75)
plt.grid()
plt.xlabel('max depth of decision tree')
plt.ylabel('score')
plt.plot(depths, cv_scores, '.g-', label='cross-validation score')
plt.plot(depths, tr_scores, '.r--', label='training score')
plt.legend()
plt.show()

'''调整划分准则'''


# 训练模型，并计算评分
def cv_score(val):
    clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return tr_score, cv_score


values = np.linspace(0, 0.005, 50)
scores = [cv_score(v) for v in values]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = values[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))

# 画出模型参数与模型评分的关系
plt.figure(figsize=(10, 6), dpi=75)
plt.grid()
plt.xlabel('threshold of entropy')
plt.ylabel('score')
plt.plot(values, cv_scores, '.g-', label='cross-validation score')
plt.plot(values, tr_scores, '.r--', label='training score')
plt.legend()
plt.show()

'''多参数联合考虑'''


# 画图代码，用于画出 gridsearch 方法中不同参数配置及评分
def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(10, 6), dpi=75)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '.--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


thresholds = np.linspace(0, 0.005, 50)
param_grid = {'min_impurity_decrease': thresholds}
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10, return_train_score=True)
clf.fit(X, y)
print("best param: {0}\nbest score: {1}".format(clf.best_params_, clf.best_score_))

plot_curve(thresholds, clf.cv_results_, xlabel='gini thresholds')

entropy_thresholds = np.linspace(0, 0.01, 50)
gini_thresholds = np.linspace(0, 0.005, 50)

param_grid = [{'criterion': ['entropy'],
               'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'],
               'min_impurity_decrease': gini_thresholds},
              {'max_depth': range(2, 10)},
              {'min_samples_split': range(2, 30, 2)}]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, return_train_score=True)
clf.fit(X, y)
print("best param: {0}\nbest score: {1}".format(clf.best_params_, clf.best_score_))
cv_result = pd.DataFrame.from_dict(clf.cv_results_)
with open('cv_result.csv', 'w') as f:
    cv_result.to_csv(f)

# 绘制决策树图形并保存
clf = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.0069387755102040816)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))

# 导出 students_dt_cut.dot 文件
with open("students_dt_cut.dot", 'w') as f:
    f = export_graphviz(clf, out_file=f)

# 导出为pdf
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("students_dt_cut.pdf")

'''预测结果分析'''
labels = data.columns.tolist()
labels = labels[:X.shape[1]]  # 更新 labels 使其长度与特征数量一致

# 特征重要性可视化
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6), dpi=75)
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [labels[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# 混淆矩阵可视化
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6), dpi=75)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 决策树可视化
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=labels,
                                class_names=['Class 0', 'Class 1'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
