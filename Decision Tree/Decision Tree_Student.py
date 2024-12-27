import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.tree import export_graphviz
from sklearn import tree
from imblearn.over_sampling import SMOTE
import seaborn as sns
import pydotplus

# 数据读取
data = pd.read_csv('student_data_DT2.csv')
# print(data)

# 丢弃无用的数据 UserId, term
data.drop(['UserId', 'term'], axis=1, inplace=True)
# print(data.head(20))

# 处理totalLearningTime缺失值,用均值填充
table = data[data['totalLearningTime'].isnull().values == False]
score_mean = table['totalLearningTime'].mean()
table = data['totalLearningTime'].fillna(value=score_mean)
data['totalLearningTime'] = table
# print(data.head(20))

# 划分数据集
y = data['isPassExam']  # 获取标签值(label)
x = data.drop(['isPassExam'], axis=1).values  # 获取特征值(feature)

# 处理数据不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x, y)


# 划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1)
print('train dataset:{0}, test dataset:{1}'.format(X_train.shape, X_test.shape))

# 构建模型
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=10)

# 训练模型和评估模型
model.fit(X_train, y_train)    # 拟合模型

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predict)
test_accuracy = accuracy_score(y_test, test_predict)
print('train accuracy:{0}, test accuracy:{1}'.format(train_accuracy, test_accuracy))

train_recall = recall_score(y_train, train_predict)
test_recall = recall_score(y_test, test_predict)
print('train recall:{0}, test recall:{1}'.format(train_recall, test_recall))

# 将决策树导出为.dot文件
with open("students_dt.dot", 'w') as f:
    f = export_graphviz(model, out_file=f)

# 或者也可以导出为pdf
dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("students_dt.pdf")
