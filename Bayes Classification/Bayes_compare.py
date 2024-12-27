import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# hide warnings
warnings.filterwarnings("ignore")

# 加载20Newsgroups数据集
data_home = "C:\\Users\\胡绍星\\scikit_learn_data"
newsgroups_data = fetch_20newsgroups(subset='train', data_home=data_home)
X = newsgroups_data.data
y = newsgroups_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用TF-IDF特征提取
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

x_train_dense = X_train_tfidf.toarray()
x_test_dense = X_test_tfidf.toarray()

# GaussianNB
print("\nTraining GaussianNB...")
gnb = GaussianNB()
gnb.fit(x_train_dense, y_train)
y_pred_gnb = gnb.predict(x_test_dense)
print("GaussianNB Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
print("Classification Report:\n", classification_report(y_test, y_pred_gnb))

# 获取GaussianNB的混淆矩阵并可视化，使其居中显示
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
plt.figure(figsize=(10, 8))
# 创建子图，并设置子图参数让其居中
ax = plt.subplot(111)
ax.set_aspect('equal')
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', xticklabels=newsgroups_data.target_names,
            yticklabels=newsgroups_data.target_names, cbar_kws={'shrink': 0.8}, square=True,
            center=np.max(cm_gnb) / 2)
plt.title('GaussianNB Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 获取GaussianNB的评估指标
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
report_gnb = classification_report(y_test, y_pred_gnb, output_dict=True)
precision_gnb = report_gnb['macro avg']['precision']
recall_gnb = report_gnb['macro avg']['recall']
f1_gnb = report_gnb['macro avg']['f1-score']

# MultinomialNB
print("\nTraining MultinomialNB...")
mnb = MultinomialNB(alpha=1.0)  # 平滑参数alpha=1.0
mnb.fit(X_train_tfidf, y_train)
y_pred_mnb = mnb.predict(X_test_tfidf)
print("MultinomialNB Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_mnb))
print("Classification Report:", classification_report(y_test, y_pred_mnb))

# 获取MultinomialNB的混淆矩阵并可视化，使其居中显示
cm_mnb = confusion_matrix(y_test, y_pred_mnb)
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
ax.set_aspect('equal')
sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Greens', xticklabels=newsgroups_data.target_names,
            yticklabels=newsgroups_data.target_names, cbar_kws={'shrink': 0.8}, square=True,
            center=np.max(cm_mnb) / 2)
plt.title('MultinomialNB Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 获取MultinomialNB的评估指标
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
report_mnb = classification_report(y_test, y_pred_mnb, output_dict=True)
precision_mnb = report_mnb['macro avg']['precision']
recall_mnb = report_mnb['macro avg']['recall']
f1_mnb = report_mnb['macro avg']['f1-score']

# ComplementNB
print("\nTraining ComplementNB...")
cnb = ComplementNB(alpha=1.0)  # 平滑参数alpha=1.0
cnb.fit(X_train_tfidf, y_train)
y_pred_cnb = cnb.predict(X_test_tfidf)
print("ComplementNB Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_cnb))
print("Classification Report:", classification_report(y_test, y_pred_cnb))

# 获取ComplementNB的混淆矩阵并可视化，使其居中显示
cm_cnb = confusion_matrix(y_test, y_pred_cnb)
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
ax.set_aspect('equal')
sns.heatmap(cm_cnb, annot=True, fmt='d', cmap='Oranges', xticklabels=newsgroups_data.target_names,
            yticklabels=newsgroups_data.target_names, cbar_kws={'shrink': 0.8}, square=True,
            center=np.max(cm_cnb) / 2)
plt.title('ComplementNB Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 获取ComplementNB的评估指标
accuracy_cnb = accuracy_score(y_test, y_pred_cnb)
report_cnb = classification_report(y_test, y_pred_cnb, output_dict=True)
precision_cnb = report_cnb['macro avg']['precision']
recall_cnb = report_cnb['macro avg']['recall']
f1_cnb = report_cnb['macro avg']['f1-score']

# BernoulliNB
print("\nTraining BernoulliNB...")
bnb = BernoulliNB(alpha=1.0)  # 平滑参数alpha=1.0
bnb.fit(X_train_tfidf, y_train)
y_pred_bnb = bnb.predict(X_test_tfidf)
print("BernoulliNB Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_bnb))
print("Classification Report:", classification_report(y_test, y_pred_bnb))

# 获取BernoulliNB的混淆矩阵并可视化，使其居中显示
cm_bnb = confusion_matrix(y_test, y_pred_bnb)
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
ax.set_aspect('equal')
sns.heatmap(cm_bnb, annot=True, fmt='d', cmap='Reds', xticklabels=newsgroups_data.target_names,
            yticklabels=newsgroups_data.target_names, cbar_kws={'shrink': 0.8}, square=True,
            center=np.max(cm_bnb) / 2)
plt.title('BernoulliNB Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 获取BernoulliNB的评估指标
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
report_bnb = classification_report(y_test, y_pred_bnb, output_dict=True)
precision_bnb = report_bnb['macro avg']['precision']
recall_bnb = report_bnb['macro avg']['recall']
f1_bnb = report_bnb['macro avg']['f1-score']

# 创建DataFrame来整理结果
results = pd.DataFrame({
    'Classifier': ['GaussianNB', 'MultinomialNB', 'ComplementNB', 'BernoulliNB'],
    'Accuracy': [accuracy_gnb, accuracy_mnb, accuracy_cnb, accuracy_bnb],
    'Precision': [precision_gnb, precision_mnb, precision_cnb, precision_bnb],
    'Recall': [recall_gnb, recall_mnb, recall_cnb, recall_bnb],
    'F1-score': [f1_gnb, f1_mnb, f1_cnb, f1_bnb]
})

# 设置显示格式，保留三位小数
pd.options.display.float_format = '{:.3f}'.format

# 打印表格形式的结果
print(results)

# 可视化部分：将数据从宽格式转换为长格式，方便seaborn绘图
results_melted = results.melt(id_vars=['Classifier'], var_name='Metric', value_name='Value')

# 设置seaborn绘图风格
sns.set(style="whitegrid")

# 创建可视化图表，绘制柱状图对比不同分类器在各指标上的表现
g = sns.catplot(data=results_melted, x='Classifier', y='Value', hue='Metric', kind='bar', palette='viridis',
                height=6, aspect=1.5)
g.despine(left=True)
g.set_axis_labels("Classifier", "Value")
g.set_titles("{col_name}")
plt.show()
