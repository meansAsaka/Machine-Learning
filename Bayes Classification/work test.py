from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, normalize=False, class_names=None):
    # 归一化混淆矩阵（如果需要）
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘制热力图
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


data_home = "C:\\Users\\胡绍星\\scikit_learn_data"
documents = fetch_20newsgroups(subset='train', data_home=data_home)

# 使用 CountVectorizer 将文本转换为词频矩阵
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents.data)
y = documents.target

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# # 使用 GaussianNB
# clf1 = GaussianNB()
# clf1.fit(X_train.toarray(), y_train)  # 转换为数组，因为 GaussianNB 只接受稠密数据
# y_pred1 = clf1.predict(X_test.toarray())
# acc1 = accuracy_score(y_test, y_pred1)
# precision1 = precision_score(y_test, y_pred1, average='macro')
# recall1 = recall_score(y_test, y_pred1, average='macro')
# f1_1 = f1_score(y_test, y_pred1, average='macro')
# cm1 = confusion_matrix(y_test, y_pred1)
# # 评估模型
# print("GaussianNB:")
# print("accuracy:", acc1)
# print("precision:", precision1)
# print("recall:", recall1)
# print("f1:", f1_1)
# plot_confusion_matrix(cm1)


# # 使用 MultinomialNB
# clf2 = MultinomialNB()
# clf2.fit(X_train, y_train)  # X_train 已经是稀疏矩阵
# y_pred2 = clf2.predict(X_test)
# acc2 = accuracy_score(y_test, y_pred2)
# precision2 = precision_score(y_test, y_pred2, average='macro')
# recall2 = recall_score(y_test, y_pred2, average='macro')
# f1_2 = f1_score(y_test, y_pred2, average='macro')
# cm2 = confusion_matrix(y_test, y_pred2)
# # 评估模型
# print("MultinomialNB:")
# print("accuracy:", acc2)
# print("precision:", precision2)
# print("recall:", recall2)
# print("f1:", f1_2)
# plot_confusion_matrix(cm2)


# 使用 ComplementNB
clf3 = ComplementNB()
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
acc3 = accuracy_score(y_test, y_pred3)
precision3 = precision_score(y_test, y_pred3, average='macro')
recall3 = recall_score(y_test, y_pred3, average='macro')
f1_3 = f1_score(y_test, y_pred3, average='macro')
cm3 = confusion_matrix(y_test, y_pred3)
# 评估模型
print("ComplementNB:")
print("accuracy:", acc3)
print("precision:", precision3)
print("recall:", recall3)
print("f1:", f1_3)
plot_confusion_matrix(cm3)

# # 使用 BernoulliNB
# vectorizer = CountVectorizer(binary=True)  # 设置为二元特征（词是否出现）
# X = vectorizer.fit_transform(documents.data)
#
# # 切分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#
# # 使用 BernoulliNB
# clf4 = BernoulliNB()
# clf4.fit(X_train, y_train)
# y_pred4 = clf4.predict(X_test)
# acc4 = accuracy_score(y_test, y_pred4)
# precision4 = precision_score(y_test, y_pred4, average='macro')
# recall4 = recall_score(y_test, y_pred4, average='macro')
# f1_4 = f1_score(y_test, y_pred4, average='macro')
# cm4 = confusion_matrix(y_test, y_pred4)
# # 评估模型
# print("BernoulliNB:")
# print("accuracy:", acc4)
# print("precision:", precision4)
# print("recall:", recall4)
# print("f1:", f1_4)
# plot_confusion_matrix(cm4)
