import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import seaborn as sns

warnings.filterwarnings('ignore')

# 读取数据，确保正确解析列
column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
data = pd.read_csv('wine.data', names=column_names)

# 数据预处理
data_num = data.shape[0]  # 数据条数
index = np.random.permutation(data_num)  # 打乱索引

# 随机划分数据集
train_index = index[:int(data_num * 0.6)]  # 训练集索引
val_index = index[int(data_num * 0.6):int(data_num * 0.8)]  # 验证集索引
test_index = index[int(data_num * 0.8):]  # 测试集索引

# 划分数据集
train_data = data.loc[train_index]
val_data = data.loc[val_index]
test_data = data.loc[test_index]

# 计算训练集的均值和标准差，归一化
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

# 转化为 numpy array
x_train = np.array(train_data.iloc[:, 1:])
y_train = np.array(train_data.iloc[:, 0])
x_val = np.array(val_data.iloc[:, 1:])
y_val = np.array(val_data.iloc[:, 0])
x_test = np.array(test_data.iloc[:, 1:])
y_test = np.array(test_data.iloc[:, 0])

# 确保标签在有效范围内 [0, 1, 2]，并且没有 -1
y_train = np.clip(y_train, 0, 2)
y_val = np.clip(y_val, 0, 2)
y_test = np.clip(y_test, 0, 2)

# 使用 TensorFlow 构建模型
model = keras.Sequential([
    keras.Input(shape=(13,)),  # 13个特征输入
    layers.Dense(64, activation='relu'),  # 第一层 64 个神经元
    layers.Dense(32, activation='relu'),  # 第二层 32 个神经元
    layers.Dense(3, activation='softmax')  # 输出层，3 个神经元，适用于三分类任务
])

# 编译模型：使用 Adam 优化器和稀疏分类交叉熵损失函数
model.compile(
    keras.optimizers.Adam(learning_rate=0.001),  # 使用 Adam 优化器，学习率为 0.001
    loss=keras.losses.SparseCategoricalCrossentropy(),  # 稀疏分类交叉熵损失函数
    metrics=[keras.metrics.SparseCategoricalAccuracy()]  # 使用 SparseCategoricalAccuracy 作为评估指标
)

# 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=64,  # 批次大小为 64
    epochs=500,  # 训练 500 个周期
    validation_data=(x_val, y_val),  # 在每个 epoch 后评估验证集
)

# 训练损失曲线
fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

# 损失曲线
ax[0].plot(history.history['loss'], label='Train Loss', color='blue')
ax[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].set_title('Train vs Validation Loss')

# 准确率曲线
ax[1].plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy', color='blue')
ax[1].plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy', color='red')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].set_title('Train vs Validation Accuracy')

plt.tight_layout()
plt.savefig('train_vs_validation_curve.png', dpi=100)  # 保存为PNG图像
plt.show()

# 评估模型在测试集上的性能
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 绘制模型预测概率的分布图
y_prob = model.predict(x_test)
plt.figure(figsize=(6, 4), dpi=100)
plt.hist(np.max(y_prob, axis=1), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Maximum Prediction Probability')
plt.xlabel('Maximum Prediction Probability')
plt.ylabel('Frequency')
plt.savefig('prediction_probability_distribution.png', dpi=100)  # 保存为PNG图像
plt.show()
