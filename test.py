import torch
import torch.nn.functional as F
import torch.nn as nn
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:57:00 2024

@author: mayuan
"""
# 分类模型标准
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))


# 构建模型
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

# 编译模型
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
"""

""" 
# softmax 与 logsoftmax
a = [[-1,0,0.5,0.5], [-1,-1,0.5,0.5]]
data = torch.tensor(a)
# data=torch.randn(2,5)
print(data)

softmax= F.softmax(data,dim=-1)
print(softmax)

log_soft1=torch.log(softmax)
print(log_soft1)
"""


batch_size = 5
nb_classes = 2
in_features = 10

model = nn.Linear(in_features, nb_classes)
criterion = nn.CrossEntropyLoss()

x = torch.randn(batch_size, in_features)
target = torch.tensor([1,1,1,1,1])
target = target.long()
# target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
print(target.shape)
print(target[0])

output = model(x)
print(output.shape)
print(output[0])
loss = criterion(output, target)
loss.backward()