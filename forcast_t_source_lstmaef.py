# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:35:56 2022

@author: Mayuan
@email: 2971589431@qq.com
"""

# 功能简介：论文方法第3步，利用迁移学习，迁移CAE特征提取编码器到LSTM，用目标域微调lstm模型中预测层，预测温度

import torch
import torch.nn as nn
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 标准化数据
import pandas as pd
from matplotlib import font_manager
import seaborn as sns
import random
from collections import Counter # 过采样后查看不同标签个数
from imblearn.over_sampling import RandomOverSampler, SMOTE  # 随机采样函数 和SMOTE过采样函数
from sklearn.metrics import accuracy_score # 计算准确率，导入预测与真实标签
from sklearn.metrics import f1_score # 计算精确率，导入预测与真实标签
from sklearn.metrics import mean_squared_error as MSE # 计算预测与真实值之间的MSE损失
from sklearn.metrics import mean_absolute_percentage_error as MAPE # 计算预测与真实值之间的MAPE损失
import os
import time


my_font=font_manager.FontProperties(fname='/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/图表/仿宋_常规.ttf',size='large')
# my_font=font_manager.FontProperties(fname='G:\privacyTest\mayuan\仿宋_常规.ttf',size='large')
# plt.matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# # 使用GPU训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# 计算核心
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# print(device)




# 训练数据处理，输入：数据变化，输出：数据变化，自编码器处理
# 窑炉数据导入与预处理（处理为数据加载器）
# 导入数据
origin_data = pd.read_csv(r'../../../数据/SG_T_YY/T_source_true.csv', encoding='gbk')
# origin_data = pd.read_csv(r'G:\\privacyTest\\mayuan\\data\\Yaolu_CAETL-LSTM_source_data.csv', encoding='gbk')

# 只是选取温度数据
shape0 = origin_data.shape[0]-1

# 获取数据列名
# features = data.drop(labels=['index', '煤耗'], axis=1)
features = origin_data.columns[:].tolist()
# 数据逐列标准化（减去均值，除方差）（后续还原反之）
# lstm中的归一化会改变范围信息，需要训练与测试数据保持一致
# print(df[features[23]].values.reshape(-1,1)) # 将数据塑性成（行，列）
data = origin_data
for i in range(len(features)):
    data[features[i]] = StandardScaler().fit_transform(origin_data[features[i]].values.reshape(-1,1))    #标准化特征 (x-u)/s

# 将原始数据重新导入
# origin_data = pd.read_csv(r'G:\\privacyTest\\mayuan\\data\\Yaolu_CAETL-LSTM_source_data.csv', encoding='gbk')
origin_data = pd.read_csv(r'../../../数据/SG_T_YY/T_source_true.csv', encoding='gbk')


# 划分训练集与测试集
train_size = 0.9 # 0.8的训练集
# train_data = data.iloc[:train_size*len(data), :]
spiltpoint = int(train_size * shape0)
train_data = data.iloc[:spiltpoint, :]
test_data = data.iloc[spiltpoint:shape0, :]


# 建立数据标签，将数据转化为有监督的形式（舍弃前面去后面，逐步向后）
# 训练数据处理，输入：数据变化，标签：下一时刻温度


'''时序步长超参数'''
time_steps = 80 # 时序已知长度

# x_train = []
# y_train = []
# for i in range(len(train_data)-time_steps):
#     X_train = [train_data.iloc[i:i+time_steps, :].values.tolist()]
#     Y_train = train_data.iloc[i+time_steps, :].values
#     x_train.append(X_train)
#     y_train.append(Y_train)

# # print('训练数据', x_train) # 40*9,时间序列为40，特征为9个
# # print('训练标签', y_train) # 1*9,最后时间点特征为9个

# x_test = []
# y_test = []
# for i in range(len(test_data)-time_steps):
#     X_test = [test_data.iloc[i:i+time_steps, :].values.tolist()]
#     Y_test = test_data.iloc[i+time_steps, :].values
#     x_test.append(X_test)
#     y_test.append(Y_test)


x_train = []
y_train = []
for i in range(len(train_data)-time_steps):
    X_train = [train_data.iloc[i:i+time_steps, :].values.tolist()]
    Y_train = train_data.iloc[i+1:i+time_steps+1, :].values
    x_train.append(X_train)
    y_train.append(Y_train)

# print('训练数据', x_train) # 40*9,时间序列为40，特征为9个
# print('训练标签', y_train) # 1*9,最后时间点特征为9个

x_test = []
y_test = []
for i in range(len(test_data)-time_steps):
    X_test = [test_data.iloc[i:i+time_steps, :].values.tolist()]
    Y_test = test_data.iloc[i+1:i+time_steps+1, :].values
    x_test.append(X_test)
    y_test.append(Y_test)


# print('测试数据', x_train) # 40*9,时间序列为40，特征为9个
# print('测试标签', y_train) # 1*9,最后时间点特征为9个


# 将数据处理为张量
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train) # 由于是做回归，于是将标签不是long。分类时标签才是long
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)

# print('x_train:', x_train)
# print('y_train:', y_train)

# 建立测试与训练数据加载器

'''BATCH_SIZE超参数(决定内存占用大小)'''
BATCH_SIZE = 64
bptt = BATCH_SIZE

# train_set = TensorDataset(x_train, y_train)
# train_loader = DataLoader(dataset=train_set,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True)

# test_set = TensorDataset(x_test, y_test)
# test_loader = DataLoader(dataset=test_set,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True)

train_set = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test_set,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

# 查看一批训练数据
dataiter = iter(test_loader)
# inputs, labels = r.nextdataite() in py3.8 torchxxx
inputs, labels = next(dataiter)  # in py3.9 torch2.0.1
inputs = inputs.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

# def get_batch(source, i):
#     seq_len = min(bptt, len(source) - i - 1)
    
#     # 语言模型训练的源数据的是将batchify的结果的切片[i:i+seq_len]
#     data = source[i:i+seq_len]
    
#     # 根据语言模型的语料定义 目标数据是将源数据向后移动一位
#     # 因为最后目标数据的切片会越界 因此使用view(-1)保证形状正常 array应该使用flat
#     target = source[i+1:i+1+seq_len]
#     # target = data
#     return data, target


# 模型训练参数
num_epochs = 200  # 选用50个epoch  # 100
learning_rate = 0.001

# 网络参数设计
input_size = 8 # 每个时间点的特征有9个，时间步长为10，还需要再去设定

'''网络超参数'''
hidden1_size = 64 # rnn 隐藏单元数
hidden2_size = 32 # rnn 隐藏单元数
num_layers = 3 # rnn 层数

"""
10 0.0001
LSTM: 第1温度，损失值：67.2477724388743
LSTM: 第2温度，损失值：27.621766715877197
LSTM: 第3温度，损失值：34.09846985162334
LSTM: 第4温度，损失值：33.807306814055245
LSTM: 第5温度，损失值：691.8933914762622
LSTM: 第6温度，损失值：23.193881496895294
LSTM: 第7温度，损失值：712.9574577742392
LSTM: 第8温度，损失值：1398.0371678219954
"""

# 定义CAE网络结构，由于需要迁移到LSTM中，因此编码器采用LSTM结构，解码器采用全连接，之后迁移到
encoding_dim = 10
input_dim = x_train[1][0].shape # 输入维度
print('输入特征维度：',input_dim)
class LSTM_RNN(nn.Module):
    
    '''lstm层结构'''
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden1_size,   # rnn 隐藏单元数
                            num_layers=num_layers,     # rnn 层数
                            batch_first=True, # If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: False
                            )
        self.output_layer1 = nn.Linear(in_features=hidden1_size, out_features=hidden2_size) # 隐藏层与线性层连接需相等
        self.output_layer2 = nn.Linear(in_features=hidden2_size, out_features=input_size)            # 多加几层全连接，网络性能会提升，哈哈哈哈

    '''前向传播'''
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # lstm_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        
        lstm_out, (h_n, h_c) = self.lstm(x, None)   # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        layers1_out = self.output_layer1(lstm_out[:, :, :])   # 选择最后时刻lstm的输出
        output = self.output_layer2(layers1_out)

        return output


# 1、使用纯净LSTM模型
model = LSTM_RNN()

model.to(device)
# print(model.buffers)

# 迁移采用源域数据训练好的lstmAE中的编码器部分
# 调整模型为预测模型（固定自编码器层，添加新的全连接与预测层）
# 固定编码器部分模型参数，不参与后续训练（√）
# for param in model.parameters():
#     param.requires_grad_(False)


# 解码器进行更新：变更为预测解码器，增加预测层
# 1、添加新的预测层（√）
# 2、更改前向传播过程 （√）

model.output_layer1 = nn.Linear(in_features=hidden1_size, out_features=hidden2_size)
model.output_layer2 = nn.Linear(in_features=hidden2_size, out_features=input_size)
model.to(device)

# print("迁移LSTMAE解码器以改为预测层")
# print(model.buffers)
# for i in model.parameters():
#     print(i)


time_start = time.time()


# 采用源域训练数据训练LSTMAE
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
# 更改为Huberloss
loss_f = nn.SmoothL1Loss()    # 采用Huber损失函数
loss_function = nn.MSELoss()  # 采用均方差损失函数
all_lstmf_loss = []
for epoch in range(num_epochs):
    total_loss = 0
    for batch, (x,y) in enumerate(train_loader):
    # for batch, i in enumerate(range(0, x_train.shape[0] - 1, bptt)):
        # 通过get_batch
        # x, y = get_batch(x_train, i)
        x =  x.to(device)
        y =  y.to(device)
        # 训练数据每批更换形状
        x = x.view(-1, time_steps, input_size)
        # print('x', x.shape)
        y_forecast = model(x)
        # print('y_forecast', y_forecast.shape)
        # print('y', y.shape)
        loss = loss_function(y_forecast, y)
        # loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        # print(step)
    total_loss /= len(train_set)
    all_lstmf_loss.append(total_loss)
    print('Epoch {}/{} : loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))


# 绘制训练损失下降过程
print('Finished Training')
plt.figure(figsize=(8, 2),dpi=400)
plt.title('LSTM-F源域训练损失下降过程',fontproperties=my_font) # 标题
plt.plot(range(num_epochs), all_lstmf_loss, color='blue', linestyle='-')
plt.legend(['train'])
plt.show()

#　time_start = time.time()
time_end = time.time()
print('-------Time:{}-------'.format(time_end - time_start))


# 导入测试集数据，进行模型损失测试，查看lstm模型回归预测性能

# 需要反标准化，还原回原来的温度
var_data = np.var(origin_data, axis=0)  # 计算每一列方差
std_data = np.std(origin_data, axis = 0)  #计算每一列标准差
mean_data = np.mean(origin_data, axis=0)  # 计算每一列均值

# 1、采用计算损失评估性能,输出预测与真实值对比
test_loss = []
lstm_pred_tempe = [[] for x in range(0,8)]
lstm_true_tempe = [[] for x in range(0,8)]
# all_MSE = [[]for i in range(14)]
for batch, (batch_x, batch_y) in enumerate(test_loader):
# for batch, i in enumerate(range(0, x_train.shape[0] - 1, bptt)):
    # 通过get_batch
    # batch_x, batch_y = get_batch(x_train, i)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    batch_x = batch_x.view(-1, time_steps, input_size)  # 将测试数据每批换形状
    output = model(batch_x)
    # 每个batch_size=64，这里每次输出64个，计算MSE，并打印效果共（0.2*43000-步长/batch_size个） 
    # print('output', output.shape)
    pred = output
    true = batch_y
    # print('true', true.shape)
    # pred = pred
    # true = true.cpu().detach().numpy()
    # 每个i=num_temperture,最后一批为剩下的
    # for i in range(pred.shape[1]):
    #     # 选取其中i列温度值
    #     tempi_pred = pred[:, i]
    #     tempi_true = true[:, i]
    #     # 将i列温度还原回标准化前的值
    #     tempi_pred = (tempi_pred * std_data[i]) + mean_data[i]
    #     tempi_true = (tempi_true * std_data[i]) + mean_data[i]
    #     # 保存绘制真实预测温度图的数据
    #     lstm_true_tempe[i].extend(tempi_true)
    #     lstm_pred_tempe[i].extend(tempi_pred)
    # print('pred', pred.shape)
    for n_t in range(pred.shape[2]):
        # 选取其中i列温度值
        tempi_pred = pred[:, 0, n_t]
        tempi_true = true[:, 0, n_t]
        # print('tempi_pred', tempi_pred.shape)
        # print('tempi_true', tempi_true.shape)
        # # 将i列温度还原回标准化前的值
        tempi_pred = (tempi_pred * std_data[n_t]) + mean_data[n_t]
        tempi_true = (tempi_true * std_data[n_t]) + mean_data[n_t]
        tempi_pred = tempi_pred.cpu().detach().numpy().tolist()
        tempi_true = tempi_true.cpu().detach().numpy().tolist()
        # 保存绘制真实预测温度图的数据
        lstm_true_tempe[n_t].extend(tempi_true)
        lstm_pred_tempe[n_t].extend(tempi_pred)
        
    # if (batch + 1) % 10 == 0:
        # print('\n')
    loss = loss_function(output, batch_y)
    test_loss.append(loss.item() * len(batch_x))  # 损失相加保存，以绘制测试集损失曲线

print('Finished Testing')
plt.figure(figsize=(8, 2),dpi=400)
plt.title('LSTM-F源域验证集损失下降过程',fontproperties=my_font) # 标题
plt.plot(range(len(test_loss)), test_loss, color='red', linestyle='-')
plt.legend(['eval'])
plt.show()



# 计算预测与真实温度之间损失，并输出
all_LSTM_MSE = []
all_LSTM_MAPE = []
for i in range(8):
    truepred_MSE = MSE(lstm_true_tempe[i], lstm_pred_tempe[i])
    truepred_MAPE = MAPE(lstm_true_tempe[i], lstm_pred_tempe[i])
    print('LSTM: 第{}温度，损失值：{}'.format(i+1, truepred_MSE))
    print('LSTM: 第{}温度，损失值：{}'.format(i+1, all_LSTM_MAPE))
    all_LSTM_MSE.append(truepred_MSE)
    all_LSTM_MAPE.append(truepred_MAPE)


# 输出各个指标真实值与预测值偏差（只输出8个指标）,表示性能
for i in range(8):
    plt.figure(figsize=(8, 2),dpi=200)
    # plt.title('LSTM: Temperure:{}, Loss:{}'.format(i+1, all_LSTM_MSE[i])) # 标题
    plt.title('LSTM:  :{}'.format(i+1)) # 标题
    plt.plot(range(len(lstm_true_tempe[i])),lstm_true_tempe[i],'-D',linewidth=0.5,markersize=2,color='royalblue',zorder=1)
    #plt.plot(range(len(lstm_pred_tempe[i])), lstm_pred_tempe[i], '-D',linewidth=0.5,markersize=2,color='orangered',zorder=2)
    #plt.scatter(range(len(lstm_true_tempe[i])), lstm_true_tempe[i], s=50, color='royalblue')
    plt.scatter(range(len(lstm_pred_tempe[i])), lstm_pred_tempe[i], s=5, color='orangered', zorder=2)
    # plt.legend(['blue is true','orange is pred']) #注释
    # plt.xlim(0, 256)
    plt.show()
    # plt.savefig('G:\\privacyTest\\mayuan\\image\\lstm_t{}.jpg'.format(i+1))

# 输出各个指标真实值与预测值准确率（只输出8个指标）,表示模型准确性
for i in range(0,8):
    plt.figure(figsize=(10, 5),dpi=800)
    plt.plot(range(len(lstm_true_tempe[i])),lstm_true_tempe[i],'--d',linewidth=1,markersize=3,color='royalblue',zorder=1)
    plt.scatter(range(len(lstm_pred_tempe[i])), lstm_pred_tempe[i], s=5, marker='o', color='orangered', zorder=2)
    plt.vlines(list(range(0,len(lstm_pred_tempe[i]))), lstm_true_tempe[i], lstm_pred_tempe[i],linewidth=0.1,color='yellow', zorder=3)
    plt.legend(['True','Forecast','Bias'],loc="upper right") #注释
    plt.title('LSTM源域验证集温度T{}预测效果:'.format(i+1),fontproperties=my_font) # 标题
    plt.xlabel('时间',fontproperties=my_font)
    plt.ylabel('温度',fontproperties=my_font)
    plt.show()



# # 导出模型进行迁移学习
# torch.save(model, 'G:\\privacyTest\\mayuan\\Net\\LSTM_TL_YaoLU_sourceforcast_net.pkl')
# # 只保存神经网络的模型参数
# torch.save(model.state_dict(), 'G:\\privacyTest\\mayuan\\Net\\LSTM_TL_YaoLU_sourceforcast_net_params.pkl')


best_model = model
# 模型保存
torch.save(best_model, './Net/LSTM_Testmachine_sourceforcast_net.pkl')
# 只保存神经网络的模型参数
torch.save(best_model.state_dict(), './Net/LSTM_Testmachine_sourceforcast_net_params.pkl')







