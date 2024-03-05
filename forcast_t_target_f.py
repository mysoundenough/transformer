import pandas as pd
import torch
import torch.nn as nn
from numpy import * 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler # 标准化数据
import time
import copy
from data import readfile
from data import dealfile

# 导入网络模型
from unembed_transformer_net_f import *


my_font=font_manager.FontProperties(fname='/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/图表/仿宋_常规.ttf',size='large')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 计算核心
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# print(device)


# 数据读取
# 源温度数据
pathfile_source_true = '../../../数据/SG_T_YY/T_source_true.csv'
pathfile_target_true = '../../../数据/SG_T_YY/T_target_true.csv'

read01 = readfile(pathfile_source_true)
read02 = readfile(pathfile_target_true)

# 换一下 目标域
origin_data_test = read01.returndata()
origin_data_train = read02.returndata()

# 数据分析
# from data import dealfile
# deal01 = dealfile(origin_data_train01)
# deal01.dealdata()
# deal02 = dealfile(origin_data_test02)
# deal02.dealdata()

# 训练数据划分验证集比率
eval_rate= 0.01
end01 = math.floor((1-eval_rate)*origin_data_train.shape[0])

# 转化为numpy的array格式
# 训练数据
train_info = np.array(origin_data_train.iloc[0:end01, :], dtype = 'float32')
#验证数据
eval_info = np.array(origin_data_train.iloc[end01:-1, :], dtype = 'float32')
# 测试数据
test_info = np.array(origin_data_test.iloc[0:-1, :], dtype = 'float32')

# 数据形成source target batchs
# 批次化数据
def batchify(data, bsz):
    """
    将数据映射为连续的数字 处理数据转化为nbatch的形式
    """
    nbatch = data.shape[0] // bsz
    
    # 数据中除去多余余数部分
    data = data[:nbatch * bsz, :]
    
    # 使用view对data进行矩阵变化
    data = data.reshape(nbatch, -1, data.shape[1], order='C')
    # print(data.shape)
    data = torch.tensor(data)
    
    return data.to(device)

# 使用batch_ify来处理训练数据
# 训练集bsz
train_batch_size = 1
# 验证bsz
eval_batch_size = 1
# 测试bsz
test_batch_size = 1

# for _ in range(4): 训练数据 成 batch后合并
train_data = batchify(train_info, train_batch_size)
# 验证数据 成batch后合并
eval_data = batchify(eval_info, eval_batch_size)
# 测试数据 成batch
test_data = batchify(test_info, test_batch_size)

# 时序长度允许最大值为100
bptt = 100

def get_batch(source, i):
    seq_len = min(bptt, len(source) - i - 1)
    
    # 语言模型训练的源数据的是将batchify的结果的切片[i:i+seq_len]
    data = source[i:i+seq_len]
    
    # 根据语言模型的语料定义 目标数据是将源数据向后移动一位
    # 因为最后目标数据的切片会越界 因此使用view(-1)保证形状正常 array应该使用flat
    target = source[i+1:i+1+seq_len]
    # target = data
    return data, target


# 设置模型超参数 初始化模型
V = 35
V2 = 8

# 词嵌入大小为200
d_model = 8

# 前馈全连接的节点数
nhid = 200

# 编码器层数量
nlayers = 1

# 多头注意力机制
nhead = 2

# 置0比率
dropout = 0.2

model = make_model(V,V2,nlayers,d_model,nhid,nhead,dropout)

# 导入源域模型的训练参数
state_dict = torch.load('./Net/Transformer_Testmachine_sourceforcast_net_params.pkl', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# 将模型迁移到gpu
model.to(device)

# 模型初始化后 接下来损失函数与优化方法选择
# 损失函数 
# 我们使用nn自带的交叉熵损失 分类问题时采用
# criterion = nn.CrossEntropyLoss()
# MSELoss函数 计算输入和输出之差的平方 可输出序列标量或均方差
criterion1 = nn.MSELoss()
criterion2 = nn.SmoothL1Loss()    # 采用Huber损失函数


# 学习率初始值为5.0
lr = 5

# 优化器选择torch自带的SGD随机梯度下降方法 并把lr传入其中
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 定义学习率调整器 使用torch自带的lr_scheduler 将优化器传入
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

# 定义训练轮数
epochs = 100 # 100

# 训练 验证 测试
# 模型训练
def train():
    """
    train model
    """
    # 模型开启训练模式
    model.train()
    # 初始损失定义为0
    total_loss = 0
    total_loss_sum = 0
    # 模型mape计算
    
    # 获得当前时间
    start_time = time.time()
    # 开始遍历批次数据
    for batch, i in enumerate(range(0, train_data.shape[0] - 1, bptt)):
        # 通过get_batch
        data, targets = get_batch(train_data, i)
        # 获得mask
        source_mask = Variable(torch.ones(data.shape[1])).to(device)
        # source_mask = subsequent_mask(data.shape[1]).to(device)
        target_mask = subsequent_mask(targets.shape[1]).to(device)
        # 设置优化器初始为0梯度
        optimizer.zero_grad()
        
        # print("data", data.shape)
        # print("targets", targets.shape)
        
        # 将数据装入model得到输出
        output = model(data, targets, source_mask, target_mask)
        # 将输入与目标传入损失函数对象
        # print("output", output.shape)
        
        loss = criterion1(output.reshape(-1), targets.reshape(-1))
        # print(loss.shape, loss)
        
        # 损失进行反向传播得到损失总和
        loss.backward()
        # 使用nn自带的clip_grad_norm_进行梯度规范化 防止出现梯度消失或者爆炸
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 模型参数更新
        optimizer.step()
        # 损失累加得到总损失
        total_loss += loss.item()
        total_loss_sum += copy.deepcopy(total_loss)
        # 日志打印间隔为200
        log_interval = 40
        if batch % log_interval == 0 and batch > 0:
            # 平均损失为 总损失 / log_interval
            cur_loss = total_loss / log_interval
            # 需要时间 当前时间 - 起始时间
            elapsed = time.time() - start_time
            # 打印 轮数 当前批次 总批次 当前学习率 训练速度(每毫秒处理多少批次)
            # 平均损失 以及困惑度 困惑度是语言模型的重要标准 计算方法
            # 交叉熵平均损失取自然对数的底数
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | mape {:8.2f}'.format(
                      epoch, batch, len(train_data) // bptt, 
                      scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                      cur_loss, cur_loss))
            
            # total_loss_sum += copy.deepcopy(total_loss)
            # 每个批次结束后总损失归0
            total_loss = 0
            # 开始时间取当前时间
            start_time = time.time()
            
    return total_loss_sum

# 模型评估
def evaluate(eval_model, data_source):
    # 模型进入评估模式
    eval_model.eval()
    # 总损失
    total_loss = 0
    with torch.no_grad():
        # 与训练步骤基本一致
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            # 获得mask
            source_mask = Variable(torch.ones(data.shape[1])).to(device)
            # source_mask = subsequent_mask(data.shape[1]).to(device)
            target_mask = subsequent_mask(targets.shape[1]).to(device)
            # 设置优化器初始为0梯度
            output = eval_model(data, data, source_mask, target_mask)
            # 对输出形状进行扁平化 变为全部词汇的概率分布
            # output_flat = output.view(-1, data.shape[0] * data.shape[1] * data.shape[2])
            # 获得评估过程的总损失
            total_loss += criterion1(output.reshape(-1), targets.reshape(-1))
            total_loss += criterion2(output.reshape(-1), targets.reshape(-1))
    # 返回每轮总损失
    return total_loss

# 模型验证评估
# 首先初始化最佳验证损失 初始值为无穷大
best_val_loss = float("inf")

# 定义最佳模型训练变量 初始值为None
best_model = model

# 训练损失下降
all_total_train_loss_sum = []
all_total_eval_loss_sum = []

# 使用for循环遍历轮数
for epoch in range(1, epochs+1):
    # 首先获得轮数开始时间
    epoch_start_time = time.time()
    # 调用训练函数
    trainloss = train()
    all_total_train_loss_sum.append(trainloss)
    # 该轮训练后我们的模型参数已经发生了变化
    # 将模型和评估数据传入评估函数
    val_loss = evaluate(model, eval_data)
    all_total_eval_loss_sum.append(val_loss.cpu())
    # val_loss = 0
    # 之后打印每轮的评估日志 分别有轮数 耗时 验证损失 验证困惑度
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid mape {:2.2f}'.format(epoch, (time.time() - epoch_start_time), 
                                     val_loss, val_loss))
    print('-' * 89)
    # 我们将比较哪一轮损失最小 赋值给best_val_loss,
    # 并取该损失下模型的best_model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model= model
    # 每轮都会对优化方法的学习率做调整
    scheduler.step()

# 绘制训练损失下降过程
print('Finished Training')
plt.figure(figsize=(8, 2),dpi=400)
plt.title('Transformer-F源域训练损失下降过程',fontproperties=my_font) # 标题
plt.plot(range(epochs), all_total_train_loss_sum, color='blue', linestyle='-')
# plt.ylim(0, 1800000)
plt.legend(['train'])
plt.show()

# 绘制训练损失下降过程
print('Finished Training')
plt.figure(figsize=(8, 2),dpi=400)
plt.title('Transformer-F源域训练损失下降过程',fontproperties=my_font) # 标题
plt.plot(range(epochs), all_total_eval_loss_sum, color='red', linestyle='-')
# plt.ylim(0, 8000)
plt.legend(['eval'])
plt.show()

# 模型测试 依然使用evaluate函数 best_model以及测试数据
test_loss = evaluate(best_model, test_data)





# 打印测试日志 包括测试损失和困惑度
print('=' * 89)
print('| End of training | test loss {:5.2f} | mape {:8.2f}'.format(test_loss, test_loss))



# 模型保存
torch.save(best_model, './Net/Transformer_Testmachine_targetforcast_net.pkl')
# 只保存神经网络的模型参数
torch.save(best_model.state_dict(), './Net/Transformer_Testmachine_targetforcast_net_params.pkl')




# 绘制损失图
# 计算预测与真实温度之间损失，并输出
all_LSTM_MSE = []
all_LSTM_MAPE = []
for i in range(8):
    truepred_MSE = MSE(lstm_true_tempe[i], lstm_pred_tempe[i])
    truepred_MAPE = MAPE(lstm_true_tempe[i], lstm_pred_tempe[i])
    print('LSTM: 第{}温度，损失值：{}'.format(i+1, truepred_MSE))
    all_LSTM_MSE.append(truepred_MSE)
    all_LSTM_MAPE.append(truepred_MAPE)


# 输出各个指标真实值与预测值偏差（只输出8个指标）,表示性能
for i in range(8):
    plt.figure(figsize=(2, 8),dpi=200)
    # plt.title('LSTM: Temperure:{}, Loss:{}'.format(i+1, all_LSTM_MSE[i])) # 标题
    plt.title('LSTM: Temperure:{}'.format(i+1)) # 标题
    plt.plot(range(len(lstm_true_tempe[i])),lstm_true_tempe[i],'-D',linewidth=0.5,markersize=2,color='royalblue',zorder=1)
    #plt.plot(range(len(lstm_pred_tempe[i])), lstm_pred_tempe[i], '-D',linewidth=0.5,markersize=2,color='orangered',zorder=2)
    #plt.scatter(range(len(lstm_true_tempe[i])), lstm_true_tempe[i], s=50, color='royalblue')
    plt.scatter(range(len(lstm_pred_tempe[i])), lstm_pred_tempe[i], s=5, color='orangered', zorder=2)
    # plt.legend(['blue is true','orange is pred']) #注释
    plt.show()

