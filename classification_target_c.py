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

from data import readfile
from data import dealfile

# 导入网络模型
from unembed_transformer_net_c import *

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 计算核心
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


# 数据读取
# 源数据
# pathfile_test01 = '../../../数据/19.FMCRD_Data/test_load0_1e_m15_200x5.csv'
# pathfile_test02 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5HI.csv'
# pathfile_test03 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5LO.csv'
# pathfile_test04 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5MED.csv'
# pathfile_train01 = '../../../数据/19.FMCRD_Data/train_load0_1e_m15_200x5.csv'
# pathfile_train02 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5HI.csv'
# pathfile_train03 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5LO.csv'
# pathfile_train04 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5MED.csv'
# undersampling10
pathfile_test01 = '../../../数据/19.FMCRD_Data/test_load0_1e_m15_200x5_undersampling100.csv'
pathfile_test02 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5HI_undersampling100.csv'
pathfile_test03 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5LO_undersampling100.csv'
pathfile_test04 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5MED_undersampling100.csv'
pathfile_train01 = '../../../数据/19.FMCRD_Data/target_data_01.csv'
pathfile_train02 = '../../../数据/19.FMCRD_Data/target_data_02.csv'
pathfile_train03 = '../../../数据/19.FMCRD_Data/target_data_03.csv'
pathfile_train04 = '../../../数据/19.FMCRD_Data/target_data_04.csv'
# pathfile_train01 = '../../../数据/19.FMCRD_Data/train_load0_1e_m15_200x5_undersampling100.csv'
# pathfile_train02 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5HI_undersampling100.csv'
# pathfile_train03 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5LO_undersampling100.csv'
# pathfile_train04 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5MED_undersampling100.csv'


read01 = readfile(pathfile_test01)
read02 = readfile(pathfile_test02)
read03 = readfile(pathfile_test03)
read04 = readfile(pathfile_test04)
read05 = readfile(pathfile_train01)
read06 = readfile(pathfile_train02)
read07 = readfile(pathfile_train03)
read08 = readfile(pathfile_train04)
origin_data_test01 = read01.returndata()
origin_data_test02 = read02.returndata()
origin_data_test03 = read03.returndata()
origin_data_test04 = read04.returndata()
origin_data_train01 = read05.returndata()
origin_data_train02 = read06.returndata()
origin_data_train03 = read07.returndata()
origin_data_train04 = read08.returndata()
# print("origin_data_test01", origin_data_test01)
# print("origin_data_test02", origin_data_test02)
# print("origin_data_test03", origin_data_test03)
# print("origin_data_test04", origin_data_test04)
# print("origin_data_train01", origin_data_train01)
# print("origin_data_train02", origin_data_train02)
# print("origin_data_train03", origin_data_train03)
# print("origin_data_train04", origin_data_train04)

# 数据分析
# from data import dealfile
# deal01 = dealfile(origin_data_train01)
# deal01.dealdata()
# deal02 = dealfile(origin_data_test02)
# deal02.dealdata()
# deal03 = dealfile(origin_data_test03)
# deal03.dealdata()
# deal04 = dealfile(origin_data_test04)
# deal04.dealdata()

# pd concat
# train_info_00 = pd.concat([origin_data_train01,origin_data_train02,origin_data_train03,origin_data_train04])
test_info_00 = pd.concat([origin_data_test01,origin_data_test02,origin_data_test03,origin_data_test04])


# 转化为numpy的array格式
# train_info_01 = np.array(origin_data_train01.iloc[0:-1:100, 0:-1], dtype = 'float32')
# train_info_02 = np.array(origin_data_train02.iloc[0:-1:100, 0:-1], dtype = 'float32')
# train_info_03 = np.array(origin_data_train03.iloc[0:-1:100, 0:-1], dtype = 'float32')
# train_info_04 = np.array(origin_data_train04.iloc[0:-1:100, 0:-1], dtype = 'float32')


# 数据均衡操作
# 对少量的样本做过采样操作
origin_data_train03 = pd.concat([origin_data_train03,origin_data_train03,origin_data_train03,origin_data_train03])
origin_data_train03 = origin_data_train03.iloc[0:origin_data_train01.shape[0], :]


# 训练数据划分验证集比率
eval_rate= 0.1
# 训练数据
end01 = math.floor((1-eval_rate)*origin_data_train01.shape[0])
train_info01 = np.array(origin_data_train01.iloc[0:end01:1, 0:-1], dtype = 'float32')
end02 = math.floor((1-eval_rate)*origin_data_train02.shape[0])
train_info02 = np.array(origin_data_train02.iloc[0:end02:1, 0:-1], dtype = 'float32')
end03 = math.floor((1-eval_rate)*origin_data_train03.shape[0])
train_info03 = np.array(origin_data_train03.iloc[0:end03:1, 0:-1], dtype = 'float32')
end04 = math.floor((1-eval_rate)*origin_data_train04.shape[0])
train_info04 = np.array(origin_data_train04.iloc[0:end04:1, 0:-1], dtype = 'float32')
# 训练数据标签
train_label01 = np.zeros(origin_data_train01.iloc[0:end01:1, -1].shape[0], dtype = 'long')
train_label02 = np.ones(origin_data_train02.iloc[0:end02:1, -1].shape[0], dtype = 'long')
train_label03 = 2 * np.ones(origin_data_train03.iloc[0:end03:1, -1].shape[0], dtype = 'long')
train_label04 = 3 * np.ones(origin_data_train04.iloc[0:end04:1, -1].shape[0], dtype = 'long')

# 验证数据
eval_info01 = np.array(origin_data_train01.iloc[end01:-1, 0:-1], dtype = 'float32')
eval_info02 = np.array(origin_data_train02.iloc[end02:-1, 0:-1], dtype = 'float32')
eval_info03 = np.array(origin_data_train03.iloc[end03:-1, 0:-1], dtype = 'float32')
eval_info04 = np.array(origin_data_train04.iloc[end04:-1, 0:-1], dtype = 'float32')
# 验证数据标签
eval_label01 = np.zeros(origin_data_train01.iloc[end01:-1, -1].shape[0], dtype = 'long')
eval_label02 = np.ones(origin_data_train02.iloc[end02:-1, -1].shape[0], dtype = 'long')
eval_label03 = 2 * np.ones(origin_data_train03.iloc[end03:-1, -1].shape[0], dtype = 'long')
eval_label04 = 3 * np.ones(origin_data_train04.iloc[end04:-1, -1].shape[0], dtype = 'long')

# 测试数据
test_info = np.array(test_info_00.iloc[0:-1:1, 0:-1], dtype = 'float32')
# 测试数据标签
test_label01 = np.zeros(origin_data_test01.shape[0], dtype = 'long')
test_label02 = np.ones(origin_data_test02.shape[0], dtype = 'long')
test_label03 = 2 * np.ones(origin_data_test03.shape[0], dtype = 'long')
test_label04 = 3 * np.ones(origin_data_test04.shape[0], dtype = 'long')


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

def label_batchify(data, bsz):
    """
    将数据映射为连续的数字 处理数据转化为nbatch的形式
    """
    nbatch = data.shape[0] // bsz
    
    # 数据中除去多余余数部分
    data = data[:nbatch * bsz]
    
    # 使用view对data进行矩阵变化
    data = data.reshape(nbatch, -1, order='C')
    data = data[:,0]
    # print(data.shape)
    data = torch.tensor(data)
    
    return data.to(device)

# 使用batch_ify来处理训练数据
# 训练集bsz
train_batch_size = 100
# 测试验证bsz
eval_batch_size = 100
# 测试bsz
test_batch_size = 100

# 训练数据标签形成批次
train_data01 = batchify(train_info01, train_batch_size)
train_data_label01 = label_batchify(train_label01, train_batch_size)
train_data02 = batchify(train_info02, train_batch_size)
train_data_label02 = label_batchify(train_label02, train_batch_size)
train_data03 = batchify(train_info03, train_batch_size)
train_data_label03 = label_batchify(train_label03, train_batch_size)
train_data04 = batchify(train_info04, train_batch_size)
train_data_label04 = label_batchify(train_label04, train_batch_size)
# 将数据标签concat
train_data = torch.cat([train_data01, train_data02], dim=0)
train_data = torch.cat([train_data, train_data03], dim=0)
train_data = torch.cat([train_data, train_data04], dim=0)
train_label = torch.cat([train_data_label01, train_data_label02], dim=0)
train_label = torch.cat([train_label, train_data_label03], dim=0)
train_label = torch.cat([train_label, train_data_label04], dim=0)

d_l = int(train_data_label01.shape[0])
t_dl = int(train_data.shape[0]/4)
# 将训练标签按顺序混淆
train_label = torch.cat([train_label[i::d_l] for i in range(t_dl)], dim=0)
# 将训练数据按顺序混淆
train_data = torch.cat([train_data[i::d_l][:,:] for i in range(t_dl)], dim=0)


# train_data = torch.cat([train_data04, train_data03], dim=0)
# train_data = torch.cat([train_data, train_data02], dim=0)
# train_data = torch.cat([train_data, train_data01], dim=0)
# train_label = torch.cat([train_data_label04, train_data_label03], dim=0)
# train_label = torch.cat([train_label, train_data_label02], dim=0)
# train_label = torch.cat([train_label, train_data_label01], dim=0)


# 验证数据 成batch后合并
eval_data01 = batchify(eval_info01, eval_batch_size)
eval_data_label01 = label_batchify(eval_label01, eval_batch_size)
eval_data02 = batchify(eval_info02, eval_batch_size)
eval_data_label02 = label_batchify(eval_label02, eval_batch_size)
eval_data03 = batchify(eval_info03, eval_batch_size)
eval_data_label03 = label_batchify(eval_label03, eval_batch_size)
eval_data04 = batchify(eval_info04, eval_batch_size)
eval_data_label04 = label_batchify(eval_label04, eval_batch_size)
# 将标签concat
eval_data = torch.cat([eval_data01, eval_data02], dim=0)
eval_data = torch.cat([eval_data, eval_data03], dim=0)
eval_data = torch.cat([eval_data, eval_data04], dim=0)
eval_label = torch.cat([eval_data_label01, eval_data_label02])
eval_label = torch.cat([eval_label, eval_data_label03])
eval_label = torch.cat([eval_label, eval_data_label04])

# 测试数据 成batch
test_data = batchify(test_info, eval_batch_size)
test_data_label01 = label_batchify(test_label01, test_batch_size)
test_data_label02 = label_batchify(test_label02, test_batch_size)
test_data_label03 = label_batchify(test_label03, test_batch_size)
test_data_label04 = label_batchify(test_label04, test_batch_size)
test_label = torch.cat([test_data_label01, test_data_label02])
test_label = torch.cat([test_label, test_data_label03])
test_label = torch.cat([test_label, test_data_label04])


# 时序长度允许最大值为35
bptt = 80

def get_batch(source, target, i):
    seq_len = min(bptt, len(source) - 1 - i)
    
    # 语言模型训练的源数据的是将batchify的结果的切片[i:i+seq_len]
    data = source[i:i+seq_len]
    
    # 根据语言模型的语料定义 目标数据是将源数据向后移动一位
    # 因为最后目标数据的切片会越界 因此使用view(-1)保证形状正常 array应该使用flat
    target = target[i:i+seq_len]
    return data, target


# 设置模型超参数 初始化模型
V = 20 # 1000 20
Variety = 4

# 编码器层数量
nlayers = 1 # 4

# 词嵌入大小为200
d_model = 14

# 前馈全连接的节点数
nhid = 20 # 2000 20

# 多头注意力机制
nhead = 2 # 7 2

# 置0比率
dropout = 0.2

model = make_model(V,Variety,nlayers,d_model,nhid,nhead,train_batch_size,dropout)

# 加载预训练的模型参数
# state_dict = torch.load('./Net/Transformer_Testmachine_sourceclassfication_net_params_xiao.pkl', map_location=torch.device('cpu'))
# state_dict = torch.load('./Net/Transformer_Testmachine_sourceclassfication_net_params_da.pkl', map_location=torch.device('cpu'))
state_dict = torch.load('./Net/Transformer_Testmachine_sourceclassfication_net_params.pkl', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# 将模型迁移到gpu
model.to(device)

# 模型初始化后 接下来损失函数与优化方法选择
# 损失函数 
# 我们使用nn自带的交叉熵损失 分类问题时采用
criterion = nn.CrossEntropyLoss()
# MSELoss函数 计算输入和输出之差的平方 可输出序列标量或均方差
# criterion = nn.MSELoss()

# 分类问题学习率初始值为0.1
lr = 0.001 # 0.2 0.01

# 优化器选择torch自带的SGD随机梯度下降方法 并把lr传入其中
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 定义学习率调整器 使用torch自带的lr_scheduler 将优化器传入
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99) # 0.99 0.9999


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
    # 训练准确率
    total = 0
    correct = 0
    predicted = None
    val_accuracy = 0.0
    # 获得当前时间
    start_time = time.time()
    # 开始遍历批次数据
    for batch, i in enumerate(range(0, train_data.shape[0] - 1, bptt)):
        
        
        # 通过get_batch
        data, targets = get_batch(train_data, train_label, i)
        # 舍弃最后一组数据
        if targets.shape[0] < bptt:
            break;
        # 获得mask
        source_mask = Variable(torch.ones(data.shape[1])).to(device)
        # source_mask = subsequent_mask(data.shape[1]).to(device)
        target_mask = subsequent_mask(data.shape[1]).to(device)
        # 设置优化器初始为0梯度
        optimizer.zero_grad()
        
        # 将数据装入model得到输出
        output = model(data, data, source_mask, target_mask)
        # 将输入与目标传入损失函数对象
        # print(output.shape)
        # print(targets.shape)
        # print(output[0])
        # print(targets[0])
        output = output.type(torch.FloatTensor)
        targets = targets.type(torch.LongTensor)
        loss = criterion(output, targets)
        # print(loss.shape, loss)
        # 损失进行反向传播得到损失总和
        loss.backward()
        
        predicted = torch.max(output.data,1)[1]
        # 打印预测
        # print(predicted)
        # print(targets)
        
        correct += (predicted == targets).sum()
        total += targets.shape[0]
        val_accuracy = float(correct * 100)/float(total)
        
        # 使用nn自带的clip_grad_norm_进行梯度规范化 防止出现梯度消失或者爆炸
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 模型参数更新
        optimizer.step()
        # 损失累加得到总损失
        total_loss += loss.item()
        # 日志打印间隔为200
        log_interval = 1
        if batch % log_interval == 0 and batch > 0:
            # 平均损失为 总损失 / log_interval
            cur_loss = total_loss / log_interval
            # 需要时间 当前时间 - 起始时间
            elapsed = time.time() - start_time
            # 打印 轮数 当前批次 总批次 当前学习率 训练速度(每毫秒处理多少批次)
            # 平均损失 以及困惑度 困惑度是语言模型的重要标准 计算方法
            # 交叉熵平均损失取自然对数的底数
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.8f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | acc {:2.5f}%'.format(
                      epoch, batch, len(train_data) // bptt, 
                      scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                      cur_loss, val_accuracy))
            
            # 训练准确率
            # 准确率
            total = 0
            correct = 0
            predicted = None
            val_accuracy = 0.0
            
            # 每个批次结束后总损失归0
            total_loss = 0
            # 开始时间取当前时间
            start_time = time.time()

# 模型评估
def evaluate(eval_model, data_source, label_source):
    """
    评估函数 包括模型验证和测试
    Parameters
    ----------
    eval_model : model的对象
        DESCRIPTION.
        每轮训练后或验证后的模型
    data_source : dataset
        DESCRIPTION.
        验证或测试数据集
    Returns
    -------
    total_loss : TYPE int
        DESCRIPTION.
        总损失
    """
    # 模型进入评估模式
    eval_model.eval()
    # 总损失
    total_loss = 0
    # 准确率
    total = 0
    correct = 0
    predicted = None
    val_accuracy = 0.0
    with torch.no_grad():
        # 与训练步骤基本一致
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, label_source, i)
            if targets.shape[0] < bptt:
                break;
            # 获得mask
            source_mask = Variable(torch.ones(data.shape[1])).to(device)
            # source_mask = subsequent_mask(data.shape[1]).to(device)
            target_mask = subsequent_mask(data.shape[1]).to(device)
            # 设置优化器初始为0梯度
            output = eval_model(data, data, source_mask, target_mask)
            # 对输出形状进行扁平化 变为全部词汇的概率分布
            # output_flat = output.view(-1, data.shape[0] * data.shape[1] * data.shape[2])
            targets = targets.type(torch.LongTensor)
            output = output.type(torch.FloatTensor)
            # print('output', output)
            # print('targets', targets)
            
            # 获得评估过程的总损失
            total_loss += criterion(output, targets)
            
            # 评估准确率
            predicted = torch.max(output.data,1)[1]
            correct += (predicted == targets).sum()
            total += targets.shape[0]
            # print('total', total)
            # print('correct', correct)
            # print('predicted', predicted)
            
    # 返回每轮总损失,准确率
    val_accuracy = float(correct * 100)/float(total)
    return total_loss, val_accuracy

# 模型验证评估
# 首先初始化最佳验证损失 初始值为无穷大
best_val_loss = float("inf")

# 定义训练轮数
epochs = 100

# 定义最佳模型训练变量 初始值为None
best_model = None

# 损失记录
lossmem = [[],[]]

# 使用for循环遍历轮数
for epoch in range(1, epochs+1):
    # 首先获得轮数开始时间
    epoch_start_time = time.time()
    # 调用训练函数
    train()
    # 该轮训练后我们的模型参数已经发生了变化
    # 将模型和评估数据传入评估函数
    val_loss, val_accuracy = evaluate(model, eval_data, eval_label)
    lossmem[0].append(val_loss)
    lossmem[1].append(val_accuracy)
    # val_loss = 0
    # 之后打印每轮的评估日志 分别有轮数 耗时 验证损失 验证困惑度
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'acc {:2.5f}%'.format(epoch, (time.time() - epoch_start_time), 
                                     val_loss, val_accuracy))
    print('-' * 89)
    # 我们将比较哪一轮损失最小 赋值给best_val_loss,
    # 并取该损失下模型的best_model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model= model
    # 每轮都会对优化方法的学习率做调整
    
    # 早停
    
    scheduler.step()

# 绘制损失下降图
# 绘制好看一点
my_font=font_manager.FontProperties(fname='/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/图表/仿宋_常规.ttf',size='large')
plt.figure(figsize=(8, 2), dpi=400)
plt.plot(lossmem[0])
plt.xlabel('Interation')
plt.ylabel('Loss')
plt.title('目标域退化状态分类训练损失下降过程', rotation=0, fontproperties=my_font)
plt.show()


# 模型测试 依然使用evaluate函数 best_model以及测试数据
test_loss, test_acc = evaluate(best_model, test_data, test_label)


# 打印测试日志 包括测试损失和困惑度
print('=' * 89)
print('| End of training | test loss {:5.2f} | test_acc {:8.2f}'.format(test_loss, test_acc))

# 模型保存
# torch.save(best_model, './Net/Transformer_Testmachine_sourceclassfication_net.pkl')
# 只保存神经网络的模型参数
torch.save(best_model.state_dict(), './Net/Transformer_Testmachine_targetclassfication_net_params.pkl')



