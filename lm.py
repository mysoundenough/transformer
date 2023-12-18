#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 02:40:28 2023

@author: mayuan
"""
# 导入时间工具包
import time

# 导入数学相关包
import math

# 导入torch相关包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入经典文本数据集包
import torchtext
from torchtext.data import Field

# 导入英文分词数据包
from torchtext.data.utils import get_tokenizer

# 导入构建完成的Transformer包
from pyitcast.transformer import TransformerModel
from transformer_net import *



# 导入wikiText数据集 并做基本处理
# 四个参数 给语料的限制 起始字符 结束字符 小写字母

# 将数据进行语料库的封装
TEXT = Field(tokenize = get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=False)

# print(TEXT)

# 使用torchtext的数据集方法导入数据
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

# print(test_txt.examples[0].text[:10000])

# 将训练集文本构建一个vocab对象 使用vocab中的stoi方法统计文本包含的不重复词汇总数
TEXT.build_vocab(train_txt)

# 设置设备
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# print(device)

# 三 构建用于模型输入的批次化数据
def batchify(data, bsz):
    """
    用于将文本数据映射为连续的数字 便于进行将数字转化为多维 并转化为指定的样式 指定的样式为
    Parameters
    ----------
    data : dataset
        data是train_txt val_txt test_txt 
    bsz : int
        bsz是batch_size 每次模型更新的数据量

    Returns
    -------
    变形后的数据

    """
    # 用于将文本数据映射为连续的数字
    data = TEXT.numericalize([data.examples[0].text])
    
    # 整除得到多个batch方向的长度
    nbatch = data.size(0) // bsz
    
    # 去掉余数部分
    data = data.narrow(0,0,nbatch * bsz)
    
    # 使用view对data进行矩阵变化
    data = data.view(bsz, -1).transpose(0,1).contiguous()
    # print(data.shape)
    # 
    return data.to(device)
    
    
# 使用batch_ify来处理训练数据
# 训练集bsz
batch_size = 20
# 测试验证bsz
eval_batch_size = 10

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# 子长度允许最大值为35
bptt = 35

def get_batch(source, i):
    """
    用于获得每个批次合理的源数据和目标数据
    参数source是通过batchify得到的train_data cal_data test_data
    Parameters
    ----------
    source : dataset
        batchify变形后的数据 行是nbatch 列是bsz 
        参数source是通过batchify得到的train_data cal_data test_data
    i : int
        具体的批次次数

    Returns
    -------
    None.

    """
    seq_len = min(bptt, len(source) - 1 - i)
    
    # 语言模型训练的源数据的是将batchify的结果的切片[i:i+seq_len]
    data = source[i:i+seq_len]
    
    # 根据语言模型的语料定义 目标数据是将源数据向后移动一位
    # 因为最后目标数据的切片会越界 因此使用view(-1)保证形状正常
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# source = test_data
# i = 1
# x,y = get_batch(source, i)




# 四 设置模型超参数和初始化模型
# 通过TEXT.vocab.stoi方法获得不重复词汇总数
ntokens = len(TEXT.vocab.stoi)

# 词嵌入大小为200
emsize = 200

# 前馈全连接的节点数
nhid = 200

# 编码器层数量
nlayers = 2

# 多头注意力机制
nhead = 2

# 置0比率
dropout = 0.2

# 将参数输入到TransformerModel中
model = TransformerModel(ntokens,emsize,nhid,nlayers,nhead,dropout).to(device)
# model = make_model(ntokens,ntokens,nlayers).to(device)

# 模型初始化后 接下来损失函数与优化方法选择
# 损失函数 我们使用nn自带的交叉熵损失
criterion = nn.CrossEntropyLoss()

# 学习率初始值为5.0
lr = 5.0

# 优化器选择torch自带的SGD随机梯度下降方法 并把lr传入其中
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 定义学习率调整器 使用torch自带的lr_scheduler 将优化器传入
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# 模型训练
def train():
    """
    train model
    """
    # 模型开启训练模式
    model.train()
    # 初始损失定义为0
    total_loss = 0
    # 获得当前时间
    start_time = time.time()
    # 开始遍历批次数据
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 通过get_batch
        data, targets = get_batch(train_data, i)
        # 设置优化器初始为0梯度
        optimizer.zero_grad()
        # 将数据装入model得到输出
        output = model(data)
        # 将输入与目标传入损失函数对象
        loss = criterion(output.view(-1, ntokens), targets)
        # 损失进行反向传播得到损失总和
        loss.backward()
        # 使用nn自带的clip_grad_norm_进行梯度规范化 防止出现梯度消失或者爆炸
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 模型参数更新
        optimizer.step()
        # 损失累加得到总损失
        total_loss += loss.item()
        # 日志打印间隔为200
        log_interval = 200
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
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // bptt, 
                      scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
        
            # 每个批次结束后总损失归0
            total_loss = 0
            # 开始时间取当前时间
            start_time = time.time()

# 模型评估
def evaluate(eval_model, data_source):
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
    with torch.no_grad():
        # 与训练步骤基本一致
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            # 对输出形状进行扁平化 变为全部词汇的概率分布
            output_flat = output.view(-1, ntokens)
            # 获得评估过程的总损失
            total_loss += criterion(output_flat, targets).item()
    # 返回每轮总损失
    return total_loss

# 模型验证评估
# 首先初始化最佳验证损失 初始值为无穷大
best_val_loss = float("inf")

# 定义训练轮数
epochs = 3

# 定义最佳模型训练变量 初始值为None
best_model = None

# 使用for循环遍历轮数
for epoch in range(1, epochs+1):
    # 首先获得轮数开始时间
    epoch_start_time = time.time()
    # 调用训练函数
    train()
    # 该轮训练后我们的模型参数已经发生了变化
    # 将模型和评估数据传入评估函数
    val_loss = evaluate(model, val_data)
    # 之后打印每轮的评估日志 分别有轮数 耗时 验证损失 验证困惑度
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), 
                                     val_loss, math.exp(min(709, val_loss))))
    print('-' * 89)
    # 我们将比较哪一轮损失最小 赋值给best_val_loss,
    # 并取该损失下模型的best_model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model= model
    # 每轮都会对优化方法的学习率做调整
    scheduler.step()


# 模型测试 依然使用evaluate函数 best_model以及测试数据
test_loss = evaluate(best_model, test_data)

# 打印测试日志 包括测试损失和困惑度
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(min(709, test_loss))))





























































