import time
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import copy
from pyitcast.transformer_utils import Batch

# 导入优化器工具包get_std_opt 该工具可以获得transformer模型的优化器
# 基于adam优化器 序列对序列的预测更有效
from pyitcast.transformer_utils import get_std_opt

# 标签平滑工具包 用于标签平滑 小幅度改变原有标签值的值域
# 因为人工标注的数据并非准确 会受到外界因素造成微小差异
# 标签平滑来弥补偏差 防止绝对依赖一条规律 防止过拟合
from pyitcast.transformer_utils import LabelSmoothing

# 导入损失计算工具包 该工具使用标签平滑后的结果计算损失
# 损失计算方法可以认为交叉熵损失函数
from pyitcast.transformer_utils import SimpleLossCompute

# 导入贪婪解码工具包greedy_decode 该工具对最终结果进行贪婪解码
# 贪婪解码的方式每次预测都选择概率最大的结果作为输出
# 它不一定多的全局最优解 但却有最高效率
from pyitcast.transformer_utils import greedy_decode

# 导入模型单轮训练工具包run_epoch,该工具将对模型使用给定的损失函数计算方法进行单轮参数更新
# 并打印每轮参数更新的损失结果
from pyitcast.transformer_utils import run_epoch


# 词嵌入 将词数字向量转化为高维向量
class Embedder(nn.Module):
    # 词表大小 词嵌入维度
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

 # 位置掩码
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 10000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        # 位置编码
        position = torch.arange(0, max_len).unsqueeze(1)
        # 变化矩阵
        div_term1 = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(0)
        div_term2 = torch.exp(torch.arange(1, d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(0)
        # pe赋值
        pe[:, 0::2] = torch.sin(position * div_term1)
        pe[:, 1::2] = torch.cos(position * div_term2)
        # 附加第一维，扩展到与x维度一致
        pe = pe.unsqueeze(0)
        # 转化为模型缓冲文件
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # print(x.shape)
        # print(self.pe[:, :x.size(1)].shape)
        x = x + Variable(self.pe[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)


# 掩码张量, 向后遮掩
def subsequent_mask(size):
    # 掩码张量后两维的维度
    attn_shape= (1, size, size)
    
    # 使用np ones生成全1矩阵  然后利用triu生成上三角矩阵  再存储为无符号整型格式
    subsquent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    # 将上三角矩阵转为 下三角矩阵
    return torch.from_numpy(1 - subsquent_mask)


# 注意力机制
def Attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # 矩阵相乘
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 掩码向后遮掩
    if mask is not None:    
        scores = scores.masked_fill(mask == 0, -1e9)
    # 得到注意力张量
    p_attn = F.softmax(scores, dim=-1)
    # dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 返回注意力值和注意力张量
    return torch.matmul(p_attn, value), p_attn
        

# clones
def clones(module, N):
    # module,N
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        # 多头数量 词嵌入纬度 随机置0率
        super(MultiHeadAttention, self).__init__()
        
        # sure one
        assert embedding_dim % head == 0
        
        # clc word vector dim
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        
        # linear q k v concat
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        
        # init tensor
        self.p_attn = None
        
        # init dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # input tensor and mask tensor
        if mask is not None:
            # mask dim ex
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        
        # use zip and view  qkv  4dim to attention == multihead
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
             for model, x in zip(self.linears, (query, key, value))]
        # clc attn and p_attn
        x, self.p_attn = Attention(query, key, value, mask, self.dropout)
        
        # 4 dim to 3 dim tanspose need contigous == concat
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.head * self.d_k)
        
        # last linear
        return self.linears[-1](x)


# forward linear 增加辅助注意力的拟合性
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # word embedding_dim in & out dim
        # d_ff: yincangceng dim
        # zhilingbilv
        super(PositionwiseFeedForward, self).__init__()
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))



# 规范化层 防止多层网络参数过大 难以收敛  深层神经网络的标准件
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # 词嵌入纬度feature， eps足够小的正数，防止除std时出现0
        super(LayerNorm, self).__init__()
        
        # 初始化a2与b2两个张量，全1和全0，使用参数封装
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        
        self.eps = eps # 防止标准差为0
        
    def forward(self, x):
        # 在函数中， 先求出最后一个纬度的均值， 保持输入输出纬度一致
        # 求出最后一个纬度的标准差， x减去均值除以标准差，防止标准差为0
        # 计算y=ax+b
        mean = x.mean(-1, keepdim=True)
        # print(mean)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2



# 子层连接结构 残差连接
# 两个子层
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        

# 编码器层
# 对输入进行特征提取
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # size: 词嵌入纬度
        # self_attn 多头注意力层
        # feed_forward 前馈全联接
        
        # 将参数传递
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        
        # 编码器有两个子层连接
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        
    def forward(self, x, mask):
        # x 上一层的张量
        # mask 掩码
        x = self.sublayers[0](x, lambda x: self.self_attn(x,x,x,mask))
        return self.sublayers[1](x, self.feed_forward)


# Encoder 编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # layer 编码器层
        # N 个数
        
        self.layers = clones(layer, N)
        
        # 初始化规范化层
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        # x 表示上一层输出的张量
        # mask 表示掩码张量
        # 使x经过n格编码器层的处理 经过规范化层输出
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

# 解码器层
# 根据输入向目标方向进行特征提取，即解码过程
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # size 词嵌入纬度
        # self_attn 多头自注意力对象
        # src_attn 常规注意力对象
        # feed_forward 前馈全联接对象
        # dropout 置0的比率
        super(DecoderLayer, self).__init__()
        
        # 将参数传入类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        
        # 按照解码层的结构使用clones
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, source_mask, target_mask):
        # x 上一层输入张量
        # memory 编码器的语意存储张量
        # source_mask 原数据的掩码张量
        # target_mask 目标数据掩码张量
        m = memory
        
        # 让x经历多头自注意力的子层连接结构
        # 采用target_mask对未来信息进行遮掩
        x = self.sublayers[0](x, lambda x: self.self_attn(x,x,x,target_mask))
        
        # 让x经过第二个子层， 常规注意力层， Q！=K=V
        # 采用source_mask遮掩掉对结果信息无用的数据
        x = self.sublayers[1](x, lambda x: self.src_attn(x,m,m,source_mask))
        
        # 让x经过前馈全连接结构
        return self.sublayers[2](x, self.feed_forward)


# 解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        
        self.linears = clones(layer, N)
        
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, m, source_mask, target_mask):
        for layer in self.linears:
            x = layer(x, m, source_mask, target_mask)
        return self.norm(x)


# 输出部分
class Generator(nn.Module):
    def __init__(self, d_model, batchsize, hid_model):
        # 词嵌入维度 词表大小
        super(Generator, self).__init__()
        
        # 定义线性层
        # 监测：增加一个三层全连接，后面判断隐藏层维度的值在阈值内
        self.project1 = nn.Linear(d_model, hid_model)
        self.project2 = nn.Linear(hid_model, d_model)

    def forward(self, x):
        # 上一层的输出张量
        # timestep = x.shape[0]
        # bsz = x.shape[1]
        # dmodel = x.shape[2]
        # print("xxxxx")
        # print(x.shape)
        # x = x.reshape(timestep, bsz * dmodel)
        # 将线性层连接到softmax层 复制不走softmax
        # 输出两个 一个复制 一个隐藏层变量
        return self.project2(F.relu(self.project1(x))), self.project1(x)


# 模型构建 编码器解码器结构类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_pos, target_pos, generator):
        # 编码器对象 解码器对象 源数据嵌入函数 目标数据嵌入函数 输出部分类别生成器对象
        super(EncoderDecoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        # 无embed层设计
        self.src_embed = source_pos
        self.tgt_embed = target_pos
        self.generator = generator
        
    def forward(self, source, target, source_mask, target_mask):
        # 源数据 目标数据 源数据掩码张量 目标数据掩码张量
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask), self.encode(source, source_mask)
        # return self.generator(self.decode(self.encode(source, source_mask), source_mask, target, target_mask))
    
    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        # 编码后的输出张量 编码后的掩码 
        # return self.generator(self.decoder(self.tgt_embed(target), memory, source_mask, target_mask))
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


# 构建模型
def make_model(source_vocab, hid_model, N=6, d_model=1024, d_ff=1024, head=8, batchsize=20, dropout=0.1):
    # 源数据的词汇总数 目标数据词汇总数 解码器与编码器层数 词嵌入维度 全连接隐藏层维度 多头注意力头数 参数置零比率
    c = copy.deepcopy
    
    # 实例化一个多头注意力机制
    attn = MultiHeadAttention(head, d_model, dropout)
    
    # 实例化一个前馈全连接
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 实例化一个位置编码
    position = PositionEncoding(d_model, dropout)
    
    # 实例化模型model
    # 编码器结构有两个子层 attn 和 ff
    # 解码器有三个 attn attn ff
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # nn.Sequential(Embedder(source_vocab, d_model), c(position)),
            # nn.Sequential(Embedder(target_vocab, d_model), c(position)),
            # 无embeding设计
            nn.Sequential(c(position)),
            nn.Sequential(c(position)),
            Generator(d_model, batchsize, hid_model)
            
            
            # 监测：增加一个三层全连接，后面判断隐藏层维度的值在阈值内
        )
    
    # 初始化参数， 如果维度大于一，那么初始化一个均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model
    

# 模型基本运行测试
"""
target_vocab = 10  # 长于源数据时序
source_vocab = 10
N = 6
d_model = 512

if __name__ == '__main__':
    model = make_model(source_vocab, target_vocab, N, d_model)
    # print(model)


x = Variable(torch.LongTensor([np.arange(1,10) for _ in range(8)]))
source_mask = subsequent_mask(x.size(1))
target_mask = subsequent_mask(x.size(1))

model_res = model(x,x,source_mask,target_mask)
# print(model_res.shape)
# print(model_res)
"""



# copy任务 模型基础测试任务 非常明显 
# 模型在短时间 小数据集内学会很重要
# 判断模型是否正常，是否具备基本学习能力
# 1构建数据集生成器 2获得模型 优化器 损失函数 3运行模型训练评估 4使用模型进行贪婪解码

# 1构建数据集
def data_generator(V, batch_size, num_batch):
    # 随机生成值+1， 每次输送给模型的数据样本数量 经过这些样本数量进行一次参数更新
    for i in range(num_batch):
        # 使用numpy中的random.randint()随机生成1-V
        # 形状分布 (batch_size, V-1)
        data = torch.LongTensor(np.random.randint(1,V,size=(batch_size, V-1)))
        
        # 将数据的第一列全部设置为1 作为起始标志
        data[:,1] = 1
        
        # copy任务生成，源数据与目标一致
        # 设置参数requires_grad=False 样本参数不参与梯度计算
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        
        # yield Batch(source, target)
        yield Batch(source.to(device), target.to(device))

# 2模型训练
# def run(model, loss, epochs=10):
#     for epoch in range(epochs):
#         # 先进入训练模式
#         model.train()
#         run_epoch(data_generator(V, batch_size, num_batch), model, loss)
        
#         # 训练结束后进入评估模式  所有参数固定不变
#         model.eval()
#         run_epoch(data_generator(V, batch_size, 5), model, loss)


# 3进行贪婪解码
# 重写run
def run(model, loss, epochs):
    start_time = time.time()
    for epoch in range(epochs):
        print("Epoch", epoch)
        # 先进入训练模式
        model.train()
        run_epoch(data_generator(V, batch_size, num_batch), model, loss)
        
        # 训练结束后进入评估模式  所有参数固定不变
        model.eval()
        run_epoch(data_generator(V, batch_size, 5), model, loss)
        
    # 模型进入评估模式
    model.eval()
    
    # 初始化一个输入张量
    source = Variable(torch.LongTensor([np.arange(1,V)])).to(device)
    # source = Variable(torch.LongTensor([np.arange(1,V)]))
    # 初始化一个掩码张量
    source_mask = Variable(torch.ones(1,1,V-1)).to(device)
    # source_mask = Variable(torch.ones(1,1,V-1))
    
    # 设定掩码最大长度 开始数字标志默认等于1
    result = greedy_decode(model, source, source_mask, max_len=V-1, start_symbol=1)
    print(result)
    print("time:", time.time() - start_time)


if __name__ == '__main__':
    
    V = 11
    N = 2
    batch_size = 20
    num_batch = 30
    epochs = 10
    res = data_generator(V,batch_size,num_batch)
    # 使用mps或者gpu
    # 计算核心
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # print(device)

    # 2构建模型 loss 优化器
    # adam优化器 序列到序列 效果更好
    model = make_model(V,V,N)
    # 将模型迁移到gpu
    model.to(device)

    # 使用工具包获得模型优化器
    model_optimizer = get_std_opt(model)

    # 使用工具包LabelSmoothing获得平滑对象
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.5)
     
    # 使用工具包SimpleLossCompute获得利用标签平滑的结果得到损失计算方法
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
    run(model, loss, epochs)




"""
# 标签平滑函数实例
crit = LabelSmoothing(size=5,padding_idx=0,smoothing=2)
predict = Variable(torch.FloatTensor([[0,0.2,0.7,0.1,0],
                                      [0,0.2,0.7,0.1,0],
                                      [0,0.2,0.7,0.1,0]]))
target = Variable(torch.LongTensor([2,1,0]))
crit(predict, target)
# 标签平滑图
plt.imshow(crit.true_dist)

"""



"""
# 测试1

d_model = 16

# # 词嵌入
embedding = Embedder(10000, d_model)
input_data = Variable(torch.LongTensor([np.arange(1,101) for _ in range(8)]))
em_res = embedding(input_data)
# print(output1)

# # 位置编码
positionencoder = PositionEncoding(d_model,0)
# emr = Variable(torch.zeros(3,100,20))
# outpe1 = positionencoder(emr)
pe_res = positionencoder(em_res)
# print(outpe1)
# print(pe_res)
# plt.figure(figsize=(15,5))
# plt.plot(np.arange(100), outpe1[0,:,4:8].data.numpy()) 
# plt.legend([])


# mask = subsequent_mask(pe_res.size(1))

# query = key = value = pe_res
# attn, p_attn = attention(query, key, value, mask)
# print("attn", attn)
# print(attn.shape)
# print("p_attn", p_attn)
# print(p_attn.shape)

# # np.triu
# print(1 - np.triu([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], k=1))

# a = np.ones((4,5))
# b = torch.from_numpy(a)
# c = torch.from_numpy(np.arange(0,10,2))
# d = torch.arange(0,10,2)
# size = 5
# sm = subsequent_mask(size)
# print(sm)
# sm = sm.unsqueeze(1)

# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])

# view transpose
# x = torch.randn(4,4)
# x.size()
# y = x.view(16)
# z = x.view(-1, 2)

# a = torch.randn(2,3)
# print(a)
# b = torch.transpose(a,0,1)
# c = a.view(-1 ,2)
# torch.equal(b,c)
# print(b)
# print(c)


# head = 8
# embedding_dim = 512
# dropout = 0.1
# query = key = value = pe_res
# multiheadattention = MultiHeadAttention(head, embedding_dim, dropout)
# mha_res = multiheadattention(query, key, value, mask)
# # print(mha_res)
# # print(mha_res.size())

# d_model = 512
# d_ff = 64
# ff = PositionwiseFeedForward(d_model, d_ff)
# ff_res = ff(mha_res)
# # print(ff_res.size())
# # print(ff_res)

# features = 512
# eps = 1e-9
# ln = LayerNorm(features, eps)
# ln_res = ln(ff_res)
# # print(ln_res.shape)
# # print(ln_res)


# sublayer
# size = 512
# dropout = 0.2
# head = 8
# d_model = 512
# d_ff = 64
# x = pe_res
# # 多头中的掩码
# mask = Variable(subsequent_mask(x.size(1)))
# # mask = Variable(torch.zeros(8,4,4))  # 出现错误

# # 子层中的多头注意力层
# self_attn = MultiHeadAttention(head, d_model)
# self_ff = PositionwiseFeedForward(d_model, d_ff)

# # 使用lambda匿名函数 获得一个函数类型的子层sublayer
# sublayer1 = lambda x: self_attn(x,x,x,mask)
# sublayer2 = lambda x: self_ff(x)

# # 定义sublayer
# sublayerconnection = SublayerConnection(size, dropout)

# x = sublayerconnection(x, sublayer1)
# sc_res = sublayerconnection(x, sublayer2)
# print(sc_res.shape)
# print(sc_res)


# EncoderLayer
# size = d_model = 512
# head = 8
# d_ff = 64
# x = pe_res
# dropout = 0.1

# self_attn = MultiHeadAttention(head, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# mask = Variable(subsequent_mask(pe_res.size(1)))
mask = Variable(torch.zeros(pe_res.size(1)))

# el = EncoderLayer(size, self_attn, ff, dropout)
# el_res = el(x, mask)
# print(el_res.shape)
# print(el_res)


# Encoder
size = d_model = 16
head = 8
d_ff = 64
dropout = 0.1
x = pe_res
c = copy.deepcopy
attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, attn, ff, dropout)
N = 8
mask = Variable(subsequent_mask(x.size(1)))
mask = Variable(torch.zeros(pe_res.size(1)))


en = Encoder(layer, N)

en_res = en(x, mask)
# print(en_res.shape)
# print(en_res)



size = d_model= 16
head = 8
d_ff = 64
dropout = 0.2

self_attn = src_attn = MultiHeadAttention(head, d_model, dropout)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)

x = pe_res
memory = en_res

mask = Variable(torch.zeros(x.size(1)))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)

# dl_res = dl(x, memory, source_mask, target_mask)
# print(dl_res.shape)
# print(dl_res)

de = Decoder(dl, N)

dn_res = de(x, memory, source_mask, target_mask)
# print(dn_res.shape)
# print(dn_res)


# x = torch.randn(30, 20)
# gen = Generator(20, 1000)
# y = gen(x)
# print(y.shape)
# print(y)

d_model = 16
vocab_size = 1000
x = dn_res

gen = Generator(d_model, vocab_size)
gen_res = gen(x)
# print(gen_res)
# print(gen_res.shape)

vocab_size = 1000
d_model = 16
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen

source = target = input_data

ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_res = ed(source, target, source_mask, target_mask)
print(ed_res.shape)
print(ed_res)









"""









