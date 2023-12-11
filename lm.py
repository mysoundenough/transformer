#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 02:40:28 2023

@author: mayuan
"""

# 导入数学相关包
import math

# 导入torch相关包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入经典文本数据集包
import torchtext

# 导入英文分词数据包
from torch.data.uitls import get_tokenizer

# 导入构建完成的Transformer包
from pyitcast.transformer import TransformerModel
from net import *



# 导入wikiText数据集 并做基本处理
# 四个参数 给语料的限制
