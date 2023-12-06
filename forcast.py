import pandas as pd
import torch
import torch.nn as nn
from numpy import * 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler # 标准化数据
import time
from data import readfile
from data import dealfile

# 计算核心
device = []
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

# 数据读取
pathfile_test01 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_load0_1e_m15_200x5.csv'
pathfile_test02 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_noisy_1e_m15_200x5HI.csv'
pathfile_test03 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_noisy_1e_m15_200x5LO.csv'
pathfile_test04 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_noisy_1e_m15_200x5MED.csv'
pathfile_train01 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_load0_1e_m15_200x5.csv'
pathfile_train02 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_noisy_1e_m15_200x5HI.csv'
pathfile_train03 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_noisy_1e_m15_200x5LO.csv'
pathfile_train04 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_noisy_1e_m15_200x5MED.csv'


# read01 = readfile(pathfile_test01)
# read02 = readfile(pathfile_test02)
# read03 = readfile(pathfile_test03)
# read04 = readfile(pathfile_test04)
read05 = readfile(pathfile_train01)
# read06 = readfile(pathfile_train02)
# read07 = readfile(pathfile_train03)
# read08 = readfile(pathfile_train04)
# origin_data_test01 = read01.returndata()
# origin_data_test02 = read02.returndata()
# origin_data_test03 = read03.returndata()
# origin_data_test04 = read04.returndata()
origin_data_train01 = read05.returndata()
# origin_data_train02 = read06.returndata()
# origin_data_train03 = read07.returndata()
# origin_data_train04 = read08.returndata()
# print("origin_data_test01", origin_data_test01)
# print("origin_data_test02", origin_data_test02)
# print("origin_data_test03", origin_data_test03)
# print("origin_data_test04", origin_data_test04)
print("origin_data_train01", origin_data_train01)
# print("origin_data_train02", origin_data_train02)
# print("origin_data_train03", origin_data_train03)
# print("origin_data_train04", origin_data_train04)

# 数据分析
from data import dealfile
deal01 = dealfile(origin_data_train01)
deal01.dealdata()
# deal02 = dealfile(origin_data_test02)
# deal02.dealdata()
# deal03 = dealfile(origin_data_test03)
# deal03.dealdata()
# deal04 = dealfile(origin_data_test04)
# deal04.dealdata()




# 网络

# 训练

# 测试

# 保存
