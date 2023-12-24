import pandas as pd
import torch
import torch.nn as nn
from numpy import * 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler # 标准化数据
import time

class readfile():
    def __init__(self, filepath):
        print("读取数据开始")
        start = time.time()
        self.data01 = []
        self.data01 = pd.read_csv(filepath)
        end = time.time()
        print("读取数据完毕，时间花费：%d", (end - start), "s", sep = "")
        
    def returndata(self):
        return self.data01

class dealfile():
    def __init__(self, data):
        self.data01 = data
    
    def showdata(self):
        print("开始统计数据")
        # 统计数据列名
        origin_data_colname = self.data01.columns[:].to_list()
        print("列：", origin_data_colname)
        rownum = self.data01.shape[0] - 1
        colnum = self.data01.shape[1] - 1
        ylabel = self.data01.iloc[0:rownum, [colnum]]
        print("ylabel", ylabel)
        # 查看数据列
        '''
        for i in range(len(origin_data_colname)):
            print(self.data01[origin_data_colname[i]])
        '''
        # 绘制数据列
        plt.figure(dpi=300, figsize=(12, 5))
        for i in range(len(origin_data_colname)):
            plt.plot(self.data01[origin_data_colname[i]])
            plt.show()
            plt.cla()
            # cur_axes.show()
    
    def undersampling_data(self, sampling_interval=100):
        self.undersamplinged_data = self.data01[::sampling_interval]
        pass
    
    def writedata(self, path):
        self.undersamplinged_data.to_csv(path_or_buf=path)
        pass

    """
# 数据读取
pathfile_test01 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_load0_1e_m15_200x5.csv'
pathfile_test02 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_noisy_1e_m15_200x5HI.csv'
pathfile_test03 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_noisy_1e_m15_200x5LO.csv'
pathfile_test04 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/test_noisy_1e_m15_200x5MED.csv'
pathfile_train01 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_load0_1e_m15_200x5.csv'
pathfile_train02 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_noisy_1e_m15_200x5HI.csv'
pathfile_train03 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_noisy_1e_m15_200x5LO.csv'
pathfile_train04 = '/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/数据/19.FMCRD_Data/train_noisy_1e_m15_200x5MED.csv'


read01 = readfile(pathfile_test01)
read02 = readfile(pathfile_test02)
read03 = readfile(pathfile_test03)
read04 = readfile(pathfile_test04)
# read05 = readfile(pathfile_train01)
# read06 = readfile(pathfile_train02)
# read07 = readfile(pathfile_train03)
# read08 = readfile(pathfile_train04)
origin_data_test01 = read01.returndata()
origin_data_test02 = read02.returndata()
origin_data_test03 = read03.returndata()
origin_data_test04 = read04.returndata()
# origin_data_train01 = read05.returndata()
# origin_data_train02 = read06.returndata()
# origin_data_train03 = read07.returndata()
# origin_data_train04 = read08.returndata()
# print("origin_data_test01", origin_data_test01)
# print("origin_data_test02", origin_data_test02)
# print("origin_data_test03", origin_data_test03)
# print("origin_data_test04", origin_data_test04)
# print("origin_data_train01", origin_data_train01)
# print("origin_data_train02", origin_data_train02)
# print("origin_data_train03", origin_data_train03)
# print("origin_data_train04", origin_data_train04)

# 数据分析
# deal01 = dealfile(origin_data_train01)
# deal01.undersampling_data(100)
# deal01.writedata(pathfile_train01 + "_undersampling100.csv")
# deal02 = dealfile(origin_data_train02)
# deal02.undersampling_data(100)
# deal02.writedata(pathfile_train02 + "_undersampling100.csv")
# deal03 = dealfile(origin_data_train03)
# deal03.undersampling_data(100)
# deal03.writedata(pathfile_train03 + "_undersampling100.csv")
# deal04 = dealfile(origin_data_train04)
# deal04.undersampling_data(100)
# deal04.writedata(pathfile_train04 + "_undersampling100.csv")

deal01 = dealfile(origin_data_test01)
deal01.undersampling_data(100)
deal01.writedata(pathfile_test01 + "_undersampling100.csv")
deal02 = dealfile(origin_data_test02)
deal02.undersampling_data(100)
deal02.writedata(pathfile_test02 + "_undersampling100.csv")
deal03 = dealfile(origin_data_test03)
deal03.undersampling_data(100)
deal03.writedata(pathfile_test03 + "_undersampling100.csv")
deal04 = dealfile(origin_data_test04)
deal04.undersampling_data(100)
deal04.writedata(pathfile_test04 + "_undersampling100.csv")



"""







