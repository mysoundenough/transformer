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
    
    def dealdata(self):
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
        