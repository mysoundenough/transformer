#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:59:29 2024
# 提供数据分析以及图片输出
@author: mayuan
"""

from data import readfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
import random



# 输出图1
# 对丝杠退化的源域数据和目标域数据相关性分布进行发布
# class c_S_T_association():
# 导入数据 
# 源域路径
pathfile_source01 = '../../../数据/19.FMCRD_Data/source_data_01.csv'
pathfile_source02 = '../../../数据/19.FMCRD_Data/source_data_02.csv'
pathfile_source03 = '../../../数据/19.FMCRD_Data/source_data_03.csv'
pathfile_source04 = '../../../数据/19.FMCRD_Data/source_data_04.csv'
# 目标域路径
pathfile_target01 = '../../../数据/19.FMCRD_Data/target_data_01.csv'
pathfile_target02 = '../../../数据/19.FMCRD_Data/target_data_02.csv'
pathfile_target03 = '../../../数据/19.FMCRD_Data/target_data_03.csv'
pathfile_target04 = '../../../数据/19.FMCRD_Data/target_data_04.csv'

# 读取数据
read01 = readfile(pathfile_source01)
read02 = readfile(pathfile_source02)
read03 = readfile(pathfile_source03)
read04 = readfile(pathfile_source04)
read05 = readfile(pathfile_target01)
read06 = readfile(pathfile_target02)
read07 = readfile(pathfile_target03)
read08 = readfile(pathfile_target04)
sourcedata01 = read01.returndata()
sourcedata02 = read02.returndata()
sourcedata03 = read03.returndata()
sourcedata04 = read04.returndata()
targetdata01 = read05.returndata()
targetdata02 = read06.returndata()
targetdata03 = read07.returndata()
targetdata04 = read08.returndata()

# 绘制分布图片
# 绘制退化曲线
# 源域和目标域时间长度不一致，做成散点图不太

# 将数据读出
datainfo = []
datainfo.append(np.array(sourcedata01.iloc[:,0:-1], dtype='float32'))
datainfo.append(np.array(sourcedata02.iloc[:,0:-1], dtype='float32'))
datainfo.append(np.array(sourcedata03.iloc[:,0:-1], dtype='float32'))
datainfo.append(np.array(sourcedata04.iloc[:,0:-1], dtype='float32'))
datainfo.append(np.array(targetdata01.iloc[:,0:-1], dtype='float32'))
datainfo.append(np.array(targetdata02.iloc[:,0:-1], dtype='float32'))
datainfo.append(np.array(targetdata03.iloc[:,0:-1], dtype='float32'))
datainfo.append(np.array(targetdata04.iloc[:,0:-1], dtype='float32'))

# 计算均值和方差
SG_data_mean = np.zeros([8,14], dtype='float32')
SG_data_std = np.zeros([8,14], dtype='float32')
for i in range(8):
    for j in range(14):
        SG_data_mean[i][j] = np.mean(datainfo[i][:,j])
        SG_data_std[i][j] = np.std(datainfo[i][:,j])

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Temperture Type')

# 标注颜色
linename = ["source01","source02","source03","source04","target01","target02","target03","target04"]
linecolor = ["#1f77b4","#ff7f0e","#2ca02c","#17becf","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#D43F3A"]

# create test data
np.random.seed(19680801)
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), dpi=1600, sharey=False)

for i in range(8):
    data = []
    
    # plot mean ax1
    for j in range(14):
        data.append(datainfo[i][:,j])
        
        # source
        if i < 4:
            ax1.set_title('(a) Mean Distribution')
            ax1.set_ylabel('Mean Values')
            ax1.violinplot(data)
        
        # target
        else :
            parts = ax1.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('#D43F3A')
                pc.set_edgecolor('black')
                pc.set_alpha(0.4)
            
            quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
            whiskers = np.array([
                adjacent_values(sorted_array, q1, q3)
                for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
            whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
            ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
            ax1.plot(inds, medians, "-", color='#D43F3A')  
    
    ax1.plot([],[], label=linename[i], color=linecolor[i], linewidth=8.0)
    ax1.legend(loc="upper left")
    
    
    # plot std ax2:
    ax2.set_title('(b) Std Distribution')
    ax2.set_ylabel('Std Values')
    ax2.plot(list(range(1,15)), SG_data_std[i], "-o", label=linename[i], color=linecolor[i])
    ax2.legend(loc="upper left")
    
    
    
    # set style for the axes
    # labels = ['T1','T2','T3','T4','T5','T6','T7','T8','Cc']
    labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
    for ax in [ax1, ax2]:
        set_axis_style(ax, labels)
    
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.tight_layout()


plt.show()


    




































