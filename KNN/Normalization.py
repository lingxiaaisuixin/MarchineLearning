#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2019/7/21 16:14
@Author : StarsDreams
@Desc :归一化
归一化公式：
    Y = (X-Xmin)/(Xmax-Xmin)
    其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
"""
from numpy import *
def autoNorm(dataSet):
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet
if __name__ == "__main__":
    dataSet = array([[40920, 8.326976, 0.953952, 3], [14488, 7.153469, 1.673904, 2], [26052, 1.441871, 0.805124, 1],
                        [75136, 13.147394, 0.428964, 1]])
    # 取前三列进行归一化处理
    dataSet = dataSet[:,[0,1,2]]
    print(autoNorm(dataSet))