#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2021/2/28 21:11
@Author : StarsDreams
@Desc :
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 读取数据
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
dataSet = pd.read_csv("data/breast-cancer-wisconsin.data",names=column_names)
# 将？替换为nan
dataSet = dataSet.replace(to_replace='?',value=np.nan)
# 删除缺失值
dataSet = dataSet.dropna()
# 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(dataSet[column_names[1:10]], dataSet[column_names[10:]], test_size=0.33)
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr.predict(x_test)
print("预测结果：",lr.score(x_test,y_test))
