#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2021/4/8 20:20
@Author : StarsDreams
@Desc :
"""
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# 读取数据
data = datasets.load_iris().data
target = datasets.load_iris().target
# 这里只取两个特征，两个类别便于画图
X = data[:, (2, 3)]
X0 = np.insert(X, 2, values=target, axis=1)
dataset = pd.DataFrame(X0, columns=['x0', 'x1', 'y'])
dataset = dataset[dataset['y'] != 2]
# 训练预测
clf = SVC(kernel='linear', C=float('inf'), probability=True)
clf.fit(dataset[['x0', 'x1']], dataset['y'])
clf_pred = clf.predict(dataset[['x0', 'x1']])
# 输出参数
x0 = np.linspace(0, 5.5, 200)
w = clf.coef_[0]
b = clf.intercept_[0]
boundry_line = -b / w[1] - w[0] / w[1] * x0
support = clf.support_vectors_
# 边界
margine = 1 / w[1]
margine_high = boundry_line + margine
margine_low = boundry_line - margine
# 画图
plt.figure(figsize=(8, 5))
plt.xlim(0, 5)
plt.ylim(0, 2)
plt.plot(dataset['x0'][dataset['y'] == 0], dataset['x1'][dataset['y'] == 0], 'ro')
plt.plot(dataset['x0'][dataset['y'] == 1], dataset['x1'][dataset['y'] == 1], 'bo')
plt.plot(x0, boundry_line, 'b', linewidth=2)
plt.plot(x0, margine_high, 'b--', linewidth=2)
plt.plot(x0, margine_low, 'b--', linewidth=2)
plt.scatter(support[:, 0], support[:, 1], s=180, facecolors='#7f7f7f')
plt.show()
