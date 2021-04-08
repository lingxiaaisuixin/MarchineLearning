#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2019/7/21 16:33
@Author : StarsDreams
@Desc :鸢尾花 使用Sklearn库KNN算法识别鸢尾花类别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
def KNN_Iris():
    """
    :return:鸢尾花类别
    """
    # 读取鸢尾花数据
    dataSet = pd.read_csv("data/Iris.csv")
    print(dataSet)
    # 查看类别
    data_Species = dataSet.groupby("Species").count()
    print(data_Species)
    # 删除Id
    dataSet = dataSet.drop(['Id'],axis=1)
    # 取特征列x和标签列y
    x = dataSet.drop(['Species'],axis=1)
    y = dataSet['Species']
    # 特征归一化
    std = StandardScaler()
    x = std.fit_transform(x)
    # 拆分数据集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
    # 使用KNN进行分类
    print(x_train.shape,x_test.shape)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train,y_train)
    knn.predict(x_test)
    print("预测结果：",knn.score(x_test,y_test))
if __name__ == "__main__":
    KNN_Iris()