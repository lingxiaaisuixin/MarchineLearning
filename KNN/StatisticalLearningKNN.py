#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2019/12/19 22:25
@Author : StarsDreams
@Desc :
"""
import numpy as np
from math import sqrt
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, data, depth=0, lchild=None, rchild=None):
        self.data = data  # 此结点
        self.depth = depth  # 树的深度
        self.lchild = lchild  # 左子结点
        self.rchild = rchild  # 右子节点


class KdTree:
    def __init__(self):
        self.KdTree = None
        self.n = 0
        self.nearest = None

    def create(self, dataSet, depth=0):
        """KD-Tree创建过程"""
        if len(dataSet) > 0:
            m, n = np.shape(dataSet)
            self.n = n - 1
            # 按照哪个维度进行分割，比如0：x轴，1：y轴
            axis = depth % self.n
            # 中位数
            mid = int(m / 2)
            # 按照第几个维度（列）进行排序
            dataSetcopy = sorted(dataSet, key=lambda x: x[axis])
            # KD结点为中位数的结点，树深度为depth
            node = Node(dataSetcopy[mid], depth)
            if depth == 0:
                self.KdTree = node
            # 前mid行为左子结点，此时行数m改变，深度depth+1，axis会换个维度
            node.lchild = self.create(dataSetcopy[:mid], depth + 1)
            node.rchild = self.create(dataSetcopy[mid + 1:], depth + 1)
            return node
        return None

    def preOrder(self, node):
        """遍历KD-Tree"""
        if node is not None:
            print(node.depth, node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

    def search(self, x, count=1):
        """KD-Tree的搜索"""
        nearest = []  # 记录近邻点的集合
        for i in range(count):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)

        def recurve(node):
            """内方法，负责查找count个近邻点"""
            if node is not None:
                # 步骤1：怎么找叶子节点
                # 在哪个维度的分割线，0,1,0,1表示x,y,x,y
                axis = node.depth % self.n
                # 判断往左走or右走，递归，找到叶子结点
                daxis = x[axis] - node.data[axis]
                if daxis < 0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)

                # 步骤2：满足的就插入到近邻点集合中
                # 求test点与此点的距离
                dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x, node.data)))
                # 遍历k个近邻点，如果不满k个，直接加入，如果距离比已有的近邻点距离小，替换掉，距离是从小到大排序的
                for i, d in enumerate(self.nearest):
                    if d[0] < 0 or dist < d[0]:
                        self.nearest = np.insert(self.nearest, i, [dist, node], axis=0)
                        self.nearest = self.nearest[:-1]
                        break

                # 步骤3：判断与垂线的距离，如果比这大，要查找垂线的另一侧
                n = list(self.nearest[:, 0]).count(-1)
                # -n-1表示不为-1的最后一行，就是记录最远的近邻点（也就是最大的距离）
                # 如果大于到垂线之间的距离，表示垂线的另一侧可能还有比他离的近的点
                if self.nearest[-n - 1, 0] > abs(daxis):
                    # 如果axis < 0，表示测量点在垂线的左侧，因此要在垂线右侧寻找点
                    if daxis < 0:
                        recurve(node.rchild)
                    else:
                        recurve(node.lchild)

        recurve(self.KdTree)  # 调用根节点，开始查找
        knn = self.nearest[:, 1]  # knn为k个近邻结点
        belong = []  # 记录k个近邻结点的分类
        for i in knn:
            belong.append(i.data[-1])
        b = max(set(belong), key=belong.count)  # 找到测试点所属的分类

        return self.nearest, b

def show_train():
    plt.scatter(x0[:, 0], x0[:, 1], c='pink', label='[0]')
    plt.scatter(x1[:, 0], x1[:, 1], c='orange', label='[1]')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')


if __name__ == "__main__":
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    data = np.array(df.iloc[:100, [0, 1, -1]])
    train, test = train_test_split(data, test_size=0.1)
    x0 = np.array([x0 for i, x0 in enumerate(train) if train[i][-1] == 0])
    x1 = np.array([x1 for i, x1 in enumerate(train) if train[i][-1] == 1])

    kdt = KdTree()
    kdt.create(train)
    kdt.preOrder(kdt.KdTree)

    score = 0
    for x in test:
        show_train()
        plt.scatter(x[0], x[1], c='red', marker='x')  # 测试点
        near, belong = kdt.search(x[:-1], 5)  # 设置临近点的个数
        if belong == x[-1]:
            score += 1
        print(x, "predict:", belong)
        print("nearest:")
        for n in near:
            print(n[1].data, "dist:", n[0])
            plt.scatter(n[1].data[0], n[1].data[1], c='green', marker='+')  # k个最近邻点
        plt.legend()
        plt.show()

    score /= len(test)
    print("score:", score)