#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2019/7/23 22:01
@Author : SundialDreams
@Desc :泰坦尼克号数据集
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def titanic_decision_tree():
    # 读取数据
    data_train = pd.read_csv('data/titanic/train.csv')
    data_test = pd.read_csv("data/titanic/test.csv")
    print(data_train.shape, '\n', data_test.shape)
    print(data_train.head())
    # x特征值，y目标值
    x = data_train[['Pclass', 'Sex', 'Age']]
    # print(x.info(verbose=True, null_counts=True))
    y = data_train['Survived']

    # 缺失值处理
    x['Age'].fillna(x['Age'].mean(), inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(x_train.shape, x_test.shape)
    # 转成数值
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    print(x_train)
    x_test = dict.fit_transform(x_test.to_dict(orient="records"))
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    print("预测的准去率：%0.4f" % clf.score(x_test, y_test))


if __name__ == "__main__":
    titanic_decision_tree()
