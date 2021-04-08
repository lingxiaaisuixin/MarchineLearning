#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2020/2/11 14:36
@Author : StarsDreams
@Desc :
"""
# 导入warnings包，利用过滤器来实现忽略警告语句
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示正负号
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
# 计算相隔天数
def day_delay(t):
    t0 = np.datetime64(basedate, 'D')
    t1 = (t - t0)
    days = (t1 / np.timedelta64(1, 'D')).astype(int)
    return days

def logistic_increase_function(t,P0):
    r = hyperparameters_r
    K = hyperparameters_K
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    exp_value = np.exp(r * (t))
    return (K * exp_value * P0) / (K + (exp_value - 1) * P0)
# plt.scatter(data['date'], data['confirmedNum'], s=35)
# plt.xlabel('date')
# plt.ylabel('confirmedNum')
# plt.xticks(rotation=90)
# plt.legend()
# plt.show()
#
# plt.plot(data['date'], data['confirmedNum'], label='确诊人数')
# plt.xticks(rotation=90)
# plt.legend()
# plt.show()
if __name__ == '__main__':
    data = pd.read_excel(r'C:\Users\38681\Desktop\epidemic Situation\疫情数据城市.xlsx')
    data = data[11:]
    print(data)
    # 首次全国统计数据日期
    global basedate
    basedate = '2020-01-21'
    basedate = datetime.date(*map(int, basedate.split('-')))
    x_data, y_data = day_delay(data['date']), data['confirmedNum']
    # 分隔训练测试集,将最后的30%数据作为测试集
    x_train, x_test, y_train, y_test = x_data[:-1 * int(len(x_data) * 0.3)], x_data[-1 * int(len(x_data) * 0.3):], y_data[:-1 * int(len(x_data) * 0.3)],y_data[-1 * int(len(x_data) * 0.3):]
    print(x_train)
    print(x_test)
    popt = None
    mse = float("inf")
    r = None
    k = None
    # 网格搜索来优化r和K参数
    for k_ in np.arange(45000, 100000, 1):
        hyperparameters_K = k_
        for r_ in np.arange(0, 1, 0.01):
            # 用最小二乘法估计拟合
            hyperparameters_r = r_
            popt_, pcov_ = curve_fit(logistic_increase_function, x_train, y_train)
            # # 获取popt里面是拟合系数
            print("K:capacity  P0:initial_value   r:increase_rate")
            print(k_, popt_, r_)

            # 计算均方误差对测试集进行验证
            mse_ = mean_squared_error(y_test, logistic_increase_function(x_test, *popt_))
            print("mse:", mse_)
            if mse_ <= mse:
                mse = mse_
                popt = popt_
                r = r_
                k = k_
    hyperparameters_K = k
    hyperparameters_r = r
    # hyperparameters_K = 49999
    # hyperparameters_r = 0.29
    print("----------------")
    print("hyperparameters_K:", hyperparameters_K)
    print("hyperparameters_r:", hyperparameters_r)
    print("----------------")
    popt, pcov = curve_fit(logistic_increase_function, x_data, y_data)
    print("K:capacity  P0:initial_value   r:increase_rate")
    print(hyperparameters_K, popt, hyperparameters_r)

    # 未来预测
    days = np.linspace(0, 35, 36)
    # future = np.array(future)
    future_predict = logistic_increase_function(days, *popt)
    print(future_predict)
    # 绘图
    days_all = []
    for d in days:
        print(d)
        tempdelta = datetime.timedelta(days=d)
        days_all.append((basedate + tempdelta).strftime('%m-%d'))
    print(days_all)
    predictdata = pd.DataFrame(future_predict,index=days_all,columns=['pre_confirmedNum'])
    print(predictdata)
    tempdate = data['date'].apply(lambda x: x.strftime('%m-%d'))
    plt.figure(figsize=(15,8))
    # x_show_data_all = [(basedate + (datetime.timedelta(days=fu))).strftime("%m-%d") for fu in future]
    plt.scatter(tempdate, data['confirmedNum'], s=40, c='green', marker='*', label="确诊人数")
    plt.plot(days_all, future_predict, 'r', marker='+', linewidth=1.5, label='预测曲线')

    plt.tick_params(direction='out', length=6, width=2, colors='b',
                   grid_color='b', grid_alpha=0.5)

    plt.xlabel('时间')
    plt.ylabel('感染人数')
    plt.xticks(days_all,rotation=90)
    plt.grid()  # 显示网格

    plt.legend()  # 指定legend的位置右下角
    plt.show()
