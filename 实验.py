import numpy as np
import math
from math import sqrt
from scipy.spatial import distance
from time import time
import matplotlib.pyplot as plt
import pylab as pl
import multiprocessing
from multiprocessing import Manager

# def get_euclidean(vec1, vec2):
#     distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
#     return 1/(1 + distance**.5)
# a = [1, 5, 4, 4, 5, 42]
# b = [1, 2, 8, 4, 2, 42]
# a = np.mat(a)
# b = np.mat(b)
# z = get_euclidean(a, b)
# print(z)


# print(r)
# print(type(r))
# dict2 = {'a': {'a': 4, 'b': 3}, 'b': {'b': 7}, 'e':{'a': 2, 'b': 22}}
# df1 = pandas.DataFrame(dict2).T.fillna(0)
# df1 = np.mat(df1)
# print(df1)
#
# w = np.mat(np.zeros((3, 3)))
# for i in range(3):
#     for j in range(3):
#         w[i, j] = get_cos(df1[i, ], df1[j, ])
# print(w)
# print(np.shape(df1)[1])

#################画图！！！！！！！！！！！！！！！
# a = {0: 784, 1: 15, 2: 83, 3: 13, 4: 16, 5: 69, 6: 92, 7: 11, 8: 41}
# # print(a)
# # print(type(a))
# # print(a[9])
# x = list(a.keys())
# y = list(a.values())
# plt.plot(x, y, label='123', color='red')
# plt.show()
#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: "Zing-p"
# Date: 2017/5/12

# data = [5, 20, 15, 25, 10]
#
# plt.bar(range(len(data)), data)
# plt.show()
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# a = 1
# b = 3
# c = 6
# d = 5
# labels = ['a', 'b', 'c', 'd']
# yy = [a, b, c, d]
# # ymajorLocator   = MultipleLocator(0.2)
# # ymajorFormatter = FormatStrFormatter('%1.1f')
# # yminorLocator   = MultipleLocator(0.05)
# plt.ylim(0,2)
# plt.bar(range(len(yy)), yy, color='red', tick_label=labels)
# plt.show()
#测试矩阵转化！！！！！！！！！！
# dict1 = {'a': {'a': 3, 'b': 8}, 'b': {'b': 2}, 'e':{'a': 2, 'b': 10}}
# print(dict1)
# df = pandas.DataFrame(dict1).T.fillna(0)
# # print(type(df))
# df = np.mat(df)
# print(df)
# a = df[0]
# print(a)
# x, y = np.shape(df)
# print(x, y)

#######################################

#原数据分片！！！！！！！！！！！！！
# trainSet = dict()
# fh_train = open('C:/Users\Administrator/Desktop/毕业设计/数据集/ml-100k/u1.base')
# for lines in fh_train:
#     user, item, score, _ = lines.strip().split('\t')
#     trainSet.setdefault(user, {})
#     trainSet[user][item] = int(score)
# # print(.trainSet.items())
# # print(trainSet)
# trainSet = pandas.DataFrame(trainSet).T.fillna(0)
# print(trainSet)
# trainSet = np.mat(trainSet)
# # print(trainSet)
# # print(type(trainSet))
# fh_train.close()
# result = trainSet[0, ]
# print(result)


#########################################

#建立字典！！！！！！！！！！！！！value ！！！！！key！！！！！！！！！！！！
# dict1 = {'a':{'b':8, 'd':20}, 'c':{'d':9}}
# print(dict1)
# print(dict1.keys())
# print(dict1.values())
# dict2 = {}
# dict2.setdefault(1, 0)
# print(dict2)
# print(dict2.items())

######################
# fo = open("C:/Users/Administrator/Desktop/毕业设计/协同过滤/result/rslt_Euc.txt", 'w')
# a = [1, 2, 3, 4]
# seq = ["哈哈哈 1\n", "菜鸟教程 2"]
# fo.writelines( str(a) + '\n' )
# fo.writelines( seq )
# fo.close()
def precision(testSet, top_10):
    rec = [[0 for i in range(10)] for i in range(8)]
    for i in range(8):
        for j in range(10):
            rec[i][j] = top_10[i][j][0] # Create two-dimensional array to store 150 people's top_10
    len_test = 0
    prec = [0 for i in range(8)]
    for i in range(8):
        hit = 0
        for j in range(10):
            t = rec[i][j]
            if (t != 0 and testSet[i, t] != 0):
                hit += 1
            else:
                pass
        prec[i] = float(hit / 10) # Calculate precision
    return prec

#Recommend top_k for user i
def recommand_k(pred, k):
    top = []
    len_pred = len(pred)
    if k >= len_pred:
        top = pred
    else:
        for i in range(k):
            top.append(pred[i])
    return top
def get_recommodation(trainSet, user, w):
    num_train_user, num_train_item = np.shape(trainSet)
    info_user_item = trainSet[user,]
    # print(info_user_item)

    not_in_item = []
    for i in range(0, num_train_item):
        if info_user_item[0, i] == 0:
            not_in_item.append(i)
    # 初始化预测字典
    pred = {}
    for m in not_in_item:
        item = np.copy(trainSet[:, m])
        for n in range(num_train_user):
            if item[n, 0] != 0:
                if m not in pred:
                    pred[m] = w[user, n] * item[n, 0]
                else:
                    pred[m] = pred[m] + w[user, n] * item[n, 0]
    return pred


trainSet = np.loadtxt(open("trainSet.csv", "rb"), delimiter=",", skiprows=0)
trainSet = np.mat(trainSet)
testSet = np.loadtxt(open("testSet.txt", "rb"), delimiter=",", skiprows=0)
testSet = np.mat(testSet)
w = np.loadtxt(open("Similarity_Euclidean.csv", "rb"), delimiter=",", skiprows=0)
w = np.mat(w)
top_10 = []
for i in range(0, 8):
    pred = get_recommodation(trainSet, i, w)
    result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
    top_10_i = recommand_k(result, 10)
    # print(top_10_i)
    top_10.append(top_10_i)
# print('欧几里得相似度推荐：')
# print(top_10)
# print('________________________________________________________')
prec = precision(testSet, top_10)
# print(prec)
rate = sum(prec) / len(prec)
# print(rate)
print("end worker_1")
print(prec)
file = open('C:/Users/Administrator/Desktop/毕业设计/协同过滤/result/rslt_Euc.txt', 'w')
seq1 = ["基于欧氏距离相似度的推荐\n"]
for i in range(8):
    file.writelines("用户" + str(i) + "推荐的电影列表")
    for j in range(10):
        file.writelines(str(top_10[i][j][0]) + '\t')
    file.writelines('推荐准确度：' + str(prec[i]))
    file.writelines('\n')
file.writelines('平均推荐精准度：' + str(rate))
file.close()