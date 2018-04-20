#coding:utf-8
import numpy as np
import math
from time import time
from numpy import linalg as la
import threading
from pylab import *
import matplotlib.pylab as plt
def precession(testSet, top_10):
    rec = [[0 for i in range(10)] for i in range(8)]
    for i in range(8):
        for j in range(10):
            rec[i][j] = top_10[i][j][0]
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
        prec[i] = float(hit / 10)
    return prec

def recommand_k(pred, k):
    top = []
    len_pred = len(pred)
    if k >= len_pred:
        top = pred
    else:
        for i in range(k):
            top.append(pred[i])
    return top

class A(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)#初始化线程
    def run(self):
        def get_recommodation(trainSet, user, w):
            num_train_user, num_train_item = np.shape(trainSet)
            info_user_item = trainSet[user,]
            # print(info_user_item)

            not_in_item = [ ]
            for i in range(0, num_train_item):
                if info_user_item[0, i] == 0:
                    not_in_item.append(i)
            # 初始化预测字典
            pred = { }
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
        print('欧几里得相似度推荐：')
        print(top_10)
        print('________________________________________________________')
        prec = precession(testSet, top_10)
        print(prec)
        rate = sum(prec) / len(prec)
        print(rate)
        return rate


class B(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)  # 初始化线程
    def run(self):
        def get_recommodation(trainSet, user, w):
            num_train_user, num_train_item = np.shape(trainSet)
            info_user_item = trainSet[user, ]

            not_in_item = [ ]
            for i in range(num_train_item):
                if info_user_item[0, i] == 0:
                    not_in_item.append(i)
            # 初始化预测字典
            pred = { }
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
        w = np.loadtxt(open("Similarity_Cos.csv", "rb"), delimiter=",", skiprows=0)
        w = np.mat(w)
        top_10 = []
        for i in range(0, 8):
            pred = get_recommodation(trainSet, i, w)
            result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
            top_10_i = recommand_k(result, 10)
            # print(top_10_i)
            top_10.append(top_10_i)
        print('余弦相似度推荐')
        print(top_10)
        print('__________________________________________________________')
        prec = precession(testSet, top_10)
        print(prec)
        rate = sum(prec) / len(prec)
        print(rate)
        return rate

class C(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)  # 初始化线程
    def run(self):
        def get_recommodation(trainSet, user, w):
            num_train_user, num_train_item = np.shape(trainSet)
            info_user_item = trainSet[user,]

            not_in_item = []
            for i in range(num_train_item):
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
        w = np.loadtxt(open("Similarity_Adcos.csv", "rb"), delimiter=",", skiprows=0)
        w = np.mat(w)
        top_10 = []
        for i in range(0, 8):
            pred = get_recommodation(trainSet, i, w)
            result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
            top_10_i = recommand_k(result, 10)
            # print(top_10_i)
            top_10.append(top_10_i)
        print('调整余弦相似度推荐：')
        print(top_10)
        print('__________________________________________')
        prec = precession(testSet, top_10)
        print(prec)
        rate = sum(prec) / len(prec)
        print(rate)
        return rate

class D(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)  # 初始化线程
    def run(self):
        def get_recommodation(trainSet, user, w):
            num_train_user, num_train_item = np.shape(trainSet)
            info_user_item = trainSet[user,]

            not_in_item = []
            for i in range(num_train_item):
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
        w = np.loadtxt(open("Similarity_Jaccard.csv", "rb"), delimiter=",", skiprows=0)
        w = np.mat(w)
        top_10 = []
        for i in range(0, 8):
            pred = get_recommodation(trainSet, i, w)
            result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
            top_10_i = recommand_k(result, 10)
            # print(top_10_i)
            top_10.append(top_10_i)
        print('Jaccard相似度推荐:')
        print(top_10)
        print('_______________________________')
        prec = precession(testSet, top_10)
        print(prec)
        rate = sum(prec) / len(prec)
        print(rate)
        return rate

class E(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)  # 初始化线程
    def run(self):
        def get_recommodation(trainSet, user, w):
            num_train_user, num_train_item = np.shape(trainSet)
            info_user_item = trainSet[user,]

            not_in_item = []
            for i in range(num_train_item):
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
        w = np.loadtxt(open("Similarity_Pearson.txt", "rb"), delimiter=",", skiprows=0)
        w = np.mat(w)
        top_10 = []
        for i in range(0, 8):
            pred = get_recommodation(trainSet, i, w)
            result = sorted(pred.items(), key=lambda d: d[1], reverse=False)
            top_10_i = recommand_k(result, 10)
            # print(top_10_i)
            top_10.append(top_10_i)
        print('Pearsin相似度推荐:')
        print(top_10)
        prec = precession(testSet, top_10)
        print(prec)
        rate = sum(prec) / len(prec)
        print(rate)
        return rate
t1 = A()  # 线程的实例化
# t1.start()
euc_rate = A()
rate1 = euc_rate.run()
t2 = B()
# t2.start()
cos_rate = B()
rate2 = cos_rate.run()
t3 = C()
# t3.start()
ad_rate = C()
rate3 = ad_rate.run()
t4 = D()
# t4.start()
Jac_rate = D()
rate4 = Jac_rate.run()
t5 = E()
# t5.start()
Pe_rate = E()
rate5 = Pe_rate.run()


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
print('______________________________')
print(rate1, rate2, rate3, rate4, rate5)
print('______________________________')

plt.xlabel(u"实现算法")
plt.ylabel(u"推荐精准度")
plt.title(u"协同过滤推荐精准度")

xx = [u"欧氏距离", u"余弦距离", u"相似余弦", u"杰卡德", u"皮尔逊系数"]
yy = [rate1, rate2, rate3, rate4, rate5]
plt.bar(range(len(yy)), yy,color='rgb',tick_label=xx)
plt.show()





