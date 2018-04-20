#coding:utf-8
import numpy as np
import math
from numpy import linalg as la
import threading
from pylab import *
import matplotlib.pylab as plt
import multiprocessing
import time
from multiprocessing import Manager
# Calculate precision with testSet and top 10 movies
def precision(testSet, top_10):
    rec = [[0 for i in range(10)] for i in range(6)]
    for i in range(6):
        for j in range(10):
            rec[i][j] = top_10[i][j][0] # Create two-dimensional array to store 150 people's top_10
    len_test = 0
    prec = [0 for i in range(6)]
    for i in range(6):
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
def worker_1(interval, return_dict):
    print("worker_1")

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
    for i in range(0, 6):
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
    file = open('C:/Users/Administrator/Desktop/毕业设计/协同过滤/result/rslt_Euc.txt', 'w')
    seq1 = ["基于欧氏距离相似度的推荐\n"]
    file.writelines(seq1)
    for x in range(6):
        file.writelines("用户" + str(x) + "推荐的电影列表")
        for j in range(10):
            file.writelines(str(top_10[x][j][0]) + '\t')
        file.writelines('推荐准确度：' + str(prec[x]))
        file.writelines('\n')
    file.writelines('平均推荐精准度：' + str(rate))
    file.close()
    time.sleep(interval)
    print("end worker_1")
    return_dict[interval] = rate

def worker_2(interval, return_dict):
    print("worker_2")

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
    w = np.loadtxt(open("Similarity_Cos.csv", "rb"), delimiter=",", skiprows=0)
    w = np.mat(w)
    top_10 = []
    for i in range(0, 6):
        pred = get_recommodation(trainSet, i, w)
        result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
        top_10_i = recommand_k(result, 10)
        # print(top_10_i)
        top_10.append(top_10_i)
    # print('余弦相似度推荐')
    # print(top_10)
    # print('__________________________________________________________')
    prec = precision(testSet, top_10)
    # print(prec)
    rate = sum(prec) / len(prec)
    # print(rate)
    time.sleep(interval)
    print("end worker_2")
    file = open('C:/Users/Administrator/Desktop/毕业设计/协同过滤/result/rslt_Cos.txt', 'w')
    seq1 = ["基于余弦相似度的推荐\n"]
    file.writelines(seq1)
    for x in range(6):
        file.writelines("用户" + str(x) + "推荐的电影列表")
        for j in range(10):
            file.writelines(str(top_10[x][j][0]) + '\t')
        file.writelines('推荐准确度：' + str(prec[x]))
        file.writelines('\n')
    file.writelines('平均推荐精准度：' + str(rate))
    file.close()
    time.sleep(interval)
    print("end worker_1")
    return_dict[interval] = rate

def worker_3(interval, return_dict):
    print("worker_3")

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
    for i in range(0, 6):
        pred = get_recommodation(trainSet, i, w)
        result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
        top_10_i = recommand_k(result, 10)
        # print(top_10_i)
        top_10.append(top_10_i)
    # print('调整余弦相似度推荐：')
    # print(top_10)
    # print('__________________________________________')
    prec = precision(testSet, top_10)
    # print(prec)
    rate = sum(prec) / len(prec)
    # print(rate)
    time.sleep(interval)
    print("end worker_3")
    file = open('C:/Users/Administrator/Desktop/毕业设计/协同过滤/result/rslt_AdCos.txt', 'w')
    seq1 = ["基于调整余弦相似度的推荐\n"]
    file.writelines(seq1)
    for x in range(6):
        file.writelines("用户" + str(x) + "推荐的电影列表")
        for j in range(10):
            file.writelines(str(top_10[x][j][0]) + '\t')
        file.writelines('推荐准确度：' + str(prec[x]))
        file.writelines('\n')
    file.writelines('平均推荐精准度：' + str(rate))
    file.close()
    time.sleep(interval)
    print("end worker_1")
    return_dict[interval] = rate

# Process 4
def worker_4(interval, return_dict):
    print("worker_4")

    def get_recommodation(trainSet, user, w):
        num_train_user, num_train_item = np.shape(trainSet) # Get dimensions of trainSet
        info_user_item = trainSet[user,] # Get user ID

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
    for i in range(0, 6):
        pred = get_recommodation(trainSet, i, w)
        result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
        top_10_i = recommand_k(result, 10)
        # print(top_10_i)
        top_10.append(top_10_i)
    # print('Jaccard相似度推荐:')
    # print(top_10)
    # print('_______________________________')
    prec = precision(testSet, top_10)
    # print(prec)
    rate = sum(prec) / len(prec)
    # print(rate)
    time.sleep(interval)
    print("end worker_4")
    file = open('C:/Users/Administrator/Desktop/毕业设计/协同过滤/result/rslt_Jac.txt', 'w')
    seq1 = ["基于杰卡德相似度的推荐\n"]
    file.writelines(seq1)
    for x in range(6):
        file.writelines("用户" + str(x) + "推荐的电影列表")
        for j in range(10):
            file.writelines(str(top_10[x][j][0]) + '\t')
        file.writelines('推荐准确度：' + str(prec[x]))
        file.writelines('\n')
    file.writelines('平均推荐精准度：' + str(rate))
    file.close()
    time.sleep(interval)
    print("end worker_1")
    return_dict[interval] = rate

# Process 5
def worker_5(interval, return_dict):
    print("worker_5")

    def get_recommodation(trainSet, user, w):
        num_train_user, num_train_item = np.shape(trainSet) # Get dimensions of trainSet
        info_user_item = trainSet[user,] # Get user ID

        # Find the items without scores
        not_in_item = []
        for i in range(num_train_item):
            if info_user_item[0, i] == 0:
                not_in_item.append(i)
        # Initializate dictionary of prediction
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

    trainSet = np.loadtxt(open("trainSet.csv", "rb"), delimiter=",", skiprows=0) # Get matrix of trainSet from local
    trainSet = np.mat(trainSet) # Trainsform the type to np.matrix
    testSet = np.loadtxt(open("testSet.txt", "rb"), delimiter=",", skiprows=0) # Get matrix of testSet from local
    testSet = np.mat(testSet)  # Trainsform the type to np.matrix
    w = np.loadtxt(open("Similarity_Pearson.csv", "rb"), delimiter=",", skiprows=0) #Get matrix of similarity from local
    w = np.mat(w) # Trainsform the type to np.matrix
    top_10 = []
    for i in range(0, 6):
        pred = get_recommodation(trainSet, i, w)
        result = sorted(pred.items(), key=lambda d: d[1], reverse=False)
        top_10_i = recommand_k(result, 10)
        # print(top_10_i)
        top_10.append(top_10_i)
    # print('Pearsin相似度推荐:')
    # print(top_10)
    prec = precision(testSet, top_10)
    # print(prec)
    rate = sum(prec) / len(prec)
    # print(rate)
    time.sleep(interval)
    print("end worker_5")
    file = open('C:/Users/Administrator/Desktop/毕业设计/协同过滤/result/rslt_Pea.txt', 'w')
    seq1 = ["基于皮尔逊相似度的推荐\n"]
    file.writelines(seq1)
    for x in range(6):
        file.writelines("用户" + str(x) + "推荐的电影列表")
        for j in range(10):
            file.writelines(str(top_10[x][j][0]) + '\t')
        file.writelines('推荐准确度：' + str(prec[x]))
        file.writelines('\n')
    file.writelines('平均推荐精准度：' + str(rate))
    file.close()
    time.sleep(interval)
    print("end worker_1")
    return_dict[interval] = rate

if __name__ == "__main__":
    manager = Manager() # Inter-process communication
    return_dict = manager.dict() # Create a dictionary to storage the 'callback'
    jobs = []
    # Create 'multiprocessing'
    p1 = multiprocessing.Process(target = worker_1, args = (2, return_dict))
    jobs.append(p1)
    p1.start() # Start processing
    p2 = multiprocessing.Process(target = worker_2, args = (2.5, return_dict)) # Send parameters to processings
    jobs.append(p2)
    p2.start() # Start processing
    p3 = multiprocessing.Process(target = worker_3, args = (3, return_dict)) # Send parameters to processings
    jobs.append(p3)
    p3.start() # Start processing
    p4 = multiprocessing.Process(target = worker_4, args = (3.5, return_dict)) # Send parameters to processings
    jobs.append(p4)
    p4.start() # Start processing
    p5 = multiprocessing.Process(target = worker_5, args = (4, return_dict)) # Send parameters to processings
    jobs.append(p5)
    p5.start() # Start processing
    # Get the return value
    for proc in jobs:
        proc.join()
    # print(return_dict.values())
    rate1 = return_dict.values()[0]
    rate2 = return_dict.values()[1]
    rate3 = return_dict.values()[2]
    rate4 = return_dict.values()[3]
    rate5 = return_dict.values()[4]
    print("The number of CPU is:" + str(multiprocessing.cpu_count())) # Outprint the count of CPU`s core

    plt.rcParams['font.sans-serif'] = ['SimHei']  # Display normal Chinese labels
    plt.xlabel(u"实现算法") #
    plt.ylabel(u"推荐精准度")
    plt.title(u"协同过滤推荐精准度") # Set title
    xx = [u"欧氏距离", u"余弦距离", u"相似余弦", u"杰卡德", u"皮尔逊系数"] # Define Abscissa
    yy = [rate1, rate2, rate3, rate4, rate5] # Define Ordinate
    plt.bar(range(len(yy)), yy, color='rgb', tick_label=xx) # Generate bar
    plt.show() # Show the bar