import numpy as np
import math
from time import time
import pylab as pl
import matplotlib.pylab as plt
from numpy import linalg as la
import threading

#u1_base: user:int___item:int___score:int___time stamp:int

###load data
def Load_data(fh_train):
    trainSet = dict()#set up training set as 'dictionary'
    fh_train = open(fh_train)
    for lines in fh_train:#split data for three parts:user,item and score
        user, item, score, _ = lines.strip().split('\t')
        trainSet.setdefault(user, {})# set up two-dimensional dictionary
        trainSet[user][item] = int(score) #first dimension is user.scond one is item
    trainSet = pandas.DataFrame(trainSet).T.fillna(0)#transform the 'dict' type to 'pandas.mat'
    trainSet = np.mat(trainSet)#transform type of 'pandas.mat' to 'numpy.mat'
    # print(trainSet)
    # print(type(trainSet))
    fh_train.close()
    return trainSet

###
def get_recommodation(trainSet, user, w, function):
    num_train_item, num_train_user = np.shape(trainSet)
    info_user_item = trainSet[user, ]

    not_in_item = []
    for i in range(num_train_user):
        if info_user_item[0, i] == 0:
            not_in_item.append(i)

    #初始化预测字典
    pred = {}
    for m in not_in_item:
        item = np.copy(trainSet[:, m])
        for n in range(num_train_item):
            if item[n, 0] != 0:
                if m not in pred:
                    pred[m] = w[user, n] * item[n, 0]
                else:
                    pred[m] = pred[m] + w[user, n] * item[n, 0]
    if function == '1':#余弦
        result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
    elif function == '2':#欧几里得
        result = sorted(pred.items(), key=lambda d: d[1], reverse=False)
    elif function == '3':#杰卡德
        result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
    elif function == '4':#调整余弦
        result = sorted(pred.items(), key=lambda d: d[1], reverse=True)
    return result, pred

###selecting function
def select_function(function, trainSet):
    if function == '1':#cosine similarity
        res = get_similarity_cos(trainSet)
    elif function == '2':#Euclidean similarity
        res = get_similarity_euclidean(trainSet)
    elif function == '3':#Jaccard similarity
        res = get_similarity_Jaccard(trainSet)
    elif function == '4':#adjustied cosine similarity
        res = get_similarity_adjusted_cos(trainSet)
    return res

###set up matrix of user-user`s similarity with Jaccard
def get_similarity_Jaccard(trainSet):
    num_train_user = np.shape(trainSet)[0]  # number of train set`s user
    # print(num_train_user)
    w = np.mat(np.zeros((num_train_user, num_train_user)))# initializate matrix
    # get the matrix from train set
    for i in range(num_train_user):
        for j in range(i, num_train_user):
            if j != i:
                w[i, j] = get_Jaccard(trainSet[i,], trainSet[j,])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w

###set up matrix of user-user`s similarity with cosine
def get_similarity_cos(trainSet):
    num_train_user = np.shape(trainSet)[0]  # number of trainset`s user
    # print(num_train_user)
    #初始化相似度矩阵
    w = np.mat(np.zeros((num_train_user, num_train_user)))
    #计算相似度矩阵
    for i in range(num_train_user):
        for j in range(i, num_train_user):
            if j != i:
                w[i, j] = get_cos(trainSet[i, ], trainSet[j, ])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w
def get_similarity_euclidean(trainSet):
    num_train_user = np.shape(trainSet)[0]  #训练集中用户数量
    # print(num_train_user)
    #初始化相似度矩阵
    w = np.mat(np.zeros((num_train_user, num_train_user)))
    #计算相似度矩阵
    for i in range(num_train_user):
        for j in range(i, num_train_user):
            if j != i:
                w[i, j] = get_euclidean(trainSet[i, ], trainSet[j, ])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w
def get_similarity_adjusted_cos(trainSet):
    num_train_user = np.shape(trainSet)[0]  #训练集中用户数量
    # print(num_train_user)
    #初始化相似度矩阵
    w = np.mat(np.zeros((num_train_user, num_train_user)))
    #计算相似度矩阵
    for i in range(num_train_user):
        for j in range(i, num_train_user):
            if j != i:
                w[i, j] = get_adjusted_cos(trainSet[i, ], trainSet[j, ])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w

def get_euclidean(vec1, vec2):
    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return 1/(1 + distance**.5)
def get_cos(vec1, vec2):
    up = float(vec1 * vec2.T)
    down = np.sqrt(vec1 * vec1.T) * np.sqrt(vec2 * vec2.T)
    # down = la.norm(vec1) * la.norm(vec2)
    return (up/down) * 0.5 + 0.5
def get_Jaccard(vec1, vec2):
    up = np.double(np.bitwise_and((vec1 != vec2), np.bitwise_or(vec1 != 0, vec2 != 0)).sum())
    down = np.double(np.bitwise_or(vec1 != 0, vec2 != 0).sum())
    distance = up/down
    return distance
def get_adjusted_cos(vec1, vec2):
    a_vec1 = np.sum(vec1, axis=1)/len(vec1) - vec1
    a_vec2 = np.sum(vec2, axis=1)/len(vec2) - vec2
    up = (a_vec1) * (a_vec2).T
    down = np.sqrt((a_vec1)*(a_vec1).T) * np.sqrt((a_vec2) * (a_vec2).T)
    return (up/down)[0, 0]

def recommand_k(pred, k):
    top = []
    len_pred = len(pred)
    if k >= len_pred:
        top = pred
    else:
        for i in range(k):
            top.append(pred[i])
    return top

if __name__ == '__main__':
    fh_train = 'C:/Users/Administrator/Desktop/毕业设计/数据集/ml-100k/u1.base'
    print('加载数据:')
    t0 = time()
    trainSet = Load_data(fh_train)
    print('加载数据完成时间' + str(time() - t0))
    print('___________________________')
    # trainSet = trainSet.T
    # print(trainSet)
    print('请选择相似度计算函数：1---余弦相似度、2---欧几里得距离、3---Jaccard相似系数、4---调整余弦。。。。。。。。。。。')
    function = input()
    print('计算相似度：')
    t0 = time()
    w = select_function(function, trainSet)
    print(w)
    print('相似度完成时间' + str(time() - t0))
    print('___________________________')
    print("推荐前10个")
    result, pred = get_recommodation(trainSet, 1, w, function)
    top_10 = recommand_k(result, 10)
    # print(top)
    print(top_10)
    x = list(pred.keys())
    y = list(pred.values())
    plt.plot(x, y, color='red')
    plt.show()




