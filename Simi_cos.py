import numpy as np
import math
from time import time
def Load_data(fh_train):
    fh_train = open(fh_train) # Open the file
    trainSet = np.zeros((944, 1683)) # Create an enmpy matrix to storage train set
    dict1 = { }
    # print(trainSet[0][0])
    for lines in fh_train:
        user, item, score, _ = lines.strip().split('\t') # Split the raw data into three parts
        dict1.setdefault(user, {})
        dict1[user][item] = float(score) # Create a dictionary to storage 'user''movie' and 'score'
        trainSet[int(user)][int(item)] = dict1[user][item]
    trainSet = np.delete(trainSet, 0, 1)
    trainSet = np.delete(trainSet, 0, 0)
    fh_train.close()
    return np.mat(trainSet)

def get_similarity_cos(trainSet):
    num_train_user = np.shape(trainSet)[0]  # 训练集中用户数量
    # print(num_train_user)
    # 初始化相似度矩阵
    w = np.mat(np.zeros((num_train_user, num_train_user))) # Initialize similarity matrix
    # Calculate the similarity
    for i in range(num_train_user):
        for j in range(i, num_train_user):
            if j != i:
                w[i, j] = get_cos(trainSet[i,], trainSet[j,])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w

def get_cos(vec1, vec2): # Calculate cosine similarity
    up = float(vec1 * vec2.T)
    down = np.sqrt(vec1 * vec1.T) * np.sqrt(vec2 * vec2.T)
    # down = la.norm(vec1) * la.norm(vec2)
    return (up / down) * 0.5 + 0.5

fh_train = 'C:/Users/Administrator/Desktop/毕业设计/数据集/ml-100k/u1.base' # Path of trainSet
trainSet = Load_data(fh_train)
t0 = time()
w = get_similarity_cos(trainSet) # Calculate similarity with 'cosine'
np.savetxt('Similarity_Cos.csv', w, delimiter = ',') # Save the similarity matrix as 'csv'
print('相似度计算完成时间' + str(time() - t0))