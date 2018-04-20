import numpy as np
import math
from time import time
# Load train set as matrix
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
    fh_train.close() # Close the file
    return np.mat(trainSet)

def get_similarity_euclidean(trainSet):
    num_train_user = np.shape(trainSet)[0]  # number of trainSet`s user
    # print(num_train_user)
    w = np.mat(np.zeros((num_train_user, num_train_user))) # Initialize similarity matrix
    # Calculate the similarity
    for i in range(num_train_user):
        for j in range(i, num_train_user):
            if j != i:
                w[i, j] = get_euclidean(trainSet[i,], trainSet[j,])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w

def get_euclidean(vec1, vec2): # Calculate Euclidean distance
    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return 1 / (1 + distance)

fh_test = 'C:/Users/Administrator/Desktop/毕业设计/数据集/ml-100k/u1.test' # Path of testSet
testSet = Load_data(fh_test)
np.savetxt('testSet.txt', testSet, delimiter = ',') # Save testSet as 'txt'

fh_train = 'C:/Users/Administrator/Desktop/毕业设计/数据集/ml-100k/u1.base' # Path of trainSet
trainSet = Load_data(fh_train)
np.savetxt('trainSet.txt', trainSet, delimiter = ',') #Save trainSet as 'txt'
t0 = time()
w = get_similarity_euclidean(trainSet) # Calculate similarity with 'Euclidean distance'
np.savetxt('Similarity_Euclidean.csv', w, delimiter = ',') # Save the similarity matrix as 'txt'
print('相似度计算完成时间' + str(time() - t0))