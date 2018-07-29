import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
import math
import collections
import random

__author__ = 'andreykoloskov'

def main():
    data = np.loadtxt('wine.data', dtype = 'float', delimiter = ',', usecols = range(1, 14))
    target = np.loadtxt('wine.data', dtype = 'int', delimiter = ',', usecols = [0]) - 1
    normalize(data)
    trainData, testData, trainHead, testHead = cross_validation.train_test_split(data, target, test_size = 0.3, random_state = 0)
    linCoef = train(trainData, trainHead)
    finishHead = test(testData, testHead, linCoef)
    printStatistic(finishHead, testHead, [0, 1, 2])

def LR(x, w):
    return  1.0 / (1.0 + (math.e ** (-np.dot(x, w))))

def GD(trainData, trainHeadNew, step, lmbd, eps):
    row, col = trainData.shape
    linCoef = trainData[random.Random().randint(0, col - 1)]
    s1 = 0
    s2 = 0
    while True:
        for j in xrange(row):
            teta = 0
            for k in xrange(col):
                teta = teta + linCoef[k] * linCoef[k]
            s1 = math.sqrt(teta)
            teta = lmbd * s1
            linCoef = linCoef - step * (LR(trainData[j], linCoef) - trainHeadNew[j]) * trainData[j] + teta
            teta = 0
            for k in xrange(col):
                teta = teta + linCoef[k] * linCoef[k]
            s2 = math.sqrt(teta)
        if (abs(s1 - s2) < eps):
            return linCoef
    return linCoef

def normalize(data):
    for i in xrange(13):
        delit = 0
        for j, k in enumerate(data):
            delit = delit + k[i]
        delit = math.sqrt(delit)
        for j, k in enumerate(data):
            k[i] = k[i] / delit
    return data 

def train(trainData, trainHead):
    linCoef = np.zeros((3, len(trainData[0])))
    for i in range(3):
        trainHeadNew = np.copy(trainHead)
        for j in range(len(trainHeadNew)):
            if (trainHeadNew[j] == i):
                trainHeadNew[j] = 1
            else:
                trainHeadNew[j] = 0
        linCoef[i] = GD(trainData, trainHeadNew, 1, 0.00001, 0.005)
    return linCoef

def test(testData, testHead, linCoef):
    finishHead = np.zeros(len(testHead), dtype = int)    
    for i in range(len(testData)):
        finishHead[i] = 0
        for j in xrange(len(linCoef)):
            if LR(testData[i], linCoef[j]) > LR(testData[i], linCoef[finishHead[i]]):
                finishHead[i] = j
    return finishHead

def printStatistic(finishHead, testHead, types):
    #Accurancy
    count = 0.0
    for i in range(len(finishHead)):
        if (finishHead[i] == testHead[i]):
            count += 1
    accurancy = count / len(finishHead)
    print 'accuracy:' + str(accurancy)
    #Precision
    tp = np.zeros(len(types))
    fn = np.zeros(len(types))
    for i in range(len(finishHead)):
        for cls in range(len(types)):
            if ((testHead[i] == types[cls]) and (finishHead[i] == types[cls])):
                tp[cls] += 1
            if ((finishHead[i] == types[cls]) and (testHead[i] != types[cls])):
                fn[cls] += 1
    precision = np.copy(tp)
    for i in range(len(tp)):
        if (tp[i] + fn[i] == 0):
            precision[i] = -1
        else:
            precision[i] /= tp[i] + fn[i]
    print 'precision:' + str(precision)
    #Recall
    tp = np.zeros(len(types))
    fn = np.zeros(len(types))
    for i in range(len(finishHead)):
        for cls in range(len(types)):
            if (testHead[i] == types[cls]):
                if (finishHead[i] == types[cls]):
                    tp[cls] += 1
                else:
                    fn[cls] += 1
    recall = np.copy(tp)
    for i in range(len(tp)):
        if (tp[i] + fn[i] == 0):
            recall[i] = -1
        else:
            recall[i] /= tp[i] + fn[i]
    print 'recall: ' + str(recall)
    
if __name__ == "__main__":
    main()
