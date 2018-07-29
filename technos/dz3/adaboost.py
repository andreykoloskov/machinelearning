from numpy import *
from sklearn import cross_validation
from sklearn import datasets

def main():
    #data = loadtxt('wine.data', dtype = 'float', delimiter = ',', usecols = range(1, 14))
    #head = loadtxt('wine.data', dtype = 'int', delimiter = ',', usecols = [0])
    #data = loadtxt('data.data', dtype = 'float', delimiter = ',', usecols = range(0, 6))
    #head = loadtxt('data.data', dtype = 'int', delimiter = ',', usecols = [6])

    iris = datasets.load_iris()
    data = iris.data[:, :4] 
    head = iris.target + 1

    print head

    trainData, testData, labelTrain, labelTest = cross_validation.train_test_split(data, head, test_size = 0.3, random_state = 0)

    finishHead = []
    for i in range(len(labelTest)):
        finishHead.append(-1)

    for i in range(labelTrain.max()):
        partCalc(labelTrain, trainData, testData, finishHead, i + 1)

    print "Result label:"
    print finishHead 
    print "" 
    print "Statistic:" 
    printStatistic(finishHead, labelTest, [1, 2, 3])


#part calculation
def partCalc(label, trainData, testData, finishHead, k):
    label1 = copy(label)
    for i in range(len(label1)):
        if label1[i] == k:
            label1[i] = 1
        else:
            label1[i] = -1

    classifier1 = train(trainData, label1, 400)
    finishHead1 = test(testData, classifier1)

    for i in range(len(finishHead)):
        if finishHead[i] == -1 and finishHead1[i] == 1:
            finishHead[i] = k


# Building weak stump function
def buildTree(d, l, D):
    #d - data, l - labels, D - vector of weights
    #vector of data
    dataMatrix = mat(d)
    #vector of labels
    labelmatrix = mat(l).T
    #m - count elements of data, n - count features
    m,n = shape(dataMatrix)
    #numstep - number of steps
    numstep = 10.0
    bestStump = {}
    bestClass = mat(zeros((n,1)))
    minErr = inf
    #loop for features
    for i in range(n):
        #minimum in this feature        
        datamin = dataMatrix[:,i].min()
        #maximum in this feature
        datamax = dataMatrix[:,i].max()
        #stepSize - step size
        stepSize = (datamax - datamin) / numstep
        #loop for steps
        for j in range(-1, int(numstep) + 1):
            #loop for left and right branches of tree
            for inequal in ['lt','gt']:
                #beginning of the current step in this feature
                threshold = datamin + float(j) * stepSize
                #stump to classify training data
                #dataMatrix - vector of data, i - ordinal number of this feature
                #threshold - beginning of the current step in the feature
                #inequal - left, right branche                
                predict = leaf(dataMatrix, i, threshold, inequal)
                err = mat(ones((m,1)))
                #if resalt of leaf == labelmatrix
                err[predict == labelmatrix] = 0
                #recalculate vector of weight
                weighted_err = D.T * err;
                #recalculated minErr, bestClass, bestStump
                if weighted_err < minErr:
                    minErr = weighted_err
                    bestClass = predict.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClass

#Use the weak stump to classify training data
#datamat - vector of data, dim - ordinal number of this feature
#threshold - beginning of the current step in the feature
#inequal - left, right branche
def leaf(datamat, dim, threshold, inequal):
    #vector of 1
    res = ones((shape(datamat)[0], 1))
    if inequal == 'lt':
        res[datamat[:, dim] <= threshold] = -1.0
    else:
        res[datamat[:, dim] > threshold] = -1.0
    return res

# Training
def train(data, label, num):
    err = 0.00001
    weakClassifiers = []
    #count of data
    m = shape(data)[0]
    #vector of weights
    D = mat(ones((m, 1)) / m)
    #null vector of data
    EnsembleClassEstimate = mat(zeros((m,1)))
    #loop for steps in adaboost
    for i in range(num):
        #build decision tree
        #bestStump - best leaf, error - error
        #classEstimate - best vector of predicats
        bestStump, error, classEstimate = buildTree(data, label, D)
        #correct
        b = float(0.5 * log((1.0 - error) / error))
        bestStump['b'] = b
        weakClassifiers.append(bestStump)
        #recalculated weight
        weightD = multiply((-1 * b * mat(label)).T, classEstimate)
        D = multiply(D, exp(weightD))
        D = D / D.sum()
        EnsembleClassEstimate += classEstimate * b
        EnsembleErrors = multiply(sign(EnsembleClassEstimate) != mat(label).T, ones((m,1)))
        errorRate = EnsembleErrors.sum() / m
        if errorRate <= err:
            break
    return weakClassifiers

# Applying adaboost classifier for a single data sample
def classify(data, classifier):
    mt = mat(data)
    m = shape(mt)[0]
    EnsembleClassEstimate = mat(zeros((m,1)))
    for i in range(len(classifier)):
        classEstimate = leaf(mt, classifier[i]['dim'], classifier[i]['threshold'], classifier[i]['ineq'])
        EnsembleClassEstimate += classifier[i]['b'] * classEstimate
    return sign(EnsembleClassEstimate)

# Testing
def test(data, classifier):
    label = []
    for i in range(shape(data)[0]):
        label.append(classify(data[i,:], classifier))
    return label


def printStatistic(finishHead, testHead, types):
    #Accurancy
    count = 0.0
    for i in range(len(finishHead)):
        if (finishHead[i] == testHead[i]):
            count += 1
    accurancy = count / len(finishHead)
    print 'accuracy:' + str(accurancy)
    #Precision
    tp = zeros(len(types))
    fn = zeros(len(types))
    for i in range(len(finishHead)):
        for cls in range(len(types)):
            if ((testHead[i] == types[cls]) and (finishHead[i] == types[cls])):
                tp[cls] += 1
            if ((finishHead[i] == types[cls]) and (testHead[i] != types[cls])):
                fn[cls] += 1
    precision = copy(tp)
    for i in range(len(tp)):
        if (tp[i] + fn[i] == 0):
            precision[i] = -1
        else:
            precision[i] /= tp[i] + fn[i]
    print 'precision:' + str(precision)
    #Recall
    tp = zeros(len(types))
    fn = zeros(len(types))
    for i in range(len(finishHead)):
        for cls in range(len(types)):
            if (testHead[i] == types[cls]):
                if (finishHead[i] == types[cls]):
                    tp[cls] += 1
                else:
                    fn[cls] += 1
    recall = copy(tp)
    for i in range(len(tp)):
        if (tp[i] + fn[i] == 0):
            recall[i] = -1
        else:
            recall[i] /= tp[i] + fn[i]
    print 'recall: ' + str(recall)

if __name__ == "__main__":
    main()
