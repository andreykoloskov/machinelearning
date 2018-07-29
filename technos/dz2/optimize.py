import argparse
import numpy as np
import pylab
from scipy.cluster import *
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from math import e
from itertools import combinations
import random
import sys
from multiprocessing import Pool
import timeit
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.metrics
from scipy.stats import linregress
from array import array



__author__ = 'andreykoloskov'


def main():
    args = parse_args()
    #input data    
    data = read_data(args.data[0])
    #mixed data
    data_mix = mix(data)
    data = [map(float, row[1:]) for row in data_mix[0:]]
    data1 = [map(float, row[0:]) for row in data_mix[0:len(data_mix) * 3 / 4]] 
    data2 = [map(float, row[0:]) for row in data_mix[len(data_mix) * 3 / 4:]]  
    #test_bgd(data, data1, data2)
    #print data
    


def mix(data):
    arr = []
    arr2 = []
    for i in xrange(len(data)):
        arr.append([random.Random().randint(0, 100000), i])
    arr1 = sorted(arr)
    for a, b in arr1:
        arr2.append(b)
    data_mix = data
    for i in xrange(len(data)):
        data_mix[arr2[i]] = data[i]
    return data_mix

def read_data(args_data):
    source = [row.strip().split(',') for row in file(args_data, 'r')]
    #data = [map(float, row[1:]) for row in source[0:]]
    data = [map(float, row[0:]) for row in source[0:]]
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs=1)
    return parser.parse_args()

if __name__ == "__main__":
    main()



def test_bgd(data, train, ytrain):
    #learning rate
    a = 0.0001
    #load data
    data = np.array(data)
    #print data
    data = sklearn.datasets.load_boston()
    print data
    #print data.data.shape
    
    #scale it, split into training and test sets
    features = sklearn.preprocessing.scale(data['data'][:,12])
    #print "1"
    #print features
    y = sklearn.preprocessing.scale(data['target'])
    train = features[:features.shape[0]/2]
    ytrain = y[:y.shape[0]/2]
    slope, intercept, r_value, p_value, std_err = linregress(train, ytrain)
    
    print 'Scipy linear regression ', intercept, slope
    
    features = np.array([np.append(1,i) for i in features])
    train = features[:features.shape[0]/2]
    test = features[features.shape[0]/2:]
    ytrain = y[:y.shape[0]/2]
    ytest = y[y.shape[0]/2:]
    
    #fit the model using bgd
    theta1, err = lr_bgd(a, train, ytrain)
    print 'BGD: ', theta1
    
    
    #measure the total error, print and plot results
    results = sum([(ytest[i]-np.dot(theta1,test[i]))**2 for i in range(test.shape[0])])
    print 'Total BGD error: ', results
    
    #plot the error
    plt.plot(err, linewidth=2)
    plt.xlabel('Iteration, i', fontsize=24)
    plt.ylabel(r'J($\theta$)', fontsize=24)
    plt.show()
    
    #plot test results
    x = [i[1] for i in test]
    plt.plot(x, ytest, 'ro', label='Data')
    
    #plot predictions
    plt.plot(x, [np.dot(theta1,i) for i in test], 'b-', linewidth=2, label='Model')
    plt.legend(loc=0)
    plt.xlabel('Scaled lower status of the population', fontsize=24)
    plt.ylabel('Scaled home price', fontsize=24)
    plt.show()
    

def lr_bgd(a, x, y, ep=0.001, max_iter=10000, debug=True):
    #if we are debugging, we'll collect the error at each iteration
    if debug: err = array('f',[])
    
    ### Initialize the algorithm state ###
    cvg = False
    m = 0
    n = x.shape[0]
    
    #intialize the parameter/weight vector
    t = np.zeros(x.shape[1])
    
    #for each training example, compute the gradient
    z = [(y[i]-np.dot(t,x[i]))*x[i] for i in range(n)]

    #update the parameters
    t = t + a * sum(z)
    
    #compute the total error
    pe = sum([(y[i]-np.dot(t,x[i]))**2 for i in range(n)])
    
    ### Iterate until convergence or max iterations ###
    while not cvg:
        z = [(y[i]-np.dot(t,x[i]))*x[i] for i in range(n)]
        t = t + a*sum(z)
        e = sum([(y[i]-np.dot(t,x[i]))**2 for i in range(n)])
        if debug: err.append(e)
        if abs(pe-e) <= ep:
            print '*** Converged, iterations: ', m, ' ***'
            cvg = True
        pe = e
        m+=1
        if m == max_iter:
            print '*** Max interactions exceeded ***'
            break
    if debug:
        return (t,err)
    else:
        return (t)

