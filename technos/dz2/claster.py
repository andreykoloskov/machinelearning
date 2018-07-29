import argparse
import numpy
import pylab
from scipy.cluster import *
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math
from itertools import combinations

__author__ = 'andreykoloskov'

def main():
    args = parse_args()
    data = read_data(args.data[0])
    #norm
    convert_data(data)
    #print data
    #K
    plt.subplot(1, 1, 1)
    plot_criterion(data, args.k)
    plt.show()
    #EM
    out = em(data, 3)
    #RAND INDEX
    data_dop = read_data_dop(args.data[0])
    out_dop = numerate(out)
    ri = randindex(data_dop, out_dop)
    #output
    #print numpy.array(out)
    print '\n'
    print 'Rand index = ' + str(ri)

def convert_data(data):
    for i in xrange(6):
        res = []
        for j, k in enumerate(data):
            res.append(k[i])
        mn = min(res)
        for j, k in enumerate(data):
            data[j][i] -= mn
        for j, k in enumerate(data):
            res.append(k[i])
        mx = max(res)
        for j, k in enumerate(data):
            data[j][i] /= mx
    return data                            

def em(data, k):
    #create first centroids
    centroids = [data[int(i * round(len(data) / k))] for i in xrange(k)]
    #iteration
    delta = 1
    eps = 0.000001
    while delta > eps:
    #for n in xrange(200):
        res = [[] for i in xrange(len(centroids))]
        #points
        for i, l in enumerate(data):
            dmin = 0
            centrmin = 0
            #centroids
            for j, k in enumerate(centroids):
                distance = 0
                for p in xrange(6):
                    distance += (k[p] - l[p]) * (k[p] - l[p])
                if (j == 0) or (j > 0 and distance < dmin):
                    dmin = distance
                    centrmin = j                
            res[centrmin].append(l)
        #recalculate centroids
        delta = 0
        for q in xrange(len(centroids)):
            newcentr = []
            for s in xrange(6):
                dcen = 0
                for r, m in enumerate(res[q]):
                    dcen += m[s]
                dcen /= len(res[q])
                newcentr.append(dcen)
            for t in xrange(6):
                delta += (centroids[q][t] - newcentr[t]) * (centroids[q][t] - newcentr[t])
            centroids[q] = newcentr
        print delta
        
    return res

def randindex(y_true, y_res):
    y_true = sorted(y_true, key=lambda st: st[1])
    y_res = sorted(y_res, key=lambda st: st[1])
    ri = 0
    combs = list(combinations(xrange(len(y_true)), 2))
    for f, s in combs:
        if (y_true[f][0] == y_true[s][0]) == (y_res[f][0] == y_res[s][0]):
            ri += 1
    return ri * 1.0 / len(combs)

def numerate(out):
    out2 = []
    for i, l in enumerate(out):
        for j, m in enumerate(l):
            k = []
            for t in xrange(6):
                k.append(m[t])
            k.insert(0, float(i + 1))
            out2.append(k)
    data2 = []
    for j, m in enumerate(out2):
        tmp = []
        tmp.append(m[0])
        tmp.append(str(m[1]) + '_' + str(m[2]) + '_' + str(m[3]) + '_' + str(m[4]) + '_' + str(m[5]) + '_' + str(m[6]))
        data2.append(tmp)
        
    return data2

def similarity(list1, list2, k):
    inp = [[] for i in xrange(k)]
    out = [[] for i in xrange(k)]
    for i, l in enumerate(list1):
        for j, m in enumerate(l):
            inp[i].append(str(m[0]) + '_' + str(m[1]) + '_' + str(m[2]) + '_' + str(m[3]) + '_' + str(m[4]) + '_' + str(m[5]))
    for i, l in enumerate(list2):
        for j, m in enumerate(l):
            out[i].append(str(m[0]) + '_' + str(m[1]) + '_' + str(m[2]) + '_' + str(m[3]) + '_' + str(m[4]) + '_' + str(m[5]))
    d = [[] for i in xrange(k)]
    i = 0
    while i < k:
        j = 0
        while j < k:
            d[i].append(len(set(inp[i]).intersection(set(out[j]))))
            j += 1
        i += 1
    sum = 0
    for i, l in enumerate(d):
        sum += max(d[i])
    return sum

def plot_criterion(data, k):
    crs = numpy.zeros(k)
    ks = numpy.zeros(k)
    for j in xrange(k):
        centroids = vq.kmeans(numpy.array(data), j + 1, iter=200)[0]
        ks[j] = j + 1
        crs[j] = kclaster(centroids, data)
    plt.xlabel('Number of clusters')
    plt.ylabel('Intra-cluster distance')
    plt.plot(ks, crs)

def kclaster(centroids, data):
    res = 0.0
    d = cdist(numpy.array(data), centroids, 'euclidean')
    for i, l in enumerate(d):
        res += l.min()
    res /= len(data)
    return res

def kmeans(data, k):
    centroids = vq.kmeans(numpy.array(data), k, iter=200)[0]
    res = [[] for i in xrange(len(centroids))]
    d = cdist(numpy.array(data), centroids, 'euclidean')
    for i, l in enumerate(d):
        res[l.tolist().index(l.min())].append(data[i])
    return res

def read_data(args_data):
    source = [row.strip().split(',') for row in file(args_data, 'r')]
    data = [map(float, row[2:]) for row in source[1:]]
    return data

def read_data2(args_data, k):
    res = [[] for i in xrange(k)]
    source = [row.strip().split(',') for row in file(args_data, 'r')]
    data = [map(float, row[1:]) for row in source[1:]]
    for i in data:
        res[int(i[0]) - 1].append(i[1:])
    return res

def read_data_dop(args_data):
    source = [row.strip().split(',') for row in file(args_data, 'r')]
    data = [map(float, row[1:]) for row in source[1:]]
    data2 = []
    for j, m in enumerate(data):
        tmp = []
        tmp.append(m[0])
        tmp.append(str(m[1]) + '_' + str(m[2]) + '_' + str(m[3]) + '_' + str(m[4]) + '_' + str(m[5]) + '_' + str(m[6]))
        data2.append(tmp)
    return data2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs=1)
    parser.add_argument('-k', dest='k', type=int, default=10, help='the max number of clusters to test')
    return parser.parse_args()

if __name__ == "__main__":
    main()

