import argparse

import numpy
import pylab
import numpy.random
import sklearn.cluster as sc
import sklearn.datasets as sd
import math
from itertools import combinations


__author__ = 'Nikolay Anokhin'


COLORS = numpy.array(['r', 'g', 'b', 'm'])
MARKERS = numpy.array(['o', 's', 'v', 'D'])


def cluster_data(x, args):
    """
    Use scikit-learn KMeans implementation to cluster data
    """
    # TODO: Implement model restart to find global max
    print "Clustering data"
    model = sc.KMeans(n_clusters=args.classes, init='random')
    y_pred = model.fit_predict(x)
    return y_pred, model.inertia_


def transform_data(x, a, b):
    """
    Apply z = Ax + b linear transformation matrix to the data set x
    """
    print "Transforming data"
    z = numpy.dot(a, x.transpose()).transpose() + b
    return z


def rand_index(y_true, y_res):
    """
    Computes rand index (measure of cluster quality)
    """
    # TODO: Implement RAND index
    "Computing rand index"
    ri = 0
    combs = list(combinations(xrange(y_true.size), 2))
    for f, s in combs:
        if (y_true[f] == y_true[s]) == (y_res[f] == y_res[s]):
	    ri += 1
    return ri * 1.0 / len(combs)    


def generate_gauss(args):
    """
    Generates random mixture of Gaussians
    """
    print "Generating 'Gaussian mixture' data set"

    x, y = sd.make_classification(n_samples=args.size,
                                  n_features=2,
                                  n_informative=2,
                                  n_redundant=0,
                                  n_classes=args.classes,
                                  n_clusters_per_class=1,
                                  random_state=args.rdm)
    return x, y


def plot_data_set(x, y_true, y_res):
    """
    Plot data points with true cluster represented by point marker
    and predicted class by point color
    """
    for y in numpy.unique(y_true):
        x_true = x[y_true == y, :]
        marker = MARKERS[y]
        colors = COLORS[y_res[y_true == y]]
        pylab.scatter(x_true[:, 0], x_true[:, 1], marker=marker, c=colors, s=50)


def plot_centroids(x, y_res):
    """
    Plot cluster centroids
    """
    for y in numpy.unique(y_res):
        x_c = x[y_res == y, :]
        mean = numpy.mean(x_c, axis=0)
        pylab.scatter(mean[0], mean[1], marker='*', c=COLORS[y], s=400)


def main():
    print "## Welcome to the Text Mining tutorial ##"
    args = parse_args()

    x, y_true = generate_gauss(args)

    pylab.figure(figsize=(12, 6))

    # Cluster and plot original data
    pylab.subplot(1, 2, 1)
    y_pred, j = cluster_data(x, args)
    rand = rand_index(y_true, y_pred)
    plot_data_set(x, y_true, y_pred)
    plot_centroids(x, y_pred)
    pylab.title('Original data: J=%.3f, rand=%.3f' % (j, rand))

    # Apply scaling transformation
    # TODO: Try rotation, shift and general linear
    z, v_true = transform_data(x, numpy.array([[4, 0], [0, 1]]), numpy.array([0, 0])), y_true
    # Cluster and plot transformed data
    pylab.subplot(1, 2, 2)
    v_pred, j = cluster_data(z, args)
    rand = rand_index(v_true, v_pred)
    plot_data_set(z, v_true, v_pred)
    plot_centroids(z, v_pred)
    pylab.title('Transformed data: J=%.3f, rand=%.3f' % (j, rand))

    pylab.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Clustering tutorial')
    parser.add_argument('-s', dest='size', type=int, default=40)
    parser.add_argument('-c', dest='classes', type=int, default=4)
    parser.add_argument('-r', dest='rdm', type=int, default=5)
    args = parser.parse_args()

    if args.classes > len(COLORS):
        raise ValueError('At most {} classes supported'.format(len(COLORS)))

    return args


if __name__ == "__main__":
    main()
