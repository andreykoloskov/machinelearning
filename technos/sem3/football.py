# ! /usr/bin/env python
import argparse

import scipy.misc as sm
import scipy.cluster.hierarchy as sh
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    print "Welcome to Clustering tutorial"
    args = parse_args()

    img_names, img_mats = read_images(args.img_dir[0])
    x = build_data_set(img_mats, args.bins)

    # Do hierarchical clustering
    link = sh.linkage(x, args.linkage)

    plt.subplot(1, 2, 1)
    plot_dendrogram(link, img_names)
    plt.subplot(1, 2, 2)

    possibles = globals().copy()
    possibles.update(locals())
    criterion_fun = possibles.get(args.criterion + "_criterion")
    plot_criterion(x, link, criterion_fun, args.k)

    plt.show()
    pass


def read_images(img_dir):
    print "Reading images from %s" % img_dir
    img_mats = []
    img_names = []
    for subdir, dirs, names in os.walk(img_dir):
        for img_name in names:
            img_names.append(img_name)
            img_path = os.path.join(subdir, img_name)
            img_mat = sm.imread(img_path)
            img_mats.append(img_mat)
    print "Successfully read %d images" % len(img_names)
    return img_names, img_mats


def build_data_set(img_mats, bins=4):
    print "Building data set from %d images" % len(img_mats)
    x = np.zeros((len(img_mats), bins ** 3))
    for i, img_mat in enumerate(img_mats):
        x[i] = _image_histogram(img_mat, bins)
    print "Building data set done"
    return x


def _image_histogram(img, bins=4):
    assert bins >= 1
    assert bins <= 256
    assert bins % 2 == 0
    bin_len = 256 / bins
    hist = np.zeros((bins, bins, bins), dtype=int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            r_bin = img[i, j, 0] / bin_len
            g_bin = img[i, j, 1] / bin_len
            b_bin = img[i, j, 2] / bin_len
            hist[r_bin, g_bin, b_bin] += 1
    return 1.0 * hist.ravel() / hist.sum()


def plot_dendrogram(link, labels):
    print "Plotting dendrogram"
    sh.dendrogram(link, labels=labels, show_leaf_counts=True)
    plt.xticks(rotation=90)


def plot_criterion(x, link, criterion_fun, k):
    print "Plotting criterion"
    crs = np.zeros(k)
    ks = np.zeros(k)
    for j in xrange(k):
        labels = sh.fcluster(link, 1.01 * link[-j-1, 2], 'distance')
        crs[j] = criterion_fun(x, labels)
        ks[j] = len(np.unique(labels))
    plt.plot(ks, crs)


def squared_criterion(x, labels):
    u_labels = np.unique(labels)
    s = 0.0
    for label in u_labels:
        xl = x[labels == label, :]
        centroid = np.mean(xl, axis=0)
        s += np.sum((xl - centroid) * (xl - centroid))
    return s/len(u_labels)


def diameter_criterion(x, labels):
    # TODO: Implement diameter criterion
    u_labels = np.unique(labels)
    s = 0.0
    for label in u_labels:
        xl = x[labels == label, :]
	d = 0
	for x1 in xl:
	    for x2 in xl:
		l = np.sqrt(np.sum((x1 - x2) * (x1 - x2)))
		if l > d:
		    d = l
	s += d
    return s / len(u_labels)
    #return np.sqrt(len(np.unique(labels)))


def silhouette_criterion(x, labels):
    # TODO: Implement silhouette criterion
    return len(np.unique(labels)) ** 2


def parse_args():
    parser = argparse.ArgumentParser(description='Hierarchical clustering')
    parser.add_argument('-b',
                        dest='bins',
                        type=int,
                        default=4,
                        help='the number of bins across each dimension in image histogram. Expected power of 2, <= 256')
    parser.add_argument('-l', dest='linkage', choices=['single', 'complete', 'average'], default='average')
    parser.add_argument('-c', dest='criterion', choices=['squared', 'diameter', 'silhouette'], default='diameter')
    parser.add_argument('-k', dest='k', type=int, default=10, help='the max number of clusters to test')
    parser.add_argument('img_dir', nargs=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
