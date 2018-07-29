import json
import argparse

import numpy as np
import pylab as pl
import sklearn.feature_extraction as sfe

__author__ = 'nikolayanokhin'


def main():
    np.set_printoptions(threshold=np.nan)

    print "Welcome to the EDA tutorial"
    args = parse_args()

    domains, data = read_data(args.data[0])

    vectorizer = sfe.DictVectorizer(sparse=True, dtype=int)
    x = vectorizer.fit_transform(data)
    print "Vectorized data %s[%dx%d]" % (type(x).__name__, x.shape[0], x.shape[1])

    td, tc = top_domain_user_counts(x, domains, args.ndomains)
    print_top_domains(td, tc)

    print "Plotting distributions"
    pl.subplot(121)
    plot_top_domains(td, tc)
    pl.subplot(122)
    plot_log_top_domains(td, tc)
    pl.show()


def plot_top_domains(top_domains, top_counts):
    """Plot domain count against domain rank in an ordinary plot

    Arguments:
    top_domains -- a list of top domains (strings) sorted according to their rank
    top_counts -- numpy array of user counts corresponding to each domain
    """
    x = np.arange(len(top_domains)) + 1
    pl.plot(x, top_counts, 'r-', lw=1)
    pl.title("Top %d domain counts" % len(top_domains))
    pl.xlabel("Domain rank")
    pl.ylabel("Unique user count")


def plot_log_top_domains(top_domains, top_counts):
    """Plot domain count against domain rank in a log-log scale plot

    Arguments:
    top_domains -- a list of top domains (strings) sorted according to their rank
    top_counts -- numpy array of user counts corresponding to each domain
    """

    x = np.log1p(np.arange(len(top_domains)))
    y = np.log1p(top_counts)
    pl.plot(x, y, 'b.', ls='None')
    plot_linear_model(x, y)
    pl.title("Top %d domain counts (log-log)" % len(top_domains))
    pl.xlabel("Log domain rank")
    pl.ylabel("Log unique user count")

    # TODO: Change this do do the actual job
    # plot_top_domains(top_domains, top_counts)
    # TODO: Call plot_linear_model here


def plot_linear_model(x, y):
    """Fit and plot a linear interpolation line, based on given x and y

    Arguments:
    x -- numpy array of x-coordinates
    y -- numpy array of y coordinates
    """
    # TODO: Change this do do the actual job
    pass


def top_domain_user_counts(mat, domains, n):
    """
    Given a CSR matrix of domain - user correspondence,
    compute user count for each domain and return top domains,
    sorted by their rank along with the counts.

    Arguments:
    mat -- CSR matrix with domains in rows and users in columns
    domains -- a list if domain names corresponding to the rows of mat
    n -- n top domains to select

    Returns:
    top_domains -- a list of top domains (strings) sorted according to their rank
    top_counts -- numpy array of user counts corresponding to each domain
    """
    # TODO: Change this do do the actual job
    res = []
    for i in mat:
        if (i in domains):
            res.append(i)
    res.sort()

    print "Counting users for each domain"
    user_counts = np.diff(mat.indptr)
    print "Sorting domains by user count"
    sort_ind = np.argsort(-user_counts)
    print "Taking %d top domains" %n
    top_counts = user_counts[sort_ind[:n]]
    top_domains = [domains[i] for i in sort_ind[:n]]
    return top_domains, top_counts



def print_top_domains(top_domains, top_counts):
    """Print given domains with their respective counts

    Arguments:
    top_domains -- a list of top domains (strings) sorted according to their rank
    top_counts -- numpy array of user counts corresponding to each domain
    """
    print "Top %d domains" % len(top_domains)
    for i in xrange(len(top_domains)):
        print "%d\t%d\t%s" % (i + 1, top_counts[i], top_domains[i])


def read_data(t_sne_path):
    """Read data set as CSR matrix with list of domain names
    """
    data = []
    domains = []
    with open(t_sne_path, 'r') as t_sne_file:
        for line in t_sne_file:
            domain, domain_features_json = line.strip().split("\t")
            domains.append(domain)
            data.append(json.loads(domain_features_json))
    print "Read %d domains" % len(domains)
    return domains, data


def parse_args():
    parser = argparse.ArgumentParser(description='Exploratory data analysis')
    parser.add_argument("-n", "--ndomains", action="store", type=int, help="Top N domains to plot", default=1000)
    parser.add_argument("data", nargs=1, help="The file that contains domain visitors in json format")
    return parser.parse_args()


if __name__ == "__main__":
    main()
