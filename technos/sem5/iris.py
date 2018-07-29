"""
This file contains misc experiments on linear classifiers
"""

import argparse
import numpy
import pylab
import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats as ss


def plot_data_set(x, y, species=None):
    print "Plotting data"
    pylab.figure(figsize=(10, 10))
    colors = numpy.array(['r', 'g', 'b'])[y]
    
    m = x.shape[1]
    for i in xrange(m):
        for j in xrange(i, m):            
            ax = pylab.subplot(m, m, m * i + j + 1)
            pylab.scatter(x[:, i], x[:, j], marker='o', c=colors, s=20, lw=0)
            pylab.xlabel("$x_%d$" % (i + 1))
            pylab.ylabel("$x_%d$" % (j + 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Take 2D data
            x_2d = x[:, [i, j]]
            x_train, x_test, y_train, y_test = train_test_split(x_2d, y, test_size=0.2)
            if species is not None and j > i:                
                # TODO: build all required models and plot decision functions           
                sgd = build_single_model(x_train, y_train, species, 'log')
                plot_decision_function(x_2d, sgd)
                print sgd.coef_, (sgd.coef_ * sgd.coef_).sum()
                # TODO: Combine models using One-VS-One and 
                # TODO: compute accuracy of the combination
                accuracy = compute_accuracy(x_test, y_test, [sgd])
                pylab.title("Accuracy: %.3f" % accuracy)
            elif species is None and j > i:
                models = []
                for k, l in [(0, 1), (1, 2), (0, 2)]:
                    sgd = build_single_model(x_train, y_train, [k, l], 'log')
                    plot_decision_function(x_2d, sgd)
                    models.append(sgd)
                accuracy = compute_accuracy(x_test, y_test, models)
                pylab.title("Accuracy: %.3f" % accuracy)
    
    pylab.show()


def build_single_model(x, y, species, loss, penalty='l2'):
    selected_items = [i for i, t in enumerate(y) if t in species]
    x, y = x[selected_items], y[selected_items]    
    sgd = SGDClassifier(loss=loss, penalty=penalty, shuffle=True, n_iter=200)
    sgd.fit(x, y)
    return sgd


def compute_accuracy(x_test, y_test, models):
    # TODO: Implement One-VS-One to improve accuracy
    pred = numpy.zeros((len(y_test), len(models)), dtype=int)
    for i, model in enumerate(models):
        pred[:, i] = model.predict(x_test)
    y_pred = numpy.zeros(len(y_test), dtype=int)
    for i in xrange(x_test.shape[0]):
        y_pred[i] = ss.mode(pred[i])[0][0]
    return accuracy_score(y_test, y_pred)  
    

def plot_decision_function(x, clf):
    x1s = numpy.linspace(x[:, 0].min(), x[:, 0].max(), 10)
    x2s = numpy.linspace(x[:, 1].min(), x[:, 1].max(), 10)

    x1, x2 = numpy.meshgrid(x1s, x2s)
    z = numpy.ones(x1.shape)

    for (i, j), v1 in numpy.ndenumerate(x1):        
        v2 = x2[i, j]                
        p = clf.decision_function([v1, v2])
        z[i, j] = p[0]

    pylab.contour(x1, x2, z, [0.0], colors='k', linestyles='dashed', linewidths=1.5)


def main():
    print "Welcome to the linear classification tutorial"
    args = parse_args()
    # Load Iris data set
    data = sklearn.datasets.load_iris()
    x, y = data["data"], data["target"]

    # Id -d option is set, print data set description, plot it and exit
    if args.show_desc:
        print data['DESCR']
        plot_data_set(x, y)
        exit(0)    

    # Plot data set along with the decision function
    plot_data_set(x, y, args.species)


def parse_args():
    parser = argparse.ArgumentParser(description='Experiments on linear classifiers applied to Iris data set')

    parser.add_argument('-d', dest='show_desc', action='store_true', default=False, help='Show data set description')

    parser.add_argument('-0', dest='species', action='append_const', const=0)
    parser.add_argument('-1', dest='species', action='append_const', const=1)
    parser.add_argument('-2', dest='species', action='append_const', const=2)

    return parser.parse_args()


if __name__ == "__main__":
    main()
