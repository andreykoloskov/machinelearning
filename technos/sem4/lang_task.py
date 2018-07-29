import argparse
import codecs
import unicodedata
import operator

import nltk
import numpy

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from random import choice

import matplotlib.pyplot as plt


__author__ = 'Nikolay Anokhin'


def read_documents(data_path):
    """
    Reads a sequence of documents from the text file
    located on a given path.

    Returns:
        A generator of tuples (LANG_CODE, unicode)
    """
    with codecs.open(data_path, 'rU', "utf-8") as data_file:
        for line in data_file:
            lang, doc = line.strip().split('\t')
            yield lang, doc


def normalise_document(doc):
    """
    Convert document to lower-case and remove accents

    Returns:
        A normalised document as unicode
    """
    # TODO: Implement normalization here
    return ''.join(c for c in unicodedata.normalize('NFD', doc.lower()) if not unicodedata.combining(c))


def tokenize_document(doc, n):
    """
    Split document in N-Grams

    Returns:
        Iterable (generator or list) of unicode n-grams
    """
    # TODO: Implement tokenization here
    tokenizer = nltk.WordPunctTokenizer()
    for token in tokenizer.tokenize(doc):
        if len(token) >= n:
            for ngram in nltk.ngrams(token, n):
                yield u"".join(ngram)


def select_features(lang_freq, top_tokens):
    """
    From each language selects top_tokens to be used as features

    Returns:
        set(unicode tokens)
    """
    features = set()
    for lang, (lid, token_freq) in lang_freq.iteritems():
        sorted_token_freq = sorted(token_freq.iteritems(), key=operator.itemgetter(1), reverse=True)
        for token, freq in sorted_token_freq[:top_tokens]:
            features.add(token)
    return features


def keep_only_features(docs, features):
    """
    Removes non-feature tokens from the document representations
    """
    for token_freq in docs:
        for token in token_freq.keys():
            if token not in features:
                del token_freq[token]


def create_model():
    """
    Initialise an NB model, supported by Sklearn

    Returns:
        Sklearn model instance
    """
    return MultinomialNB()


def validate_model(model, x, y, folds=10):
    """
    Computes cross-validation score for the given data set and model.
    Plots ROC curve for the class specified by class_ind

    Returns:
        A numpy.array of accuracy scores.
    """
    scores = cross_val_score(model, x, y, cv=folds)
    return scores


def plot_roc(model, x, y, class_ind=0):
    """
        Compute and plot ROC cureve for one class
    """
    # Compute train-test split. Train model and do test prediction
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=0)
    fit = model.fit(x_train, y_train)
    y_prob = fit.predict_proba(x_test)
    # TODO: Use y_tes, y_prob to compute ROC curve
    # TODO: compute real TPR, FPR
    fpr = numpy.linspace(0.0, 1.0, 100) 
    tpr = fpr * fpr
    roc_auc = 42
    # Plot ROC curve
    plt.figure()
    plt.fill_between(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class index %s' % class_ind)
    plt.show()


def main():
    print "## Welcome to the Text Mining tutorial ##"

    args = parse_args()

    # A list of dicts, each representing one document in format:
    # {feature1: count1, ...}
    docs = []
    # Language code for each dict (0-based)
    langs = []
    # A list of tuples, each tuple corresponds to one language
    # First compunent is the code of the language, second is its token frequencies
    # Contains entries like {lang_code: (lang_id, {token_frequencies})}
    lang_freq = {}
    for lang, doc in read_documents(args.ds_path[0]):
        normalized_doc = normalise_document(doc)

        token_freq = {}
        for token in tokenize_document(normalized_doc, args.n):
            token_freq[token] = 1 + token_freq.get(token, 0)
            if lang not in lang_freq:
                print "Found language %s: %d" % (lang, len(lang_freq))
                lang_freq[lang] = (len(lang_freq), {})
            lang_freq[lang][1][token] = 1 + lang_freq[lang][1].get(token, 0)

        docs.append(token_freq)
        langs.append(lang_freq[lang][0])

    # Select top n features for each lang
    features = select_features(lang_freq, args.top_tokens)
    # Remove from documents all features except the selected
    keep_only_features(docs, features)

    # Transform documents to numpy matrix
    dv = DictVectorizer()
    x = dv.fit_transform(docs).todense()
    y = numpy.array(langs)

    model = create_model()
    # Print cross-validated accturacy
    scores = validate_model(model, x, y)
    print "Model mean accuracy: {}".format(numpy.mean(scores))
    # Plot ROC
    plot_roc(model, x, y, class_ind=args.class_ind)


def parse_args():
    parser = argparse.ArgumentParser(description='Experiments with Text Mining')    
    parser.add_argument('-n', dest='n', help='Stands for n that is used in n-grams', type=int)
    parser.add_argument('-t', dest='top_tokens', help='Top tokens to take in each language', type=int, default=10)
    parser.add_argument('-i', dest='class_ind', help='Index of the class to plot', type=int, default=3)
    parser.add_argument('ds_path', nargs=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
