import numpy as np

def tfn(X, labels, lower=0.02, upper=0.15, dtype=np.float32):
    """ normalized term frequency-inverse document frequency """
    N, D = X.shape
    Nj = (X > 0).sum(axis=0) # number of documents containing word j = 1, ..., D

    # filtering features for being top rare or too abundant
    l = np.floor(lower * N) # minimum count of documents for a feature to occur in
    u = np.ceil(upper * N)  # maximum count of documents for a feature to occur in
    idx = np.where((Nj >= l) & (Nj <= u))[0]
    if len(idx) < D:
        print('tfn: discarding ' + str(D - len(idx)) + ' words (too rare, too abundant) from dictionary.')
    X, Nj = X[:,idx], Nj[idx]

    # check if any documents lost all non-zero features in the feature-filter step
    idx = np.where(X.sum(axis=1)>0)[0]
    if len(idx) < N:
        print('tfn: discarding ' + str(N - len(idx)) + ' documents due to having only zero word counts in remaining dictionary')
    X, labels = X[idx,:], labels[idx]
    N, D = X.shape

    # compute actual tfn matrix
    gj = np.log(N/Nj)
    si = 1. / np.sqrt(((X * gj.reshape(1,D))**2).sum(axis=1))

    return X * np.outer(si, gj), labels

def filter_features_against_stopwords(X, dictionary):
    """ filter word dictionary and word-document occurence matrix against a list of stopwords. """
    stopwords = np.loadtxt('data/stoplist_smart.txt', dtype=str)
    idx = np.invert(np.isin(dictionary, stopwords)).flatten()

    return X[:,idx], dictionary[idx]
