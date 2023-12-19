import numpy as np

def tfn(X, remove_dead_features=True, dtype=np.float32):
    """ normalized term frequency-inverse document frequency """
    N, D = X.shape
    Nj = (X > 0).sum(axis=0) # number of documents containing word j = 1, ..., D

    if remove_dead_features:
        idx = np.where(Nj > 0)[0]
        if len(idx) < N:
            print('tfn: discarding ' + str(N - len(idx)) + ' words with zero occurence from dictionary.')
        sub_sample_features = False
        if sub_sample_features:
            D_max = 200
            D = np.min((D,D_max))
            idx__ = np.argsort(Nj)
            idx = idx__[-D:]
        D = len(idx)
        X, Nj = X[:,idx], Nj[idx]

    gj = np.log(N/Nj)
    si = 1. / np.sqrt(((X * gj.reshape(1,D))**2).sum(axis=1))

    return X * np.outer(si, gj)

def filter_features_against_stopwords(X, dictionary):
    """ filter word dictionary and word-document occurence matrix against a list of stopwords. """
    stopwords = np.loadtxt('data/stoplist_smart.txt', dtype=str)
    idx = np.invert(np.isin(dictionary, stopwords)).flatten()

    return X[:,idx], dictionary[idx]
