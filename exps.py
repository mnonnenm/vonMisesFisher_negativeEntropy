import re
import numpy as np
import pyreadr
import matplotlib.pyplot as plt
import scipy.sparse

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups

from vMFne.utils_text import filter_features_against_stopwords, tfn
from run_exps import run_spkmeans, run_softmovMF, run_hardmovMF
from run_exps import run_softBregmanClustering, run_hardBregmanClustering
from vMFne.logpartition import gradΦ

def load_classic3(classic300=False, permute_order=True):
    """ Dataset loader for pre-processed classic3 datasets found online """

    np.random.seed(0)

    # load dataset

    datasource = 'cclust_package'
    assert datasource in ['ZILGM', 'cclust_package']

    if datasource == 'ZILGM':
        # retrieved from https://rdrr.io/github/bbeomjin/ZILGM/man/classic3.html
        # N = 3890, D = 5896, but highly reduntant, e.g. 'cir', 'circul', 'circular', 'circulatori'
        classic3 = pyreadr.read_r('data/classic3.RData')['classic3']
        X_raw = classic3.to_numpy()
        D_raw = X_raw.shape[1]
        labels = X_raw[:,-1]
        X_raw = X_raw[:,1:-1].astype(dtype=np.float32)
        word_lens = np.array([ len(classic3.keys()[1:-1][i]) for i in range(len(classic3.keys())-2) ])

    elif datasource == 'cclust_package':
        # retrieved from https://github.com/franrole/cclust_package/blob/master/datasets/classic3.mat
        # N = 3891, D = 4303, a bunch of which are 2-letter (not even words), but otherwise seems sensible 
        import scipy.io

        classic3 = scipy.io.loadmat('data/classic3.mat')
        X_raw = classic3['A'].toarray()
        D_raw = X_raw.shape[1]
        labels = classic3['labels']
        word_lens = np.array([ len(classic3['ms'][i,0][0]) for i in range(classic3['ms'].shape[0]) ])
        dictionary = classic3['ms']

        # remove 2-letter words
        #idx = word_lens > 2
        #X_raw, dictionary = X_raw[:,idx], dictionary[idx]
        #word_lens = word_lens[idx]
        N, D = X_raw.shape

    if classic300: # subsample 100 documents from each class for a total of N=300
        idx = np.concatenate([np.random.permutation(np.where(labels==k)[0])[:100] for k in range(3)])
        X_raw, labels = X_raw[idx], labels[idx]

    X_raw, dictionary = filter_features_against_stopwords(X_raw, dictionary)
    X, labels = tfn(X_raw, labels, lower=8/N, upper=0.15, dtype=np.float32)

    N, D = X.shape
    if permute_order:
        idx = np.random.permutation(N)
        X, labels = X[idx], labels[idx]

    print('\n')
    print('selecting D=' + str(D) + ' features out of ' + str(D_raw) + ' features in full dataset.')
    print('\n')
    
    return X, labels, dictionary


def load_classic3_sklearn(classic300=False, seed=0, permute_order=True, 
                          sparse_datamatrix=False, min_df=7, max_df=0.15):
    """ Dataset loader for TF-IDF applied to a copy of the original classic3 dataset """

    np.random.seed(0)

    # all data (3891 abstract) found in 3 large formatted text files
    # retrieved January 2024 form 
    # http://ir.dcs.gla.ac.uk/resources/test_collections/cran/
    # http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/
    # http://ir.dcs.gla.ac.uk/resources/test_collections/medl/
    root = 'data/'
    fns = ['MED.ALL', 'CISI.ALL', 'cran.all.1400']

    data,labels = [], []
    for i,fn in enumerate(fns):
        dataset = open(root + fn, "r").read()

        dataset = re.sub(re.compile("\n"), " ", dataset)
        # use .I (flag for new abstract) to separate the individual documents
        dataset = re.sub(re.compile("\\.I\s[0-9]+"), "\n ", dataset)
        vectorizer = CountVectorizer(token_pattern='\n.*')
        dataset = [string[2:] for string in vectorizer.build_tokenizer()(dataset)]
        if fn[:4] == 'CISI':
             # remove extra header
            dataset = [re.sub(re.compile("\s+\\.T\s"), "", string) for string in dataset]
            # remove authors
            dataset = [re.sub(re.compile("\s+\\.A\s.*\s\\.W\s+"), ". ", string) for string in dataset]
            # remove pre-processing results
            dataset = [re.sub(re.compile("\s\\.X.*"), "", string) for string in dataset]
        else: # MED and CRAN
            # remove header
            dataset = [re.sub(re.compile(".*\\.W\s+"), "", string) for string in dataset]
        data += dataset
        labels = np.concatenate((labels, i * np.ones(len(dataset), dtype=np.int32)))

    tokenizer = CountVectorizer()
    stopwords = np.loadtxt('data/stoplist_smart.txt', dtype=str).tolist()
    stopwords = tokenizer.fit(stopwords).get_feature_names_out().tolist()

    if classic300: # subsample 100 documents from each class for a total of N=300
        idx = np.concatenate([np.random.permutation(np.where(labels==k)[0])[:100] for k in range(20)])
        data, labels = [data[i] for i in idx], labels[idx]

    # TF-IDF, filtering for features with at least min_df occurences across all documents and 
    # which occur in at most 15% of all documents.
    vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(data)
    dictionary = vectorizer.vocabulary_

    # remove dead documents (i.e. those that don't contain a single word in the current vocabulary)
    idx = np.where(X.sum(axis=-1)>0.)[0]
    X, labels = X[idx], labels[idx]

    N, D = X.shape
    if permute_order:
        idx = np.random.permutation(N)
        X, labels = X[idx], labels[idx]

    print('\n')
    print('(N,D) = ', (N,D))
    print('\n')

    X = X if sparse_datamatrix else X.astype(np.float32).toarray()
    
    return X, labels, dictionary


def run_all_algs(fn_root, version, X, K_range, n_repets, max_iter, seed, verbose, κ_max, Ψ0):

    N,D = X.shape
    print('N,D', (N,D))
    μ_norm_max = np.linalg.norm(gradΦ(κ_max * np.ones(D)/np.sqrt(D)))

    if verbose: 
        print('done loading data.')
        print('μ_norm_max', μ_norm_max)

    if verbose:
        print('running spherical K-means fits')
    fn = fn_root + 'spkmeans_' + str(n_repets) + 'repets_seed_' + str(seed) + '_v' + str(version) + '_'
    run_spkmeans(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose)


    if verbose:
        print('running soft moVMF fits')
    fn = fn_root + 'softmovMF_' + str(n_repets) + 'repets_seed_' + str(seed) + '_no_tying_' + '_v' + str(version) + '_'
    run_softmovMF(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=False, κ_max=κ_max, init_with_spkmeans=False)

    if verbose:
        print('running soft Bregman Clustering fits')
    fn = fn_root + 'softBregClust_' + str(n_repets) + 'repets_seed_' + str(seed) + '_no_tying_' + '_v' + str(version) + '_'
    run_softBregmanClustering(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=False, μ_norm_max=μ_norm_max, init_with_spkmeans=False, Ψ0=Ψ0)

    if verbose:
        print('running soft moVMF fits with tied variances')
    fn = fn_root + 'softmovMF_' + str(n_repets) + 'repets_seed_' + str(seed) + '_with_tying_' + '_v' + str(version) + '_'
    run_softmovMF(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=True, κ_max=κ_max, init_with_spkmeans=False)

    if verbose:
        print('running soft Bregman Clustering fits with tied variances')
    fn = fn_root + 'softBregClust_' + str(n_repets) + 'repets_seed_' + str(seed) + '_with_tying_' + '_v' + str(version) + '_'
    run_softBregmanClustering(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=True, μ_norm_max=μ_norm_max, init_with_spkmeans=False, Ψ0=Ψ0)

    if verbose:
        print('running hard moVMF fits')
    fn = fn_root + 'hardmovMF_' + str(n_repets) + 'repets_seed_' + str(seed) + '_no_tying_' + '_v' + str(version) + '_'
    run_hardmovMF(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=False, κ_max=κ_max, init_with_spkmeans=False)

    if verbose:
        print('running hard Bregman Clustering fits')
    fn = fn_root + 'hardBregClust_' + str(n_repets) + 'repets_seed_' + str(seed) + '_no_tying_' + '_v' + str(version) + '_'
    run_hardBregmanClustering(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=False, μ_norm_max=μ_norm_max, init_with_spkmeans=False, Ψ0=Ψ0)

    if verbose:
        print('running hard moVMF fits with tied variances')
    fn = fn_root + 'hardmovMF_' + str(n_repets) + 'repets_seed_' + str(seed) + '_with_tying_' + '_v' + str(version) + '_'
    run_hardmovMF(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=True, κ_max=κ_max, init_with_spkmeans=False)

    if verbose:
        print('running hard Bregman Clustering fits with tied variances')
    fn = fn_root + 'hardBregClust_' + str(n_repets) + 'repets_seed_' + str(seed) + '_with_tying_' + '_v' + str(version) + '_'
    run_hardBregmanClustering(fn, X, K_range=K_range, n_repets=n_repets, max_iter=max_iter, seed=seed, verbose=verbose, 
                      tie_norms=True, μ_norm_max=μ_norm_max, init_with_spkmeans=False, Ψ0=Ψ0)


def run_all_classic3(fn_root='results/classic3_', n_repets=10, K_range=[2,3,4,5,6,7,8,9,10,11],
                 seed=0, max_iter=100, κ_max=10000., Ψ0=[None, 0.], version='0',
                 classic300=False, verbose=False, min_df=7, max_df=0.15):

    X, labels, dictionary = load_classic3_sklearn(classic300=classic300, min_df=min_df, max_df=max_df)
    run_all_algs(fn_root, version, X, K_range, n_repets, max_iter, seed, verbose, κ_max, Ψ0)


def load_news20_sklearn(subset='all', remove=('headers',), news20_small=False, seed=0,
                          permute_order=True, sparse_datamatrix=False, min_df=6, max_df=0.15):

    np.random.seed(seed)

    tokenizer = CountVectorizer()
    stopwords = np.loadtxt('data/stoplist_smart.txt', dtype=str).tolist()
    stopwords = tokenizer.fit(stopwords).get_feature_names_out().tolist()
    news20 = fetch_20newsgroups(subset=subset, remove=remove)
    labels = 1 * news20.target
    data = news20.data

    if news20_small: # subsample 100 documents from each class for a total of N=2000
        # for small dataset, remove the shortest messages first to avoid all-zero TF-ID vectors
        idx = np.where([len(doc) >= 20 for doc in news20.data])[0]
        data, labels = [data[i] for i in idx], labels[idx]

        idx = np.concatenate([np.random.permutation(np.where(labels==k)[0])[:100] for k in range(20)])
        data, labels = [data[i] for i in idx], labels[idx]

    # TF-IDF, filtering for features with at least min_df occurences across all documents and 
    # which occur in at most 15% of all documents.
    vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(data)

    dictionary = vectorizer.vocabulary_

    # remove dead documents (i.e. those that don't contain a single word in the current vocabulary)
    idx = np.where(X.sum(axis=-1)>0.)[0]
    X, labels = X[idx], labels[idx]

    N, D = X.shape
    if permute_order:
        idx = np.random.permutation(N)
        X, labels = X[idx], labels[idx]

    print('\n')
    print('(N,D) = ', (N,D))
    print('\n')

    X = X if sparse_datamatrix else X.astype(np.float32).toarray()

    return X, labels, dictionary


def load_news20_manual(only_train_data=False, news20_small=False, permute_order=True):

    np.random.seed(0)

    data_train = np.loadtxt('data/20news_preprocessed/train.data', dtype=int)
    data_train = scipy.sparse.coo_array((data_train[:,2], (data_train[:,0]-1, data_train[:,1]-1))).todense()

    if only_train_data:
        data = data_train
        labels = np.loadtxt('data/20news_preprocessed/train.label', dtype=int)
    else:
        data_test = np.loadtxt('data/20news_preprocessed/test.data', dtype=int)
        data_test = scipy.sparse.coo_array((data_test[:,2], (data_test[:,0]-1, data_test[:,1]-1))).todense()
        data_train = np.concatenate([data_train,
                                     np.zeros((data_train.shape[0], data_test.shape[1]-data_train.shape[1]),dtype=data_train.dtype)],
                                    axis=1)
        data = np.concatenate([data_train, data_test], axis=0)
        labels = np.concatenate([np.loadtxt('data/20news_preprocessed/train.label', dtype=int),
                                 np.loadtxt('data/20news_preprocessed/test.label', dtype=int)], axis=0)

    N, D_raw = data.shape

    dictionary = np.loadtxt('data/20news_preprocessed/vocabulary.txt', dtype=str)
    dictionary = dictionary[:D_raw] # training data alone does not contain whole dictionary actually

    data, dictionary = filter_features_against_stopwords(data, dictionary)
    labels = labels[data.sum(axis=1) > 0] # kick out that one document whose only occuring features 
    data = data[data.sum(axis=1) > 0]     # are the stopwords 'more', 'say' and 'need' ...

    X, labels = tfn(data, labels, upper=0.15, lower=7/N, dtype=np.float32)

    if news20_small: # subsample 100 documents from each class for a total of N=2000
        idx = np.concatenate([np.random.permutation(np.where(labels==k+1)[0])[:100] for k in range(20)])
        X, labels = X[idx], labels[idx]

    N, D = X.shape
    if permute_order:
        idx = np.random.permutation(N)
        X, labels = X[idx], labels[idx]

    print('\n')
    print('selecting D=' + str(D) + ' features out of ' + str(D_raw) + ' features in full dataset.')
    print('\n')

    return X, labels, dictionary


def run_all_news20(fn_root='results/news20_', n_repets=10, K_range=[4,8,12,16,20,24,28,32,36,40], 
                 seed=0, max_iter=100, κ_max=10000., Ψ0=[None, 0.], version='0', 
                 news20_small=False, verbose=False, min_df=6, max_df=0.15):
    
    X, labels, dictionary = load_news20_sklearn(news20_small=news20_small, min_df=min_df, max_df=max_df)
    run_all_algs(fn_root, version, X, K_range, n_repets, max_iter, seed, verbose, κ_max, Ψ0)
