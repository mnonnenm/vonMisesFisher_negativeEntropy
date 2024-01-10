import numpy as np
import pyreadr
import matplotlib.pyplot as plt
import scipy.sparse

from vMFne.utils_text import filter_features_against_stopwords, tfn
from run_exps import run_spkmeans, run_softmovMF, run_softBregmanClustering
from vMFne.logpartition import gradΦ


def load_classic3(classic300=False):

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

    X_raw, dictionary = filter_features_against_stopwords(X_raw, dictionary)
    X, labels = tfn(X_raw, labels, lower=8/N, upper=0.15, dtype=np.float32)

    if classic300: # subsample 100 documents from each class for a total of N=300
        idx = np.concatenate([np.random.permutation(np.where(labels==k)[0])[:100] for k in range(3)])
        X, labels = X[idx], labels[idx]

    N, D = X.shape
    idx = np.random.permutation(N)
    X, labels = X[idx], labels[idx]

    print('\n')
    print('selecting D=' + str(D) + ' features out of ' + str(D_raw) + ' features in full dataset.')
    print('\n')
    
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


def run_all_classic3(fn_root='results/classic3_', n_repets=10, K_range=[2,3,4,5,6,7,8,9,10,11],
                 seed=0, max_iter=100, κ_max=10000., Ψ0=[None, 0.], version='0',
                 classic300=False, verbose=False):

    X, labels, dictionary = load_classic3(classic300=classic300)
    run_all_algs(fn_root, version, X, K_range, n_repets, max_iter, seed, verbose, κ_max, Ψ0)


def load_news20(only_train_data=False, news20_small=False):

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
    idx = np.random.permutation(N)
    X, labels = X[idx], labels[idx]

    print('\n')
    print('selecting D=' + str(D) + ' features out of ' + str(D_raw) + ' features in full dataset.')
    print('\n')

    return X, labels, dictionary


def run_all_news20(fn_root='results/news20_', n_repets=10, K_range=[4,8,12,16,20,24,28,32,36,40], 
                 seed=0, max_iter=100, κ_max=10000., Ψ0=[None, 0.], version='0', 
                 only_train_data=False, news20_small=False, verbose=False):

    
    X, labels, dictionary = load_news20(only_train_data=only_train_data, news20_small=news20_small)
    run_all_algs(fn_root, version, X, K_range, n_repets, max_iter, seed, verbose, κ_max, Ψ0)
