from vMFne.bregman_clustering import spherical_kmeans
from vMFne.bregman_clustering import softBregmanClustering_vMF, posterior_marginal_vMF_mixture_Ψ
from vMFne.moVMF import moVMF, posterior_marginal_vMF_mixture_Φ
from vMFne.negentropy import gradΨ
import numpy as np

def run_spkmeans(fn, X, K_range, n_repets, max_iter=100, seed=0, verbose=False):

    assert np.allclose(np.linalg.norm(X,axis=-1), 1.)

    for K in K_range: # for each number of clusters ...
        if verbose:
            print('K='+str(K))
        all_c = []
        for ii in range(n_repets): # for n_repets different random initializations 
            np.random.seed(seed+ii)
            _, w, c = spherical_kmeans(X, K=K, max_iter=max_iter, verbose=False)
            all_c.append(1 * c)
            if verbose:
                print(' - ' + str(ii+1) + '/' + str(n_repets))
        np.save(fn + '_K_' + str(K), np.stack(all_c,axis=0))


def run_softmovMF(fn, X, K_range, n_repets, max_iter=100, seed=0, verbose=False, 
                  tie_norms=False, κ_max=np.inf, init_with_spkmeans=False):

    assert np.allclose(np.linalg.norm(X,axis=-1), 1.)
    N,D = X.shape

    for K in K_range: # for each number of clusters ...
        if verbose:
            print('K='+str(K))
        all_ηs, all_w, all_LL, all_c  = [], [], [], []
        for ii in range(n_repets): # for n_repets different random initializations 
            np.random.seed(seed+ii)
            if init_with_spkmeans:
                _, w, c = spherical_kmeans(X, K=K, max_iter=max_iter, verbose=False)
                all_c.append(1 * c)            
                μs = np.stack([X[c==k].mean(axis=0) for k in range(K)],axis=0)
                ηs = gradΨ(μs, D=D, Ψ0=1e-6)
            else:
                ηs, w = None, None
            ηs, w, LL = moVMF(X, K=K, max_iter=max_iter, verbose=False, 
                              ηs_init=ηs, w_init=w, tie_norms=tie_norms, κ_max=κ_max)
            all_ηs.append(1. * ηs)
            all_w.append(1. * w)
            all_LL.append(1. * LL)
            
            if verbose:
                print(' - ' + str(ii+1) + '/' + str(n_repets))

        np.savez(fn + '_K_' + str(K), out={
            'N' : N,
            'D' : D,
            'K' : K,
            'c' : np.stack(all_c,axis=0) if init_with_spkmeans else None,
            'LLs' : np.stack(all_LL,axis=0),
            'w' : np.stack(all_w,axis=0),
            'ηs' : np.stack(all_ηs,axis=0)
        }) 


def run_softBregmanClustering(fn, X, K_range, n_repets, max_iter=100, seed=0, verbose=False, 
                  tie_norms=False, μ_norm_max=0.99, init_with_spkmeans=False, Ψ0=[None, 0.]):

    assert np.allclose(np.linalg.norm(X,axis=-1), 1.)
    N,D = X.shape

    for K in K_range: # for each number of clusters ...
        if verbose:
            print('K='+str(K))
        all_μs, all_w, all_LL, all_c  = [], [], [], []
        for ii in range(n_repets): # for n_repets different random initializations 
            np.random.seed(seed+ii)
            if init_with_spkmeans:
                _, w, c = spherical_kmeans(X, K=K, max_iter=max_iter, verbose=False)
                all_c.append(1 * c)            
                μs = np.stack([X[c==k].mean(axis=0) for k in range(K)],axis=0)
            else:
                μs, w = None, None
            μs, w, LL = softBregmanClustering_vMF(X, K=K, max_iter=max_iter, verbose=False, Ψ0=Ψ0,
                              μs_init=μs, w_init=w, tie_norms=tie_norms, μ_norm_max=μ_norm_max)
            all_μs.append(1. * μs)
            all_w.append(1. * w)
            all_LL.append(1. * LL)
            
            if verbose:
                print(' - ' + str(ii+1) + '/' + str(n_repets))

        np.savez(fn + '_K_' + str(K), out={
            'N' : N,
            'D' : D,
            'K' : K,
            'c' : np.stack(all_c,axis=0) if init_with_spkmeans else None,
            'LLs' : np.stack(all_LL,axis=0),
            'w' : np.stack(all_w,axis=0),
            'μs' : np.stack(all_μs,axis=0),
            'Ψ0' : Ψ0
        }) 