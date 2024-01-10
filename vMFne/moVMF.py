import numpy as np
from vMFne.negentropy import gradΨ
from vMFne.logpartition import vMF_loglikelihood_Φ
import scipy.special


def log_joint_vMF_mixture_Φ(X,w,ηs):

    logpxh = np.log(w).reshape(1,ηs.shape[0]) + vMF_loglikelihood_Φ(X,ηs,incl_const=False)

    return logpxh


def posterior_marginal_vMF_mixture_Φ(X,w,ηs):

    logpxh = log_joint_vMF_mixture_Φ(X,w,ηs)
    logpx = scipy.special.logsumexp(logpxh,axis=1).reshape(X.shape[0],1)

    return np.exp(logpxh - logpx), logpx


def em_M_step_Φ(X, post, κ_max=np.inf, tie_norms=False):

    D = X.shape[-1]

    w = post.mean(axis=0)
    nalphas = post.sum(axis=0)
    mus = post.T.dot(X)
    mu_norms = np.linalg.norm(mus,axis=1)
    rbar = mu_norms / nalphas
    mus = mus / mu_norms.reshape(-1,1)  # unit-norm 'mean parameter' vectors
    κs = np.minimum(rbar * (D - rbar**2) / (1 - rbar**2), κ_max)
    if tie_norms:
        κs = κs.mean() * np.ones_like(κs)
    ηs = mus * κs.reshape(-1,1)

    return w, ηs


def init_EM_Φ(X, K):

    N, D = X.shape

    idx = np.random.permutation(N)
    μs_init = np.zeros((K,D))
    for k in range(K-1):
        μs_init[k] = X[idx[k*N//K : (k+1)*N//K]].mean(axis=0)
    μs_init[K-1] = X[idx[(K-1)*N//K:]].mean(axis=0) # last partition can be larger

    return gradΨ(μs_init,D=D)


def softMoVMF(X, K, max_iter=50, w_init=None, ηs_init=None, verbose=False, κ_max=np.inf, tie_norms=False):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    ηs = init_EM_Φ(X, K) if ηs_init is None else ηs_init

    if verbose:
        print('initial w:', w)
        print('inital kappa:', np.linalg.norm(ηs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute cluster responsibilities
        post, logpx = posterior_marginal_vMF_mixture_Φ(X=X,w=w,ηs=ηs)
        LL[ii] = logpx.sum()

        # M-step:
        w, ηs = em_M_step_Φ(X, post, κ_max=κ_max, tie_norms=tie_norms)
        
        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)
            print('kappa:', np.linalg.norm(ηs,axis=-1))

    return ηs, w, LL[:ii+1]


def hardMoVMF(X, K, max_iter=50, w_init=None, ηs_init=None, verbose=False, κ_max=np.inf, tie_norms=False):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning        
    ηs = init_EM_Φ(X, K) if ηs_init is None else ηs_init

    if verbose:
        print('initial w:', w)
        print('inital kappa:', np.linalg.norm(ηs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute (hardened) cluster responsibilities
        logpxh = log_joint_vMF_mixture_Φ(X=X,w=w,ηs=ηs)
        post = (logpxh >= (np.max(logpxh,axis=1).reshape(-1,1) * np.ones(1,K)))

        logpx = scipy.special.logsumexp(logpxh,axis=1).reshape(N,1)
        LL[ii] = logpx.sum()

        # M-step:
        w, ηs = em_M_step_Φ(X, post, κ_max=κ_max, tie_norms=tie_norms)

        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)
            print('kappa:', np.linalg.norm(ηs,axis=-1))

    return ηs, w, LL[:ii+1]
