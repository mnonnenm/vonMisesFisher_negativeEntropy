import numpy as np
from vMFne.negentropy import DBregΨ
import scipy.special

def vMF_loglikelihood_Ψ(x,μ,D=2,Ψ0=None):
    # log p(x|μ) = - D_\Psi(x||\mu) + \Psi(x) + some constant in dimensionality D.
    # x : D-dim. vector or N-D matrix
    # μ : D-dim. vector or K-D matrix
    return - DBregΨ(x,μ,D=D,treat_x_constant=True,Ψ0=Ψ0) # N x K


def log_joint_vMF_mixture_Ψ(X,w,μs,Ψ0=0.):

    K,D = μs.shape
    logpxh = np.log(w).reshape(1,K) + vMF_loglikelihood_Ψ(X,μs,D,Ψ0=Ψ0)

    return logpxh


def posterior_marginal_vMF_mixture_Ψ(X,w,μs,Ψ0=0.):

    N = X.shape[0]
    logpxh = log_joint_vMF_mixture_Ψ(X,w,μs,Ψ0=Ψ0)
    logpx = scipy.special.logsumexp(logpxh,axis=1).reshape(N,1)

    return np.exp(logpxh - logpx), logpx


def em_M_step_Ψ(X, post, μ_norm_max=0.99, tie_norms=False):

    K = post.shape[-1]

    w = post.mean(axis=0)
    μs = post.T.dot(X) / post.sum(axis=0).reshape(K,1)
    μs_norm = np.linalg.norm(μs,axis=-1)
    if np.any(μs_norm>μ_norm_max):
        idx = μs_norm>μ_norm_max
        μs[idx,:] = μ_norm_max * μs[idx,:] / μs_norm[idx].reshape(-1,1)
    if tie_norms:
        μs = μs / μs_norm.reshape(K,1) * μs_norm.mean()

    return w, μs


def init_EM_Ψ(X, K):

    N, D = X.shape

    idx = np.random.permutation(N)
    μs_init = np.zeros((K,D))
    for k in range(K-1):
        μs_init[k] = X[idx[k*N//K : (k+1)*N//K]].mean(axis=0)
    μs_init[K-1] = X[idx[(K-1)*N//K:]].mean(axis=0) # last partition can be larger

    return μs_init


def softBregmanClustering_vMF(X, K, max_iter=100, w_init=None, μs_init=None, verbose=False, Ψ0=[0., 1e-6],
                              tie_norms=False, μ_norm_max=0.99):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    μs = init_EM_Ψ(X, K) if μs_init is None else μs_init
    assert np.all(np.linalg.norm(μs,axis=-1) <= 1.0)

    if verbose:
        print('initial w:', w)
        print('initial ||μs||:', np.linalg.norm(μs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute cluster responsibilities
        post, logpx = posterior_marginal_vMF_mixture_Ψ(X=X,w=w,μs=μs,Ψ0=Ψ0)
        LL[ii] = logpx.sum()

        # M-step:
        w, μs = em_M_step_Ψ(X, post, μ_norm_max=μ_norm_max, tie_norms=tie_norms)

        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)
            print('||μs||:', np.linalg.norm(μs,axis=-1))

    return μs, w, LL[:ii+1]


def hardBregmanClustering_vMF(X, K, max_iter=100, w_init=None, μs_init=None, verbose=False, Ψ0=[0., 1e-6],
                              tie_norms=False, μ_norm_max=0.99):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    μs = init_EM_Ψ(X, K) if μs_init is None else μs_init
    assert np.all(np.linalg.norm(μs,axis=-1) <= 1.0)

    if verbose:
        print('initial w:', w)
        print('initial ||μs||:', np.linalg.norm(μs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute (hardened) cluster responsibilities
        logpxh = log_joint_vMF_mixture_Ψ(X,w,μs,Ψ0=Ψ0)
        post = (logpxh >= (np.max(logpxh,axis=1).reshape(-1,1) * np.ones(1,K)))

        logpx = scipy.special.logsumexp(logpxh,axis=1).reshape(N,1)
        LL[ii] = logpx.sum()

        # M-step:
        w, μs = em_M_step_Ψ(X, post, μ_norm_max=μ_norm_max, tie_norms=tie_norms)

        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)
            print('||μs||:', np.linalg.norm(μs,axis=-1))

    return μs, w, LL[:ii+1]


def spherical_kmeans(X, K, max_iter=100, w_init=None, μs_init=None, verbose=False):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    μs = init_EM_Ψ(X, K) if μs_init is None else μs_init
    μs = μs / np.linalg.norm(μs,axis=-1).reshape(K,1) # centroids are on sphere surface

    if verbose:
        print('initial w:', w)

    for ii in range(max_iter):

        # E-step: - compute cluster assignments via cosine similarity
        c = np.argmax(X.dot(μs.T),axis=-1) # X*μs' is N x K, c is N-dim.

        # M-step:
        w = np.array([np.mean(c==k) for k in range(K)])
        μs = np.stack([X[c==k].mean(axis=0) for k in range(K)],axis=0)
        μs_norm = np.linalg.norm(μs,axis=-1).reshape(K,1)
        μs_norm[μs_norm==0] = 1.
        μs = μs / μs_norm

        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)

    return μs, w, c
