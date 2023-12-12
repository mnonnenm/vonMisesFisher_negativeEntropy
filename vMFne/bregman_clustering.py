import numpy as np
from vMFne.negentropy import DBregΨ
import scipy.special

def vMF_loglikelihood_Ψ(x,μ,D=2,Ψ0=None):
    # log p(x|μ) = - D_\Psi(x||\mu) + \Psi(x) + some constant in dimensionality D.
    # x : D-dim. vector or N-D matrix
    # μ : D-dim. vector or K-D matrix
    return - DBregΨ(x,μ,D=D,treat_x_constant=True,Ψ0=Ψ0) # N x K

def posterior_marginal_vMF_mixture_Ψ(X,w,μs,Ψ0=0.):

    N,K,D = X.shape[0], μs.shape[0], X.shape[1]
    LL_k = vMF_loglikelihood_Ψ(X,μs,D,Ψ0=Ψ0)
    logpxh = np.log(w.reshape(1,K)) + LL_k
    logpx = scipy.special.logsumexp(logpxh,axis=1).reshape(N,1)

    return np.exp(logpxh - logpx), logpx

def softBregmanClustering_vMF(X, K, max_iter=100, w_init=None, μs_init=None, verbose=False, Ψ0=[0., 1e-6]):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    if μs_init is None:
        #idx = np.random.permutation(N)
        #μs_init = np.zeros((K,D))
        #for k in range(K-1):
        #    μs_init[k] = X[idx[k*N//K : (k+1)*N//K]].mean(axis=0)
        #μs_init[K-1] = X[idx[(K-1)*N//K:]].mean(axis=0) # last partition can be larger
        #μs_init = 0.9 * μs_init # pull means towards origin (make clusters broader)
        μs_init = np.random.normal(size=(K,D))
        μs_init = μs_init / (100. * np.linalg.norm(μs_init,axis=-1).reshape(-1,1))

    μs = μs_init 

    if verbose:
        print('initial w:', w)
        print('inital ||μs||:', np.linalg.norm(μs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute cluster responsibilities
        post, logpx = posterior_marginal_vMF_mixture_Ψ(X=X,w=w,μs=μs,Ψ0=Ψ0)
        LL[ii] = logpx.sum()

        # M-step:
        w = post.mean(axis=0)
        μs = post.T.dot(X) / post.sum(axis=0).reshape(K,1)

        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)
            print('||μs||:', np.linalg.norm(μs,axis=-1))

    return μs, w, LL[:ii+1]
