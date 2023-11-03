import numpy as np
from vMFne.negentropy import vMF_loglikelihood_Ψ


def posterior_marginal_vMF_mixture(X,w,μs):
    N,K,D = X.shape[0], μs.shape[0], X.shape[1]
    LL_k = vMF_loglikelihood_Ψ(X,μs,D)
    pxh = w.reshape(1,K) * np.exp(LL_k)
    px = pxh.sum(axis=1).reshape(N,1)
    return pxh / px, px # N-by-K

def softBregmanClustering_vMF(X, K, max_iter=100, w_init=None, μs_init=None, verbose=False):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    if μs_init is None:
        idx = np.random.permutation(N)
        μs_init = np.zeros((K,D))
        for k in range(K-1):
            μs_init[k] = X[idx[k*N//K : (k+1)*N//K]].mean(axis=0)
        μs_init[K-1] = X[idx[(K-1)*N//K:]].mean(axis=0) # last partition can be larger
        μs_init = 0.9 * μs_init # pull means towards origin (make clusters broader) 
    μs = μs_init 

    if verbose:
        print(r'$w_0$:', w)
        print('r$||μs_0||:$', np.linalg.norm(μs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute cluster responsibilities
        post, px = posterior_marginal_vMF_mixture(X=X,w=w,μs=μs)
        LL[ii] = np.log(px).sum()

        # M-step:
        w = post.mean(axis=0)
        μs = post.T.dot(X) / post.sum(axis=0).reshape(K,1)

        if verbose:
            print('w:', w)
            print('||μs||:', np.linalg.norm(μs,axis=-1))

    return μs, w, LL[:ii+1]
