import numpy as np
from vMFne.negentropy import gradΨ
import mpmath

def log_besseli(v,z):
    return np.float32(mpmath.log(mpmath.besseli(v, z)))

def ratio_besseli(v,z):
    return np.float32(mpmath.besseli(v, z) / mpmath.besseli(v-1, z))

def vMF_loglikelihood_Φ(X,ηs):

    ηs = np.atleast_2d(ηs)
    assert ηs.ndim == 2
    K,D = ηs.shape 
    κs = np.linalg.norm(ηs,axis=-1)
    log_Id = np.array([log_besseli(D/2.-1, κ) for κ in κs])
    LL = X.dot(ηs.T) - D/2. * np.log(2*np.pi) + ((D/2. - 1.) * np.log(κs) - log_Id).reshape(1,K)
    return LL # N-by-K

def posterior_marginal_vMF_mixture_Φ(X,w,ηs):

    N,K,D = X.shape[0], ηs.shape[0], X.shape[1]
    LL_k = vMF_loglikelihood_Φ(X,ηs)
    pxh = w.reshape(1,K) * np.exp(LL_k)
    px = pxh.sum(axis=1).reshape(N,1)

    return pxh / px, px # N-by-K

def moVMF(X, K, max_iter=50, w_init=None, ηs_init=None, verbose=False):

    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    if ηs_init is None:
        idx = np.random.permutation(N)
        μs_init = np.zeros((K,D))
        for k in range(K-1):
            μs_init[k] = X[idx[k*N//K : (k+1)*N//K]].mean(axis=0)
        μs_init[K-1] = X[idx[(K-1)*N//K:]].mean(axis=0) # last partition can be larger
        μs_init = 0.9 * μs_init # pull means towards origin (make clusters broader) 
        ηs_init = gradΨ(μs_init,D=D)
    ηs = ηs_init

    if verbose:
        print('initial w:', w)
        print('inital kappa:', np.linalg.norm(ηs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute cluster responsibilities
        post, px = posterior_marginal_vMF_mixture_Φ(X=X,w=w,ηs=ηs)
        LL[ii] = np.log(px).sum()

        # M-step:
        w = post.mean(axis=0)

        nalphas = post.sum(axis=0)
        mus = post.T.dot(X)
        mu_norms = np.linalg.norm(mus,axis=1)
        rbar = mu_norms / nalphas
        mus = mus / mu_norms.reshape(-1,1)  # unit-norm 'mean parameter' vectors
        κs = rbar * (D - rbar**2) / (1 - rbar**2)
        ηs = mus * κs.reshape(-1,1)
        #μs = mus/ post.sum(axis=0).reshape(K,1)
        #ηs = gradΨ(μs,D=D)
        
        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)
            print('kappa:', np.linalg.norm(ηs,axis=-1))

    return ηs, w, LL[:ii+1]

def vMF_entropy_Φ(ηs):

    ηs = np.atleast_2d(ηs)
    K,D = ηs.shape
    κs = np.linalg.norm(ηs,axis=-1)    
    H = - (D/2.-1.) * np.log(κs) + D/2. * np.log(2.0*np.pi) 
    log_I = np.array([log_besseli(D/2.-1, κ) for κ in κs])
    ratio_I = np.array([ratio_besseli(D/2, κ) for κ in κs])
    H = H + log_I - κs * ratio_I
    
    return H
