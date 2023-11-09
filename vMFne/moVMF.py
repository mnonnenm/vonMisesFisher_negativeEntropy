import numpy as np
from vMFne.negentropy import gradΨ
import mpmath

def log_besseli(v,z):
    return np.float32(mpmath.log(mpmath.besseli(v, z)))

def ratio_besseli(v,z):
    return np.float32(mpmath.besseli(v, z) / mpmath.besseli(v-1, z))

def Φ(κs, D):
    log_Id = np.array([log_besseli(D/2.-1, κ) for κ in κs])
    return log_Id - (D/2. - 1.) * np.log(κs)

def gradΦ(ηs):
    ηs = np.atleast_2d(ηs)
    K,D = ηs.shape
    κs = np.linalg.norm(ηs,axis=-1)
    μs = ηs * (np.array([ratio_besseli(D/2,κ) for κ in κs]) / κs).reshape(-1,1)
    return μs

def logχ(D) :
    return - D/2. * np.log(2*np.pi)

def vMF_loglikelihood_Φ(X,ηs):

    ηs = np.atleast_2d(ηs)
    assert ηs.ndim == 2
    K,D = ηs.shape 
    κs = np.linalg.norm(ηs,axis=-1)
    LL = X.dot(ηs.T) + logχ(D) - Φ(κs, D).reshape(1,K)
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
    H = Φ(κs, D) - logχ(D) - κs * A(κs,D)

    return H

def A(κs,D):

    return np.array([ratio_besseli(D/2,κ) for κ in κs])

def dA(κs,D):

    return 1- A(κs,D)**2 - (D-1) * A(κs,D) / κs

def banerjee_44(rbar,D):
    """ Approximately solve for Ψ'(rbar) = κ in notation rbar = ||μ||, κ = ||η||. 
    Taken from eq. (4.4) of
    Banerjee, Arindam, et al. 
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions." 
    Journal of Machine Learning Research 6.9 (2005).
    """
    return rbar * (D- rbar**2) / (1-rbar**3)

def newtonraphson(κs_init,D,rbar,max_iter=100, atol=1e-12):
    κs = κs_init
    diffs = np.zeros((max_iter+1, len(κs)))
    f = (A(κs,D) - rbar)
    diffs[0] = f
    df = dA(κs,D)
    for i in range(max_iter):
        κs = κs - f/df
        f = (A(κs,D) - rbar)
        df = dA(κs,D)
        diffs[i+1] = f
        if np.all(np.abs(diffs) < atol):
            diffs = diffs[:i+2]
            break

    return κs, diffs

def invert_gradΦ(μs_norm,D,max_iter=10, atol=1e-12):
    rbar = μs_norm
    κs_est_44 = banerjee_44(rbar,D)
    κs_est, diffs = newtonraphson(κs_init=κs_est_44, D=D, rbar=rbar, 
                                  max_iter=max_iter, atol=atol)

    return κs_est, diffs
