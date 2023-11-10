import numpy as np
import mpmath
from vMFne.negentropy import gradΨ

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

def vMF_loglikelihood_Φ(X,ηs,incl_const=True):

    ηs = np.atleast_2d(ηs)
    assert ηs.ndim == 2
    K,D = ηs.shape 
    κs = np.linalg.norm(ηs,axis=-1)
    LL = X.dot(ηs.T) - Φ(κs, D).reshape(1,K)
    if incl_const:
        LL = LL + logχ(D)
    return LL # N-by-K


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

def invgradΦ(μs_norm,D,max_iter=10, atol=1e-12):
    rbar = μs_norm
    κs_est_44 = banerjee_44(rbar,D)
    κs_est, diffs = newtonraphson(κs_init=κs_est_44, D=D, rbar=rbar, 
                                  max_iter=max_iter, atol=atol)

    return κs_est, diffs