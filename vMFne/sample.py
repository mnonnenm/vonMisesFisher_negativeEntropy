import numpy as np
from vMFne.utils_angular import cart2spherical, spherical_rotMat

def sample_w(N, kappa=1., D=3):
    assert D == 3 # sampling w more complicated for other dimensions

    u = np.random.rand(N) # uniform
    w = np.log(2*u*np.sinh(kappa) + np.exp(-kappa))/kappa # inverse CDF of p(w)

    return w

def sample_vMF_Ulrich(N, m, kappa, D=3):
    # samples from von Mises-Fisher distribution over S^(D-1) according to Ulrich (1984) 

    # sample w 
    w = sample_w(N, kappa=kappa, D=D)

    # sample v from orthogonal subspace S^(D-2)
    v = np.random.normal(size=(D-1,N))
    v = v/np.sqrt(((v**2).sum(axis=0))).reshape(1,N)

    # stack scaled (w,v) and rotate results
    x = np.concatenate([np.sqrt(1-w**2) * v, w.reshape(1,-1)],axis=0)
    m_spherical = cart2spherical(m)
    M = spherical_rotMat(m_spherical)

    return M.dot(x).T
