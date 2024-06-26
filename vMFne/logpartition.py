import numpy as np
import mpmath
from vMFne.negentropy import gradΨ, banerjee_44

def log_besseli(v,z):
    """ Logarithm of Bessel function of the first kind and order v. """
    return np.float32(mpmath.log(mpmath.besseli(v, z)))


def ratio_besseli(v,z):
    """ Ratio of Bessel functions of the first kind and orders v,v-1. """
    try:
        return np.float32(mpmath.besseli(v, z) / mpmath.besseli(v-1, z))
    except:
        with mpmath.extraprec(n=1000):
            r = np.float32(mpmath.besseli(v, z) / mpmath.besseli(v-1, z))

    return r


def Φ(κs, D):
    """
    Radial profile of the log-partition function of the D-dim. von Mises-
    Fisher distribution. Due to radial symmetry, Φ(η) = Φ(||η||) = Φ(κ).

    Parameters
    ----------
    κs : K-dim. array_like
        Norms of K-many natural parameters η.
    D : integer, >=2
        Dimension of vMF distribution and natural parameters η.

    Returns
    -------
    Φ : K-dim. array
        Φ(κs[k]), k=1,..,K, for vMF log-partition Φ in D dimensions.

    """
    log_Id = np.array([log_besseli(D/2.-1, κ) for κ in κs])

    return log_Id - (D/2. - 1.) * np.log(np.clip(κs,1e-15,None)) + logχ(D)


def gradΦ(ηs):
    """ Gradient of log-partition function of the von Mises-Fisher distribution. """
    ηs = np.atleast_2d(ηs)
    K,D = ηs.shape
    κs = np.linalg.norm(ηs,axis=-1)
    μs = ηs * (np.array([ratio_besseli(D/2,κ) for κ in κs]) / κs).reshape(-1,1)

    return μs


def logχ(D) :
    """ Base measure of the von Mises-Fisher distribution in exponential family form. """
    return - D/2. * np.log(2*np.pi)


def vMF_loglikelihood_Φ(X,ηs):
    """
    Log-likelihood of von Mises-Fisher distribution with natural parameter η.

    If ηs is a K-D array, will return the log-likelihoods of x for all K
    natural parameter vectors η[k] simultaneously in an N-K matrix.

    log p(x|η) = x'η - Φ(η) + logχ(D)
    where η is the vMF log-partition function and χ the base measure.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    ηs : K-by-D array_like
        Collection of K natural parameters for which log p(X|η) will be computed.

    Returns
    -------
    logp : N-by-K array
        log p(X[n]|η[k]) for all n=1,...,N, k=1,..,K

    """
    ηs = np.atleast_2d(ηs)
    assert ηs.ndim == 2
    K,D = ηs.shape 
    κs = np.linalg.norm(ηs,axis=-1)
    LL = X.dot(ηs.T) - Φ(κs, D).reshape(1,K)

    return LL


def vMF_entropy_Φ(ηs):
    """
    Entropy of von Mises-Fisher distribution with natural parameter η.

    If ηs is a K-D array, will return the entopies for all K
    natural parameter vectors η[k] simultaneously in a K-dim. array.

    Parameters
    ----------
    ηs : K-by-D array_like
        Collection of K natural parameters for which log p(X|η) will be computed.

    Returns
    -------
    H : K-dim. array
        H[X|η[k]] for all k=1,..,K

    """
    ηs = np.atleast_2d(ηs)
    K,D = ηs.shape
    κs = np.linalg.norm(ηs,axis=-1)    
    H = Φ(κs, D) - κs * A(κs,D)

    return H


def A(κs,D):
    """
    Ratio of Bessel functions of the first kind and orders D,D-1.
    Works with vector-valued arguments κs.

    Parameters
    ----------
    κs : K-dim. array_like
        Arguments for Bessel function of the first kind.
    D : integer, >=2
        Order of Bessel function of the first kind.
    Returns
    -------
    A : K-dim. array
        Besseli(κ,D) / Besseli(κ,D-1) for κ in 1:K, K=len(κs).

    """
    return np.array([ratio_besseli(D/2,κ) for κ in κs])


def dA(κs,D):
    """
    Derivative of ratio of Bessel functions of the first kind
    and orders D,D-1. Works with vector-valued arguments κs.

    Parameters
    ----------
    κs : K-dim. array_like
        Arguments for Bessel function of the first kind.
    D : integer, >=2
        Order of Bessel function of the first kind.

    Returns
    -------
    dA : K-dim. array
        d/dκ[Besseli(κ,D) / Besseli(κ,D-1)] for κ in 1:K, K=len(κs).

    """
    return 1- A(κs,D)**2 - (D-1) * A(κs,D) / κs


def newtonraphson(κs_init,D,rbar,max_iter=100, atol=1e-12):
    """
    Numerical root finder for
    A(κ,D) - r,
    where A(κ,D) = Besseli(κ,D)/Besseli(κ,D-1)
    is the derivative of the radial profile von Mises-Fisher
    log-partition function, and r=||μ|| is the norm of the
    associated mean parameter μ.

    Uses the Newton-Raphson algorithm to refine an initial guess.

    Parameters
    ----------
    κs_init : K-dim. array_like
        Initial guess for the solutions of A(κ,D) = r.
    D : integer, >=2
        Order of Bessel function of the first kind.
    rbar : K-dim. array_like
        Desired outcomes for A(κ,D).
    max_iter : non-negative integer
        Maximal number of iterations.
    atol : float, >= 0.0
        tolerance for convergence criterion |A(κ,D) - r| < atol

    Returns
    -------
    κs : K-dim. array
        Approximate solutions of A(κ,D) = r.
    diffs: K-dim. array of length <= len(max_iter)
        |A(κs,D) - r| across all iterations until convergence.

    """
    κs = κs_init
    diffs = np.zeros((max_iter+1, len(κs)))
    if max_iter > 0:
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
    """
    Numerical inversion of the gradient of the log-partition
    function Φ of the D-dimensional von Mises-Fisher distribution.

    Uses Newton-Rapshon on ||∇Φ(η)|| = μs_norm to find ||η|| = κ.
    Since the vMF log-partition is radially symmetric, κ is the 
    only non-trivial aspect of natural parmeter η to figure out.

    Parameters
    ----------
    μs_norm : K-dim. array_like
        L2 norms of mean parameters μs_norm = ||μ||.
    D : integer, >=2
        Dimension of the natural parameter η and vMF distribution.
    max_iter : non-negative integer
        Maximal number of Newton-Rapshon iterations.
    atol : float, >= 0.0
        tolerance for convergence criterion |A(κ,D) - r| < atol

    Returns
    -------
    κs_est : K-dim. array
        Approximate solutions of ||η|| for which ||∇Φ(η)||=μs_norm.
    diffs: K-dim. array of length <= len(max_iter)
        |A(κs,D) - r| across all iterations until convergence.

    """
    rbar = μs_norm
    κs_est_44 = banerjee_44(rbar,D) # initial guess for κs
    κs_est, diffs = newtonraphson(κs_init=κs_est_44, D=D, rbar=rbar, 
                                  max_iter=max_iter, atol=atol)
    return κs_est, diffs


def invgradΦ_base(μs_norm,D, order=2, K=20):
    """
    Numerical inversion of the gradient of the log-partition
    function Φ of the D-dimensional von Mises-Fisher distribution.

    Uses truncated Newton-Rapshon on ||∇Φ(η)|| = μs_norm to find 
    ||η|| = κ. Since the vMF log-partition is radially symmetric, 
    κ is the only aspect of natural parmeter η to figure out.
    
    Approximates ||∇Φ(η)|| = f(||η||) using the Perron continuous
    fraction representation. 

    Parameters
    ----------
    μs_norm : K-dim. array_like
        L2 norms of mean parameters μs_norm = ||μ||.
    D : integer, >=2
        Dimension of the natural parameter η and vMF distribution.
    order : non-negative integer
        Number of Newton-Rapshon iterations.
    K : non-negative integer
        Number of iterations to numerically approximate ||∇Φ(η)||.

    Returns
    -------
    κs_est : K-dim. array
        Approximate solutions of ||η|| for which ||∇Φ(η)||=μs_norm.

    """
    κs_est = truncated_newtonraphson_perron(μs_norm,D,order,K)
    return κs_est


def perron_cf_rec(u,v,w,ρ,x,xp,s,k):
    if k==1:
        v = s + x + 0.5
        u = (s + x) * v
        w = xp * (s + 0.5)
        ρ = w / ((s+xp) * v - w)
    else:
        u = u + v
        v = v + 0.5
        w = w + xp
        t = w * (1. + ρ)
        ρ = t / (u - t)
    return u,v,w,ρ


def A_perron_cf(x,D,K):
    A = np.empty_like(x)
    idx = (x >= 1e-6)
    s = D/2.
    xp = 0.5 * x[idx]
    p,psum = 1.0, 1.0
    u,v,w,ρ = None, None, None, None
    k = 1
    not_converged = True
    while not_converged:
        u,v,w,ρ = perron_cf_rec(u,v,w,ρ,x[idx],xp,s,k)
        p = ρ * p
        psum = psum + p 
        k = k+1
        not_converged = k < K

    A[idx] = psum / (1. + 2. * s/x[idx])
    nidx = np.invert(idx)
    if np.any(nidx):
        A[nidx] = x[nidx] / D - x[nidx]**3 / (D**2 * (D + 2)) + 2. * x[nidx]**5 / (D**3 * (D + 2) * (D + 4))
    return A


def truncated_newtonraphson_perron(rbar,D,order=2,K=5):
    κ = banerjee_44(rbar,D)
    for order in range(order):
        Aκ = A_perron_cf(κ,D,K=K)
        f = Aκ - rbar
        df = 1. - Aκ**2 - (D-1) * Aκ/κ
        ddf = 2. * Aκ**3 + 3. * (D-1) * Aκ**2/κ + (D*(D-1)/κ**2 - 2.0) * Aκ - (D-1)/κ
        κ = κ - 2. * f * df / (2. * df**2 - f * ddf)
    return κ


def logbesseli_hornik(v,x):
    x2v12 = np.sqrt(x**2 + (v+1)**2)
    return x2v12 + (v+0.5)*np.log((2.0*v+1.5)*x/((v+0.5+x2v12)*(2.0*v+2.0))) - np.log(x/2.)/2. - np.log(2.0*np.pi)/2
