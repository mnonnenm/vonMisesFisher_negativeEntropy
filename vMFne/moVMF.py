import numpy as np
from vMFne.negentropy import Ψ
from vMFne.logpartition import vMF_loglikelihood_Φ, invgradΦ_base
import scipy.special

def log_joint_vMF_mixture_Φ(X,w,ηs):
    """
    Log-joint distribution of observed variable X and latent Z for a 
    mixture distribution with K von Mises-Fisher mixture components in
    natural parametrization.

    log p(X, Z | η[1:K], w[1:K]) for latent cluster assignment Z = 1,..,K, 
    where η_k are natural parameters of the different mixture components,
    and w_k the corresponding mixture weights.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    w : K-dim. array_like
        Mixture weights. Must be non-negative and sum to 1.
    ηs : K-by-D array_like
        Collection of K natural parameters for which log p(X|η) is computed.

    Returns
    -------
    logpxh : N-by-K array
        log p(X[n], Z=k | |η[1:K], w[1:K]) for all n=1,...,N, k=1,..,K.

    """
    K = ηs.shape[0]

    logpxz = np.log(np.clip(w,1e-15,None)).reshape(1,K) + vMF_loglikelihood_Φ(X,ηs,incl_const=False)

    return logpxz


def posterior_marginal_vMF_mixture_Φ(X,w,ηs):
    """
    Posterior distribution of latent Z for observed variable X under a 
    mixture distribution with K von Mises-Fisher mixture components in natural
    parametrization. Additionally returns the (log-)marginal probability of X. 

    Posterior p(Z|X,η[1:K],w[1:K]) for latent cluster assignment Z = 1,..,K, 
    where η[k] are the mean parameters of the different mixture components,
    and w[k] the corresponding mixture weights.

    Log-marginal log p(X | η[1:K], w[1:K]) summed over all possible latent Z.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    w : K-dim. array_like
        Mixture weights. Must be non-negative and sum to 1.
    ηs : K-by-D array_like

    Returns
    -------
    pz_x : N-by-K array
        p(Z=k | X[n], |η[1:K], w[1:K]) for all n=1,...,N, k=1,..,K.
    logpx : N-dim array
        log p(X[n], |η[1:K], w[1:K]) for all n=1,...,N.

    """
    N = X.shape[0]

    logpxz = log_joint_vMF_mixture_Φ(X,w,ηs)
    logpx = scipy.special.logsumexp(logpxz,axis=1).reshape(N,1)

    return np.exp(logpxz - logpx), logpx


def em_M_step_Φ(X, post, κ_max=np.inf, tie_norms=False):
    """
    M-step of Expectation Maximization algorithm for mixture of 
    von Mises-Fisher distribution in natural parametrization, and data X.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    post : N-by-K array_like
        Posterior p(Z=k | X[n], |η[1:K], w[1:K]) for all n=1,...,N, k=1,..,K.
    κ_max : float, > 0.0
        Maximal permissible L2 norm κ = ||η|| for learned vMF parameters.
        Very large norms can lead to long compute times in the E-step due to
        the vMF likelihood p(X|η) involving Bessel functions with argument κ.
    tie_norms: bool
        Flag whether to set all natural-parameter norms κ[k] to be identical.
        Default is False, i.e. each vMF components has its own variance.

    Returns
    -------
    w : D-dim. array
        Mixture weights for mixture of vMF distributions. 
    ηs : K-by-D array
        Collection of K mean parameters for K-many vMF mixture components.

    """
    D = X.shape[-1]

    w = post.mean(axis=0)
    nalphas = post.sum(axis=0)
    mus = X.T.dot(post).T # double-transpose for X.dot() in case X is sparse
    mu_norms = np.linalg.norm(mus,axis=1)
    rbar = mu_norms / nalphas
    mus = mus / mu_norms.reshape(-1,1)  # unit-norm 'mean parameter' vectors
    κs_est = invgradΦ_base(rbar, D, order=2, K=50)
    κs = np.minimum(κs_est, κ_max)
    if tie_norms:
        κs = κs.mean() * np.ones_like(κs)
    ηs = mus * κs.reshape(-1,1)

    return w, ηs


def init_EM_Φ(X, K, uniform_norm=False, Ψ0=None):
    """
    Computes mixture component natural paramters η[1:K] to initialize vMF mixture
    model from data X, by using a random partitioning of the N datapoints in X.

    Will first compute initial values for mean parameters μ[k] and from this
    compute natural parameters η[k] via (the gradient of) the Legendre
    transform of the log-partition function.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    K : integer >= 2.
        Number of mixture components.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)] for numerical evaluation of Ψ(μ), where Ψ is the 
        Legendre-transform of the log-partition function of the vMF distribution.

    Returns
    -------
    ηs_init : K-by-D array
        Collection of K mean parameters (D-dim vectors with L2 norm <= 1.0).
    """
    N, D = X.shape

    idx = np.random.permutation(N)
    μs_init = np.zeros((K,D))
    for k in range(K-1):
        μs_init[k] = X[idx[k*N//K : (k+1)*N//K]].mean(axis=0)
    μs_init[K-1] = X[idx[(K-1)*N//K:]].mean(axis=0) # last partition can be larger
    if uniform_norm:
        # in high-dim. spaces, X[n].T.dot(η)=0 for random η, and then hardMoFM will
        # assign all X[n] to η[k] with largest |η[k]| in the very first round...
        μs_norm = np.linalg.norm(μs_init,axis=-1)
        μs_init = μs_init / μs_norm.reshape(-1,1) * μs_norm.mean()

    _, ηs_init = Ψ(μs_init,D=D,Ψ0=[0., 0.], return_grad=True)

    return ηs_init


def softMoVMF(X, K, max_iter=50, w_init=None, ηs_init=None, verbose=False, 
              tie_norms=False, κ_max=np.inf, Ψ0=None):
    """
    Full Expectation Maxizmization for mixture model with von Mises-Fisher 
    components, with vMF mixture components in natural parametrization.

    Alternates between Expectation (E-) steps and Maximization (M-) steps.
    In the E-step, distributions over latent variables Z[n]=1,..,K are 
    computed, which represent assignments to any of the K.
    In the M-step, optimal model parameters η[1:K] and w[1:K] are found
    given X and the current estimates of Z.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    K : integer >= 2.
        Number of mixture components.
    max_iter : integer
        Maximum number of EM iterations.
    w_init : K-dim array_like or None.
        Initialization for mixture weights. Must be non-negative and sum to 1.0.
        If handed as None, will be initialized as uniform: w_init[k] = 1/K.
    ηs_init : K-by-D array
        Initialization for K natural parameters for K-many vMF mixture components.
        If handed as None, will be computed from a random partitioning of data X.
    verbose: bool
        Flag for printing intermediate results.
    tie_norms: bool
        Flag whether to set all natural-parameter norms ||η[k]|| to be identical.
        Default is False, i.e. each vMF components has its own variance.
    κ_max : float, > 0.0
        Maximal permissible L2 norm κ = ||η|| for learned vMF parameters.
        Very large norms can lead to long compute times in the E-step due to
        the vMF likelihood p(X|η) involving Bessel functions with argument κ.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)] for numerical evaluation of Ψ(μ), where Ψ is the 
        Legendre-transform of the log-partition function of the vMF distribution.

    Returns
    -------
    ηs : K-by-D array
        Collection of fitted natural parameters (D-dim vectors).
    w : K-dim array
        Fitted mixture weights.
    LL : array of length <= max_iter
        marginal (i.e. observed) log-likelihoods for whole data X over iterations.
    """
    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning
    ηs = init_EM_Φ(X, K, Ψ0=Ψ0) if ηs_init is None else ηs_init

    if verbose:
        print('initial w:', w)
        print('inital kappa:', np.linalg.norm(ηs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    ii = 0
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


def hardMoVMF(X, K, max_iter=50, w_init=None, ηs_init=None, verbose=False, 
              tie_norms=False, κ_max=np.inf, Ψ0=None):
    """
    Restricted Expectation Maxizmization for mixture model with von Mises-Fisher 
    components, with vMF mixture components in natural parametrization.
    Differs from full EM algorithm by hard assignments in the E-step (see below).

    Alternates between Expectation (E-) steps and Maximization (M-) steps.
    In the E-step, latent variables Z[n]=1,..,K are esimtated, which represent 
    assignments to any of the K. In full EM, latent variables are represented by
    the full distribution p(Z[n]=k|X[n], η[1:K], w[1:K]). In this restricted EM
    with hard assignments, Z[n] = argmax_k p(Z[n]=k | X[n], η[1:K], w[1:K]).
    In the M-step, optimal model parameters η[1:K] and w[1:K] are found
    given X and the current estimates of Z, as for full EM. 

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    K : integer >= 2.
        Number of mixture components.
    max_iter : integer
        Maximum number of EM iterations.
    w_init : K-dim array_like or None.
        Initialization for mixture weights. Must be non-negative and sum to 1.0.
        If handed as None, will be initialized as uniform: w_init[k] = 1/K.
    ηs_init : K-by-D array
        Initialization for K natural parameters for K-many vMF mixture components.
        If handed as None, will be computed from a random partitioning of data X.
    verbose: bool
        Flag for printing intermediate results.
    tie_norms: bool
        Flag whether to set all natural-parameter norms ||η[k]|| to be identical.
        Default is False, i.e. each vMF components has its own variance.
    κ_max : float, > 0.0
        Maximal permissible L2 norm κ = ||η|| for learned vMF parameters.
        Very large norms can lead to long compute times in the E-step due to
        the vMF likelihood p(X|η) involving Bessel functions with argument κ.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)] for numerical evaluation of Ψ(μ), where Ψ is the 
        Legendre-transform of the log-partition function of the vMF distribution.

    Returns
    -------
    ηs : K-by-D array
        Collection of fitted natural parameters (D-dim vectors).
    w : K-dim array
        Fitted mixture weights.
    LL : array of length <= max_iter
        marginal (i.e. observed) log-likelihoods for whole data X over iterations.

    """
    N,D = X.shape

    w = np.ones(K)/K if w_init is None else w_init
    assert np.all(w >= 0.)
    assert np.allclose(w.sum(), 1.)

    # initialize clusters on means of random partitioning        
    ηs = init_EM_Φ(X, K, Ψ0=Ψ0, uniform_norm=True) if ηs_init is None else ηs_init


    if verbose:
        print('initial w:', w)
        print('inital kappa:', np.linalg.norm(ηs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    ii = 0
    for ii in range(max_iter):

        # E-step: - compute (hardened) cluster responsibilities
        logpxh = log_joint_vMF_mixture_Φ(X=X,w=w,ηs=ηs)
        post = (logpxh >= (np.max(logpxh,axis=1).reshape(-1,1) * np.ones((1,K))))

        logpx = scipy.special.logsumexp(logpxh,axis=1).reshape(N,1)
        LL[ii] = logpx.sum()

        # M-step:
        w, ηs = em_M_step_Φ(X, post, κ_max=κ_max, tie_norms=tie_norms)

        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)
            print('kappa:', np.linalg.norm(ηs,axis=-1))

    return ηs, w, LL[:ii+1]
