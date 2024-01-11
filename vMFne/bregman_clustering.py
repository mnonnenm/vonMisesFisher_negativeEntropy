import numpy as np
from vMFne.negentropy import DBregΨ
import scipy.special

def vMF_loglikelihood_Ψ(X,μ,D,Ψ0=None):
    """
    Log-likelihood of von Mises-Fisher distribution with mean parameter μ.

    This means that the mixture components p(X | μ[k]) are parameterized by
    μ[k] = E[x | μ[k]] for vMF-distributed random variable x, simplifying 
    maximum likelihood from a data matrix X when data is assumed to be iid.

    If μ is a K-D array, will return the log-likelihoods of x for all K
    mean parameter vectors μ[k] simultaneously in an N-K matrix.

    log p(x|μ) = - D_Ψ(x||μ) + Ψ(x) + constant(D)
    where Ψ is the Legendre-transform of the vMF log-partition function, and 
    D_Ψ(x||y) is the Bregman divergence for convex function Ψ.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    μs : K-by-D array_like
        Collection of K mean parameters for which log p(X|μ) will be computed.
    D : int
        Dimensionality of hypersphere for von Mises-Fisher distribution.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)], which is used for numerical evaluation of Ψ(μ).

    Returns
    -------
    logp : N-by-K array
        log p(X[n]|μ[k]) for all n=1,...,N, k=1,..,K

    """
    logp = - DBregΨ(X,μ,D=D,treat_x_constant=True,Ψ0=Ψ0)

    return logp


def log_joint_vMF_mixture_Ψ(X,w,μs,Ψ0=0.):
    """
    Log-joint distribution of observed variable X and latent Z for a 
    mixture distribution with K von Mises-Fisher mixture components in mean
    parametrization.

    This means that the mixture components p(X | μ[k]) are parameterized by
    μ[k] = E[x | μ[k]] for vMF-distributed random variable x, simplifying 
    maximum likelihood from a data matrix X when data is assumed to be iid.

    log p(X, Z | μ[1:K], w[1:K]) for latent cluster assignment Z = 1,..,K, 
    where μ_k are the mean parameters of the different mixture components,
    and w_k the corresponding mixture weights.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    w : K-dim. array_like
        Mixture weights. Must be non-negative and sum to 1.
    μs : K-by-D array_like
        Collection of K mean parameters for which log p(X|μ) will be computed.
    D : int
        Dimensionality of hypersphere for von Mises-Fisher distribution.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)] for numerical evaluation of Ψ(μ).

    Returns
    -------
    logpxh : N-by-K array
        log p(X[n], Z=k | |μ[1:K], w[1:K]) for all n=1,...,N, k=1,..,K.

    """
    K,D = μs.shape

    logpxz = np.log(w).reshape(1,K) + vMF_loglikelihood_Ψ(X,μs,D,Ψ0=Ψ0)

    return logpxz


def posterior_marginal_vMF_mixture_Ψ(X,w,μs,Ψ0=None):
    """
    Posterior distribution of latent Z for observed variable X under a 
    mixture distribution with K von Mises-Fisher mixture components in mean
    parametrization. Additionally returns the (log-)marginal probability of X. 

    This means that the mixture components p(X | μ[k]) are parameterized by
    μ[k] = E[x | μ[k]] for vMF-distributed random variable x, simplifying 
    maximum likelihood from a data matrix X when data is assumed to be iid.

    Posterior p(Z|X,μ[1:K],w[1:K]) for latent cluster assignment Z = 1,..,K, 
    where μ[k] are the mean parameters of the different mixture components,
    and w[k] the corresponding mixture weights.

    Log-marginal log p(X | μ[1:K], w[1:K]) summed over all possible latent Z.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    w : K-dim. array_like
        Mixture weights. Must be non-negative and sum to 1.
    μs : K-by-D array_like
        Collection of K mean parameters for which log p(X|μ) will be computed.
    D : int
        Dimensionality of hypersphere for von Mises-Fisher distribution.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)] for numerical evaluation of Ψ(μ)

    Returns
    -------
    pz_x : N-by-K array
        p(Z=k | X[n], |μ[1:K], w[1:K]) for all n=1,...,N, k=1,..,K.
    logpx : N-dim array
        log p(X[n], |μ[1:K], w[1:K]) for all n=1,...,N.

    """
    N = X.shape[0]

    logpxz = log_joint_vMF_mixture_Ψ(X,w,μs,Ψ0=Ψ0)
    logpx = scipy.special.logsumexp(logpxz,axis=1).reshape(N,1)

    return np.exp(logpxz - logpx), logpx


def em_M_step_Ψ(X, post, μ_norm_max=0.99, tie_norms=False):
    """
    M-step of Expectation Maximization algorithm for mixture of 
    von Mises-Fisher distribution in mean parametrization, and data X.

    This means that the mixture components p(X | μ[k]) are parameterized by
    μ[k] = E[x | μ[k]] for vMF-distributed random variable x, simplifying 
    maximum likelihood from a data matrix X when data is assumed to be iid.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    post : N-by-K array_like
        Posterior p(Z=k | X[n], |μ[1:K], w[1:K]) for all n=1,...,N, k=1,..,K.
    μ_norm_max : float, 0.0 < μ_norm_max <= 1.0
        Maximal permissible L2 norm ||μ|| for learned vMF mean parameters.
        Norms very close to 1.0 can lead to long compute times in the E-step,
        depending on how the vMF likelihood p(X|μ) is evaluated.
    tie_norms: bool
        Flag whether to set all mean-parameter norms ||μ[k]|| to be identical.
        Default is False, i.e. each vMF components has its own variance.

    Returns
    -------
    w : D-dim. array
        Mixture weights for mixture of vMF distributions. 
    μs : K-by-D array
        Collection of K mean parameters for K-many vMF mixture components.

    """
    K = post.shape[-1]

    w = post.mean(axis=0)
    μs = X.T.dot(post).T / post.sum(axis=0).reshape(K,1) # in case X is sparse
    μs_norm = np.linalg.norm(μs,axis=-1)
    if np.any(μs_norm>μ_norm_max):
        idx = μs_norm>μ_norm_max
        μs[idx,:] = μ_norm_max * μs[idx,:] / μs_norm[idx].reshape(-1,1)
    if tie_norms:
        μs = μs / μs_norm.reshape(K,1) * μs_norm.mean()

    return w, μs


def init_EM_Ψ(X, K):
    """
    Computes mixture component mean paramters μ[1:K] to initialize vMF mixture
    model from data X, by using a random partitioning of the N datapoints in X.

    In mean parametriztaion, mixture components p(X | μ[k]) are parameterized
    by μ[k] = E[x | μ[k]] for vMF-distributed random variable x, simplifying 
    maximum likelihood from a data matrix X when data is assumed to be iid.

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    K : integer >= 2.
        Number of mixture components.
        
    Returns
    -------
    μs_init : K-by-D array
        Collection of K mean parameters (D-dim vectors with L2 norm <= 1.0).
    """
    N, D = X.shape

    idx = np.random.permutation(N)
    μs_init = np.zeros((K,D))
    for k in range(K-1):
        μs_init[k] = X[idx[k*N//K : (k+1)*N//K]].mean(axis=0)
    μs_init[K-1] = X[idx[(K-1)*N//K:]].mean(axis=0) # last partition may be larger

    return μs_init


def softBregmanClustering_vMF(X, K, max_iter=100, w_init=None, μs_init=None,
                              verbose=False, Ψ0=[0., 1e-6],
                              tie_norms=False, μ_norm_max=0.99):
    """
    Full Expectation Maxizmization for mixture model with von Mises-Fisher 
    components, with vMF mixture components in mean parametrization.

    This means that the mixture components p(X | μ[k]) are parameterized by
    μ[k] = E[x | μ[k]] for vMF-distributed random variable x, simplifying 
    maximum likelihood from a data matrix X when data is assumed to be iid.

    Alternates between Expectation (E-) steps and Maximization (M-) steps.
    In the E-step, distributions over latent variables Z[n]=1,..,K are 
    computed, which represent assignments to any of the K.
    In the M-step, optimal model parameters μ[1:K] and w[1:K] are found
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
    μs_init : K-by-D array
        Initialization for K mean parameters for K-many vMF mixture components.
        If handed as None, will be computed from a random partitioning of data X.
    verbose: bool
        Flag for printing intermediate results.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)] for numerical evaluation of Ψ(μ), where Ψ is the 
        Legendre-transform of the log-partition function of the vMF distribution.
    tie_norms: bool
        Flag whether to set all mean-parameter norms ||μ[k]|| to be identical.
        Default is False, i.e. each vMF components has its own variance.
    μ_norm_max : float, 0.0 < μ_norm_max <= 1.0
        Maximal permissible L2 norm ||μ|| for learned vMF mean parameters.
        Norms very close to 1.0 can lead to long compute times in the E-step,
        depending on how the vMF likelihood p(X|μ) is evaluated.

    Returns
    -------
    μs : K-by-D array
        Collection of fitted mean parameters (D-dim vectors with L2 norm <= 1.0).
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


def hardBregmanClustering_vMF(X, K, max_iter=100, w_init=None, μs_init=None,
                              verbose=False, Ψ0=[0., 1e-6],
                              tie_norms=False, μ_norm_max=0.99):
    """
    Restricted Expectation Maxizmization for mixture model with von Mises-Fisher 
    components, with vMF mixture components in mean parametrization.
    Differs from full EM algorithm by hard assignments in the E-step (see below).

    In mean parametrization the mixture components p(X | μ[k]) are parameterized
    by μ[k] = E[x | μ[k]] for vMF-distributed random variable x, simplifying 
    maximum likelihood from a data matrix X when data is assumed to be iid.

    Alternates between Expectation (E-) steps and Maximization (M-) steps.
    In the E-step, latent variables Z[n]=1,..,K are esimtated, which represent 
    assignments to any of the K. In full EM, latent variables are represented by
    the full distribution p(Z[n]=k|X[n], μ[1:K], w[1:K]). In this restricted EM
    with hard assignments, Z[n] = argmax_k p(Z[n]=k | X[n], μ[1:K], w[1:K]).
    In the M-step, optimal model parameters μ[1:K] and w[1:K] are found
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
    μs_init : K-by-D array
        Initialization for K mean parameters for K-many vMF mixture components.
        If handed as None, will be computed from a random partitioning of data X.
    verbose: bool
        Flag for printing intermediate results.
    Ψ0 : tuple or list float or None, or [None, float]
        Gives [Ψ(0), Ψ'(0)] for numerical evaluation of Ψ(μ), where Ψ is the 
        Legendre-transform of the log-partition function of the vMF distribution.
    tie_norms: bool
        Flag whether to set all mean-parameter norms ||μ[k]|| to be identical.
        Default is False, i.e. each vMF components has its own variance.
    μ_norm_max : float, 0.0 < μ_norm_max <= 1.0
        Maximal permissible L2 norm ||μ|| for learned vMF mean parameters.
        Norms very close to 1.0 can lead to long compute times in the E-step,
        depending on how the vMF likelihood p(X|μ) is evaluated.

    Returns
    -------
    μs : K-by-D array
        Collection of fitted mean parameters (D-dim vectors with L2 norm <= 1.0).
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
    μs = init_EM_Ψ(X, K) if μs_init is None else μs_init
    assert np.all(np.linalg.norm(μs,axis=-1) <= 1.0)

    if verbose:
        print('initial w:', w)
        print('initial ||μs||:', np.linalg.norm(μs,axis=-1))

    LL = np.zeros(max_iter) # likelihood (up to multiplicative constant)
    for ii in range(max_iter):

        # E-step: - compute (hardened) cluster responsibilities
        logpxh = log_joint_vMF_mixture_Ψ(X,w,μs,Ψ0=Ψ0)
        post = (logpxh >= (np.max(logpxh,axis=1).reshape(-1,1) * np.ones((1,K))))

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
    """
    Spherical K-means algorithm for data X on the hypersphere. 

    Iterates between a) partitioning the data X[1:N] by finding the closest
    cluster mean out of K many clusters, and b) setting the cluster means to
    be the centroids of all data points X[n] assigned to that cluster.

    Differs from 'standard' K-means by using cosine similarity instead of
    Euclidean distance as the measure of closeness, and by the cluster centroids
    being forced onto the surface of the hypersphere (i.e. L2 norm equal to 1.0).

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
    μs_init : K-by-D array
        Initialization for K cluster means. All μs_init[k] must have unit L2 norm.
        If handed as None, will be computed from a random partitioning of data X.
    verbose: bool
        Flag for printing intermediate results.

    Returns
    -------
    μs : K-by-D array
        Collection of fitted cluster means (D-dim vectors with L2 norm = 1.0).
    w : K-dim array
        Fitted mixture weights.
    c : N-dim array
        Cluster assignment for each input datapoint.

    """
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

        # 'E-step' - compute cluster assignments via cosine similarity
        c = np.argmax(X.dot(μs.T),axis=-1) # X*μs' is N x K, c is N-dim.

        # 'M-step'
        w = np.array([np.mean(c==k) for k in range(K)])
        μs = np.stack([X[c==k].mean(axis=0) for k in range(K)],axis=0)
        μs_norm = np.linalg.norm(μs,axis=-1).reshape(K,1)
        μs_norm[μs_norm==0] = 1.
        μs = μs / μs_norm

        if verbose:
            print(' #' + str(ii+1) + '/' + str(max_iter))
            print('w:', w)

    return μs, w, c
