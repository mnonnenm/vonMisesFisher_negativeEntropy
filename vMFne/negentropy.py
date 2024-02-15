from scipy import integrate
from scipy import special as spspecial
import numpy as np


def banerjee_44(rbar,D):
    """
    Approximate Ψ'(||μ||) = Ψ'(rbar) = ||η|| = κ, where Φ is the
    log-partition of the von Mises-Fisher distribution in D dimensions, Ψ is
    its Legendre transform, η the natural parameter, μ the mean parameter
    and Ψ(||μ||) the radial profile of the radially symmetric function Ψ.

    Taken from eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).

    Parameters
    ----------
    rbar : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.

    Returns
    -------
    κs : K-dim. array
        Numerical approximations to Ψ'(rbar[k]) = κs[k], k = 1,...,K.

    """
    return rbar * (D-rbar**2) / (1-rbar**2)

def dΨ_base(μ_norm, D, mode='ODE_refined'):
    """
    Approximate Ψ'(||μ||) = Ψ'(μ_norm) = ||η|| = κ, where Φ is the
    log-partition of the von Mises-Fisher distribution in D dimensions, Ψ is
    its Legendre transform, η the natural parameter, μ the mean parameter
    and Ψ(||μ||) the radial profile of the radially symmetric function Ψ.

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    mode: string
        Optional argument to select method of approximation.
    Returns
    -------
    κs : K-dim. array
        Numerical approximations to κs[k] = Ψ'(μ_norm[k]), k = 1,...,K.

    """
    assert mode in ['B44', 'ODE_refined']
    if mode=='B44':
        return dΨbase_Banerjee_plus(μ_norm, D)
    elif mode=='ODE_refined':
        return dΨbase_ODE_refined(μ_norm, D)


def dΨbase_Banerjee_plus(μ_norm, D):
    """
    Approximate Ψ'(||μ||) = Ψ'(μ_norm) = ||η|| = κ, where Φ is the
    log-partition of the von Mises-Fisher distribution in D dimensions, Ψ is
    its Legendre transform, η the natural parameter, μ the mean parameter
    and Ψ(||μ||) the radial profile of the radially symmetric function Ψ.

    Builds on eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).

    Adds two additional terms to the approximation that further improve the
    approximation to the (intractable) Ψ'(||μ||) for large D.

    For small D (roughly D<10), this approximation is not good !

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.

    Returns
    -------
    κs : K-dim. array
        Numerical approximations to κs[k] = Ψ'(μ_norm[k]), k = 1,...,K.

    """
    dΨ = banerjee_44(μ_norm,D=D) # eq. (4.4) of Banerjee et al. (2005)
    dΨ = dΨ + 0.08 * (np.sin(np.pi*μ_norm)**2)/np.pi - 0.5 * μ_norm**2 # correction

    return dΨ

def dΨbase_ODE_refined(μ_norm, D):
    """
    Approximate Ψ'(||μ||) = Ψ'(μ_norm) = ||η|| = κ, where Φ is the
    log-partition of the von Mises-Fisher distribution in D dimensions, Ψ is
    its Legendre transform, η the natural parameter, μ the mean parameter
    and Ψ(||μ||) the radial profile of the radially symmetric function Ψ.

    Builds on eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).

    Results from plugging an estimate for Ψ'' into the second-order ODE, 
    Ψ'(||μ||) = (D-1) ||μ|| / ( 1 -  ||μ||^2 - 1/Ψ''(||μ||)  )
    Here we take the estimate Ψ'' from the derivative of eq. (4.4).

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.

    Returns
    -------
    κs : K-dim. array
        Numerical approximations to κs[k] = Ψ'(μ_norm[k]), k = 1,...,K.

    """

    return (D-1.)*( μ_norm/(1.-μ_norm**2) + μ_norm/(μ_norm**4+(D-2.)*μ_norm**2+D-1.) )


def Ψ_base(μ_norm, D, mode='ODE_refined'):
    """
    Approximate Ψ(||μ||), where Ψ is the Legendre transform of the log-partition
    for the von Mises-Fisher distribution in D dimensions.

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    mode: string
        Optional argument to select method of approximation.

    Returns
    -------
    Ψμ : K-dim. array
        Numerical approximations to Ψ(μ_norm[k]), k = 1,...,K.

    """
    assert mode in ['B44', 'ODE_refined']
    if mode=='B44':
        return Ψbase_Banerjee_plus(μ_norm, D)
    elif mode=='ODE_refined':
        return Ψbase_ODE_refined(μ_norm, D)



def Ψbase_Banerjee_plus(μ_norm, D):
    """
    Approximate Ψ(||μ||), where Ψ is the Legendre transform of the log-partition
    for the von Mises-Fisher distribution in D dimensions.

    Builds on the anti-derivative of eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).

    Adds two additional terms to the approximation that further improve the
    approximation to the (intractable) Ψ(||μ||) for large D.

    For small D (roughly D<10), this approximation is not good !

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.

    Returns
    -------
    Ψμ : K-dim. array
        Numerical approximations to Ψ(μ_norm[k]), k = 1,...,K.

    """
    μ2 = μ_norm**2
    # anti-derivative of eq. (4.4) in Banerjee et al. (2005)
    Ψμ = 0.5 *((1-D) * np.log(1 - μ2) + μ2)
    # anti-derivatives of additive correction terms
    Ψμ = Ψμ + 0.02/np.pi * (2.*μ_norm - np.sin(2.*np.pi*μ_norm)/np.pi) - (μ_norm**3)/6.

    return Ψμ


def Ψbase_ODE_refined(μ_norm, D):
    """
    Approximate Ψ(||μ||), where Ψ is the Legendre transform of the log-partition
    for the von Mises-Fisher distribution in D dimensions.

    Builds on eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).

    Results from plugging an estimate for Ψ'' into the second-order ODE, 
    Ψ'(||μ||) = (D-1) ||μ|| / ( 1 -  ||μ||^2 - 1/Ψ''(||μ||)  )
    Here we take the estimate Ψ'' from the derivative of eq. (4.4).

    We then find the analytical antriderivative of the resulting Ψ'(||μ||).

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.

    Returns
    -------
    Ψμ : K-dim. array
        Numerical approximations to Ψ(μ_norm[k]), k = 1,...,K.

    """
    sqrt, v, μ_norm2 = np.sqrt(D**2/4. - 2.*D + 2.), D/2. - 1., μ_norm**2
    log1 = np.log(v + μ_norm2 - sqrt) 
    log2 = np.log(v + μ_norm2 + sqrt)
    offset = np.log(v+sqrt) - np.log(v-sqrt)
    return (D-1.) * ((log1 - log2 + offset)/(4.*sqrt) - 0.5 * np.log(1.-μ_norm2))

def comp_norm(μ, D):
    """ Compute norms ||μ|| of collection of mean parameters μ. """
    # could also replace with np.linalg.norm(μ,axis=1)
    assert μ.shape[-1] == D
    μ = μ.reshape(1,D) if μ.ndim == 1 else μ
    assert μ.ndim == 2

    return np.sqrt((μ**2).sum(axis=-1))


def Ψ(μ, D, Ψ0=None, t0=0., return_grad=False, solve_delta=True):
    """
    Negative entropy Ψ(μ) = Ψ(||μ||), where Ψ is (the radial profile of)
    the Legendre transform of the log-partition for the von Mises-Fisher
    distribution in D dimensions.

    Solves a second-order ODE to compute Ψ(μ) and (optionally) ∇Ψ(μ).
    Optionally can solve only for the difference of the ODE to an
    approximate reference solution - in the extreme case when called with
    optional argument Ψ0[1]=0.0 will directly return the approximate
    reference solution without involving any ODE solver, resulting in a
    very quick approximate result (better for larger D, i.e. D > 10).

    Reference solution builds on the anti-derivative of eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).

    Parameters
    ----------
    μ : K-by-D. array_like
        Collection of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    Ψ0 : tuple or list float or [None, float]
        Gives [Ψ(t0), Ψ'(t0)]. First entry is computed if provided as None.
        If the second entry Ψ0[1] = Ψ'(t0) = 0.0, the function will not
        solve any ODE and instead return the (approximate) reference solution.
    t0 : float, 0.0 <= t0 <= 1.0
        Starting point for the numerical solver for the radial profile
        Ψ(||μ||) defined for 0.0 < ||μ|| < 1.0.
    return_grad : bool
        Flag whether to also return the gradients ∇Ψ(μ) alongside Ψ(μ).
    solve_delta : bool
        Flag whether to solve only for the difference to a reference
        solution

    Returns
    -------
    Ψμ : K-dim. array
        Numerical approximations to Ψ(μ[k]), k = 1,...,K.
    gradΨμ : K-by-D array (optional, only if return_grad=True)
        Numerical approximations to ∇Ψ(μ[k]), k = 1,...,K.

    """
    μ_norm = comp_norm(μ, D=D)
    y0 = [Ψ0, 1e-6] if np.ndim(Ψ0)==0 else Ψ0 
    assert len(y0) == 2
    if Ψ0[1] > 0:
        if solve_delta:
            Ψμ, dΨμ = solve_delta_Ψ_dΨ(μ_norm, D=D, y0=y0, t0=t0)
            dΨμ = dΨμ + dΨ_base(μ_norm, D=D, mode='B44')
            Ψμ = Ψμ + Ψ_base(μ_norm, D=D, mode='B44')
        else:
            Ψμ, dΨμ = solve_Ψ_dΨ(μ_norm, D=D, y0=y0, t0=t0)
    else:
        Ψμ =  Ψ_base(μ_norm, D=D, mode='ODE_refined')
        dΨμ = dΨ_base(μ_norm, D=D, mode='ODE_refined') if return_grad else None

    if return_grad:
        return Ψμ, _gradΨ(dΨμ, μ, μ_norm, D=D)
    else:
        return Ψμ


def gradΨ(μ, D, Ψ0=None, t0=0.):
    """
    Gradient ∇Ψ(μ) of the negative entropy Ψ(μ) = Ψ(||μ||), where Ψ
    is (the radial profile of) the Legendre transform of the log-partition
    for the von Mises-Fisher distribution in D dimensions.

    Solves a first-order ODE to compute ∇Ψ(μ).

    Parameters
    ----------
    μ : K-by-D. array_like
        Collection of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    Ψ0 : float or None
        Gives Ψ'(t0). Will be computed if provided as None.
    t0 : float, 0.0 <= t0 <= 1.0
        Starting point for the numerical solver for the radial profile
        Ψ'(||μ||) defined for 0.0 < ||μ|| < 1.0.

    Returns
    -------
    gradΨμ : K-by-D array (optional, only if return_grad=True)
        Numerical approximations to ∇Ψ(μ[k]), k = 1,...,K.

    """
    μ_norm = comp_norm(μ, D=D)
    y0 = [1e-6] if Ψ0 is None else [Ψ0]
    dΨ =  solve_dΨ(μ_norm, D=D, y0=y0, t0=t0)
    gradΨμ = _gradΨ(dΨ, μ, μ_norm, D)

    return gradΨμ


def _gradΨ(dΨ, μ, μ_norm, D):
    """
    Gradient ∇Ψ(μ) from Ψ'(||μ||). Ψ is (the radial profile of) the
    Legendre transform of the log-partition for the von Mises-Fisher
    distribution in D dimensions.

    Parameters
    ----------
    dΨ : K-dim. array_like
        Ψ'(||μ[k]||) for  mean parameters μ[k], k = 1,...,K.
    μ : K-by-D. array_like
        Collection of mean parameters μ[k], k = 1,...,K.
    μ_norm : K-dim. array_like
        Pre-computed L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.

    Returns
    -------
    gradΨμ : K-by-D array (optional, only if return_grad=True)
        Numerical approximations to ∇Ψ(μ[k]), k = 1,...,K.

    """
    return μ * (dΨ / μ_norm).reshape(*μ.shape[:-1], 1)

def hessΨ(μ, D, Ψ0=None):
    """
    Hessian ∇∇Ψ(μ), where Ψ is (the radial profile of) the Legendre
    transform of the log-partition for the von Mises-Fisher distribution
    in D dimensions.

    Parameters
    ----------
    μ : K-by-D. array_like
        Collection of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    Ψ0 : float or None
        Gives Ψ'(t0). Will be computed if provided as None.

    Returns
    -------
    hessΨμ : K-by-D-by-D array
        Numerical approximations to ∇∇Ψ(μ[k]), k = 1,...,K.

    """
    μ_norm = comp_norm(μ, D=D)
    y0 = [1e-6] if Ψ0 is None else [Ψ0]
    dΨ =  solve_dΨ(μ_norm, D, y0=y0)
    ddΨ = vMF_ODE_first_order(μ_norm, dΨ, D)
    hessΨμ = _hessΨ(ddΨ, dΨ, μ, μ_norm, D)

    return hessΨμ

def _hessΨ(ddΨ, dΨ, μ, μ_norm, D):
    """
    Gradient ∇Ψ(μ) from Ψ''(||μ||) and Ψ'(||μ||). Ψ is (the radial profile
    of) the Legendre transform of the log-partition for the von Mises-Fisher
    distribution in D dimensions.

    Parameters
    ----------
    dΨ : K-dim. array_like
        Ψ'(||μ[k]||) for  mean parameters μ[k], k = 1,...,K.
    ddΨ : K-dim. array_like
        Ψ''(||μ[k]||) for  mean parameters μ[k], k = 1,...,K.
    μ : K-by-D. array_like
        Collection of mean parameters μ[k], k = 1,...,K.
    μ_norm : K-dim. array_like
        Pre-computed L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.

    Returns
    -------
    hess : K-by-D-by-D array
        Numerical approximations to ∇∇Ψ(μ[k]), k = 1,...,K.

    """
    out_shape = [*np.ones(np.ndim(μ_norm),dtype=np.int32),D,D]
    μμT = np.matmul(μ.reshape(*μ.shape,1), μ.reshape(*μ.shape[:-1],1,μ.shape[-1]))
    I_D = np.eye(D).reshape(out_shape)
    hess = (dΨ/μ_norm).reshape(-1,1,1) * I_D + (ddΨ/μ_norm**2 - dΨ/μ_norm**3).reshape(-1,1,1) * μμT

    return hess


def DBregΨ(X, μ, D, treat_x_constant=False, Ψ0=None):
    """
    Bregman divergence D_Ψ(x||μ), where Ψ is (the radial profile of)
    the Legendre transform of the log-partition for the von Mises-Fisher
    distribution in D dimensions.

    Note that for either ||X[n]||=1 or ||μ[k]||=1, it is D_Ψ(X[n]||μ[k])=Inf
    unless X[n]=μ[k], so for X on S^(D-1), consider using treat_x_constant=True!

    Parameters
    ----------
    X : N-by-D array_like or scipy.sparse matrix
        Input matrix with unit-norm data vectors in the rows.
    μ : K-by-D. array_like
        Collection of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    treat_x_constant : bool
        Flag whether to omit the term Ψ(X) and instead return D_Ψ(X||μ)-Ψ(X).
    Ψ0 : tuple or list float or [None, float]
        Gives [Ψ(t0), Ψ'(t0)]. First entry is computed if provided as None.
        If the second entry Ψ0[1] = Ψ'(t0) = 0.0, the function will not
        solve any ODE and instead return the (approximate) reference solution.

    Returns
    -------
    dΨdμ_x_μ : N-by-K array
        Bregman divergences D_Ψ(X[n]||μ[k]), for n=1,...,N, k = 1,...,K.
        If treat_x_constant=True, instead returns D_Ψ(X[n]||μ[k]) - Ψ(X[n]) !

    """
    Ψμ, dΨdμ = Ψ(μ, D=D, return_grad=True, Ψ0=Ψ0)
    dΨdμ_x_μ = ((dΨdμ*μ).sum(axis=-1) - Ψμ).reshape(1,-1) - X.dot(dΨdμ.T) # N x K
    if treat_x_constant:
        return dΨdμ_x_μ
    else:
        return dΨdμ_x_μ + Ψ(X, D=D, Ψ0=Ψ0, t0=t0)


def vMF_ODE_first_order(t, x, D):
    """ First-order ODE for Ψ''(||μ||) = f(Ψ', ||μ||, D). """
    return x / ((1 - t**2) * x + (1-D) * t )


def vMF_ODE_second_order(t, x, D):
    """ Second-order ODE for Ψ''(||μ||) = f(Ψ', ||μ||, D). Allows solving for Ψ, Ψ'."""
    u, v = x

    return [v, v / ((1 - t**2) * v + (1-D) * t )]


def solve_Ψ_dΨ(μ_norm, D, t0=0., y0=[None, 1e-6], rtol=1e-12, atol=1e-12):
    """
    ODE solver for Ψ(||μ||) and its derivative Ψ'(||μ||), where Ψ is
    (the radial profile of) the Legendre transform of the log-partition
    for the von Mises-Fisher distribution in D dimensions.

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    t0 : float, 0.0 <= t0 <= 1.0
        Starting point for the numerical solver for the radial profile
        Ψ(||μ||) defined for 0.0 < ||μ|| < 1.0.
    y0 : tuple or list float or [None, float]
        Gives [Ψ(t0), Ψ'(t0)]. First entry is computed if provided as None.
    rtol : float, > 0.0
        Relative tolerance for ODE solver.
    atol : float, > 0.0
        Absolute tolerance for ODE solver.

    Returns
    -------
    Ψμ : K-dim. array
        Numerical approximations to Ψ(μ_norm[k]), k = 1,...,K.
    dΨμ : K-dim. array
        Numerical approximations to Ψ'(μ_norm[k]), k = 1,...,K.

    """
    assert np.all(μ_norm < 1.0)
    if y0[0] is None:
        y0[0] = (D/2-1) * np.log(2) + spspecial.loggamma(D/2) # Ψ(0) = - Φ(0)

    def f(t, x):
        return vMF_ODE_second_order(t,x,D=D)

    if np.ndim(μ_norm) > 0:
        μ_norm, idx, idx_inv = np.unique(μ_norm, return_index=True, return_inverse=True)
        Ψμ, dΨμ = np.empty_like(μ_norm), np.empty_like(μ_norm)
        if np.any(μ_norm <= t0):
            out = integrate.solve_ivp(f, t_span=[t0, μ_norm[0]], t_eval=μ_norm[μ_norm <= t0][::-1],
                                      y0=y0, rtol=rtol, atol=atol)
            Ψ1, dΨ1 = out.y[0][::-1], out.y[1][::-1]
        else:
            Ψ1, dΨ1 = [], []
        if np.any(t0 < μ_norm):
            out = integrate.solve_ivp(f, t_span=[t0, μ_norm[-1]], t_eval=μ_norm[μ_norm > t0],
                                      y0=y0, rtol=rtol, atol=atol)
            Ψ2, dΨ2 = out.y[0], out.y[1]
        else:
            Ψ2, dΨ2 = [], []
        Ψμ = np.concatenate((Ψ1,Ψ2))[idx_inv]  # vectors
        dΨμ = np.concatenate((dΨ1,dΨ2))[idx_inv] #
    else:
        t_span = [t0, μ_norm] if t0 < μ_norm else [t0, t0-μ_norm]
        f = f_fw if t0 < μ_norm else f_bw
        out = integrate.solve_ivp(f, t_span=t_span,
                                  y0=y0, rtol=rtol, atol=atol)
        Ψμ, dΨμ = out.y[0][-1], out.y[1][-1] # scalars
        
    return Ψμ, dΨμ


def solve_dΨ(μ_norm, D, t0=0., y0=[1e-6], rtol=1e-12, atol=1e-12):
    """
    ODE solver for Ψ'(||μ||), where Ψ is (the radial profile of)
    the Legendre transform of the log-partition for the
    von Mises-Fisher distribution in D dimensions.

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    t0 : float, 0.0 <= t0 <= 1.0
        Starting point for the numerical solver for the radial profile
        Ψ'||μ||) defined for 0.0 < ||μ|| < 1.0.
    y0 : float
        Gives Ψ'(t0).
    rtol : float, > 0.0
        Relative tolerance for ODE solver.
    atol : float, > 0.0
        Absolute tolerance for ODE solver.

    Returns
    -------
    dΨμ : K-dim. array
        Numerical approximations to Ψ'(μ_norm[k]), k = 1,...,K.

    """
    assert np.all(μ_norm < 1.0)

    def f(t, x):
        return vMF_ODE_first_order(t,x,D=D)

    if np.ndim(μ_norm) > 0:
        μ_norm, idx, idx_inv = np.unique(μ_norm, return_index=True, return_inverse=True)
        if np.any(μ_norm <= t0):
            out = integrate.solve_ivp(f, t_span=[t0, μ_norm[0]], t_eval=μ_norm[μ_norm <= t0][::-1],
                                      y0=y0, rtol=rtol, atol=atol)
            dΨ1 = out.y[0][::-1]
        else:
            dΨ1 = []
        if np.any(t0 < μ_norm):
            out = integrate.solve_ivp(f, t_span=[t0, μ_norm[-1]], t_eval=μ_norm[μ_norm > t0],
                                      y0=y0, rtol=rtol, atol=atol)
            dΨ2 = out.y[0]
        else:
            dΨ2 = []
        dΨμ = np.concatenate((dΨ1,dΨ2))[idx_inv] #
    else:
        t_span = [t0, μ_norm] if t0 < μ_norm else [t0, t0-μ_norm]
        f = f_fw if t0 < μ_norm else f_bw
        out = integrate.solve_ivp(f, t_span=t_span,
                                  y0=y0, rtol=rtol, atol=atol)
        dΨμ = out.y[0][-1]

    return dΨμ


def vMF_delta_ODE_second_order(t, x, D):
    """
    Second-order ODE for Ψ''(||μ||) = f(Ψ', ||μ||, D). Allows solving for Ψ, Ψ'.

    Actually uses a reference approximate solution to Ψ,Ψ' (with known Ψ'') and
    only represents the *difference* in the right-hand sides of the respective ODEs,
    meaning this ODE represents an ODE on the differences between the approximate
    and true solutions to Ψ''(||μ||) = f(Ψ', ||μ||, D).

    """
    u, v = x   # = Ψ', Ψ''
    t2 = t**2  # = ||μ||**2 = μ' * μ
    mt2 = (1.0 - t2)
    mt22 = mt2**2

    dx = v
    ddx = 1./mt2 + (1.-D)/mt22*(1.+t2 - t/(0.5*t*(2.-t)+0.08/np.pi*np.sin(np.pi*t)**2+v)) - 1. + t - 0.08*np.sin(2.*np.pi*t)

    return [dx, ddx]


def solve_delta_Ψ_dΨ(μ_norm, D, t0=0., y0=[None, 1e-6], rtol=1e-12, atol=1e-12):
    """
    ODE solver for Ψ(||μ||) and its derivative Ψ'(||μ||), where Ψ is
    (the radial profile of) the Legendre transform of the log-partition
    for the von Mises-Fisher distribution in D dimensions.

    Solves only for the *difference* between the actual ODE solution and a
    reference approximation to Ψ, Ψ', meaning that for poor ODE solution
    one can hopefully fall back to the (approximate) reference solution.

    Reference solution builds on the anti-derivative of eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).

    Parameters
    ----------
    μ_norm : K-dim. array_like
        L2 norms of mean parameters μ[k], k = 1,...,K.
    D : integer, >=2
        Dimensionality of vMF distribution and parameters η and μ.
    t0 : float, 0.0 <= t0 <= 1.0
        Starting point for the numerical solver for the radial profile
        Ψ(||μ||) defined for 0.0 < ||μ|| < 1.0.
    y0 : tuple or list float or [None, float]
        Gives [Ψ(t0), Ψ'(t0)]. First entry is computed if provided as None.
    rtol : float, > 0.0
        Relative tolerance for ODE solver.
    atol : float, > 0.0
        Absolute tolerance for ODE solver.

    Returns
    -------
    Ψμ : K-dim. array
        Numerical approximations to Ψ(μ_norm[k]), k = 1,...,K.
    dΨμ : K-dim. array
        Numerical approximations to Ψ'(μ_norm[k]), k = 1,...,K.

    """
    assert np.all(μ_norm < 1.0)
    if y0[0] is None:
        y0[0] = (D/2-1) * np.log(2) + spspecial.loggamma(D/2) # Ψ(0) = - Φ(0)

    if y0[1] == 0:
        Ψ = y0[0] * np.ones_like(μ_norm)
        dΨ = np.zeros_like(μ_norm)
        return Ψ, dΨ

    def f(t, x):
        return vMF_delta_ODE_second_order(t,x,D=D)

    if np.ndim(μ_norm) > 0:
        μ_norm, idx, idx_inv = np.unique(μ_norm, return_index=True, return_inverse=True)
        Ψ, dΨ = np.empty_like(μ_norm), np.empty_like(μ_norm)
        if np.any(μ_norm <= t0):
            out = integrate.solve_ivp(f, t_span=[t0, μ_norm[0]], t_eval=μ_norm[μ_norm <= t0][::-1],
                                      y0=y0, rtol=rtol, atol=atol)
            Ψ1, dΨ1 = out.y[0][::-1], out.y[1][::-1]
        else:
            Ψ1, dΨ1 = [], []
        if np.any(t0 < μ_norm):
            out = integrate.solve_ivp(f, t_span=[t0, μ_norm[-1]], t_eval=μ_norm[μ_norm > t0],
                                      y0=y0, rtol=rtol, atol=atol)
            Ψ2, dΨ2 = out.y[0], out.y[1]
        else:
            Ψ2, dΨ2 = [], []
        Ψ = np.concatenate((Ψ1,Ψ2))[idx_inv]  # vectors
        dΨ = np.concatenate((dΨ1,dΨ2))[idx_inv] #
    else:
        t_span = [t0, μ_norm] if t0 < μ_norm else [t0, t0-μ_norm]
        f = f_fw if t0 < μ_norm else f_bw
        out = integrate.solve_ivp(f, t_span=t_span,
                                  y0=y0, rtol=rtol, atol=atol)
        Ψ, dΨ = out.y[0][-1], out.y[1][-1] # scalars

    return Ψ, dΨ
