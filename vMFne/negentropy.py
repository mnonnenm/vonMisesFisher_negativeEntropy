from scipy import integrate
from scipy import special as spspecial
import numpy as np


def banerjee_44(rbar,D):
    """ Approximately solve for Ψ'(rbar) = κ in notation rbar = ||μ||, κ = ||η||.
    Taken from eq. (4.4) of
    Banerjee, Arindam, et al.
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
    Journal of Machine Learning Research 6.9 (2005).
    """
    return rbar * (D-rbar**2) / (1-rbar**2)

def dΨ_base(μ_norm, D):
    # explicit approximation to Ψ'(|μ|), better for large D
    dΨ = banerjee_44(μ_norm,D=D) # eq. (4.4) of Banerjee et al. (2005)
    dΨ = dΨ + 0.08 * (np.sin(np.pi*μ_norm)**2)/np.pi - 0.5 * μ_norm**2 # correction

    return dΨ

def Ψ_base(μ_norm, D):
    # explicit approximation to Ψ(|μ|), better for large D
    μ2 = μ_norm**2
    Ψ = 0.5 *((1-D) * np.log(1 - μ2) + μ2) # anti-derivative of eq. (4.4) in Banerjee et al. (2005)
    Ψ = Ψ + 0.02/np.pi * (2.0*μ_norm - np.sin(2.0*np.pi*μ_norm)/np.pi) - (μ_norm**3)/6.0 # correction

    return Ψ

def vMF_ODE_first_order(t, x, D):

    return x / ((1 - t**2) * x + (1-D) * t )

def vMF_ODE_second_order(t, x, D):
    u, v = x

    return [v, v / ((1 - t**2) * v + (1-D) * t )]

def solve_Ψ_dΨ(μ_norm, D, t0=0., y0=[None, 1e-6], rtol=1e-12, atol=1e-12):

    assert np.all(μ_norm < 1.0)
    if y0[0] is None:
        y0[0] = (D/2-1) * np.log(2) + spspecial.loggamma(D/2) # Ψ(0) = - Φ(0)

    def f(t, x):
        return vMF_ODE_second_order(t,x,D=D)

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

def solve_dΨ(μ_norm, D, t0=0., y0=[1e-6], rtol=1e-12, atol=1e-12):

    assert np.all(μ_norm < 1.0)

    def f(t, x):
        return vMF_ODE_first_order(t,x,D=D)

    if np.ndim(μ_norm) > 0:
        idx = np.argsort(μ_norm)  # tbd: catch repeats in norm, 
        μ_norm = (1.*μ_norm[idx]) #      solve_ivp doesn't like those
        dΨ = np.empty_like(μ_norm)
        out = integrate.solve_ivp(f, t_span=[t0, μ_norm[-1]], t_eval=μ_norm,
                                  y0=y0, rtol=rtol, atol=atol)
        dΨ[idx] = out.y[0] # vector
    else:
        t_span = [t0, μ_norm] if t0 < μ_norm else [μ_norm, t0]
        out = integrate.solve_ivp(f, t_span=t_span,
                                  y0=y0, rtol=rtol, atol=atol)
        dΨ =  out.y[0][-1] # scalar

    return dΨ


def vMF_delta_ODE_second_order(t, x, D):
    # ODE to compute the difference between [Ψ(|μ|),Ψ'(|μ|)] and our explicit approximations to it.
    u, v = x
    t2 = t**2
    mt2 = (1.0 - t2)
    mt22 = mt2**2

    dx = v
    ddx = 1./mt2 + (1.-D)/mt22*(1.+t2 - t/(0.5*t*(2.-t)+0.08/np.pi*np.sin(np.pi*t)**2+v)) - 1. + t - 0.08*np.sin(2.*np.pi*t)

    return [dx, ddx]

def solve_delta_Ψ_dΨ(μ_norm, D, t0=0., y0=[None, 1e-6], rtol=1e-12, atol=1e-12):

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

def comp_norm(μ, D):
    assert μ.shape[-1] == D
    μ = μ.reshape(1,D) if μ.ndim == 1 else μ
    assert μ.ndim == 2

    return np.sqrt((μ**2).sum(axis=-1))

def Ψ(μ, D, Ψ0=None, t0=0., return_grad=False, solve_delta=True):
    μ_norm = comp_norm(μ, D=D)
    y0 = [Ψ0, 1e-6] if np.ndim(Ψ0)==0 else Ψ0 
    assert len(y0) == 2
    if solve_delta:
        Ψ, dΨ = solve_delta_Ψ_dΨ(μ_norm, D=D, y0=y0, t0=t0)
        dΨ = dΨ + dΨ_base(μ_norm, D=D)
        Ψ = Ψ + Ψ_base(μ_norm, D=D)
    else:
        Ψ, dΨ = solve_Ψ_dΨ(μ_norm, D=D, y0=y0, t0=t0)

    if return_grad:
        return Ψ, _gradΨ(dΨ, μ, μ_norm, D=D)
    else:
        return Ψ

def gradΨ(μ, D, Ψ0=None, t0=0.):
    μ_norm = comp_norm(μ, D=D)
    y0 = [1e-6] if Ψ0 is None else [Ψ0]
    dΨ =  solve_dΨ(μ_norm, D=D, y0=y0, t0=t0)

    return  _gradΨ(dΨ, μ, μ_norm, D)

def _gradΨ(dΨ, μ, μ_norm, D=2):
    return μ * (dΨ / μ_norm).reshape(*μ.shape[:-1], 1)

def hessΨ(μ, D, Ψ0=None):
    μ_norm = comp_norm(μ, D=D)
    y0 = [1e-6] if Ψ0 is None else [Ψ0]
    dΨ =  solve_dΨ(μ_norm, D, y0=y0)
    ddΨ = vMF_ODE_first_order(μ_norm, dΨ, D)

    return _hessΨ(ddΨ, dΨ, μ, μ_norm, D)

def _hessΨ(ddΨ, dΨ, μ, μ_norm, D):
    out_shape = [*np.ones(np.ndim(μ_norm),dtype=np.int32),D,D]
    μμT = np.matmul(μ.reshape(*μ.shape,1), μ.reshape(*μ.shape[:-1],1,μ.shape[-1]))
    I_D = np.eye(D).reshape(out_shape)
    hess = (dΨ/μ_norm).reshape(-1,1,1) * I_D + (ddΨ/μ_norm**2 - dΨ/μ_norm**3).reshape(-1,1,1) * μμT

    return hess

def DBregΨ(x, μ, D, treat_x_constant=False, Ψ0=None):
    # Bregman divergence D_Ψ(x||μ) for vMF distribution on S^(D-1).
    # Note that for either ||x||=1 or ||μ||=1, it is D_Ψ(x||μ) = Inf unless x=μ,
    # so for x on S^(D-1), consider using treat_x_constant=True !
    # x : D-dim. vector or N-D matrix
    # μ : D-dim. vector or K-D matrix
    Ψμ, dΨdμ = Ψ(μ, D=D, return_grad=True, Ψ0=Ψ0)
    dΨdμ_x_μ = ((dΨdμ*μ).sum(axis=-1) - Ψμ).reshape(1,-1) - x.dot(dΨdμ.T) # N x K
    if treat_x_constant:
        return dΨdμ_x_μ
    else:
        return dΨdμ_x_μ + Ψ(x, D=D, Ψ0=Ψ0, t0=t0)
