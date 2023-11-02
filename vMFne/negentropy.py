from scipy import integrate
import numpy as np

def vMF_ODE_first_order(t, x, D=2):
    return x / ((1 - t**2) * x + (1-D) * t )

def vMF_ODE_second_order(t, x, D=2):
    u, v = x
    return [v, v / ((1 - t**2) * v + (1-D) * t )]

def solve_Ψ_dΨ(μ_norm, D=2, y0=[0.0, 1e-8], rtol=1e-12, atol=1e-12):

    assert np.all(μ_norm < 1.0)

    def f(t, x):
        return vMF_ODE_second_order(t,x,D=D)

    if np.ndim(μ_norm) > 0:
        idx = np.argsort(μ_norm)  # tbd: catch repeats in norm,
        μ_norm = (1.*μ_norm[idx]) #      solve_ivp doesn't like those
        Ψ, dΨ = np.empty_like(μ_norm), np.empty_like(μ_norm)
        out = integrate.solve_ivp(f, t_span=[0, μ_norm[-1]], t_eval=μ_norm,
                                  y0=y0, rtol=rtol, atol=atol)
        Ψ[idx] = out.y[0]
        dΨ[idx] = out.y[1]
    else:
        out = integrate.solve_ivp(f, t_span=[0, μ_norm],
                                  y0=y0, rtol=rtol, atol=atol)
        Ψ, dΨ = out.y[0][-1], out.y[1][-1]
        
    return Ψ, dΨ

def solve_dΨ(μ_norm, D=2, y0=[1e-8], rtol=1e-12, atol=1e-12):

    assert np.all(μ_norm < 1.0)

    def f(t, x):
        return vMF_ODE_first_order(t,x,D=D)

    if np.ndim(μ_norm) > 0:
        idx = np.argsort(μ_norm)  # tbd: catch repeats in norm, 
        μ_norm = (1.*μ_norm[idx]) #      solve_ivp doesn't like those
        dΨ = np.empty_like(μ_norm)
        out = integrate.solve_ivp(f, t_span=[0, μ_norm[-1]], t_eval=μ_norm,
                                  y0=y0, rtol=rtol, atol=atol)
        dΨ[idx] = out.y[0]
    else:
        out = integrate.solve_ivp(f, t_span=[0, μ_norm],
                                  y0=y0, rtol=rtol, atol=atol)
        dΨ =  out.y[0][-1]

    return dΨ

def comp_norm(μ, D=2):
    assert μ.shape[-1] == D
    μ = μ.reshape(1,D) if μ.ndim == 1 else μ
    assert μ.ndim == 2
    return np.sqrt((μ**2).sum(axis=-1))

def Ψ(μ, D=2, return_grad=False):
    μ_norm = comp_norm(μ, D=D)
    Ψ, dΨ = solve_Ψ_dΨ(μ_norm, D)
    if return_grad:
        return Ψ, _gradΨ(dΨ, μ, μ_norm, D=D)
    else:
        return Ψ

def gradΨ(μ, D=2):
    μ_norm = comp_norm(μ, D=D)
    dΨ =  solve_dΨ(μ_norm, D)
    return  _gradΨ(dΨ, μ, μ_norm, D)

def _gradΨ(dΨ, μ, μ_norm, D=2):
    return μ * (dΨ / μ_norm).reshape(*μ.shape[:-1], 1)

def hessΨ(μ, D=2):
    μ_norm = comp_norm(μ, D=D)
    dΨ =  solve_dΨ(μ_norm, D)
    ddΨ = vMF_ODE_first_order(μ_norm, dΨ, D)
    return _hessΨ(ddΨ, dΨ, μ, μ_norm, D)

def _hessΨ(ddΨ, dΨ, μ, μ_norm, D=2):
    out_shape = [*np.ones(np.ndim(μ_norm),dtype=np.int32),D,D]
    μμT = np.matmul(μ.reshape(*μ.shape,1), μ.reshape(*μ.shape[:-1],1,μ.shape[-1]))
    I_D = np.eye(D).reshape(out_shape)
    hess = (dΨ/μ_norm).reshape(-1,1,1) * I_D + (ddΨ/μ_norm**2 - dΨ/μ_norm**3).reshape(-1,1,1) * μμT
    return hess

def DBregΨ(x,μ,D=2,treat_x_constant=False):
    # Bregman divergence D_Ψ(x||μ) for vMF distribution on S^(D-1).
    # Note that for either ||x||=1 or ||μ||=1, it is D_Ψ(x||μ) = Inf unless x=μ,
    # so for x on S^(D-1), consider using treat_x_constant=True !
    Ψμ, dΨdμ = Ψ(μ, D=D, return_grad=True)
    if treat_x_constant:
        return (dΨdμ*(x - μ)).sum(axis=-1) - Ψμ
    else:
        return (dΨdμ*(x - μ)).sum(axis=-1) - Ψμ + Ψ(x, D=D)

def vMF_loglikelihood_Ψ(x,μ,D=2):
    # log p(x|μ) = - D_\Psi(x||\mu) + \Psi(x) + some constant in dimensionality D.
    return - DBregΨ(x,μ,D=D,treat_x_constant=True)

