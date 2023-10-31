# vonMisesFisher_negativeEntropy

Development repository for "An ordinary differential equation for the negative
entropy function of the von Mises-Fisher distribution."

The von Mises-Fisher distribution vMF($m, \kappa)$ in D dimensions is an exponential family if written in form 

$p(x|\eta) = h(x) e^{\eta^\top{}x - \Phi(\eta)}$

with $h(x)$ uniform over the hypersphere $S^{D-1}$, $\eta = \kappa m$ and 

$\Phi(\eta) = \Phi(||\eta||) = \log I_{D/2−1}(||\eta||) − (D/2 − 1) \log ||\eta||.$

As a minimal exponential family, it can equivalently be written as 

$p(x|\mu) = h(x) e^{\nabla{}\Psi(\mu)^\top{}(x - \mu) + \Psi(\mu)}$
with negative entropy $\Psi(\mu)$ and mean parameter $\mu = \nabla\Phi(\eta)$. The function $\Psi(\mu)$ defined for all $||\mu|| < 1$ however is generally unknown. 

Here we derive a second-order ODE on $\Psi(||\mu||)$, the radial profile of $\Psi(\mu)$, which can be used to compute quantities $\Psi(\mu)$, $\nabla\Psi(\mu)$, $\nabla^2\Psi(\mu)$, and hence work with the von Mises-Fisher distribution in mean-parameterized exponential family form. 

We show several direct applications that open up with the mean-parameterized form, including (soft) Bregman clustering.

Implementation of (gradient of / Hessian of) negative entropy for the D-dimensional von Mises-Fisher distribution, alongside several simple applications to statistics and machine learning, in Python. 
