Code repository for the study

"A solution for the mean parametrization of the von Mises-Fisher distribution"

## Contents

This code repository contains an implementation of (gradient / Hessian of) the log-partition function and negative entropy function for the D-dimensional [von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution), alongside code for fitting mixtures of vMF distribution in both natural and mean parametrization, all in Python. Sparse matrix support allows fast execution in high-dimensional feature spaces when features are sparse, as is common in document embeddings such as [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).   

We make use of the mean-parameterized form in Bregman clustering for document clustering in high-dimensional (>25k dim) text embedding spaces, for the classical document datasets ['20-newsgroups'](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) and ['classic3'](http://ir.dcs.gla.ac.uk/resources/test_collections/).

The [example notebook](https://github.com/vmf-negentropy/vmf-negentropy/blob/master/example_bregman_clustering.ipynb) contains a small toy example of Bregman clustering that fits a mixture of von Mises-Fisher distribution in mean paramtrization to several hundred data points in $1000$ dimensions within a fraction of a second. 
The [experiments notebook](https://github.com/vmf-negentropy/vmf-negentropy/blob/master/experiments.ipynb) can be used to reproduce all figures and numerical results of the manuscript.


## Background

The von Mises-Fisher distribution vMF($m, \kappa)$ in D dimensions is a natural exponential family 

$\log p(x|\eta) = \eta^\top{}x - \Phi(\eta) - \log h(x) $

with base measure $h(x)$ uniform over the hypersphere $S^{D-1}$, natural parameter $\eta = \kappa m$ and log-partition function

$\Phi(\eta) = \Phi(||\eta||) = \log I_{D/2−1}(||\eta||) − (D/2 − 1) \log ||\eta||.$

As a natural exponential family with strictly convex log-partition function, the vMF family of distributions can equivalently be written as 

$\log p(x|\mu) = \nabla{}\Psi(\mu)^\top{}(x - \mu) + \Psi(\mu) - \log h(x)$

with 'negative entropy' function 

$\Psi(\mu) = \max_\eta \ \mu^\top\eta - \Phi(\eta) = - H[X|\mu] + const.$ 

and mean parameter $\mu = \nabla\Phi(\eta) = \mathbb{E}[X]$. 
The mean-parameter form of exponential families is convenient for maximum likelihood or variants thereof such as expectation maximization, but the required function $\Psi(\mu)$ defined for all $||\mu|| < 1$ and its gradient are not known in closed form.

We derived a second-order ODE on $\Psi(||\mu||)$, the radial profile of $\Psi(\mu)$, which can be used to compute quantities $\Psi(\mu)$, $\nabla\Psi(\mu)$, $\nabla^2\Psi(\mu)$, and hence work with the von Mises-Fisher distribution in mean-parameterized exponential family form. We also provide fast closed-form approximations to the solutions of the ODE for large dimensions $D$.  
