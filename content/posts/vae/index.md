+++
title = 'Variational Autoencoders'
date = 2024-04-10T23:41:23+05:30
draft = false
math = true
ShowToc= true
+++
# Introduction
An auto-encoder learns an efficient representation of data. It "compresses" said data into lower dimensional vector and also learns to reconstruct it from said vector called latent vector.

In variational auto-encoders, apart from also learning to re-construct from the compressed vector, it also learns a mapping to an underlying "latent space". This mapping can then help us to generate new variations of the data.
# Problem Statement
{{< image src="image.png" alt="Directed Graphical Model" title="Fig 1. Directed Graphical Model" >}}
$\mathbf{z}$ is a continuous random variable sampled from some prior distribution \(p_\theta(\mathbf{z})\).
- A data point $\mathbf{x}^{(i)}$ is generated from conditional distribution $p_\theta(\mathbf{x}|\mathbf{z})$. 
- Therefore, $p_\theta(\mathbf{x}|\mathbf{z})$ acts as a decoder distribution with $\theta$ as its parameters.
- And $p_\theta(\mathbf{z}|\mathbf{x})$ is the encoder distribution.
### Our Goal
It is maximise the likelihood: $p_\theta(\mathbf{x})$, i.e parameters $\theta$ are good at generating data data resembling $\mathbf{x}$. This gives rise to a plethora of problems.
- $p_\theta(\mathbf{x})= \int_{\mathbf{z}}p_\theta(\mathbf{x},\mathbf{z})d\mathbf{z}= \int_{\mathbf{z}}p_\theta(\mathbf{z})p_\theta(\mathbf{x}|\mathbf{z})d\mathbf{z}$ is intractable, meaning it may be theoretically possible to solve the integral but in practical sense, it takes too much time or resources to be useful.

**Example:**
- $\mathbf{z}\in\mathbb{R}^n$ for which the integral can expand to $\int_{z_1}\int_{z_2}...\int_{z_n}p_\theta(\mathbf{x},\mathbf{z})dz_1dz_2...dz_n$  
- This integral become quite hard to solve when $\mathbf{z}$ has 100's of dimensions.

Another way to compute $p_\theta(\mathbf{x})$ is through the posterior $p_\theta(\mathbf{z}|\mathbf{x})$, but that too requires we know $p_\theta(\mathbf{x})$. From Bayes theorem:
$$p_\theta(\mathbf{z}|\mathbf{x})=\frac{p_\theta(\mathbf{z})p_\theta(\mathbf{x}|\mathbf{z})}{p_\theta(\mathbf{x})} \tag{1}$$
So instead of trying to find the true posterior, we find some surrogate or approximate posterior ~ $q_\phi(\mathbf{z}|\mathbf{x})$.

$q_\phi(\mathbf{z}|\mathbf{x})$ acts a decoder distribution with $\phi$ as its parameters.
# Variational Inference
Method used in statistics to estimate the posterior of a model by considering a surrogate posterior ~ $q_\phi(\mathbf{z}|\mathbf{x})$.
## ELBO as Loss Function
Value we want to maximise:
$$p_\theta(\mathbf{x})$$
Which is similar to maximising:
$$\log p_\theta(\mathbf{x})$$
Hence we can perform some mathematics on this log-likelihood:

$$
\begin{aligned} 
\log p_\theta(\mathbf{x})&=\log p_\theta(\mathbf{x})\\
&= \log p_\theta(\mathbf{x}) \cdot \int q_\phi(\mathbf{z}|\mathbf{x})d\mathbf{z} \\
&= \int\log p_\theta(\mathbf{x})q_\phi(\mathbf{z}|\mathbf{x})d\mathbf{z}\\
&= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x})]\\
&= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[\log\frac{p_\theta(\mathbf{z})p_\theta(\mathbf{x}|\mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]\\
&=\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[\log \frac{p_\theta(\mathbf{x},\mathbf{z})q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})q_\phi(\mathbf{z}|\mathbf{x})}\right]\\
&=\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] +
\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right] \\
&=\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] + D_{KL}\left(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z}|\mathbf{x})\right)
\end{aligned}
$$

Therefore:

$$
\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log\frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] + D_{\text{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})\right) \tag{2}
$$

Rearranging the above equation,
$$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] = 
\log p_\theta(\mathbf{x}) - 
D_{KL}\left(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z}|\mathbf{x})\right) \tag{3}$$
The term \(\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]\) is called the **Evidence Lower Bound** or **ELBO**. So, maximising this term with respect to parameters $\theta$ and $\phi$ 
- **maximises** $\log p_\theta(\mathbf{x})$,
- and **minimises** $D_{KL}\left(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z}|\mathbf{x})\right)$, the KL-Divergence between the surrogate and true posterior.

Observing the ELBO further:
$$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p_\theta(\mathbf{z})p_\theta(\mathbf{x}|\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$
Therefore after breaking up the log values we get,
$$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[p_\theta(\mathbf{x}|\mathbf{z})\right]+ \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p_\theta(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]
\tag{4}
$$

The right term in the above expression is just the negative of the KL-divergence between $p_\theta(\mathbf{z})$ and $q_\phi(\mathbf{z}|\mathbf{x})$. Therefore our loss function comes out to be:
$$\mathcal{L}(\theta,\phi,\mathbf{x}) = 
\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[p_\theta(\mathbf{x}|\mathbf{z})\right] -
D_{KL}\left(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z})\right) \tag{5}$$
The **problem** with this loss function is that we can't run back-propagation through it.
## Why?
According to the Leibniz-Integral rule:
$$\frac{d}{dt}\int_a^bf(x,t)dx=\int_a^b\frac{\partial}{\partial t}f(x,t)dx \tag{i}$$
Hence, we can calculate the $\nabla_\theta\mathcal{L}$ by taking the gradient operator inside the expectation but can't do the same for $\nabla_\phi\mathcal{L}$ as
- Support of the integral ($x$ for $(\text{i})$) is $q_\phi(\mathbf{z}|\mathbf{x})$ in our case,
- is not a function of $\theta$.
- But is a function of $\phi$.

# Re-Parametrisation Trick
## Law of the Unconscious Statistician
Given a random variable $X$, its expectation can be calculated as:
$$\mathbb{E}_{p(x)}[X]=\int_xx\cdot p(x)dx$$
So given a transformation $f(.)$,
$$Y=f(X)$$
we want to calculate the expectation of $Y$, which is
$$\mathbb{E}_{g(y)}[Y]=\mathbb{E}_{p(x)}[f(X)]=\int_xf(x)\cdot p(x)dx$$
## Re-Parametrisation
$\mathbf{z}$ is set to be a deterministic function $\mathbf{z}=g_\phi(\mathbf{x},\epsilon)$, where
- $\mathbf{x}$ is a data-point,
- $\epsilon$ is random variable $\epsilon \sim p(\epsilon)$,
- $\phi$ is the set of parameters of the model.

Therefore we can rewrite $(4)$ as 
$$\mathbb{E}_{p(\epsilon)}\left[p_\theta(\mathbf{x}|g_\phi(\mathbf{x},\epsilon)\right]+
\mathbb{E}_{p(\epsilon)}\left[\log\frac{p_\theta(g_\phi(\mathbf{x},\epsilon))}{q_\phi(g_\phi(\mathbf{x},\epsilon)|\mathbf{x})} \right]
\tag{6}$$
# Variational Auto-encoder
Framework for an autoencoder in the paper:
- Prior over latent variables, $p_\theta(\mathbf{z})$ be multivariate isotropic gaussian - $\mathcal{N}(\mathbf{z};\mathbf{0},\mathbf{I})$.
- Let $p_\theta(\mathbf{x}|\mathbf{z})$ be a multivariate gaussian (in-case of real value data) or Bernoulli (in case of binary data).
- Let the intractable but true posterior be approximate gaussian with an approximate diagonal covariance:
$$\log q_\phi(\mathbf{z}|\mathbf{x})=\log \mathcal{N}(\mathbf{z};\boldsymbol{\mu};\boldsymbol{\sigma}^2\mathbf{I})$$
where $\boldsymbol{\sigma}$ and $\boldsymbol{\mu}$ will be calculated by the encoder network.

Therefore, we sample the posterior $\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})$ using
$$\mathbf{z}=g_\phi(\mathbf{x},\epsilon)=\boldsymbol{\sigma}\odot\epsilon + \boldsymbol{\mu}$$ where $\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$.
## Loss
### Computing the KL-Divergence
Now from eq $(5)$, we compute the KL-divergence between the posterior and the prior. So equation for multivariate gaussian:
$$\mathcal{N}(\mathbf{x};\boldsymbol{\mu},\boldsymbol{\Sigma})=
\frac{1}{2^{n/2} |\boldsymbol{\Sigma}|}\exp \left(
-\frac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^{\intercal}  \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})
\right)$$
where $\mathbf{x},\boldsymbol{\mu}\in\mathbb{R}^n$ and $\boldsymbol{\Sigma}\in\mathbb{R}^{n\times n}$.

So,
- $p_1 = q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1) = \mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)$
- $p_2 = p_\theta(\mathbf{z}) =\mathcal{N}(\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_2) =\mathcal{N}(\mathbf{0},\mathbf{I})$ 

Now the KL divergence:
$$\begin{aligned}
D_{KL}\left(q_\phi(\mathbf{z}|\mathbf{x}) || p_\theta(\mathbf{z})\right) & = \frac{1}{2}
\left[
\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}-n+\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + 
(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^{\intercal}  \boldsymbol{\Sigma}^{-1}_2(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
\right]\\
& = \frac{1}{2}
\left[
\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}-n+
\text{tr}(\mathbf{I}^{-1}\boldsymbol{\sigma}^2) + 
(\mathbf{0}-\boldsymbol{\mu})^{\intercal}  \boldsymbol{\Sigma}^{-1}_2(\mathbf{0}-\boldsymbol{\mu})
\right]\\
& = \frac{1}{2}
\left[
\log\frac{1}{|\boldsymbol{\sigma}^2|}-n+
\text{tr}(\mathbf{I}^{-1}\boldsymbol{\sigma}^2) +  
(\mathbf{0}-\boldsymbol{\mu}^{\intercal}\mathbf{I}^{-1})(\mathbf{0}-\boldsymbol{\mu})
\right]\\
& = \frac{1}{2}
\left[
-\log|\boldsymbol{\sigma}^2|-n+
\text{tr}(\boldsymbol{\sigma}^2) + \boldsymbol{\mu}^{\intercal}\boldsymbol{\mu}
\right]\\
& = \frac{1}{2}
\left[
-\sum_{j}(\log\sigma^2_j+1)+
\sum_j\sigma_j^2 + \sum_j\mu_j^2
\right]\\
\end{aligned}
$$

### Final Loss
Therefore our final loss can be written as:
$$\mathcal{L}=\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[p_\theta(\mathbf{x}|\mathbf{z})\right]-\frac{1}{2}
\left[
\sum_{j}(\log\sigma^2_j+1)-
\sum_j\sigma_j^2 - \sum_j\mu_j^2
\right]$$
where, \(\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[p_\theta(\mathbf{x}|\mathbf{z})\right]\) is the reconstruction loss. There many options we can consider as reconstruction loss depending on scenario - MSE, MAE, BCE losses. 

# Implementation
You can find an implementation of VAEs (with $\mathcal{N}(1,2)$ prior) trained on various loss functions in this [repository](https://github.com/Sasopsy/SAiDL-2024-Assignment/tree/master/VAEs). The derivation of the KL-divergence with $\mathcal{N}(1,2)$ as prior and various findings using different VAE configurations can be found in this [report](https://github.com/Sasopsy/SAiDL-2024-Assignment/blob/master/VAEs/report/report.pdf).


# References
- [Umar Jamil's VAE Video](https://www.youtube.com/watch?v=iwEzwTTalbg&t=953s)
- [Kapil Sachdeva's Reparametrisation Trick Video](https://www.youtube.com/watch?v=nKM9875PVtU)
- [Auto-encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
