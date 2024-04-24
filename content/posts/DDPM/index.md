+++
title = 'The Math of DDPMs'
date = 2024-04-24T18:47:49+05:30
draft = false
math = true
ShowToc= true
+++
# Introduction
This blog post dives deep into the mathematics behind Denoising Diffusion Probabilistic Models, breaking down the objective function that make DDPMs work. I tried to "hand-wave" as little math as possible in this post and tried my best to cover all the steps in this derivation. 
# DDPMs
Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain to slowly add random noise to data and learn to reverse the diffusion to get the original data sample back from the noise. The idea is to generate images from noise by iteratively removing "noise" from it. 
# Forward Diffusion Process
Given a sample datapoint $\mathbf{x}_0\sim q(\mathbf{x})$, forward diffusion process is addition of small amounts of gaussian noise to sample in $T$ time-steps, producing a sequence of noisy samples $\mathbf{x}_1,...\mathbf{x}_T$ controlled a by a variance schedule \(\{\beta_t\in(0,1)\}_{t=1}^{T}\).   
$$\boxed{q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t|\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})} \tag{1}$$
$$q(\mathbf{x}_{1:T}|\mathbf{x}_0)=\prod_{t=1}^{T}q(\mathbf{x}_t|\mathbf{x}_{t-1}) \tag{2}$$
The data sample loses its features and becomes more noisy as it propagates through the forward process. 
## Result 1
*As $T\rightarrow \infty$, $\mathbf{x}_T$ is equivalent to an isotropic Gaussian*. 

*Proof.*
Let us assume a constant schedule:
$$\beta_t=\beta$$
We know, 
$$q(\mathbf{x}_T|\mathbf{x}_{T-1})=\mathcal{N}(\mathbf{x}_T|\sqrt{1-\beta}\mathbf{x}_{T-1},\beta\mathbf{I})$$
According to the re-parametrisation trick,
$$\mathbf{x}_T=\sqrt{1-\beta}\mathbf{x}_{T-1}+\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I})$$
Therefore expanding on this,
$$
\begin{aligned}
&\mathbf{x}_T=\sqrt{1-\beta}\mathbf{x}_{T-1}+\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I})\\
&\mathbf{x}_T= \sqrt{1-\beta}(\sqrt{1-\beta}\mathbf{x}_{T-2}+\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I})) + \sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I})\\
&\mathbf{x}_T=\sqrt{1-\beta}^2\mathbf{x}_{T-2}+\sqrt{1-\beta}\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I}) +\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I})\\
&\mathbf{x}_T=\sqrt{1-\beta}^2(\sqrt{1-\beta}\mathbf{x}_{T-3}+\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I}))+\sqrt{1-\beta}\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I}) +\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I})\\
&\mathbf{x}_T=\sqrt{1-\beta}^T\mathbf{x}_{0}+\sqrt{1-\beta}^{T-1}\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I}) +...+\sqrt{1-\beta}\sqrt{\beta}+\sqrt{\beta}\mathcal{N}(\mathbf{0},\mathbf{I})\\
&\mathbf{x}_T=\sqrt{1-\beta}^T\mathbf{x}_{0}+\mathcal{N}(\mathbf{0},\mathbf{I})(\sqrt{1-\beta}^{T-1}\sqrt{\beta} +...+\sqrt{1-\beta}\sqrt{\beta}+\sqrt{\beta})\\
\end{aligned}$$
The second term is GP sum with a common ratio $\sqrt{1-\beta}$. Therefore,
$$\mathbf{x}_T=\sqrt{1-\beta}^T\mathbf{x}_{0}+\sqrt\beta\frac{1-\sqrt{1-\beta}^{T-1}}{1-\sqrt{1-\beta}}\mathcal{N}(\mathbf{0},\mathbf{I})$$
Since, $\beta<<1$ and $T\rightarrow\infty$,
$$\mathbf{x}_T\approx\mathcal{N}(\mathbf{0},\mathbf{I})$$
## One-Shot Forward Diffusion
Instead of performing the forward diffusion process $T$ times, we can formulate an equation that can compute it at a single go.

Let us assume,
$$\boxed{\alpha_t=1-\beta_t}\tag{3}$$
Therefore, we can write our forward process of $t$-th step as
$$\mathbf{x}_t=\sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$$
where, $\epsilon_t\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. Noise from all time-steps are samples from a normal distribution.

Expanding on this,
$$
\begin{aligned}
&\mathbf{x}_t=\sqrt{\alpha_t}\mathbf{x}_{t-1}+
\sqrt{1-\alpha_t}\epsilon_t \\
&\mathbf{x}_t=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+
\sqrt{1-\alpha_{t-1}}\epsilon_{t-1})+
\sqrt{1-\alpha_t}\epsilon_t \\
&\mathbf{x}_t=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+
\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\epsilon_{t-1}+
\sqrt{1-\alpha_t}\epsilon_t \\
\end{aligned}
$$
Now, we know that when
$$
\begin{aligned}
&X\sim N(\mu_{X},\sigma_{X}^{2})\\
&Y\sim N(\mu_{Y},\sigma_{Y}^{2})\\
&Z=X+Y\\
\end{aligned}
$$
Then,
$$Z\sim N(\mu_{X}+\mu_{Y},\sigma _{X}^{2}+\sigma_{Y}^{2})$$
So,
$$\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\epsilon_{t-1}+
\sqrt{1-\alpha_t}\epsilon_t=\sqrt{\alpha_t(1-\alpha_{t-1})+1-\alpha_t}\epsilon$$
And hence, coming back to our formulation
$$\begin{aligned}
&\mathbf{x}_t=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+
\sqrt{\alpha_t-\alpha_t\alpha_{t-1}+1-\alpha_t}\epsilon \\
&\mathbf{x}_t=\sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2}+
\sqrt{1-\alpha_t\alpha_{t-1}}\epsilon \\
\end{aligned}
$$
If we keep iteratively doing this, we arrive at a formula,
$$
\mathbf{x}_t=\sqrt{\alpha_t\alpha_{t-1}...\alpha_2\alpha_1}\mathbf{x}_{0}+
\sqrt{1-\alpha_t\alpha_{t-1}...\alpha_2\alpha_1}\epsilon \\
$$
Let,
$$\boxed{\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i}\tag{4}$$
Finally, 
$$\boxed{
\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_{0}+
\sqrt{1-\bar{\alpha}_t}\epsilon} \tag{5}\\
$$
# Reverse Diffusion Process
Through the reverse diffusion process, we want to sample from \(q(\mathbf{x}_{t-1}|\mathbf{x}_{t})\) to able to recreate the original image from a Gaussian noise $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$.

Unfortunately, we cannot compute \(q(\mathbf{x}_{t-1}|\mathbf{x}_{t})\) as it requires the entire data distribution to compute.

So we learn a model $p_\theta$ to approximate these conditional probabilities.
$$p_\theta(\mathbf{x}_{0:T})=p_\theta(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \tag{6}$$
We model our approximate reverse diffusion process parametrising the mean and variance of the noise to be estimated. We assume it is a **normal distribution**.
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})=\mathcal{N}(\mathbf{x}_{t-1}|\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\boldsymbol{\Sigma}_\theta(\mathbf{x}_t,t))$$
This kind of setup is very similar to a VAE where we want to maximise 
$$\log p_\theta(\mathbf{x}_0)$$
Hence,
$$\begin{aligned}
\log p_\theta(\mathbf{x}_0) &= 
\log p_\theta(\mathbf{x}_0)\cdot 
\int q(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T}\\
&= \int \log p_\theta(\mathbf{x}_0) q(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T}\\
&=\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}
\left[ \log p_\theta(\mathbf{x}_0)  \right]
\end{aligned}$$
Now the term \(\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}[ \log p_\theta(\mathbf{x}_0)]\) can be further be broken down using Bayes rule. So from $(2)$ and $(6)$
$$\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}
\left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}  \right] \tag{7}
$$
Eq $(7)$ is for a single point in our dataset. We want to find the expectation of the likelihood over the entire distribution,
$$\mathbb{E}_{q(\mathbf{x}_0)}\left[ \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}
\left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}  \right] \right]
$$
which can be written as
$$\int\int\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}q(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T}\cdot q(\mathbf{x}_0)d\mathbf{x}_0$$
Since \(q(\mathbf{x}_{0:T})=q(\mathbf{x}_0)q(\mathbf{x}_{1:T}|\mathbf{x}_0)\), our final log-likelihood can be written as 
$$
\mathbb{E}_{q(\mathbf{x}_{0:T})}
\left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}  \right] \tag{8}
$$
Maximising eq $(8)$ is equivalent to minimising:
$$
\mathbb{E}_{q(\mathbf{x}_{0:T})}
\left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}  \right] = \mathcal{L}_{\text{VLB}}\tag{9}
$$
## Simplifying the Likelihood
So taking the \(\mathcal{L}_{\text{VLB}}\),
$$\begin{aligned}
\mathcal{L}_{\text{VLB}} &= \mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] 
\end{aligned}$$
So expanding using $(6)$ an $(2)$ and taking \(q=q(\mathbf{x}_{0:T}) \)
$$\begin{aligned}
&\mathbb{E}_{q} \left[ \log \frac{\prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)} \right]\\
=& \mathbb{E}_{q} \left[ - \log p_\theta(\mathbf{x}_T) + \sum_{t=1}^{T} \log \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)} \right] \\
\end{aligned}$$
Now separating the summation (starting it from $t=2$ and stripping out $t=1$)
$$\begin{aligned}
\mathbb{E}_{q} \left[ - \log p_\theta(\mathbf{x}_T) + \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_{t} | \mathbf{x}_{t-1}, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)} + \log \frac{q(\mathbf{x}_1 | \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 | \mathbf{x}_1)} \right] \\
\end{aligned}$$
Note that \(q(\mathbf{x}_{t} | \mathbf{x}_{t-1})=q(\mathbf{x}_{t} | \mathbf{x}_{t-1},\mathbf{x}_0)\) as $q$ is a Markov chain. Conditioning it on \(\mathbf{x}_0\) won't change anything. Now focusing on the summation, using Bayes rule, we get
$$
\begin{aligned}
&\mathbb{E}_{q} \left[ - \log p_\theta(\mathbf{x}_T) + \sum_{t=2}^{T} \log \left( \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)} \cdot \frac{q(\mathbf{x}_t | \mathbf{x}_0)}{q(\mathbf{x}_{t-1} | \mathbf{x}_0)} \right) + \log \frac{q(\mathbf{x}_1 | \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 | \mathbf{x}_1)} \right]\\
=&\mathbb{E}_{q} \left[ - \log p_\theta(\mathbf{x}_T) + \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)} + \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_t | \mathbf{x}_0)}{q(\mathbf{x}_{t-1} | \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 | \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 | \mathbf{x}_1)} \right] \\
\end{aligned}$$
Now observing the term 
$$\sum_{t=2}^{T} \log \frac{q(\mathbf{x}_t | \mathbf{x}_0)}{q(\mathbf{x}_{t-1} | \mathbf{x}_0)}=\frac{q(\mathbf{x}_2| \mathbf{x}_0)...q(\mathbf{x}_{T-1} | \mathbf{x}_0)q(\mathbf{x}_T | \mathbf{x}_0)}{q(\mathbf{x}_1 | \mathbf{x}_0)q(\mathbf{x}_2 | \mathbf{x}_0)...q(\mathbf{x}_{T-1} | \mathbf{x}_0)}$$
There's clear recursive cancellation that takes place leaving us with our final term
$$\log\frac{q(\mathbf{x}_T | \mathbf{x}_0)}{q(\mathbf{x}_1 | \mathbf{x}_0)}$$
Therefore plugging it back into our $\mathcal{L}_{\text{VLB}}$ 
$$
\begin{align*}
&\mathbb{E}_q \left[ - \log p_\theta(\mathbf{x}_T) + \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} + \log \frac{q(\mathbf{x}_T|\mathbf{x}_0)}{q(\mathbf{x}_1|\mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1|\mathbf{x}_0)}{p_\theta(\mathbf{x}_0|\mathbf{x}_1)} \right] \\
=&\mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_T|\mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \\
=&\mathbb{E}_q \left[ D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) \| p_\theta(\mathbf{x}_T)) + \sum_{t=2}^{T} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
\end{align*}
$$
Let us divide the final term into smaller chunks
- \(L_T=D_{KL}(q(\mathbf{x}_T | \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))\) 
- \(L_{t}=D_{KL}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t))\) where \(2\leq t \leq T-1\)
- \(L_0=\log p_\theta(\mathbf{x}_0 | \mathbf{x}_1)\)
## PDF of of $L_t$ 
We have the general term for a reverse diffusion process:
$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$$
This is a tractable expression as we have the reverse process being conditioned on $\mathbf{x}_0$ - a single datapoint and not the entire data distribution.

Using Bayes theorem, we can write this as:
$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)=\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}$$
For our forward process, \(q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})\) is a Markov chain, hence, \(q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})=q(\mathbf{x}_{t}|\mathbf{x}_{t-1})\). So our given pdf can easily be written from xexpressions we already have. \(q(\mathbf{x}_{t}|\mathbf{x}_{0})\) and \((\mathbf{x}_{t-1}|\mathbf{x}_{0})\)  is equation $(5)$.
$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)=\frac{\mathcal{N}(\mathbf{x}_{t}|\sqrt{\alpha_t}\mathbf{x}_{t-1},(1-\alpha_t)\mathbf{I}) \mathcal{N}(\mathbf{x}_{t-1}|\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0},(1-\bar{\alpha}_{t-1})\mathbf{I})}{\mathcal{N}(\mathbf{x}_{t}|\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0},(1-\bar{\alpha}_{t})\mathbf{I})}$$
**Our goal is to find a gaussian pdf** with mean a function of \(\mathbf{x}_t\) and \( \mathbf{x}_0 \) (during training we'll have access to both values). So expanding the above and take only the exponential terms, we get
$$\exp\left(-\frac{1}{2}\left[\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{1-\alpha_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1-\bar{\alpha}_t}\right]\right)
$$
Our first strategy will be to take all the $\mathbf{x}_{t-1}$ terms and group them together. So, we expand the above terms:
1. \(\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{1-\alpha_t} = \frac{\mathbf{x}_t^2 - 2\mathbf{x}_t\sqrt{\alpha_t}\mathbf{x}_{t-1} + \alpha_t \mathbf{x}_{t-1}^2}{1-\alpha_t}\)
2. \(\frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} = \frac{\mathbf{x}_{t-1}^2 - 2\mathbf{x}_{t-1}\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \bar{\alpha}_{t-1} \mathbf{x}_0^2}{1-\bar{\alpha}_{t-1}}\)
3. \(\frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1-\bar{\alpha}_t} = \frac{\mathbf{x}_t^2 - 2\mathbf{x}_t\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \bar{\alpha}_t \mathbf{x}_0^2}{1-\bar{\alpha}_t}\)

Now we separate the the \(\mathbf{x}_{t-1}\) terms and the rest. The \( \mathbf{x}_{t-1}\) terms in the exponential are
$$\begin{aligned}
&\mathbf{x}_{t-1}^2 \left( \frac{1-\bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} \right) - 2\mathbf{x}_{t-1} \left( \frac{\sqrt{\alpha_t}\mathbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1 - \bar{\alpha}_{t-1}} \right)\\
=&\mathbf{x}_{t-1}^2 \left( \frac{1-\bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} \right) - 2\mathbf{x}_{t-1} \left( \frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{(1 - \bar{\alpha}_{t-1})(1 - \alpha_t)}\right)\\
=& f(\mathbf{x}_{t-1})
\end{aligned}
$$
The rest are
$$\begin{aligned}
&\frac{\mathbf{x}_t^2}{1 - \alpha_t} + \frac{\bar{\alpha}_{t-1} \mathbf{x}_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{\mathbf{x}_t^2}{1 - \bar{\alpha}_t} - \frac{\bar{\alpha}_t \mathbf{x}_0^2}{1 - \bar{\alpha}_t} + \frac{2\sqrt{\bar{\alpha}_t}\mathbf{x}_0\mathbf{x}_t}{1 - \bar{\alpha}_t}\\
=& \frac{1-\bar{\alpha_t}-1+\alpha_t}{(1-\bar{\alpha_t})(1-\alpha_t)}\mathbf{x}_t^2 + \frac{\bar{\alpha}_{t-1}-\bar{\alpha}_{t-1}\bar{\alpha}_{t}-\bar{\alpha}_{t}+\bar{\alpha}_{t-1}\bar{\alpha}_{t}}{(1-\bar{\alpha}_{t-1})(1-\bar{\alpha}_{t})}\mathbf{x}_0^2 + \frac{2\sqrt{\bar{\alpha}_t}\mathbf{x}_0\mathbf{x}_t}{1 - \bar{\alpha}_t}\\
=& \frac{\alpha_t-\bar{\alpha_t}}{(1-\bar{\alpha_t})(1-\alpha_t)}\mathbf{x}_t^2 + \frac{\bar{\alpha}_{t-1}-\bar{\alpha}_{t}}{(1-\bar{\alpha}_{t-1})(1-\bar{\alpha}_{t})}\mathbf{x}_0^2+\frac{2\sqrt{\bar{\alpha}_t}\mathbf{x}_0\mathbf{x}_t}{1 - \bar{\alpha}_t}\\
=&\frac{\alpha_t(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha_t})(1-\alpha_t)}\mathbf{x}_t^2 + \frac{\bar{\alpha}_{t-1}(1-\alpha_t)}{(1-\bar{\alpha}_{t-1})(1-\bar{\alpha}_{t})}\mathbf{x}_0^2+\frac{2\sqrt{\bar{\alpha}_t}\mathbf{x}_0\mathbf{x}_t}{1 - \bar{\alpha}_t}\\
\end{aligned}$$
The above is a squared sum. Hence, it can be written as
$$\begin{aligned}
&\frac{1}{1-\bar{\alpha}_t}\left(\sqrt{\frac{\alpha_t(1-\bar{\alpha}_{t-1})}{1-\alpha_t}}\mathbf{x}_t +  \sqrt{\frac{\bar{\alpha}_{t-1}(1-\alpha_t)}{1-\bar{\alpha}_{t-1}}}\mathbf{x}_0 \right)^2\\
=& \frac{1}{(1-\bar{\alpha}_t)(1 - \bar{\alpha}_{t-1})(1 - \alpha_t)}\left((1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 \right)^2\\
=&f(\mathbf{x}_{t},\mathbf{x}_0)
\end{aligned}$$
We want our \(\mathbf{x}_{t-1}^2\) to be separate as its the random variable we want to sample. So we common out \(\frac{1-\bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}\) from both \(f(\mathbf{x}_{t-1})\) and \(f(\mathbf{x}_{t},\mathbf{x}_0)\). Therefore doing so from \(f(\mathbf{x}_{t-1})\) will yield
$$\begin{aligned}
&\mathbf{x}_{t-1}^2 - 
2\mathbf{x}_{t-1} \left( \frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_t}\right)\\
=&C(\mathbf{x}_{t-1})
\end{aligned}$$
and from \(f(\mathbf{x}_{t},\mathbf{x}_0)\) will yield
$$\begin{aligned}
&\frac{1}{(1-\bar{\alpha}_t)^2}\left((1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 \right)^2\\
=&\left(\frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{(1-\bar{\alpha}_t)} \right)^2 \\
=& C(\mathbf{x}_t,\mathbf{x}_0)
\end{aligned}$$
Note: the above two terms **aren't** \(f(\mathbf{x}_{t-1})\) and \(f(\mathbf{x}_{t},\mathbf{x}_0)\). Zooming out for a moment to our original term in the exponential,
$$\begin{aligned}
&\exp\left(-\frac{1}{2}\left[\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{1-\alpha_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1-\bar{\alpha}_t}\right]\right)\\
=&\exp\left(-\frac{1}{2}\left[f(\mathbf{x}_{t-1})+f(\mathbf{x}_{t},\mathbf{x}_0)\right]\right) \\
=& \exp-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}}\left[C(\mathbf{x}_{t-1})+C(\mathbf{x}_{t},\mathbf{x}_0)\right]\right)
\end{aligned}$$
Clearly, \(C(\mathbf{x}_{t-1})+C(\mathbf{x}_{t},\mathbf{x}_0)\) is a squared sum. Hence, we can write the above exponential as
$$\exp-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}}\left[\mathbf{x}_{t-1}-\frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{(1-\bar{\alpha}_t)}\right]^2\right)$$
Now if we evaluate the normalisation terms of our three pdfs:
1. $\frac{1}{\sqrt{(1-\alpha_t)2\pi}}$ 
2. $\frac{1}{\sqrt{(1-\bar{\alpha}_{t-1})2\pi}}$ 
3. \(\frac{1}{\sqrt{(1-\bar{\alpha}_{t})2\pi}}\)

and compute the final normalisation term we get, 
$$\frac{1}{\sqrt{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}2\pi}}$$
Our final formula for $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ is 
$$\frac{1}{\sqrt{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}2\pi}}\exp-\frac{1}{2}\left(\frac{1}{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}}\left[\mathbf{x}_{t-1}-\frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{(1-\bar{\alpha}_t)}\right]^2\right)$$
which is clearly the pdf of a gaussian. Hence, 
$$\boxed{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)=\mathcal{N}\left(\mathbf{x}_{t-1}| \frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{(1-\bar{\alpha}_t)},\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}\right)}$$
## Re-parameterisation of Mean
Our mean $\mu_\theta(\mathbf{x}_t,t)$ is given by
$$\mu(\mathbf{x}_t,t) = \frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t+(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{(1-\bar{\alpha}_t)}$$
We know
$$\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_{0}+
\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t $$
Hence, we can write $\mathbf{x}_0$ as 
$$\mathbf{x}_0=\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t\right)$$
Plugging the above into our mean
$$\begin{aligned}
\mu(\mathbf{x}_t,t) =& \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}\mathbf{x}_t + (1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}\left( \frac{\mathbf{x}_t}{\sqrt{\bar{\alpha}_t}} - \frac{\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}} \right)}{1 - \bar{\alpha}_t} \\
=&\frac{1}{\sqrt{\alpha}_t} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_t \right)
\end{aligned}
$$
# Final Training Loss
Recall that we need to learn a neural network to approximate the conditioned probability distributions in the reverse diffusion process,  
$$p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t))$$
We would like to train  $\mu_{\theta}$ to predict \(\mu(\mathbf{x}_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right)\). Because \(\mathbf{x}_t\) is available as input at training time, we can re-parameterise the Gaussian noise term instead to make it predict \(\boldsymbol{\epsilon}_t\) from the input \(\mathbf{x}_t\) at time step $t$
$$\mu_{\theta}(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_{\theta}(\mathbf{x}_t, t) \right)$$
Therefore $L_t$ is the KL-Divergence between \(p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)\) and \(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)\). KL-Divergence between two normal distributions - \(\mathcal{N}(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1)\) and \(\mathcal{N}(\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_2)\) can be written as 
$$\frac{1}{2}
\left[
\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}-n+\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + 
(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^{\intercal}  \boldsymbol{\Sigma}^{-1}_2(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
\right]$$
Therefore,
$$\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}=0$$
and
$$\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1)=\mathbf{I}$$
Since 
$$\Sigma_{\theta}(\mathbf{x}_t, t)=\Sigma(\mathbf{x}_t, t)=\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$
for which $\mathbf{I}$ and $n$ can be ignore in our final loss term as they aren't parameterised by $\theta$. We have
$$\frac{1}{2}
\left[ 
(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^{\intercal}  \boldsymbol{\Sigma}^{-1}_2(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
\right]$$
Let's rewrite the above with our terms we derived in the earlier sections
$$\begin{aligned}
&\frac{1}{2}
\left[ 
(\boldsymbol{\mu}(\mathbf{x}_t,t)-\mu_\theta(\mathbf{x}_t,t))^{\intercal}  \boldsymbol{\Sigma}^{-1}_\theta(\boldsymbol{\mu}(\mathbf{x}_t,t)-\boldsymbol{\mu}_\theta(\mathbf{x}_t,t))
\right]\\
=&\left[ \frac{1}{2\|\boldsymbol{\Sigma}_\theta\|_2^2} \left\| \mu_{\theta}(\mathbf{x}_t, \mathbf{x}_0) - \mu_{\theta}(\mathbf{x}_t, t) \right\|_2^2 \right]\\
\end{aligned}$$
Plugging in $\mu_{\theta}(\mathbf{x}_t, t)$ and $\mu(\mathbf{x}_t,t)$, we get
$$\frac{(1 - \alpha_t)^2}{2\alpha_t (1 - \bar{\alpha}_t)\|\boldsymbol{\Sigma}_\theta\|_2^2} \left\| \boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_{\theta} \left( \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t \right) \right\|_2^2 
$$
## Simplification
The authors of the original DDPM suggested to use only the norm term and ignore the weighing term as empirically it performed better. So we get, 
$$
\boxed{L_t^{\text{simple}} = \mathbb{E}_{t\sim\mathcal{U}[1,T], \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_{\theta}\left( \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t \right) \right\|_2^2 \right]}
$$
which is our final objective function.
# Algorithm
## Algorithm 1: Training
**repeat**

1. $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
2.  $t \sim \text{Uniform}(\{1, \ldots, T\})$
3. $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$
4. Take gradient descent step on $\nabla_{\theta} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t)\|^2$ 

**until converged**
## Algorithm 2: Sampling
1. $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. for $t = T, \ldots, 1$ do
	1. $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ if $t > 1$, else $\mathbf{z} = \mathbf{0}$
	2. \(\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}\)
3. end for
4. return $\mathbf{x}_0$
# Implementation 
For the model, a U-Net was used where $t$ was fed in as a positional embedding in its intermediate layers. A implementation of DDPM can be found [here](https://github.com/Sasopsy/Diffusion-Models).
# References
(1) Jonathan Ho et al. [“Denoising diffusion probabilistic models.”](https://arxiv.org/abs/2006.11239) arxiv Preprint arxiv:2006.11239 (2020).

(2) Lilian Weng. (Jul 2021). ["What are diffusion models? Lil’Log"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

(3) Tushar Kumar. (Nov 2023). ["Denoising Diffusion Probabilistic Models | DDPM Explained"](https://www.youtube.com/watch?v=H45lF4sUgiE).
