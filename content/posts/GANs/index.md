+++
title = 'Generative Adversarial Networks'
date = 2024-04-12T02:49:45+05:30
draft = false
math = true
ShowToc= true
+++
# Introduction
The main idea: To train generators using really good classifiers.
## Intuitive Explanation
Imagine a counterfeiter tasked with creating fake paintings and a detective tasked with discerning fake from real paintings. Both of them are thrown in a game where they have to out compete each other. Through this, both of them will get have to get better in the methods they use to win the game.
## Mathematical Explanation
There are two models: A generative model $G$ that tries to capture the data distribution and a discriminative model $D$ that estimates the probability of a particular sample came from the training data. Both $G$ and $D$ are put in this minimax adversarial game with a value function $V(G,D)$.
# Framework
Both $G$ and $D$ are neural networks. 

To learn $p_g$ which is the distribution of the generator's over data $\mathbf{x}$, we define a **prior distribution** $p_{\mathbf{z}}(\mathbf{z})$ where the input noise variable is sampled from. This is then mapped to the data space as $G(\mathbf{z};\theta_{g})$ where $G$ is a neural network with parameters $\theta_{g}$.

$D(\mathbf{x},\theta_d)$ on the other hand just outputs a single scaler ranging between $0$ and $1$ which represents the probability that the given piece of data is from our data distribution $p_d$ rather than $p_g$.

Here's the framework for training both of these neural nets:
$$\min_{G}\max_{D} V(G,D)=
\mathbb{E}_{\mathbf{x}\sim p_{data}}[\log D(\mathbf{x})]+
\mathbb{E}_{\mathbf{z}\sim p_{z}(\mathbf{z})}[\log(1-D(G(\mathbf{z})))]$$
One important observation can be made from this is that the generator doesn't directly 'look' at the data. It tries to learn the distribution of it from just the 'feedback' of the discriminator.
## Why does this work?
What does $D$ predict?
$$D(\mathbf{x};\theta_d)=P(\mathbf{x}\textnormal{ is real})$$
So it's goal is to maximise,
$$\log D(\mathbf{x})$$
when the given sample is from the training set ($\mathbf{x}$ is a real data point). Conversely, the generators role is to minimise this as it's trying to fool the discriminator.

Similarly, the discriminator's goal is to maximise the second term when the data is from the generator.
$$\log(1-D(G(\mathbf{z})))$$
The term $1-D(G(\mathbf{z}))$ represents how fake the data is. So when the data is from the generator, the discriminator would want to maximise the log-likelihood of this term. Conversely, the generator would want to minimise this.

For training the generator we use the term,
$$-\log(D(G(\mathbf{z})))$$
is used instead of $\log(1-D(G(\mathbf{z})))$. The reason is covered in a later section.

# Training
For training the adversarial net, discriminator is not started out fully trained. If so was the case, it would be hard for the generator to catch up. Instead, they are trained simultaneously with one step for optimising the discriminator and then one step for the generator.
## Algorithm
***
**for** number of iterations **do**
  - Sample minibatch of $m$ noise samples $\set{\mathbf{z}^{(1)},\mathbf{z}^{(2)},...,\mathbf{z}^{(m)}}$ from noise prior $p_g(\mathbf{z})$.
  - Sample minibatch of $m$ data samples $\set{\mathbf{x}^{(1)},\mathbf{x}^{(2)},...,\mathbf{x}^{(m)}}$ from training set.
  - Update the discriminator by descending its stochastic gradient: $$\nabla_{\theta_d}\frac{1}{m}\sum_{i=1}^{m}\left[-\log D(\mathbf{x^{(i)}}) - \log(1-D(G(\mathbf{z^{(i)}})))\right]$$
  - Sample minibatch of $m$ noise samples $\set{\mathbf{z}^{(1)},\mathbf{z}^{(2)},...,\mathbf{z}^{(m)}}$ from noise prior $p_g(\mathbf{z})$.
  - Update the generator by descending its stochastic gradient: $$\nabla_{\theta_g}\frac{1}{m}\sum_{i=1}^{m}\left[\log(1-D(G(\mathbf{z^{(i)}})))\right]$$
end **for**
***
## Better Loss for Generator
In practice, the given algorithm may not provide sufficient gradient for the generator in the initial stages of training. Early in training, $D$ has an easier time in rejecting samples from the generator. Hence, $\log(1-D(G(\mathbf{z})))$ saturates because $D(G(\mathbf{z})) \approx 0$ causing the generator to receive weak gradient signals.

Going deeper into why,
$$\frac{d\log(1-D(G(\mathbf{z})))}{dD(G(\mathbf{z}))}=-\frac{1}{1-D(G(\mathbf{z}))}\approx-1 \space\space.$$
In the final layer of $D$, we apply [sigmoid](Logistic%20Regression.md#Math#Formulation) to get a probability score
$$D(G(\mathbf{z}))=\sigma(\hat{\mathbf{y}}) \space\space,$$
where $\hat{\mathbf{y}}$ is the pre-activation prediction of our discriminator and $\sigma(x)$ is the sigmoid activation function. Taking the [derivative](Logistic%20Regression.md#Math#Gradient%20Descent#Sigmoid)
$$\frac{dD(G(\mathbf{z}))}{d\hat{\mathbf{y}}}=\sigma(\hat{\mathbf{y}})(1-\sigma(\hat{\mathbf{y}}))=D(G(\mathbf{z}))\left[1-D(G(\mathbf{z}))\right] \space\space .$$
Since, $D(G(\mathbf{z})) \approx 0$,
$$\frac{dD(G(\mathbf{z}))}{d\hat{\mathbf{y}}}\approx0 \space\space.$$
This is a very weak gradient which can stunt training for our generator.

In order to avoid this, the authors proposed to focus on maximising 
$$\log(D(G(\mathbf{z}))$$
Here as well $D(G(\mathbf{z^{(i)}})) \approx 0$, and similarly like our previous case,
$$\frac{dD(G(\mathbf{z}))}{d\hat{\mathbf{y}}}=D(G(\mathbf{z}))\left[1-D(G(\mathbf{z}))\right]\approx0$$
But,
$$-\frac{d\log(D(G(\mathbf{z})))}{dD(G(\mathbf{z}))}=-\frac{1}{D(G(\mathbf{z}))}\approx-\infty$$
This result tending to $-\infty$ can help offset the closeness to $0$ we get with $dD(G(\mathbf{z}))/d\hat{\mathbf{y}}$ when we apply the chain rule, thus providing overall stronger gradients for our generator to learn with.

**A more intuitive explanation:**
As discussed before $1-D(G(\mathbf{z}))$ represents how fake the data is. So instead of the generator minimising the probability of an image being fake to the discriminator, it focuses on maximising the probability of an image being real.

**Note:**
Also, note that in the original research paper, the authors suggested to train the discriminator for $k$ times and the generator $1$ time in an alternating fashion. They proposed $k$ to be a hyper-parameter and went with $k=1$.
# Theoretical Results
## Result 1
*For a fixed $G$, the optimal discriminator $D$ is:*
$$D^{*}_G(\mathbf{x})=\frac{p_d(\mathbf{x})}{p_d(\mathbf{x})+p_g(\mathbf{x})}$$
where, $p_d$ is the distribution of the data and $p_g$ is the distribution of the generator.

*Proof.* 
We have our loss function:
$$L(G, D) = \int \left(p_{\mathbf{d}}(\mathbf{x}) \log(D(\mathbf{x})) + p_{\mathbf{g}}(\mathbf{x}) \log(1 - D(\mathbf{x})) \right) \, d\mathbf{x}$$
where $\mathbf{x}$ can be both, generated and real samples. This integral represents an expectation over the input space. If we can minimise the quantity inside the integral for each $\mathbf{x}$, then the entire integral is minimised because the integral sums up these individual minima over all $\mathbf{x}$.

Therefore differentiating the quantity inside with $D(\mathbf{x})$ and equating it to $0$:

$\Rightarrow \frac{d}{dD(\mathbf{x})} \left( p_{\mathbf{d}}(\mathbf{x}) \log(D(\mathbf{x})) + p_{\mathbf{g}}(\mathbf{x}) \log(1 - D(\mathbf{x})) \right) = 0$ 

$\Rightarrow \frac{p_{\mathbf{d}}(\mathbf{x})}{D(\mathbf{x})} - \frac{p_{\mathbf{g}}(\mathbf{x})}{1 - D(\mathbf{x})} = 0$

$\Rightarrow p_{\mathbf{d}}(\mathbf{x})(1 - D(\mathbf{x})) = p_{\mathbf{g}}(\mathbf{x})D(\mathbf{x})$

$\Rightarrow p_{\mathbf{d}}(\mathbf{x}) = D(\mathbf{x})(p_{\mathbf{d}}(\mathbf{x}) + p_{\mathbf{g}}(\mathbf{x}))$

$\Rightarrow D^*(\mathbf{x}) = \frac{p_{\mathbf{d}}(\mathbf{x})}{p_{\mathbf{d}}(\mathbf{x}) + p_{\mathbf{g}}(\mathbf{x})}$
## Reformulation
Therefore, the main minmax game can be reformulated as:
$$C(G)=\max_D V(G,D)=
\mathbb{E}_{\mathbf{x}\sim p_{d}}[\log D^*_G(\mathbf{x})]+
\mathbb{E}_{\mathbf{x}\sim p_{g}}[\log(1-D^*_G(\mathbf{x})]$$
Hence, through this we can say that the training objective of the discriminator is maximising the log likelihood of the conditional probability $P(Y=1|\mathbf{x})$ when $\mathbf{x}$ comes from $p_d$ and $P(Y=0|\mathbf{x})$ when $\mathbf{x}$ comes from $p_g$.

Now, putting $D^{*}_G(\mathbf{x})=\frac{p_d(\mathbf{x})}{p_d(\mathbf{x})+p_g(\mathbf{x})}$ in the above equation, we get
$$C(G)=
\mathbb{E}_{\mathbf{x}\sim p_{d}}\left[\log \frac{p_d(\mathbf{x})}{p_d(\mathbf{x})+p_g(\mathbf{x})}\right]
+
\mathbb{E}_{\mathbf{x}\sim p_{g}}\left[\log\frac{p_g(\mathbf{x})}{p_d(\mathbf{x})+p_g(\mathbf{x})}\right]$$
## Result 2
*The global minimum for training criterion of the generator $C(G)$ is achieved if and only if $p_g=p_d$. At that point, $C(G)$ takes a value $-\log4$.*

*Proof.* 
Given
$$\begin{aligned} 
C(G)&=\mathbb{E}_{\mathbf{x}\sim p_{d}}\left[\log \frac{p_d(\mathbf{x})}{p_d(\mathbf{x})+p_g(\mathbf{x})}\right]
+
\mathbb{E}_{\mathbf{x}\sim p_{g}}\left[\log\frac{p_g(\mathbf{x})}{p_d(\mathbf{x})+p_g(\mathbf{x})}\right]\\
&=\mathbb{E}_{\mathbf{x}\sim p_{d}}\left[\log\frac{1}{2}\cdot\frac{p_d(\mathbf{x})}{\frac{p_d(\mathbf{x})+p_g(\mathbf{x})}{2}}\right]
+
\mathbb{E}_{\mathbf{x}\sim p_{g}}\left[\log\frac{1}{2}\cdot\frac{p_g(\mathbf{x})}{\frac{p_d(\mathbf{x})+p_g(\mathbf{x})}{2}}\right]\\
&=\mathbb{E}_{\mathbf{x}\sim p_{d}}\left[\log\frac{p_d(\mathbf{x})}{\frac{p_d(\mathbf{x})+p_g(\mathbf{x})}{2}}\right]
+
\mathbb{E}_{\mathbf{x}\sim p_{g}}\left[\log\cdot\frac{p_g(\mathbf{x})}{\frac{p_d(\mathbf{x})+p_g(\mathbf{x})}{2}}\right]
+2\log\frac{1}{2}\\
&=-\log4 +
\mathbb{E}_{\mathbf{x}\sim p_{d}}\left[\log\frac{p_d(\mathbf{x})}{\frac{p_d(\mathbf{x})+p_g(\mathbf{x})}{2}}\right]
+
\mathbb{E}_{\mathbf{x}\sim p_{g}}\left[\log\cdot\frac{p_g(\mathbf{x})}{\frac{p_d(\mathbf{x})+p_g(\mathbf{x})}{2}}\right]
\\
\end{aligned}
$$
Therefore,
$$C(G)=-\log4 + KL\left(p_d\bigg|\bigg|\frac{p_d+p_g}{2}\right) + KL\left(p_g\bigg|\bigg|\frac{p_d+p_g}{2}\right)$$
where, $KL$ is the Kullback-Leibler divergence. The expression after $-\log4$ is twice of the Jensen-Shannon divergence ($JSD$) between $p_g$ and $p_d$:
$$C(G)=-\log4+2\cdot JSD(p_d||p_g)$$
Since $JSD$ between the two distributions is always non-negative, or zero only when they are equal. Therefore,
$$C^*=-\log4$$
and the only solution to this is $p_g=p_d$.

When $p_g=p_d$, from result 1 we get
$$D^{*}_G(\mathbf{x})=\frac{1}{2}$$
Therefore, from the reformulation in result 1 we get,
$$C(G)=\log\frac{1}{2}+\log\frac{1}{2}=-\log4$$

# References
(1) Ian Goodfellow et al. [“Generative adversarial nets.”](https://arxiv.org/pdf/1406.2661.pdf) NIPS, 2014.

(2) Alec Radford et al. ["UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS."](https://arxiv.org/pdf/1511.06434.pdf) ICLR 2016

(3) Lilian Weng. (2017) ["From GAN to WGAN"](https://lilianweng.github.io/posts/2017-08-20-gan/#:~:text=The%20loss%20function%20of%20the,a%20much%20smoother%20value%20space.)
