---
layout: post
title:  "Backpropagation Pt II"
date:   2018-04-18 19:12:22 +0200
comments: true
categories: [deep-learning, backpropagation]
excerpt_separator: <!--more-->
---
This is the continuation of my [backpropagation post]({% post_url 2018-04-13-backpropagation-i %}). This time, I will recap how I implemented backpropagation for batch normalization.

<!--more-->

## Batch Normalization

Batch normalization is a technique used to reduce the dependency between weight initialization and training results. With a BN layer, it is possible to ensure roughly unit gaussian distributed outputs before a nonlinear activation function such as ReLU is is applied. Or at least, that's the case if you apply BN before ReLU, but I have also seen people use BN after ReLU in which case the activations are unit gaussian. It would be interesting to know in which situations BN placement makes a difference and why. For now, let's return to the topic of backpropagation.

Batch normalization takes the input, which consists of samples $$x_i \in \mathbb{R}^D, i\in\{1,\dots,N\}$$ and normalizes every feature column by

$$x^{\text{bn}}_{ik} = \gamma_k \cdot \frac{x_{ik}-\overline{x_k}}{\sqrt{\sigma^2_k + \epsilon}} + \beta_k \quad , \quad  k\in\{1,\dots,D\}$$

where $$\overline{x_k}$$ and $$\sigma^2_k$$ refer to the mean and variance of the $$k$$-th column. The parameters $$\gamma_k$$ and $$\beta_k$$ are learned by backpropagation.

Our first goal is to calculate the derivative

$$\frac{dx^{\text{bn}}_{ik}}{dx_{lk}} \quad , \quad l\in\{1,\dots,N\} .$$

Both the column mean and variance are functions of $$x_{lk}$$, so we need to derive $$d\overline{x_k}/dx_{lk}$$ and $$d\sigma^2_k/dx_{lk}$$ first. For the column mean, this is straightforward:

$$\frac{d\overline{x_k}}{dx_{lk}} = \frac{1}{N} \quad .$$

For the variance, a bit more effort is required
\begin{align}
\frac{d\sigma_{k}^2}{dx_{lk}} &= \frac{d}{dx_{lk}} \frac{\sum_{i=1}^N (x_{ik}-\overline{x_k})^2}{N} \newline
&= \frac{2 \sum_{i=1}^N (x_{ik}-\overline{x_k}) \cdot (\delta(i=l)-\frac{1}{N})}{N} \newline
&= \frac{2 \big((x_{lk}-\overline{x_k})-\frac{1}{N} \sum_{i=1}^N (x_{ik}-\overline{x_k}) \big)}{N} \newline
&= \frac{2(x_{lk}-\overline{x_k})}{N} 
\end{align}

We can now use the above expressions to calculate the complete derivation with the quotient rule

\begin{align}
\frac{dx\_{ik}^{\text{bn}}}{dx\_{lk}} &= \gamma_k \cdot \Bigg( \frac{\delta(i=l)-1/N}{\sqrt{\sigma^2_k + \epsilon} } - \frac{(x\_{ik}-\overline{x_k})\cdot (x\_{lk}-\overline{x_k})}{N\sqrt{\sigma^2_k + \epsilon}^3}\Bigg)
\end{align}

The above expression describes the inner gradient, which still needs to be chained to the upstream gradient $$dout/dx^{\text{bn}}_{jk}$$ for $$j\in\{1,\dots,N\}, k\in\{1,\dots,D\}$$:

\begin{align}
\frac{dout}{dx\_{lk}} &= \sum_{j=1}^N \Big( \frac{dout}{dx^{\text{bn}}\_{jk}} \cdot \frac{dx\_{jk}^{\text{bn}}}{dx\_{lk}} \Big) \newline 
&= \sum_{j=1}^N \Bigg( \frac{dout}{dx^{\text{bn}}\_{jk}} \cdot \gamma_k \cdot\Bigg( \frac{\delta(j=l)-1/N}{\sqrt{\sigma^2_k + \epsilon} } - \frac{(x\_{jk}-\overline{x_k})\cdot (x\_{lk}-\overline{x_k})}{N\sqrt{\sigma^2_k + \epsilon}^3} \Bigg) \Bigg)\newline 
&= \frac{\gamma\_k}{\sqrt{\sigma^2_k + \epsilon}} \cdot \Bigg( \sum\_{j=1}^N  \frac{dout}{dx^{\text{bn}}\_{jk}} \cdot \big( \delta(j=l)-1/N \big) - \sum\_{j=1}^N \frac{dout}{dx^{\text{bn}}\_{jk}} \cdot \frac{(x\_{jk}-\overline{x_k})\cdot (x\_{lk}-\overline{x_k})}{N (\sigma^2_k + \epsilon)} \Bigg) \newline 
&= \frac{\gamma\_k}{\sqrt{\sigma^2_k + \epsilon}} \cdot \Bigg( \frac{dout}{dx^{\text{bn}}\_{lk}} - \frac{1}{N} \sum\_{j=1}^N \frac{dout}{dx^{\text{bn}}\_{jk}} - \frac{1}{N} \sum\_{j=1}^N \frac{dout}{dx^{\text{bn}}\_{jk}} \cdot \frac{(x\_{jk}-\overline{x_k})\cdot (x\_{lk}-\overline{x_k})}{\sigma^2_k + \epsilon} \Bigg)
\end{align}

The gradient with respect to $$\gamma_k$$ is given by 

\begin{align}
\frac{dout}{d\gamma_k} &= \sum_{j=1}^N \Big( \frac{dout}{dx^{\text{bn}}\_{jk}} \cdot \frac{dx\_{jk}^{\text{bn}}}{d\gamma_{k}} \Big) \newline
&= \sum_{j=1}^N \Big( \frac{dout}{dx^{\text{bn}}\_{jk}} \cdot \frac{x\_{jk}-\overline{x_k}}{\sqrt{\sigma_k^2+\epsilon}} \Big) \quad .
\end{align}

The gradient with respect to $$\beta_k$$ is given by

\begin{align}
\frac{dout}{d\beta_k} = \sum_{j=1}^N \frac{dout}{dx^{\text{bn}}_{jk}} \quad .
\end{align}

Plugging everything in will make the implementation of a BN layer look something like this in Python:
{% highlight python %}
import numpy as np

def bn_forward(X,gamma,beta,eps):
  ''' 
    Input
    - X: NxD matrix
    - gamma: D-dim. scale vector
    - beta: D-dim. shift vector
    Returns
    - out: input X after batch normalization
    - cache: tuple (X,gamma,beta,eps)
  '''
  mu = X.mean(axis=0)
  sigma2 = X.var(axis=0)
  norm = (X-mu)/np.sqrt(sigma2+eps)
  cache = (X, gamma, beta, eps)
  out = gamma*norm+beta
  return out, cache

def bn_backward(dout,cache):
  ''' 
    Input
    - dout: NxD matrix of upstream gradients
    - cache: tuple (X,gamma,beta,eps)
    Returns
    - dX: gradient w.r.t. X
    - dbeta: gradient w.r.t. beta
    - dgamma: gradient w.r.t. gamma
  '''
  X, gamma, beta, eps = cache
  mu = X.mean(axis=0)
  norm = (X-mu)/np.sqrt(sigma2+eps)
  sigma2 = X.var(axis=0)
  dgamma = np.sum(dout*norm,axis=0)
  dbeta = np.sum(dout,axis=0)
  dX = gamma/np.sqrt(var)*(dout - np.mean(dout,axis=0) - np.mean(dout*norm,axis=0)*norm)
  return dX, dbeta, dgamma
  
{% endhighlight %}

### Further Reading
[1] [Deriving Batch-Norm Backprop equations][01] by Chris Yeh is an alternative derivation of the vectorized equations

[01]: https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
