---
layout: post
title:  "Backpropagation Pt II"
date:   2018-04-18 19:12:22 +0200
comments: true
categories: deep-learning backpropagation
---
This is the continuation of my [backpropagation post]({% post_url 2018-04-13-backpropagation-i %}). This time, I will recap how I implemented backpropagation for batch normalization.

## Batch Normalization

Batch normalization is a technique used to reduce the dependency between weight initialization and training results. With a BN layer, it is possible to ensure roughly unit gaussian distributed outputs before a nonlinear activation function such as ReLU is is applied. Or at least, that's the case if you apply BN before ReLU, but I have also seen people use BN after ReLU in which case the activations are unit gaussian. It would be interesting to know in which situations BN placement makes a difference and why. For now, let's return to the topic of backpropagation.

Batch normalization takes the input, which consists of samples $$x_i \in \mathbb{R}^D, i\in\{1,\dots,N\}$$ and normalizes every feature column by

$$x^{\text{norm}}_{ik} = \gamma_k \cdot \frac{x_{ik}-\overline{x_k}}{\sqrt{\sigma^2_k + \epsilon}} + \beta_k \quad , \quad  k\in\{1,\dots,D\}$$

where $$\overline{x_k}$$ and $$\sigma^2_k$$ refer to the mean and variance of the $$k$$-th column. The parameters $$\gamma_k$$ and $$\beta_k$$ are learned by backpropagation.

Our goal is to calculate the derivative

$$\frac{dx^{\text{norm}}_{ik}}{dx_{lk}} \quad , \quad l\in\{1,\dots,N\} .$$

The first thing to note is that both the column mean and variance are functions of $$x_{lk}$$. Therefore we need to derive $$d\overline{x_k}/dx_{lk}$$ and $$\sigma^2_k/dx_{lk}$$ first. For the column mean, this is straightforward:

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
\frac{x_{ik}^{norm}}{dx_{lk}} &= \gamma_k \cdot \frac{(\delta(i=l)-1/N)}{\sqrt{\sigma^2_k + \epsilon} } - \frac{(x\_{ik}-\overline{x_k})\cdot (x\_{lk}-\overline{x_k})}{N\sqrt{\sigma^2_k + \epsilon}^3}
\end{align}

The above expression describes the inner gradient, which still needs to be chained to the upstream gradient $$dout/dx^{\text{norm}}_jk, j\in\{1,\dots,N\}, k\in\{1,\dots,D\}$$:

\begin{align}
\frac{dout}{dx_{lk}} = \sum_{j=1} \big( \frac{dout}{dx^{\text{norm}}_{jk}} \¢dot \frac{dx_{jk}^{norm}}{dx_{lk}} \big) \quad .
\end{align}

The inner gradient with respect to $$\gamma_k$$ is given by 

\begin{align}
\frac{dx_{jk}^{norm}}{d\gamma_k} = \sum_{j=1}^N \frac{x\_{jk}-\overline{x_k}}{\sqrt{\sigma_k^2+\epsilon}}
\end{align}

and chaining it to the upstream gradient yields 

\begin{align}
\frac{dout}{d\gamma_k} = \sum_{h=1}^N \big( \frac{dout}{dx^{\text{norm}}_{hk}} \cdot \frac{dx_{hk}^{norm}}{d\gamma_{k}} \big) \quad .
\end{align}

The gradient with respect to $$\beta_k$$ is given by

\begin{align}
\frac{dout}{d\beta_k} = \sum_{j=1}^N \frac{dout}{dx^{\text{norm}}_{jk}} \quad .
\end{align}

The implementation of a BN layer in python is given by
{% highlight python %}
import numpy as np

def bn_forward(X,gamma,beta,eps):
  ''' 
    X: NxD matrix
    gamma: Dx1 scale vector
    beta: Dx1 shift vector
  '''
  N = X.shape[0]
  mu = X-X.mean(axis=1)
  sigma2 = 1/N * (X-mu).dot(X-mu)
  norm = (X-mu)/np.sqrt(sigma2+eps)
  cache = (X, gamma, beta, eps)
  return norm.dot(gamma)+beta, cache

def bn_backward(dout,cache):
  X, gamma, beta, eps = cache
  #todo
{% endhighlight %}