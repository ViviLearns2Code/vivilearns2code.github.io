---
layout: post
title:  "Backpropagation Pt I"
date:   2018-04-13 20:24:00 +0200
comments: true
categories: deep-learning backpropagation
---
One of the first things which I looked into was backpropagation. Although it is "just" an application of the (multivariate) chain rule, it took me a while to actually write it down in code. I have seen many sources derive update formulas but what really helped me understand backpropagation was the introduction of computation graphs in Stanford's CS231n class. After doing the assignments 1 and 2 of CS231n, I am more confident that I can implement backpropagation without losing myself in formulas too much. 

As an example, I will explain how I implemented backpropagation for Softmax Loss and [Batch Normalization]({% post_url 2018-04-18-backpropagation-ii %}).

## Softmax Loss 
Let's assume that we have $$N$$ training sample pairs $$\{(x_i,y_i) | i=1,\dots ,N\}$$ for classification. The sample $$x_i$$ is a feature vector and there are a total of $$C$$ classes, meaning $$y_i \in \{1,\dots,C\}$$. The goal is to correctly predict the class given a feature vector.

We can forward the training samples through several layers of a neural network. In each layer, the layer input is multiplied with the layer's weight matrix and an activation function is applied to each element of the resulting matrix. The dimension of the layer output is $$N \times size_{layer}$$. The output of the final layer of a neural network for classification has dimension $$N\times C$$. We will call the elements of this output matrix _scores_ and use the notation $$s^{*}_{ij}$$. These scores are [normalized][1] and then plugged in to the softmax function to yield the probability of each class. Finally, a (cross-entropy) loss is calculated based on these probabilities.

$$s_{ij} = s^{*}_{ij} - \max_{k=1,\dots ,C}{\{ s_{ik} \}}$$

$$\text{softmax}(x_i,y_i) = \frac{\exp(s_{iy_i})}{\sum_{j=1}^{C} \exp(s_{ij})} $$

$$\text{loss} = -\frac{1}{N} \sum_{i=1}^N \log(\text{softmax}(x_i,y_i))$$

Note that each (normalized) score $$s_{ij}$$ is a function of a training sample $$x_i$$ and a weight vector $$w_j$$ associated with class $$j$$.

For backpropagation, we would like to calculate the gradient that flows through the loss function. The gradient can then be used as upstream gradient for backpropagation through the layers which came before the loss function. 

$$\frac{d\text{loss}}{ds^{*}_{ij}} \quad \text{for}\ i\in\{1,\dots,N\}\ \text{and}\ j\in\{1,\dots ,C\}$$ 

The following graph shows the computation graph for a normalized score $$s_{11}$$.

![Softmax1]({{"/images/BackpropSoftmax.svg"}})

The normalized scores are then used to calculate the softmax probability of the actual class for each sample before the loss calculation.

![Softmax2]({{"/images/BackpropSoftmax2.svg"}})

 The red arrows in the computation graphs indicate the gradient flow. The gradient from the loss function is propagated to the softmax nodes. Each softmax node receives the gradient
 
 $$\frac{d\text{loss}}{d\text{softmax}} = -\frac{1}{N \cdot \text{softmax}(x_i,y_i)} \quad .$$

 To continue with backpropagation, we first need to know what, given $$(x_i,y_i)$$, the derivative of a softmax function is w.r.t a normalized score $$s_{ij}$$ (note that scores $$s_{kj}$$ with $$k\neq i$$ do not contribute to the calculation of $$\text{softmax}(x_i,y_i)$$). Application of the quotient rule yields

 \begin{align}
 \frac{d\text{softmax}(x_i,y_i)}{ds_{ij}} &= \frac{\delta(y_i=j) \cdot \exp(s_{iy_i}) \cdot \sum_{c=1}^C \exp(s_{ic}) - \exp(s_{iy_i}) \cdot \exp(s_{ij})}{ \big( \sum_{c=1}^N \exp(s_{ic}) \big) ^2} \newline
 &= \text{softmax}(x_i,y_i) \cdot (\delta(y_i=j) - \text{softmax}(x_i,j) ) \quad ,
 \end{align}

 where $$\delta(A)$$ is $$1$$ if condition $$A$$ is true and $$0$$ otherwise.
 Each softmax node chains their inner gradient to the upstream gradient from the loss function and propagates the resulting gradient to the corresponding normalized scores:
 
 $$\frac{d\text{loss}}{ds_{ij}} = -\frac{\text{softmax}(x_i,y_i) \cdot (\delta(y_i=j) - \text{softmax}(x_i,j) )}{N \cdot \text{softmax}(x_i,y_i)} \quad .$$

 From there, the gradient is propagated to both the `MAX` node and the corresponding unnormalized scores. The dashed red arrow indicates the gradient flow from the `MAX` node to the largest unnormalized score in the corresponding row. Note that the largest unnormalized score of a sample $$x_i$$ receives gradient flows from all normalized scores $$s_{ic}$$ because it was used for normalizing the $$i$$-th row.

 $$\frac{d\text{loss}}{ds^{*}_{ij}} = \frac{d\text{loss}}{ds_{ij}} -\delta(j=\underset{c=1,\dots,C}{\text{argmax}}\{s^{*}_{ic}\})\cdot \sum_{c=1}^C \frac{d\text{loss}}{ds_{ic}}$$

 The implementation of the softmax function then looks as follows

{% highlight python %}
import numpy as np

def softmax_loss(X,y):
    ''' 
      X: NxD score matrix (not the sample matrix)
      y: N dimensional class vector 
    '''
    stable = X-np.max(X,axis=1,keepdims=True) #NxC
    softmax = np.exp(stable)/np.sum(np.exp(stable),axis=1,keepdims=True) #Nx1
    loss = -np.mean(np.log(softmax[np.arange(N),y])) #scalar
    dsoftmax = -1/(N*softmax[np.arange(N),y]) #Nx1
    dstable = -softmax[np.arange(N),y]*softmax 
    dstable[np.arange(N),y] += softmax[np.arange(N),y] 
    dstable *= dsoftmax #NxC
    dX = dstable
    dX[np.arange(N),np.argmax(X,axis=1)] -= np.sum(dstable,axis=1,keepdims=True) #NxC
    return loss, dX

{% endhighlight %}

[1]: http://cs231n.github.io/linear-classify/
