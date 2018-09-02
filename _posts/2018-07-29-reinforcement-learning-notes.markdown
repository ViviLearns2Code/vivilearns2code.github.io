---
layout: post
title:  "Reinforcement Learning Notes"
date:   2018-07-29 18:31:43 +0100
comments: true
categories: reinforcement-learning
---
[Part 14 of Stanford's CS231n lecture][3] was about Reinforcement Learning. The content of the slides was very dense, so I looked up a couple of concepts to understand the concepts better.

## The Setting
In a reinforcement learning setting, an agent interacts with its environment and receives feedback for every interaction. The feedback comes in the form of numeric rewards which depend on the actions of the agent in every time step. The goal is to maximize the total reward by choosing the best course of actions. This dynamic optimization problem can be formalized as a Markov Decision Process (MDP). Possible applications are for example algorithms that learn to play games or robots that learn to move.

The components of a MDP are

* A set of environment states $$\mathcal{S}$$. The state at a timestep $$t$$ contains all the information needed to make the next decision, i.e. no knowledge about past states is required. A terminal/absorbing state is a state with zero rewards that cannot be left. 
* A set of actions $$\mathcal{A}(s)$$ available at state $$s$$
* Rewards $$R_{t+1} \in \mathbb{R}$$ as feedback to choosing an action $$A_{t}$$ in state $$S_{t}$$ in time step $$t$$
* State transition probabilities $$p(S_t=s',R_t=r \vert S_{t-1}=s,A_{t-1}=a)$$ to characterize the dynamics of a MDP
* A policy $$\pi$$, where $$\pi(a \vert s)$$ specifies the probability with which action $$a$$ is chosen in state $$s$$.

The agent chooses actions at each time step and transitions from one state to the next until they arrive at a terminal state $s_T$. The sequence of states, actions and rewards $$s_0,a_0,r_1\ldots,s_T$$ defines an episode.  

### Definitions
The total reward in time step $$t$$ is the discounted sum of single rewards:

\begin{align}
G_t := \sum_{k=0}^T \gamma^k R_{t+k+1}
\end{align}

The action-value function is defined as the expected value of $$G_t$$ given a state $$s$$ and action $$a$$ in time step $$t$$ and following policy $$\pi$$ starting in $$t+1$$.

\begin{align}
q_{\pi}(s,a) := \mathbb{E}\_{\pi}[G_t \vert S_t=s, A_t=a]
\end{align}

The state-value function $$v_{\pi}$$ is defined as the expected value of $$G_t$$ given a state $$s$$ in time step $$t$$ and following policy $$\pi$$

\begin{align}
v_{\pi}(s) := \mathbb{E}\_{\pi} [G_t \vert S_t=s] = \sum_{a \in \mathcal{A}(s)} \pi(a \vert s) q_{\pi}(s,a)
\end{align}

Note that the expectation depends on all states the agent ends up in and all actions the agent chooses under policy $$\pi$$. 

Both state-value and action-value functions offer a way to compare to policies. For example, if for two policies $$\pi$$, $$\pi'$$ the relationship $$v_{\pi}(s) \le v_{\pi'}(s)$$ holds for all states $$s$$, the policy $$\pi'$$ is at least as good as $$\pi$$.

The optimization problem is to find an optimal policy $$\pi$$ that maximizes the state-value function

\begin{align}
\pi^{*} = \underset{\pi}{\mathrm{argmax}} \ v_{\pi}(s) \quad \forall s \in \mathcal{S}
\end{align}

For Value Iteration and Q-Learning, a value function is learned so we can infer the optimal policy from the values. In Policy Gradient methods, a parametrized policy is learned directly. Actor-Critic methods combine Policy Gradients and value function learning.

## Finding the Optimal Policy
Every policy satisfies the Bellman equations
\begin{align}
v_\pi(s) &= \mathbb{E}\_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t=s]\notag \newline 
&= \sum_{a} \pi(a|s) \sum_{s'} p(S_{t+1}=s' \vert S_{t}=s,A_{t}=a) \ldots \notag \newline
& \ldots \sum_{r} p(R_{t}=r \vert S_{t}=s,A_{t}=a, S_{t+1}=s') [r+\gamma v_{\pi}(s')] \notag \newline
&= \sum_{a} \pi(a|s) \sum_{s',r} p(S_{t+1}=s',R_{t}=r|s,a)[r+\gamma v_{\pi}(s')] 
\end{align}
\begin{align}
q_\pi(s,a) &= \mathbb{E}\_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t=s, A_t=a] \notag \newline
&= \sum_{s'} p(S_{t+1}=s',R_{t}=r|S_{t}=s,A_{t}=a)\ldots \notag \newline
&\ldots [r+\gamma \sum_{a'} \pi(a' \vert s') q_{\pi}(s',a')]
\end{align}
Every optimal policy $$\pi^{*}$$ satisfies the Bellman Optimality Equations
\begin{align}
v_{\pi^{\*}}(s) &= \underset{a}{\mathrm{max}} \ q_{\pi^{*}}(s,a) \notag \newline
&= \underset{a}{\mathrm{max}} \ \sum_{s',r} p(S_{t+1}=s',R_{t}=r \vert S_t=s, A_t = a)[r+\gamma v_{\pi^{\*}}(s')] \newline
q_{\pi^{\*}}(s,a) &= \sum_{s',r} p(S_{t+1}=s',R_t=r \vert S_t=s,A_t=a)[r+\gamma \underset{a'}{\mathrm{max}} \ q_{\pi^{\*}}(s',a')]
\label{eq:bell1} \end{align}

The Bellman equations can be exploited to find the optimal policy. Both Value Iteration and Q-Learning make use of them. 

### Value Iteration
The algorithm described in chapter 4.4 of [Sutton & Barto][1] uses the Bellman optimality equation as update rule
\begin{align}
v_{k+1}(s) = \underset{a}{\mathrm{max}} \ \sum_{s',r} p(S_{t+1}=s',R_{t}=r \vert S_t=s, A_t = a)[r+\gamma v_k(s')] \ \forall s\in\mathcal{S}
\end{align} 
What happens in the update rule is that the state-values which evaluate the current policy are approximated at every state $$s$$. At the same time, the policy is iteratively improved by taking the maximizing action in state $$s$$ (greedy policy). The iterations continue until $$\| v_k(s) - v_{k+1}(s) \|$$ is small enough for all states $$s$$. Value Iteration is a truncated version of Policy Iteration, where the approximation of the state-value function requires more than one iteration before the policy is improved by taking the maximizing action (see chapter 4.3 of [Sutton & Barto][1]). 

In the CS231n lecture, value iteration is not carried out for the state-value function, but for the action-value function, which leads to the update rule
\begin{align}
q_{k+1}(s,a) = \sum_{s',r} p(S_{t+1}=s',R_{t}=r \vert S_t=s, A_t = a)[r+\gamma \underset{a'}{\mathrm{max}} \ q_k(s',a')]
\end{align} 
The final state/action-values implicitly encode the optimal policy, because they can be used to choose the optimal (greedy) action $$\underset{a}{\mathrm{argmax}} \ q_{\pi}(s,a)$$ in every time step. There is not much stochasticity in the encoded policy unless there are ties for the maximal value.

### Q-Learning
Q-Learning does not require full knowledge of the MDP dynamics because it uses the observed samples instead of the transition probabilities. The action-value function of an optimal policy $$q_{\pi^{*}}$$ is approximated using the update rule

\begin{align}
q_{i+1}(s_t,a_t) = q_{i}(s_t,a_t) + \alpha [r_{t+1}+\gamma \underset{a'}{\mathrm{max}} \ q_{i}(s_{t+1},a_{t+1})-q_{i}(s_t,a_t))] \label{eq:qlearn}
\end{align}

The update rule tries to move the current action-value $$q_{i}(s_t,a_t)$$ closer to the target $$r_{t+1}+\gamma \underset{a'}{\mathrm{max}}$$ to fulfill the Bellman equation \eqref{eq:bell1}, except that there is no expectation involved anymore. After an action is chosen (the next section describes how), the transition state $$s_{t+1}$$ and the reward $$r_{t+1}$$ are immediately observed and used for a greedy update.

A requirement for Q-Learning to work is that every state-action pair is visited frequently enough. This cannot be guaranteed if we always greedily choose the maximizing action as in Value Iteration. Chapter 5.4 of [Sutton & Barto][1] introduces an $$\epsilon$$-greedy policy to ensure this. Basically, the next action is chosen according to a greedy policy with probability $$1-\epsilon$$ and with probability $$\epsilon$$ it is chosen randomnly. Q-Learning is called an off-policy algorithm, because the action choice is $$\epsilon$$-greedy, while the action-value update is done according to a greedy policy.

Deep Q-Learning is the topic of the paper [Playing Atari with Deep Reinforcement Learning][2], which is also referenced by the CS231n course. In this paper, the action-value function of the optimal policy is approximated by a neural network $$q(s,a;\theta) \approx q_{\pi{*}}(s,a)$$. This moves the complexity of calculating the action-value for all action/state pairs (which can be a lot) to calculating the parameters $$\theta$$. Since the action-value function fulfills the Bellman optimality equation \eqref{eq:bell1}, the neural network approximation should eventually also fulfill the equation across all state-action pairs. Therefore, the ideal loss function in iteration $$i$$ is

\begin{align}
L(\theta_i) &= \frac{1}{2} \mathbb{E}\_{S_t,A_t} \Big[ \big( \mathbb{E}\_{S_{t+1},R_{t+1}} [ R_{t+1} +\gamma \underset{A_{t+1}}{\mathrm{max}} \ q(S_{t+1},A_{t+1};\theta_{i-1}) \vert  S_t, A_t] - q(S_t,A_t;\theta_i)\big )^2 \Big]
\end{align}

with the gradient
\begin{align}
\nabla_{\theta_i} L(\theta_i) &= -\mathbb{E}\_{S_t,A_t} \Big[ \Big( \mathbb{E}\_{S_{t+1},R_{t+1}} [ R_{t+1} +\gamma \underset{A_{t+1}}{\mathrm{max}} \ q(S_{t+1},A_{t+1};\theta_{i-1}) \vert  S_t, A_t] - q(S_t,A_t;\theta_i)\Big) \notag \newline 
&\ldots \nabla_{\theta_i} q(S_t,A_t,\theta_{i})\Big]
\end{align}

Since calculating the exact expectation is intractable, we use calculate the loss for mini-batches of $$N$$ observations. These observations are randomnly drawn from a so-called replay memory $$\mathcal{D}$$ which contains all previously experienced transitions $$(s_t,a_t,r_{t+1},s_{t+1})$$. Drawing from the replay memory has the advantage that you don't pick the transitions in the sequence in which they were experienced to avoid correlations. 

\begin{align}
L(\theta_i) = \frac{1}{2N} \sum\_{(s_t,a_t,r_{t+1},s_{t+1})\in \mathcal{D}} \big(r_{t+1} &+\gamma \underset{a'}{\mathrm{max}} \ q(s_{t+1},a';\theta_{i-1}) - q(s_t,a_t;\theta_i)\big)^2
\end{align}

The neural network parameters $$\theta$$ are updated with gradient descent with learning parameter $$\alpha$$

\begin{align}
\theta_{i+1} = \theta_i + \alpha \frac{1}{N} \sum\_{(s_t,a_t,r_{t+1},s_{t+1})\in \mathcal{D}} \big(r_{t+1} &+ \gamma \underset{a'}{\mathrm{max}} \ q(s_{t+1},a';\theta_{i-1}) \notag \newline
& - q(s_t,a_t;\theta_i)\big) \nabla_{\theta_i} q(s_t,a_t,\theta_{i})
\end{align}

This is similar to the update rule in \eqref{eq:qlearn}, except for a few differences: 
* we update $$\theta$$ instead of the action-value itself
* we use mini-batches instead of a single observation
* we move the current action-value $$q(s_t,a_t,\theta_{i})$$ towards a Bellman target that is based on the previous approximation $$r_{t+1}+\gamma \underset{a'}{\mathrm{max}} \ q(s_{t+1},a';\theta_{i-1})$$

### Policy Gradients
In Policy Gradients, we learn a parametrized, naturally non-deterministic policy $$\pi_{\theta}$$ that can also deal with continuous action spaces. For this topic, I found [Berkeley's CS294 lecture videos][5] especially helpful.

We introduce
\begin{align}
\pi_{\theta}(\tau) := p(s_0) \prod_{t=0}^T \pi_{\theta}(a_t \vert s_t)p(s_{t+1},r_{t+1} \vert s_t, a_t)
\end{align}
as the probability of a certain trajectory $$\tau$$ (sequence of states, actions, rewards) under policy $$\pi_{\theta}$$. The trajectory distribution is not only determined by the policy, but also by the (unknown) model transition probabilities. 

The (discounted) reward under a trajectory $$\tau$$ is $$r(\tau)$$. The expected reward is
\begin{align}
J(\theta) := v_{\pi_\theta} = \mathbb{E}\_{\tau}\Big[r(\tau) \Big]
\end{align}

For gradient ascent, we need to calculate 
\begin{align}
\nabla_{\theta} J(\theta) &= \nabla_{\theta} \mathbb{E}\_{\tau}\Big[ r(\tau) \Big] \notag \newline
&= \int_\tau r(\tau) \nabla_{\theta} \pi_{\theta}(\tau) d\tau \label{eq:reinforce1}
\end{align}
but the problem is that we do not know the dynamics $$p(s_{t+1},r_{t+1} \vert s_t, a_t)$$ which is part of $$\pi_{\theta}(\tau) $$. The REINFORCE trick allows us to rewrite the term in \eqref{eq:reinforce1} as 
\begin{align}
\nabla_{\theta} J(\theta) &= \mathbb{E}\_{\tau}\Big[ r(\tau) \frac{\nabla_{\theta} \pi_{\theta}(\tau)}{\pi_{\theta}(\tau)} \Big] \notag \newline
&= \mathbb{E}\_{\tau}\Big[ r(\tau) \nabla_{\theta} \log{\pi_{\theta}(\tau)} \Big] \notag \newline
&= \mathbb{E}\_{\tau}\Big[ r(\tau) \sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(a_t\vert s_t)} \Big] \label{eq:reinforce2}
\end{align}

For every iteration of our gradient ascent algorithm, we sample $$N$$ trajectories under the current policy $$\pi_{\theta}$$ and evaluate the total discounted return of that trajectory. The update rule is

\begin{align}
\theta_{i+1} = \theta_i + \alpha \frac{1}{N} \sum_{j}^N \Big( r(\tau^{(j)}) \sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(a_t^{(j)} \vert s_t^{(j)})} \Big) \label{eq:reinforce3}
\end{align}

As this excellent [blog post][4] explains, the expression on the right hand side in \eqref{eq:reinforce2} can be written as

\begin{align}
\mathbb{E}\_{\tau} \Big[ r(\tau) \sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(a_t\vert s_t)} \Big] =  \mathbb{E}\_{\tau}\Big[ \sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(a_t\vert s_t)} \Big( \sum_{k = t}^T \gamma^{k-t} r_{k+1} \Big) \Big]
\end{align}

This connection is called causality, as future actions will not influence past rewards. An update which considers causality is given by 
\begin{align}
\theta_{i+1} = \theta_i + \alpha \frac{1}{N} \sum_{j=0}^N \Big( \sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(a_t^{(j)} \vert s_t^{(j)}) \Big( \sum_{k = t}^{T-1} \gamma^{k-t} r_{k+1}^{(j)} \Big)} \Big) \label{eq:reinforce4}
\end{align}

REINFORCE trains slowly due to high variance of the gradient estimates. If we look at the update in \eqref{eq:reinforce3}, we see that the policy is updated in a way that raises a trajectory's probability scaled by $$r(\tau)$$. Good trajectories with high rewards are made more probable than bad trajectories. However, scaling with the raw value of observed trajectory rewards is not very stable. Imagine that we have one bad trajectory $$\tau_{bad}$$ with reward $$r_{bad} \lt 0$$ and a good trajectory $$\tau_{good}$$ with reward $$r_{good} \gt 0$$. Then, the update will make the policy assign less probability to $$\tau_{bad}$$ and more to $$\tau_{good}$$. However, if we shift all rewards upward by some constant $$c$$ such that both rewards become strictly positive, the policy will be updated in a way that assigns more probability to $$\tau_{bad}$$ and even more to $$\tau_{good}$$. If we shift them downward by a constant such that $$r_{good}$$ becomes zero, the policy will be moved into a direction that decreases the probability of $$\tau_{bad}$$, and that direction can be either away from, or toward $$\tau_{good}$$. Although the difference between the two trajectory rewards is always the same, the effect of the update changes every time.

Using the update in \eqref{eq:reinforce4} helps mitigate the variance problem because we have fewer summands. Furthermore, instead of relying on the raw, absolute rewards, we are more interested in how the trajectories compare to each other. This leads to the introduction of a baseline $$b$$ around which we center the returns

\begin{align}
\theta_{i+1} = \theta_i + \alpha \frac{1}{N} \sum_{j=0}^N \Big( \sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(a_t^{(j)} \vert s_t^{(j)}) \Big( \sum_{k = t}^{T-1} \gamma^{k-t} r_{k+1}^{(j)} - b(s_t^{(j)})\Big)} \Big) \label{eq:reinforce5}
\end{align}

### Actor-Critic Methods
A natural choice for the baseline function is the expected reward under $$\pi$$ at state $$s_t$$. The expected reward is, by definition, given by the state-value function $$v_{\pi}(s_t)$$. Going a step further, the observed sample reward in \eqref{eq:reinforce5} can be replaced with the expected reward given state $$s_t$$ and action $$a_t$$. And this is, by definition, the action-value function $$q_{\pi}(s_t,a_t)$$.

\begin{align}
\theta_{i+1} &= \theta_i + \alpha \frac{1}{N} \sum_{j=0}^N \Big( \sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(a_t^{(j)} \vert s_t^{(j)}) \Big( q_{\pi}(s_t^{(j)},a_t^{(j)}) - v_{\pi}(s_t^{(j)})\Big)} \Big)
\end{align}

Using the true expectations instead of sample rewards reduces variance of our gradient estimates. We can interpret the update rule as the value function $$q_{\pi}(s_t,a_t)$$ (critic) evaluating how good the action $$a_t$$ chosen by the policy (actor) is compared to the average $$v_{\pi}(s_t)$$. The value functions $$v_{\pi}$$ and $$q_{\pi}$$ can be learned with Q-Learning.

[1]: http://incompleteideas.net/book/bookdraft2017nov5.pdf
[2]: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf 
[3]: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf
[4]: https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/
[5]: http://rail.eecs.berkeley.edu/deeprlcourse-fa17/]
