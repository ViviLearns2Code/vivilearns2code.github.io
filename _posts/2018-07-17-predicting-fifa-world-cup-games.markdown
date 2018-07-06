---
layout: post
title:  "Predicting FIFA World Cup Games"
date:   2018-07-17 10:04:43 +0200
comments: true
categories: project
---
As a football enthusiast and a beginner in machine learning, the FIFA World Cup was the perfect "first project". I decided to build a model which would predict the exact number of goals scored by each team in a game. A lot of literature which I have found regarding football predictions is focused on predicting the correct tendency (win/draw/loss) and the models are trained on and applied to football league systems such as the English Premier League. The world cup only takes place every four years, which reduces the amount of available data. It is divided into a group stage and a knockout stage and teams which have never encountered each other before often produce surprising results. 

## Collecting Data
For data collection, I first checked if there was a database or website which recorded enough data about the past international tournaments. I ended up with transfermarkt.de which has detailed world cup data starting from 2006. I wrote a scraper with [scrapy][1] to retrieve the following information 
* player market values
* player ages
* team fragmentation (how many different clubs are present in a national team)
* results from past encounters between two teams
* actual results
* gametype (group stage, semifinal,...)

Since there have only been three world cups since 2006, the amount of data is limited (especially the knockout stage games are few). I therefore decided to scrape the European, Asian, African, South American and North American championships as well. This did not require much additional effort because transfermarkt pages all follow the same pattern and no additional scraper was neessary. At the same time, these continental tournaments are the type of football events most similar to the world cup.

What was unexpected was that transfermarkt sometimes returned different data on different days. You could open the same team lineup page on different days and you would get different data. Some inconcistencies were manually corrected after scraping. Missing player market values were imputed with the lowest market value within a player's team.

The final, aggregated dataset can be seen here [TODO: link to github file]

A heatmap plot of the spearman rank correlations shows that goal count is monotonically related to market values and historic results. The p-values of both features are [TODO]. 

TODO: Heatmap plot

## Modelling
I chose to do a negative binomial regression on the obtained data set. The PMF of a random variable $$X \sim Nb(r,p)$$ is given by 

\begin{equation}
P(X = k) = \binom{r+k-1}{k} p^r (1-p)^k = \frac{\Gamma(r+k)}{\Gamma(r)k!} p^r (1-p)^k \label{eq:pmf}
\end{equation}

In many papers, the Poisson distribution is used to model goal counts. However, the negative binomial distribution can be viewed as a generalization of the Poisson distribution. A random variable $$X \sim Nb(r,p)$$ can be seen as a random variable $$X \vert \Lambda=\lambda \sim Po(\lambda)$$ where $$\Lambda$$ is a gamma-distributed random variable with shape parameter $$r$$ and scale parameter $$(1-p)/p$$.

The PMF can therefore be equivalently written as

\begin{align}
P(X = k) &= \int_0^\infty P(X = k \vert \Lambda=\lambda) \cdot f_{\Gamma(r,(1-p)/p)}(\lambda) \ d\lambda \nonumber \newline
&= \int_0^\infty \frac{\lambda^k \exp{(-\lambda)}}{k!} \cdot \frac{\lambda^{r-1} \exp{\big(-\frac{\lambda p}{1-p}\big)}}{\Gamma(r)\big(\frac{1-p}{p}\big)^r} \ d\lambda \nonumber \newline
&= \frac{p^r(1-p)^{-r}}{\Gamma(r)k!} \int_0^\infty \lambda^{r+k-1} \exp{(-\frac{\lambda}{1-p})}  \ d\lambda \nonumber \newline
& = \frac{p^r(1-p)^{-r}}{\Gamma(r)k!} (1-p)^{r+k-1} \int_0^\infty \frac{\lambda^{r+k-1}}{(1-p)^{r+k-1}} \exp{(-\frac{\lambda}{1-p})}  \ d\lambda \nonumber \newline
& = \frac{p^r(1-p)^{-r}}{\Gamma(r)k!} (1-p)^{r+k-1} \int_0^\infty t^{r+k-1} \exp{(-t)} (1-p) \ dt \nonumber \newline
& = \frac{p^r(1-p)^{-r}}{\Gamma(r)k!} (1-p)^{r+k-1} (1-p) \Gamma(r+k) \nonumber \newline
& = \frac{\Gamma(r+k)}{\Gamma(r)k!} p^r(1-p)^k 
\end{align}
where we [integrated by substitution][4] with $$\phi(t)=t(1-p)$$. 

The mean and variance are given by 
\begin{align}
E[X] &= E_{\Lambda}[E_X[X\vert\Lambda]] = E_{\Lambda}[\Lambda] = \frac{r(1-p)}{p} \label{eq:mu} \newline
V[X] &= E_{\Lambda}[V_X(X\vert\Lambda)] + V_{\Lambda}[E_X[X\vert\Lambda]] = E_{\Lambda}[\Lambda] + V_\Lambda[\Lambda] \nonumber\newline
&= \frac{r(1-p)}{p} + \frac{r(1-p)^2}{p^2} = \mu + r^{-1}\mu^2
\end{align}

Here we see that the variance can be larger than the mean for negative binomially distributed random variables. Large $$r$$ indicate similarity to a [Poisson distribution][8].

An equivalent representation of the PMF that depends on the mean $$\mu:=E[X]$$ instead of the probability $$p$$ can be derived if one solves equation \eqref{eq:mu} for $$p$$ and plugs it into \eqref{eq:pmf}.

$$P(X = k) = \frac{\Gamma(r+k)}{\Gamma(r)k!} \Big(\frac{r}{\mu+r}\Big)^r \Big(\frac{\mu}{\mu+r}\Big)^k$$

## Implementation
The python library [statsmodels][2] offers built-in support for negative binomial regression (as part of their GLM module). In statsmodels, the parameter $$\alpha := r^{-1}$$ is a hyperparameter that the developer needs to set. This means that it only regresses the mean conditioned on the input data and does not fit the variance. Furthermore, the target variable is only one dimensional, so we would have to regress the mean for both teams separately. 

Instead of using statsmodels, I implemented a neural network which would regress the mean $$\mu$$ and $$\alpha$$ of both teams. In other words, the output layer has 4 neurons. I defined a negative binomial likelihood loss, similar to [this implementation in Tensorflow][3]. But instead of using one linear layer (which corresponds to a generalized linear model), I use more linear layers with ReLu functions. This way, the expectation is no longer a linear function of the features but can be learned as a more complex function with nonlinearities. 

TODO: Insert code snippet loss function

The learned paramaters $$\mu$$ and $$\alpha$$ determine a team's goal distribution. The assumption is that both teams' goal counts are conditionally independent, so the distribution of a game outcome (the joint distribution of both teams' goal counts) can be modelled as the product of each team's goal distribution. In this implementation, I limit the number of goals to a maximum of 10. The outcome distribution is captured in a 11x11 matrix, where rows represent team A and columns team B. The trace of the matrix represents the probability of a draw. The sum of all elements in the strictly upper triangular matrix represents the probability that team A loses. The sum of all elements in the strictly lower triangular matrix represents the probability that team A wins.

TODO: Insert code snippet from mu, alpha to NB pmf, prob matrix

## Results
I evaluated the trained models on a validation set by calculating how accurately the model predicts a game's tendency (win/draw/loss). The model for the first match day of the group phase was trained on data from previous tournaments. The models for the following match days were trained on data from previous tournaments and from the preceding games of this world cup. A total of 7 models were trained: 3 for the group stage, 1 for the round of sixteen, 1 for the quarterfinals, 1 for the semfinals and 1 for the finals. The table below summarizes the results.

|               | Top-5 Accuracy | Top-3 Accuracy | Top-1 Accuracy | Tendency      | 
|:-------------:| :------------: |:--------------:|:--------------:|:-------------:|
| Group Match 1 | TODO           | TODO           | TODO           |               |
| Group Match 2 | TODO           | TODO           | TODO           |               |
| Group Match 3 | TODO           | TODO           | TODO           |               |
| Round of 16   | TODO           | TODO           | TODO           |               |
| Quarterfinal  | TODO           | TODO           | TODO           |               |
| Semifinal     | TODO           | TODO           | TODO           |               |
| 3rd Place     | TODO           | TODO           | TODO           |               |
| Final         | TODO           | TODO           | TODO           |               |

For top-n accuracy, I count a prediction as correct if the actual result is included in the n most probable outcomes. For tendency accuracy, a prediction is counted as correct if the outcome distribution assigns the most probability to the tendency of the actual outcome.

Out of curiosity, I also trained standard sklearn classifiers (Gradient Tree Boosting, Random Forest, Logistic Regression) and a classifier neural network to see how accurately tendencies can be predicted. The result was an accuracy of roughly 50% for the group stage and 70% for the knockout stage on the validation set. I did not use any world cup 2018 data for the experiment.

TODO

## For Future Reference
What I did not notice was that the market values of players increased immensely from one tournament to the next. According to statista, the most valuable team in Brazil 2014 was worth [622 million Euros][5]. The most valuable team in Russia 2018 was worth [1080 million euros][6]. I noticed this when playing around with statsmodels. I fit the model on data from previous world cups and made it predict results for world cup 2018 data. It ended up predicting very high means because the inverse of the used link function, the exponential function, is sensitive to changes of the linear predictor. For example, let $$w = [0.1;0.2;-0.01]^T$$ the learned coefficients and $$x_1 = [1;2;5]^T, x_2 = [1;10;5]^T$$ two different feature vectors. The predicted mean values the differ considerably:
\begin{align}
\mu_1 &= \exp{(w^T x_1)} = \exp{( [1 2 5])} = \exp{(0.45)} \approx 1.57 \nonumber\newline 
&\ll 7.77 \approx \\exp{(2.05)} = \exp{(w^T x_2)} = \mu_2
\end{align} 
While inference using statsmodels produced useless results, the neural network approach was not as sensitive to the market value changes. For future projects, it would be a good idea to remove trends of such time series to prevent instabilities.

Another point for improvement is that the collected data lacks information on a team's current form. The used dataset includes historical encounters between two teams that face each other in a game, but these encounters are often very sparse. Games that took place years ago are not the best indicator for a team's current form. For example, Germany looked very good on paper with high market values and good historical records against its group stage rivals. The team still exited this world cup at the group stages. This outcome would not have been as surprising if one looked at Germany's most recent test games. 

While the above points refer to the quality of the dataset, there is also something that can be improved on model side. Currently, I admittedly don't know how much I can trust the black box neural network, especially when it outputs results which seem counterintuitive to a seasoned football fan. I calculated the gradients of the predicted average $$\mu$$ and $$\alpha$$ w.r.t. the input features, but it did not make me trust the model. The paper [Understanding Black-box Predictions via Influence Functions][7] introduces the concept of influence functions which identify the training data points that influenced the prediction of a test datapoint the most. I am curious to find out if it can be applied to my football predictions!

[1]: https://scrapy.org/
[2]: https://www.statsmodels.org/stable/glm.html#module-statsmodels.genmod.generalized_linear_model
[3]: https://github.com/gokceneraslan/neuralnet_countmodels/blob/master/Count%20models%20with%20neuralnets.ipynb
[4]: https://en.wikipedia.org/wiki/Integration_by_substitution
[5]: https://www.statista.com/statistics/268728/value-of-the-national-teams-at-the-2010-world-cup/
[6]: https://www.statista.com/statistics/865737/value-of-the-national-teams-at-the-2018-world-cup/
[7]: https://arxiv.org/abs/1703.04730
[8]: https://en.wikipedia.org/wiki/Negative_binomial_distribution#Poisson_distribution