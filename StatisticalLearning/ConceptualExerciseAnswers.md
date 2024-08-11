---
documentclass: report
fontfamily: heuristica
---

An Introduction to Statistical Learning, Conceptual Exercises
=============================================================

Chapter 2
---------

### Exercise 1

For each part, indicate whether we would expect the performance of a flexible statistical learning method to be better
or worse than an inflexible method. Justify your answer.

#### Answer to Exercise 1

(a). The sample size $n$ is extremely large, and the number of predictors $p$ is small.

I think that in this case, barring patterns in the data, an inflexible method would be better. With so much data, there
is definitely the risk that a very flexible method would become too responsive to small differences within the data
points, essentially missing the forest for the trees. Assuming there *is* a pattern to the data, an increased number
of data points will likely mean that more of those data points break away from that pattern due to random variation and
error. This means a more flexible method may respond to that random variation, obscuring the pattern.

Counterargument: A large amount of data means it is much less likely that outliers will affect the model, since there
is likely to be vastly more points that aren't outliers. This can also be seen in the law of large numbers, where
increasing data sets trend to be more like a normal distribution. The prevailing thought here is that more flexibility
will work better with more data. (Well, prevailing in the sense that the answers I looked up agreed.)

(b). The number of predictors $p$ is extremely large, and the number of observations $n$ is small.

In this case, a flexible method becomes better, because with a low number of observations, it is much more important
to consider each observation and avoiding being proactive in assuming a relationship between any one. Thus, a flexible
method which will be more sensitive to each observation seems more likely to find a good model for the underlying
pattern.

Counterargument: Similar to part (a), less data points means that any outliers will have a much larger effect on the
model. Plus with the increased number of predictors, that provides more chances for noise in a predictor to be
considered signal. I'm not sold on the part (a) counterargument, but this one is pretty solid, I think and is the
actual right answer.

(c). The relationship between the predictors and the response is highly non-linear.

In this case, a more flexible solution is more likely to do better as it has the flexibility to account for any type
of non-linearity, especially disjoint non-linearity. At the same time, there exist non-linear inflexible solutions, and
in theory the data may fit that even better, but I am taking the term "highly" to imply that it does not match any
common known function in such a simple way.

(d). The variance of the error terms, i.e. $\sigma^2 = \mathrm{Var}(\epsilon)$, is extremely high.

In this case, a more flexible solution is more likely to do worse, as the increased flexibility means that it is more
likely to follow the additional irreducible error in the data, likely leading to over-fitting.

### Exercise 2

Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in
inference or prediction. Finally, provide $n$ and $p$.

#### Answer to Exercise 2

(a). We collect a set of data of the top 500 firms in the US. For each firm we record profit, number of employees,
industry, and the CEO salary. We are interested in understanding which factors affect CEO salary.

With the main response being the quantitative value of CEO salary, this is most likely a regression problem. Rather
than predicting CEO salary we are more concerned with understanding what factors affect it, so this is also an
inference problem. The number of companies is 500, so that is $n$, and the number of predictors is 3 (profit, number of
employees, industry) so that is $p$.

(b). We are considering launching a new product and wish to know whether it will be a *success* or a *failure*. We
collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a
success or failure, price charged for the product, marketing budget, competition price, and 10 other variables.

In this case the response is a qualitative value of success/failure, making this a classification problem. We are
directly involved in predicting the outcome of our new product making this a prediction problem. There are 20 products
in the study so $n$ is 20, and there are 13 predictors mentioned so that is $p$. Note that the measure of
success/failure for each product is *not* a predictor, as it is the response.

(c). We are interested in predicting the percent change in the USD/Euro exchange rate in relation to the weekly changes
in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the percent change in
the USD/Euro, the percent change in the US market, the percent change in the British market, and the percent change in
the German market.

The response variable is the percent change in the USD/Euro exchange rate, and is quantitative, so this is a regression
problem. We are directly trying to predict this variable, so it is a prediction problem. There are (usually) 52 weeks in
a year, and this applies for 2012 so $n$ is 52. Our three predictors are the percent change in the US, British, and
German markets making $p$ equal to 3.

### Exercise 3

Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error
curves on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The
x-axis should represent the amount of flexibility in the method, and the y-axis should represent the values for each
curve. There should be five curves. Also, explain why each of the curves has that shape.

#### Answer to Exercise 3

In lieu of making a drawing, I'll just explain each curve.

The bias will start off high at low flexibility and then quickly diminish in size as flexibility increases. This is
because more flexible solutions will have less of a bias towards a particular model, while very inflexible solutions
will have very high bias.

The variance will start very low at low flexibility and then increase as flexibility increases. With increased
flexibility, individual data points will have a much higher effect on the model leading to increased variance as we
change training data sets.

Training error will start high with low flexibility and increase greatly as flexibility increases. The increased
flexibility allows the model to better fit the training data, leading to less training error.

Test error will have a U-shape, starting high with low flexibility, decreasing and then increasing. This is due to the
effect of the bias and variance components. When flexibility is low, bias is high and dominates to make test error high.
As flexibility increases, the bias drops to a low level like the variance, leading to a low test error. And then as
variance begins to increase, it will begin to dominate and increase the test error again.

Irreducible error will be flat. Irreducible error is exactly the error that cannot be removed from any change in method
and on the similar charts in the textbook was represented as the minimum possible test MSE, that is a flat line.

### Exercise 4

(a). Describe three real-life applications in which *classification* might be useful. Describe the response, as well
as the predictors. Is the goal for each application inference or prediction? Explain your answer.

(b). Describe three real-life applications in which *regression* might be useful. Describe the response, as well
as the predictors. Is the goal for each application inference or prediction? Explain your answer.

(c). Describe three real-life applications in which *cluster analysis* might be useful.

#### Answer to Exercise 4

(a).

1. Parsing hand-written numbers, where the response is one of 10 digits. This is a prediction problem as it is
   predicting an answer. It is classification because the numeric value of the digits is not important.
2. Determining what species a specimen is from a photo. The response is the species of the specimen, it is a prediction
   problem as it is predicting an answer. It is classification because there is a qualitative response.
3. Investigating what factors lead to political party identification. Political identification is the qualitative
   response, making it classification. However, in examining factors rather than prediction it is an inference problem.

(b).

1. Determining which factors affect salary among software developers. Salary is the response, and is a quantitative
   value, making this a regression problem. As we are not predicting salary, but looking at the factors involved, it is
   an inference problem.
2. Predicting estrogen levels for people undergoing HRT based on their vital factors and prescriptions. The level of
   estrogen is a quantitative value, making this a regression problem. We are attempting to predict what the levels will
   be so this is a prediction problem.
3. Predicting how long a player will play a video game based on their ratings and playtimes of other games. The response
   is the number of hours of play, a quantitative value, making this a regression problem. We are predicting, so this is
   a prediction problem.

(c).

1. Determining which type of users tend to follow certain big accounts on social media.
2. Investigating if there are patterns in how players play a video game based on time spent in different activities
   within the game.
3. Grouping grocery store customers based on their shopping habits and purchases into groups.

### Exercise 5

What are the advantages or disadvantages of a very flexible (versus a less flexible) approach for regression or
classification? Under what circumstances might a more flexible approach be preferred to a less flexible approach? When
might a less flexible approach be preferred?

#### Answer to Exercise 5

The flexibility of statistical learning methods essentially describes the ability for the method to adapt to the data
when generating the model from the data. More flexible methods will tend to follow the data more closely, which can
often lead to better accuracy of prediction or classification. However, as flexibility increases too much, expected
variation is followed too closely, leading to over-fitting.

Flexible methods have a greater range of possible models, which can be very useful when the relationships of the data do
not follow a simple model such as a linear model, and may even require very disjoint models for different subsets of the
data. However, in cases where it's more likely that the data does follow a known model an inflexible method with a
matching model will be less likely to be affected by the random noise of data. Additionally, as mentioned in Exercise 1,
when there are very few data points, the risk of over-fitting is high so inflexible models are likely to be less
susceptible to outliers in the data.

### Exercise 6

Describe the differences between a parametric and a non-parametric statistical learning approach. What are the
advantages of a parametric approach to regression or classification (as opposed to a non-parametric approach)? What are
its disadvantages?

#### Answer to Exercise 6

A parametric approach is one where a functional model is chosen in advance and the process is in determining the best matching coefficients or other values for that functional model. A non-parametric approach is one where such a model is *not* chosen, instead the process is completely based in trying to match the data. Parametric models have an advantage in interpretability because they have a clear model to refer back to, but tend to be less flexible than non-parametric models, flexibility being the the main advantage of non-parametric models at the cost of interpretability.

### Exercise 7

The table below provides a training data set containing six observations, three predictors, and one qualitative response variable.

| Obs. | $X_{1}$ | $X_{2}$ | $X_{3}$ | $Y$   |
|------|---------|---------|---------|-------|
| $1$  | $0$     | $3$     | $0$     | Red   |
| $2$  | $2$     | $0$     | $0$     | Red   |
| $3$  | $0$     | $1$     | $3$     | Red   |
| $4$  | $0$     | $1$     | $2$     | Green |
| $5$  | $-1$    | $0$     | $1$     | Green |
| $6$  | $1$     | $1$     | $1$     | Red   |

Suppose we wish to use this data set to make a prediction for $Y$ when
$X_{1} = X_{2} = X_{3} = 0$ using K-nearest neighbors.

#### Answer to Exercise 7

(a) Compute the Euclidean distance between each observation and the test point $X_{1} = X_{2} = X_{3} = 0$.

| Obs. | Distance                  |
|------|---------------------------|
| $1$  | $3$                       |
| $2$  | $2$                       |
| $3$  | $\sqrt{10} \approx 3.162$ |
| $4$  | $\sqrt{5} \approx 2.236$  |
| $5$  | $\sqrt{2} \approx 1.414$  |
| $6$  | $\sqrt{3} \approx 1.732$  |

(b) What is our prediction with $K = 1$? Why?

Green, because the closest observation is observation $5$, and it is Green.

(c) What is our prediction with $K = 3$? Why?

Red, because the three closest observations are then $5$, $6$, and $2$ in that order, and two of those are Red while one
is Green.

(d) If the Bayes decision boundary in this problem is highly non-linear, then would we expect the *best* value for $K$ to be large or small? Why?

We would expect it to be somewhat small because as $K$ increases, the model becomes less flexible and thus will be unlikely to follow the boundary closely.
