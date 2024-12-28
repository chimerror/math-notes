---
documentclass: report
fontfamily: heuristica
---

Notes on Statistical Learning
=============================

Introduction
------------

This is a series of notes on Statistical Learning originally created for my own edification. However, it may be
useful for others. It assumes very little pre-knowledge, but also does not go too deeply into areas that I feel are
reasonably well known. These topics will be covered in sections with the title "generally", to signal that it is a
very brief overview.

Observations and Variables
--------------------------

A data set useful for statistical learning is made of several observations of several variables.

### Observations

An **observation** is an individual row of a data set, which could represent a single subject or a single survey
response or a single month of data. Its exact meaning will vary based on the data set in question.

### Variables

Observations are made up of one or more **variables**, the columns of the data set, which could represent the
measurements or values for that particular observation. The exact meaning of variables will vary based on the data set
in question.

### Quantitative and Qualitative Variables

As a computational process, all variables are represented as numbers, either integers or real numbers. However,
variables are not just **quantitative**, where the number itself represents the variable. They can also be
**qualitative** (also called **categorical**), representing a non-numeric value. Examples of qualitative variables
include: which neighborhood of a city the observation is from, sex, and race. That is, any such variable that represents
one out of a non-ranked set of **classes**. We can also consider categorical data that *does* have an ordering, such as
the categories of "Small", "Medium", and "Large" where the difference between Medium and Small may not necessarily be
equal to the difference between Large and Medium despite there being an order.

Representing Data as an Matrix
------------------------------

Say we have 1000 observations of 13 variables. We can notate the number of observations with $n$, and the number of
variables with $p$. So for our example, $n = 1000$ and $p = 13$.

We can arrange the data in a large $n \times p$ matrix, $\mathbf{X}$, where each element $x_{ij}$ is the $j$th
variable of the $i$th observation ($i = 1, 2, \ldots, n$ and $j = 1, 2, \ldots, p$):

$$
\mathbf{X} = \left(
    \begin{array}{cccc}
        x_{11} & x_{12} & \ldots & x_{1p} \\
        x_{21} & x_{22} & \ldots & x_{2p} \\
        \vdots & \vdots & \ddots & \vdots \\
        x_{n1} & x_{n2} & \ldots & x_{np} \\
    \end{array} \right).
$$

Each row of the matrix is one observation, and each column is a variable.

What Is Statistical Learning?
-----------------------------

Let's say we have an $n \times p$ matrix of $n$ observations $\mathbf{X}$. Each row can be thought of as a vector of $p$
variables each of which is $X_{i}$. The group of these variables, $X$ makes up our **input variables**. Let's also say
we have an **output variable** (also called a **response variable**), $Y$.

Now, let us assume there is some fixed but unknown function $f(X)$, that forms a relationship between $X$ and $Y$, such
that:

\begin{equation}
    Y = f(X) + \epsilon\label{eq:1}.
\end{equation}

Here $\epsilon$ is an **error term**, a random value independent of $X$, with mean zero, while $f$ is the **systematic**
information that $X$ provides about $Y$. Because $f$ takes arguments from the entire group of variables, $X$, it will
likely be a function of many variables. Thus, **statistical learning**, is a method for estimating what $f$ might be
based using $X$ as **training data**. That is, we want to find $\hat{f}$, an estimate of $f$. Since the error term
$\epsilon$ has mean zero, we can use $\hat{Y}$ for a prediction of $Y$:

$$
    \hat{Y} = \hat{f}(X).
$$

In this case, there are two quantities that affect the accuracy of $\hat{Y}$ as a prediction. The first,
**reducible error** is due to the differences between $\hat{f}$ and $f$. Using statistical learning, we can attempt to
reduce this error by finding more accurate estimates of $\hat{f}$ for $f$.

However, as noted in equation $\eqref{eq:1}$, $Y$ also depends on $\epsilon$. $\epsilon$ cannot be reduced by improving
$\hat{f}$ to better estimate $f$, thus it is *irreducible error*. Despite $\epsilon$ having mean zero, $\epsilon$ may
contain unmeasured or even unmeasurable variables that affect $Y$. Thus, irreducible error is not guaranteed to be zero.

Assuming $\hat{f}$ and $X$ are fixed, we can calculate the average, or **expected value** of the squared difference
between $Y$ and $\hat{Y}$:

\begin{eqnarray*}
    E(Y-\hat{Y})^2 & = & E|f(X) + \epsilon - \hat{f}(X)|^2 \\
                   & = & |f(X) - \hat{f}(X)|^2 + \mathrm{Var}(\epsilon).
\end{eqnarray*}

Here $|f(X) - \hat{f}(X)|^2$ is the **reducible error**, while the variance (this is described later) of $\epsilon$,
$\mathrm{Var}(\epsilon)$, is the **irreducible error**. It is important to remember that while the reducible error can
be minimized (which will be our goal), the irreducible error will provide an upper bound for predictions of $Y$. What
this bound actually is usually unknown.

Prediction and Inference: the Why
---------------------------------

While the section before discussed $\hat{Y}$ as a **prediction** of $Y$, a method to generate estimates of $Y$ given
various $X$ as input, this is not always our major concern. We are sometimes more concerned with **inference**, that is,
the relationship between the individual variables of $X$ (that is, $X_{1},\ldots,X_{p}$) and $Y$. We want to know
exactly how $Y$ changes as $X$ changes. Inference can be concerned with the existences of relationships, the directions
and sizes of relationships, or even if the relationships can be summarized with a linear equation.

While for prediction the exact form of $\hat{f}$ is unimportant compared to the need for accurate predictions of $Y$,
for inference, the exact form of $\hat{f}$ is exactly what is important to us. This means inference tends to favor more
interpretable models for $\hat{f}$ even if that leads to less accurate predictions for $Y$. Thus, different statistical
learning techniques are more or less desirable based on which of prediction or inference (or both!) we are attempting to
solve.

Parametric and Non-Parametric: the How
--------------------------------------

One way to begin finding estimates for $f$ is to first select a model for $\hat{f}$, such as this linear one:

$$
    \hat{f}(X) = \beta_{0} + \beta_{1}X_{1}+ \beta_{2}X_{2} + \ldots + \beta_{p}X_{p}.
$$

Having made this assumption, rather than predicting an entire $p$-dimensional function, we only need to find the $p+1$
coefficients $\beta_{0},\beta_{1},\ldots,\beta_{p}$. We do this by using the training data in $X$ to **fit** or
**train** the model. That is choosing the **parameters** $\beta_{0},\beta_{1},\ldots,\beta_{p}$ to best match $Y$, given
the values in $X$. This method of fitting to a model is a **parametric** method.

Of course by choosing a model, we run a risk of choosing a model that is far from the actual function $f$. We can
attempt to get a better fit of our model to $f$ by choosing more **flexible** models, which generally have more
parameters to estimate. Flexible models also run the risk of **overfitting** the data, too closely following the errors
or **noise** in the data, treating them as meaningful.

Instead of assuming a functional form in advance, **non-parametric** methods make no assumptions about $f$. Instead,
estimates are based on the data itself, within some range of **smoothness** for the fit. By increasing the smoothness of
the fit, the generated fit can follow training data more closely. While this may be more accurate, the same risk of
overfitting exists here.

In general, we are seeking the correct balance between flexibility, allowing for more accurate predictions, and
**interpretability**, the ability to describe the relationships of individual variables of $X$ to $Y$. In general,
flexibility and interpretability are negatively related. That is, more flexibility tends to reduce the interpretability
of the method. However, that does not mean that a more flexible model *always* makes better predictions, since more
flexible models are more prone to overfitting.

Supervised and Unsupervised
---------------------------

So far we have assumed that we have paired observations of $Y$ for each observation of $X$. However, that is not always
the case. If we do *not* have observations of $Y$, we can still determine relationships either between the variables
(for example, does weight and height have a relationship), or between the observations. In the second case, we are often
attempting to see which observations are most like other observations, that is, we are attempting to use
**cluster analysis** to cluster the observations together into groups.

This second case, with no response variable, is referred to as **unsupervised learning**, while the first is
**supervised learning**. There may even be cases where we have the response variable for *some* of the observations in
$X$, but not all of them. This is **semi-supervised learning**, naturally, and is an advanced topic.

Regression and Classification
-----------------------------

We can further divide methods based on if the response variable is quantitative or qualitative as defined above.
Problems with a quantitative response are usually referred to as **regression** problems, while ones with a qualitative
response are **classification** problems. However, this division is not always so clear. A method called logistic
regression is typically used with a two-class or binary qualitative response, and can be seen as classification.
However, as it also estimates the class probabilities, a quantitative value, it can be seen as regression.

For this purpose we are generally only concerned with the nature of the response variable rather than the input
variables. Of course input variables can be either qualitative or quantitative, too. However, in general, most methods
can be applied regardless of the input variable, as long as qualitative input variables are properly **coded**.

Measuring Regression Fit Quality through Mean Squared Error
-----------------------------------------------------------

We are of course, desirous to be able to evaluate the performance of a statistical learning method. In the context of
regression methods (we will go over classification methods later), this can be measured through
**mean square error (MSE)**. Given $n$ observations of input variables $x_{i}$ for response variables $y_{i}$ with
prediction function $\hat{f}$, this is:

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_{i} - \hat{f}(x_{i}))^2.
$$

This is the sum of the squared differences of the actual observed response, $y_{i}$, and the predicted response,
$\hat{f}(x_{i})$ for all observations. The better that $\hat{f}$ is at predicting $f$, the smaller these differences
should be, by more closely matching $y_{i}$.

However, this calculation here would be of the closeness of an $\hat{f}$ given the *training data*, or the
**training MSE**. While this has some importance, we're really more concerned with how $\hat{f}$ performs on unseen
*test data*.

That is, given some test observation $(x_{0}, y_{0})$ that was not part of the original set of observations
$\{(x_{1}, y_{1}), \ldots, (x_{n}, y_{n})\}$, we'd rather know how close $\hat{f}(x_{0})$ is to $y_{0}$. And given a
whole set of test observations, we can calculate the **test MSE**:

$$
\mathrm{Ave}(y_{0} - \hat{f}(x_{0}))^2.
$$

The average of the squared differences of actual observed test response $y_{0}$ and predicted response $\hat{f}(x_{0})$
for all test observations. That is, assuming we *have* a set of test observations to use.

If we don't, we might start with minimizing the training MSE, but there is nothing to guarantee that the method with the
lowest training MSE will also minimize the test MSE. The particular problem is our old foe overfitting. By being very
flexible in generating $\hat{f}$, there's a greater chance that we too closely follow the noise of the given test data,
leading to our $\hat{f}$ attempting to follow trends that are not actually there. That is, going back to our original
assumption of $Y$ being related to $X$ by some unknown function $f$ plus some error term $\epsilon$ we made in
$\eqref{eq:1}$, there is nothing that guarantees that $\hat{f}$ is following $f$ instead of $\epsilon$.

Bias and Variance in Regression
-------------------------------

It can be proven (but not here) that the expected test MSE can be decomposed into three quantities:

* the **variance** of $\hat{f}(x_{0})$, the amount of change in $\hat{f}$ when estimating it with different training
  data sets
* the squared **bias** of $\hat{f}(x_{0})$, where bias refers to the error introduced by the mismatch between the chosen
  model and the real-life problem
* the variance of the error term $\epsilon$, $\mathrm{Var}(\epsilon)$, that is, the irreducible error

Note that while the irreducible error also uses the term "variance", most of the discussion in this section will be
referring to the variance of the *predicted function*, because that variance *can* be changed based on method
selection.

This composition can be summarized in this equation:

$$
E(y_{0} - \hat{f}(x_{0}))^2 = \mathrm{Var}(\hat{f}(x_{0})) + [\mathrm{Bias}(\hat{f}(x_{0}))]^2 + \mathrm{Var}(\epsilon).
$$

The variance of the method is a natural result of using training data. Different selections of training data will cause
the eventual predicted function to differ. Ideally, this difference is small. In general, a more flexible method will
follow the training data more closely, leading to a higher variance as small changes in training data can lead to large
changes in $\hat{f}$.

Bias is also a natural result of approximating complicated real-life behavior with a simpler model. If we choose to only
consider linear models as we do with linear regression, then any non-linear behavior will be poorly approximated, no
matter how much more data we use to train it. In essence, we've foreclosed better matching models from consideration.
Flexibility of method tends to *negatively* correlate with bias, in that more flexible methods tend to be *less* biased
as they are more willing to follow the data.

So as flexibility increases, variance increases and bias decreases, with the different rates of change determining if
the test MSE will decrease or increase. The tendency is that bias tends to decrease relatively quickly compared to
the increase in variance *early on*. But eventually, the decrease in bias begins to diminish and the increase in
variance begins to dominate. When graphing the test MSE against the flexibility, this will produce a U-shape, where the
test MSE first decreases and then increases. The challenge then is how best to balance the trade-off between bias and
variance. Choosing low-bias methods tends to result in high variance, and choosing low-variance methods tends to result
in high bias, but we ideally want our chosen method to be low in both.

Note that in practice, $f$ is unobserved and unknown, and thus explicit computation of the test MSE is impossible.
Despite this, the bias-variance trade-off still applies. Many of the methods that will be discussed are very flexible,
and thus have very low bias. However, if the data itself *is* biased, for example, they really point to a truly linear
relationship, then the less flexible linear regression will tend to do a better job despite being highly biased.

Measuring Classification Quality through Error Rate
---------------------------------------------------

Many of the concepts for model accuracy that we have discussed in the context of regression models also apply to
classification models. We are still seeking to estimate a function $f$ by creating a prediction function $\hat{f}$ based
on training observations $\{(x_{1}, y_{1}), \ldots, (x_{n}, y_{n})\}$, but now $y_{1}, \ldots, y_{n}$ are qualitative.
It doesn't make sense to consider the MSE since despite being coded as a number there is no ordering of qualitative
data. So the **error rate** is preferred:

\begin{equation}
    \frac{1}{n} \sum_{i=1}^n I(y_{i} \neq \hat{y}_{i})\label{eq:2}.
\end{equation}

$\hat{y}_{i}$ refers to which class label is predicted for the $i$th observation by $\hat{f}$.
$I(y_{i} \neq \hat{y}_{i})$, then is the **indicator variable** which equals $1$ when $y_{i} \neq \hat{y}_{i}$ and $0$
when $y_{i} = \hat{y}_{i}$. That is a result of $0$ means $\hat{f}$ classified the observation correctly, while a
result of $1$ means it misclassified it. Thus, this gives us the proportion of misclassifications.

Equation $\eqref{eq:2}$ is more exactly the **training error rate** as it is calculated on the training data. This is
fine, but more interesting is the **test error rate**. For a set of test observations $(x_{0}, y_{0})$, the test error
rate is:

$$
\mathrm{Ave}(I(y_{0} \neq \hat{y}_{0})).
$$

Here $\hat{y}_{0}$ is the predicted class label for the $0$th test observation. The aim is to pick a classification
model that minimizes this value. To illustrate this further, let's demonstrate using two simple classifiers.

### The Bayes Classifier

It can be proven (though, still not here) that the test error rate can be minimized if we are able to create the
**Bayes classifier** by being able to calculate the conditional distribution of the data. This classifier assigns each
observation to the most likely class given the predictor values. That is, given predictor vector $x_{0}$, we pick
whichever class $j$ has the highest value for the conditional probablity $\Pr(Y = j|X = x_{0})$. In a two-class case
where the classes are, say, $1$ or $2$, the Bayes classifier would assign $x_{0}$ to class $1$ if
$\Pr(Y = 1|X = x_{0}) > 0.5$ and class $2$ otherwise.

Let us further assume in our two-class problem that there are two predictors that make up $X$, $X_{1}$ and $X_{2}$. If
we chart these two predictors against each other, we can draw a boundary where the probability of assigning to either
class is 50% called the **Bayes decision boundary**. In short, this boundary divides the space into an area where a
predictor vector will be assigned to class $1$ and an area where a predictor will be assigned to class $2$.

As mentioned, the Bayes classifier will produce the lowest possible test error rate, the **Bayes error rate**, which is
much like the irreducible error. For $X = x_{0}$ this would be $1 - \max_{j}\Pr(Y = j|X = x_{0})$. And
extending over all $X$:

$$
1 - E(\max_{j}\Pr(Y = j|X)).
$$

### K-Nearest Neighbors Classification

Given real data, it is generally impossible to compute the Bayes classifier since we don't actually know the conditional
distribution of the data. But it still has some use as a standard to compare other methods against, which instead
attempt to estimate the conditional distribution, and then assign a class based on this estimate.

For example, the **K-nearest neighbors (KNN) classifier** is based on a constant positive integer $K$ and a test
observation $x_{0}$. It starts by identifying the $K$ closet points in the training data to $x_{0}$, $\mathcal{N}_{0}$.
From this, the conditional probability for class $j$ can be estimated by the fraction of points in $\mathcal{N}_{0}$
that are in class $j$:

$$
\Pr(Y = j|X = x_{0}) = \frac{1}{K} \sum_{i \in \mathcal{N}_{0}}I(y_{i} = j).
$$

Whichever class $j$ has the highest probability is what $x_{0}$ is assigned to. We can graph the decision boundary for
KNN just the same as we did with the Bayes decision boundary, and despite the simplicity of the method, the KNN decision
boundary can get very close to the ideal Bayes decision boundary. However, this depends on the choice of $K$. As $K$
grows, KNN becomes less flexible since more of the training data affects the boundary. If $K=1$, it is likely that the
boundary will be too flexible and overfit to close data, while at the other extreme of say, $K=100$, the boundary will
not be flexible enough to better follow the Bayes boundary. These issues will result in higher test error rates,
producing the familiar U-shape where both extremes have high error. Picking $K$ so that the test error is minimized
becomes the goal.

Simple Linear Regression
------------------------

**Linear regression** is a very simple approach for supervised learning, in particular predicting a quantitative
response. It has a long history and is a staple of statistics textbooks, but despite its age, there is value in being
aware of it because many more modern approaches for statistical learning are based on it.

Let's start out with **simple linear regression** which can be used to match a single predictor $X$ to a quantitative
response $Y$. One of the first assumptions of linear regression is that the relationship between $X$ and $Y$ is
linear or at least approximately linear. That is:

\begin{equation}
    Y \approx \beta_{0} + \beta_{1}X\label{eq:3},
\end{equation}

where $\beta_{0}$ and $\beta_{1}$ are unknown constants representing the intercept and slope of the linear model. The
**intercept**, $\beta_{0}$, represents the value $Y$ will have when $X$ is $0$. The **slope**, $\beta_{1}$, represents
the change in $Y$ for each increase of $X$. So for example, if $Y$ represents hormone levels in international units per
liter (IU/L) after given a certain dosage of HRT in mg, $\beta_{0}$ is the expected level with no treatment, while
$\beta_{1}$ is the expected increase (one hopes) in IU/L for every additional mg of dosage.

The pair of constants is often known as the **coefficients** or **parameters** of the model (and thus linear
regression is a parametric method). However, these constants generally remain unknown and our hope is to estimate them
using the training data, computing:

$$
\hat{y} = \hat{\beta}_{0} + \hat{\beta}_{1}x,
$$

where $\hat{y}$ is a prediction of $Y$ on the basis of $X = x$, the hat symbol denoting an estimated value of a
parameter or the predicted value of the response. This means that given a set of $n$ observations of measurements of
$X$ and $Y$, $(x_{1}, y_{1}), (x_{2}, y_{2}), \ldots, (x_{n}, y_{n})$, we want to find $\hat{\beta}_{0}$ and
$\hat{\beta}_{1}$ that fit the data well, such that the resulting line is close to the data points. There are many ways
to determine closeness but the most common approach is **least squares**.

Given prediction $\hat{y}_{i} = \hat{\beta}_{0} + \hat{\beta}_{1}x_{i}$ for the $i$th observation, we define the $i$th
**residual** as $e_{i} = y_{i} - \hat{y}_i$, the difference between the observed value of $y_{i}$ and the prediction,
$\hat{y}_{i}$. Taking the residuals for all observations, we can calculate the **residual sum of squares** (RSS) as

$$
\mathrm{RSS} = e_{1}^2 + e_{2}^2 + \ldots + e_{n}^2,
$$

which can be expanded out to

$$
\mathrm{RSS} = (y_{1} - \hat{\beta}_{0} - \hat{\beta}_{1}x_{1})^2 + (y_{2} - \hat{\beta}_{0} - \hat{\beta}_{2}x_{2})^2 +
\ldots + (y_{n} - \hat{\beta}_{0} - \hat{\beta}_{n}x_{n})^2.
$$

This RSS is exactly what we are going to minimize with our choice of coefficients. Using calculus, it can be shown that
the desired **least squares coefficient estimates** are:

<!-- markdownlint-disable MD049 -->
\begin{eqnarray*}
    \hat{\beta}_{1} & = & \frac{\sum_{i = 1}^n (x_{i} - \bar{x}) (y_{i} - \bar{y})}{\sum_{i = 1}^n (x_{i} - \bar{x})^2} \\
    \hat{\beta}_{0} & = & \bar{y} - \hat{\beta}_{1}\bar{x},
\end{eqnarray*}
<!-- markdownlint-enable MD049 -->

where $\bar{y} = \frac{1}{n} \sum_{i = 1}^n y_{i}$ and $\bar{x} = \frac{1}{n} \sum_{i = 1}^n x_{i}$ are the sample
means.

Assessing Linear Regression Coefficient Estimates using Standard Error
----------------------------------------------------------------------

When we gave equation $\eqref{eq:3}$ earlier as the suspected true relationship between $X$ and $Y$, we overlooked
the error term, $\epsilon$ as we defined back in equation $\eqref{eq:1}$. Adding this back in produces the
**population regression line**:

$$
Y = \beta_{0} + \beta_{1}X + \epsilon,
$$

the best linear approximation of the true relationship. Note that this is subtly different from the least squares line
we calculated earlier. The difference comes down to the fact that usually rather than measuring the entire population
we are attempting to describe, we are instead working with a smaller sample of that population to *estimate* for the
entire population. A good comparison is the difference between the population mean, $\mu$, that we'd get if we had
observations for the entire population versus the sample mean, $\hat{\mu} = \bar{y}$.

It is very likely the population mean and sample mean differ, but the sample mean should be a good estimate nonetheless
assuming our sampling is **unbiased**, that is, not systematically over- or under-estimating the true value. If we took
many different samples and calculated the sample mean, the average of all of these sample means will approach the
population mean assuming lack of bias. Likewise, with linear regression, we are calculating estimates of the population
regression line coefficients $\beta_{0}$ and $\beta_{1}$ of the population regression line with our least squares line
values of $\hat{\beta}_0$ and $\hat{\beta}_1$.

Returning to the estimation of the population mean as an analogy, we can use the **standard error** of a sample mean
estimate, $\hat{\mu}$, to judge how close it is to the population mean, $\mu$:

$$
\mathrm{Var}(\hat{\mu}) = \mathrm{SE}(\hat{\mu})^2 = \frac{\sigma^2}{n}
$$

where in this case, $\sigma$ is the standard deviation of our observations of $Y$. And likewise for linear regression,
we can calculate how close our estimated coefficients are:

<!-- markdownlint-disable MD049 -->
\begin{eqnarray*}
    \mathrm{SE}(\hat{\beta}_{0})^2 & = & \sigma^2\left[\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i = 1}^n (x_{i} - \bar{x})^2}\right], \\
    \mathrm{SE}(\hat{\beta}_{1})^2 & = & \frac{\sigma^2}{\sum_{i = 1}^n (x_{i} - \bar{x})^2},
\end{eqnarray*}
<!-- markdownlint-enable MD049 -->

where $\sigma^2 = \mathrm{Var}(\epsilon)$, given the assumption that the errors for each observation are uncorrelated,
and have common variance $\sigma^2$. We don't usually actually know $\sigma$, so we estimate it using the
residual standard error based on the data, $\hat{\sigma} = \mathrm{RSE} = \sqrt{\mathrm{RSS}/(n -2)}$ (which is more
fully described in a later section). Technically, this means we should write these standard errors as
$\widehat{\mathrm{SE}}$ to mark them as estimates, but this is usually dropped for simplicity of notation.

Determining Linear Regression Coefficient Confidence Intervals
--------------------------------------------------------------

The standard error can be used to calculate **confidence intervals**, a range of values that, with a certain probability
(usually 95%), should contain the true value of a parameter. These are calculated as an upper and lower limit. So for
$\hat{\beta}_1$:

$$
\hat{\beta}_1 \pm 2 \cdot \mathrm{SE}(\hat{\beta}_1),
$$

meaning that there is a 95% chance that the interval
$[\hat{\beta}_1 - 2 \cdot \mathrm{SE}(\hat{\beta}_1), \hat{\beta}_1 + 2 \cdot \mathrm{SE}(\hat{\beta}_1)]$ contains the
true value, $\beta_1$. And likewise for $\hat{\beta}_0$ the confidence interval is:

$$
\hat{\beta}_0 \pm 2 \cdot \mathrm{SE}(\hat{\beta}_0).
$$

Though note these formulas are based both on the assumption that the errors are Gaussian, and the factor $2$ is a
stand-in for a more complicated value based on $n$, the 97.5% quantile of a t-distribution with $n - 2$ degrees of
freedom. The assumption is usually warranted, and we can more accurately calculate the necessary factor using a
statistical package as we will do in practice.

Hypothesis Testing of Linear Regression
---------------------------------------

Further, we can use the standard error to do **hypothesis testing**. The most common such test involves testing the
**null hypothesis** ($H_{0}$) that there is no relationship between $X$ and $Y$ against the **alternative hypothesis**
($H_{a}$) that there *is* some relationship between them. Or mathematically, testing $H_{0}: \beta_1 = 0$ versus
$H_{a}: \beta_1 \neq 0$. Or rephrasing, we want to make sure our estimate $\hat{\beta}_{1}$ is far enough from $0$ that
we are confident that $\beta_1$ is non-zero.

How far this needs to be is dependent on $\mathrm{SE}(\hat{\beta}_{1})$. The smaller this value is, the more accurate
our estimate is, and the closer $\hat{\beta}_{1}$ can be to zero while still concluding that $\beta_{1}$ is non-zero. In
practice, this is calculated using the **t-statistic**:

$$
t = \frac{\hat{\beta}_1 - 0}{\mathrm{SE}(\hat{\beta}_1)}
$$

which measures the number of standard deviations that $\hat{\beta}_1$ is from $0$. We can also calculate the
**p-value**, the probability of that we would have an observation with an absolute value equal to or larger than $|t|$,
assuming that $\beta_1 = 0$. A small p-value implies that it is very unlikely that we would have seen such the
association between the predictor and response if indeed there was no relation between the predictor and response. Given
a sufficiently small p-value, we can confidently reject the null hypothesis, and conclude that there is indeed a
relationship. Typical cut-offs for p-values are 5% or 1%, though there is much more to be said on this topic.

Assessing Linear Regression Fit Quality
---------------------------------------

After rejecting the null hypothesis, we are confident that there is a relationship, but it still is worth quantifying
how well our regression *fits* the the data. There are two quantities generally used for this, the residual standard
error and the $R^2$ statistic.

### Residual Standard Error

Our model attempts to account for the general case of error in our observations using the error term $\epsilon$,
accounting for a myriad of reasons it may not actually be possible to estimate the response based on the predictors. We
estimate the standard deviation of $\epsilon$ using the **residual standard error** (RSE), the average amount that the
response may deviate from the true population regression line. For simple linear regression, RSE can be calculated using
RSS:

$$
\mathrm{RSE} = \sqrt{\frac{1}{n - 2}\mathrm{RSS}} = \sqrt{\frac{1}{n - 2}\sum_{i = 1}^{n} (y_{i} - \hat{y}_{i})^2}.
$$

Note this is the formula for *simple* linear regression, where there is only one predictor. We will cover *multiple*
linear regression using more than one predictor in a bit, where the idea is the same but the formula is slightly
different.

The RSE is a measure of the *lack of fit* of the model, where a small value implies a better fit.

### $R^2$ Statistic

A downside of the RSE is that it is measured in the units of the response, so what a good value will be depends on what
exactly is being measured by the response. An alternative is the **$R^2$ statistic**, which is instead the proportion
of variance explained by the regression. Being a proportion it takes a value between $0$ and $1$, so it can be compared
between different responses more easily. The formula is:

$$
R^2 = \frac{\mathrm{TSS} - \mathrm{RSS}}{\mathrm{TSS}} = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}
$$

where $\mathrm{TSS} = \sum(y_{i} - \bar{y})^2$, the **total sum of squares**, and RSS is defined above. TSS is a measure
of the total variance in the response, the amount the response varies on its own. This makes RSS then the amount of
unexplained variability after performing the regression. So $\mathrm{TSS} - \mathrm{RSS}$ is the variability that is
accounted for by the regression, and thus $R^2$ is the proportion of variability in $Y$ that can be explained by $X$
through the regression. The closer $R^2$ is to $0$, the less that is explained by $X$, and as it approaches $1$, more
of the variability is accounted for by $X$ through the regression. Low values of $R^2$ might indicate the linear model
is incorrect, that there is high error variance $\sigma^2$, or both.

While $R^2$ does indeed make comparison easier as it will always be between $0$ and $1$, it does not necessarily mean
for all applications that similar values are equally good or bad. For relationships that are known to be linear or very
close to linear, a higher value can be expected versus cases where the relationship is definitely not linear where
lower values can be acceptable.

Another measure of the linear relationship between $X$ and $Y$ is the **correlation**, defined as:

$$
r = \mathrm{Cor}(X, Y) = \frac{\sum_{i = 1}^{n} (x_{i} - \bar{x})(y_{i} - \bar{y})}{\sqrt{\sum_{i = 1}^{n}(x_{i} - \bar{x})^2}\sqrt{\sum_{i = 1}^{n}(y_{i} - \bar{y})^2}}.
$$

However, the use of this for accessing the fit quality is limited because in fact, for simple linear regression it can
be shown that $R^2 = r^2$. Further, correlation does not extend to multiple linear regression, which will be introduced
in the next section. $R^2$ *does* work for multiple linear regression on the other hand.

Multiple Linear Regression
--------------------------

Simple linear regression works well for predicting a response based on a single predictor, but in practice, there are
often multiple predictors we'd like to look at. For example, hormone replacement therapy usually consists of a
cocktail of drugs. So rather than just prescribing estrogen injections to increase estrogen levels, doctors will tend
to also prescribe an anti-androgen like spironolactone as well as other hormones such as progesterone. It may be
worthwhile measuring the effect on estrogen levels for different dosages of each component of the cocktail, each dosage
serving as a predictor for the response of the final measured estrogen levels.

While three separate simple linear regressions could be run, it would be hard to combine the three separate linear
regression equations to get a good single prediction of the response. Additionally, each simple linear regression for
a particular drug would ignore the effects of the other drugs, which could lead to misleading results. Instead, we can
use **multiple linear regression**, by giving each predictor its own slope giving a model of the form:

$$
Y = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + \cdots + \beta_{p}X_{p} + \epsilon
$$

for $p$ distinct predictors where $X_{j}$ is the $j$th predictor and $\beta_{j}$ is the association between that
variable and the response, that is the average effect on $Y$ given 1 unit increase in $X_{j}$ while holding all other
predictors fixed.

Like simple linear regression, we attempt to estimate these coefficients from the training data using the formula:

$$
\hat{y} = \hat{\beta}_{0} + \hat{\beta}_{1}x_{1} + \hat{\beta}_{2}x_{2} + \cdots + \hat{\beta}_{p}x_{p}
$$

choosing the multiple least squares regression coefficients by minimizing the sum of squared residuals:

<!-- markdownlint-disable MD049 -->
\begin{eqnarray*}
    \mathrm{RSS} & = & \sum_{i = 1}^{n} (y_{i} - \hat{y}_{i})^2 \\
     & = & \sum_{i = 1}^{n} (y_{i} - \hat{\beta}_{0} - \hat{\beta}_{1}x_{i1} - \hat{\beta}_{2}x_{i2} - \cdots - \hat{\beta}_{p}x_{ip})^2.
\end{eqnarray*}
<!-- markdownlint-enable MD049 -->

However, the forms for the multiple least squares regression coefficients are complicated and most easily represented in
matrix form. As such, they are not provided here, and the use of a statistical software package is recommended to
calculate them.

Note that it can often turn out that a multiple regression may imply there is no relationship between a particular
predictor and the response, while a simple regression *does* imply such a relationship. This is very common, because
it may turn out that the predictor is correlated with another predictor that has a more direct relationship. For
example, a simple regression of ski accidents per day versus liters of eggnog consumed per day would likely show a
positive relationship between the two. However, eggnog (at least non-alcoholic eggnog) has no real relationship with the
number of ski accidents besides both sharply increasing in the wintertime. If a multiple regression pulling in
temperature is performed, eggnog consumption would likely no longer remain a significant predictor, as temperature more
directly impacts the amount of skiing people do, and thus the number of accidents that occur.

Hypothesis Testing of Multiple Linear Regression
------------------------------------------------

Much like in simple linear regression, we will want to consider if there is indeed a relationship between the response
and the predictors in multiple linear regression, we can use hypothesis testing to do this. In the simple linear
regression case, we only need to check if $\beta_{1} = 0$, but in multiple linear regression we will likely need to
check that *all* regression coefficients are zero. Thus our null hypothesis, $H_{0}$ would be
$H_{0}: \beta_1 = \beta_2 = \cdots = \beta_p = 0$, and our alternative hypothesis, $H_{a}$ would be that at least one
$\beta_j$ is non-zero.

This can be calculated using the **F-statistic**:

$$
F = \frac{(\mathrm{TSS} - \mathrm{RSS})/p}{\mathrm{RSS}/(n - p - 1)}
$$

where $\mathrm{TSS} = \sum(y_{i} - \bar{y})^2$ and $\mathrm{RSS} = \sum(y_{i} - \hat{y}_{i})^{2}$ as in simple linear
regression.

Given that linear model assumptions are correct, it can be shown (though not here) that:

$$
E(\mathrm{RSS}/(n - p - 1)) = \sigma^{2}
$$

and when $H_{0}$ is true:

$$
E((\mathrm{TSS} - \mathrm{RSS})/p) = \sigma^{2}.
$$

So if $H_{0}$ is indeed true, and there is no relationship between the response and predictors, the F-statistic will be
close to $1$. If instead $H_{a}$ is true, $E{(\mathrm{TSS} - \mathrm{RSS})/p} > \sigma^{2}$, making $F$ be greater than
$1$. Though this implies that we can readily reject $H_{0}$ when $F$ is much greater than $1$, a bit of care needs to
be taken when $F$ is closer to $1$. Whether or not we can reject $H_{0}$ when $F$ is greater than but close to $1$
depends on the number of observations, $n$, and the number of predictors, $p$. For example, a larger $n$ allows us to
more readily reject the null hypothesis for $F$ closer to $1$, and smaller $n$ require higher values of $F$ to do the
same.

A clearer understanding of this comes as a result that if we assume $H_{0}$ to be true, and that the errors
$\epsilon_{i}$ to have a normal distribution, then the F-statistic follows an F-distribution. This remains approximately
true for large sample sizes, even if the errors are *not* normally-distributed. The p-value for the F-distribution can
be calculated for a particular F-statistic using any statistical software package. This p-value can then be used to
decide to reject $H_{0}$ or not.

In some cases, we may instead want to make our null hypothesis that only a subset $q$ of the coefficients are zero, thus
giving us $H_{0}: \beta_{p-q+1} = \beta_{p-q+2} = \cdots = \beta_{p} = 0$, where the coefficients that we are
considering part of $q$ are at the end for convenience. In this case, we can fit a second model that does not include
our chosen coefficients. If that second model has a residual sum of squares of $\mathrm{RSS}_{0}$, then the appropriate
F-statistic will be:

$$
F = \frac{(\mathrm{RSS}_{0} - \mathrm{RSS})/q}{\mathrm{RSS}/(n - p - 1)}.
$$

For each predictor used as part of a multiple linear regression, individual p-values and the relevant t-statistics are
calculated. Both of these end up being exactly equivalent to an F-test omitting that single variable from the model,
thus as if $q=1$, reporting the **partial effect** of each predictor when added to the model. One *might* assume that
these partial effects would be sufficient for hypothesis testing versus the entire F-statistic, figuring that if any
individual p-value is very small, we can accept $H_{a}$, that one of the predictors is thus related to the response.
However, this reasoning is flawed, especially as the number of available predictors increase. In a situation where
$H_{0}$ was true, there will be about 5% of the p-values that end up below $0.05$ by pure chance. So given 100
predictors, we would likely see five small p-values even if there was no relationship between any of them. The
F-statistic accounts for the number of predictors, and thus does not have this same problem.

However, there are limits to the power of the F-statistic. It works best if the number of predictors are small,
especially when the number of predictors, $p$, are smaller than the number of observations, $n$. However, if $p > n$,
then there are more coefficients to estimate than we have observations. We cannot even fit a multiple linear regression
model using least squares in this case, so we cannot use the F-statistic and other concepts discussed so far. This case
of a high $p$ is referred to as a **high-dimensional** setting, inviting the use of other approaches like forward
selection, which will be introduced in the next section, and discussed further in a later section.

Variable Selection in Multiple Linear Regression
------------------------------------------------

After calculating the F-statistic and determining that there is at least one predictor related to the response, a
natural follow-up question is *which* predictor. As mentioned, individual p-values are not sufficient for this, and
instead we must perform the task of **variable selection**, that is determining which predictors are associated with
the response in order to recreate the model using only those predictors.

An ideal solution for this would be trying all possible combinations of predictors and judging each model based on some
statistics useful for judging the quality of the model, said statistics to be covered in more detail in a later section.
However this exhaustive process is not tractable as the number of predictors increase. The total number of models for
$p$ predictors is $2^p$, which increases exponentially. For example, a mere 13 predictors would require 8,192 models to
be checked.

Instead we must use some methods to limit which models we check, and there are three classical approaches:

* **Forward selection**: Beginning with the **null model**, the model with an intercept but no predictors, we first
  do $p$ simple linear regressions and add the predictor with the lowest RSS to the null model. Then we do $p - 1$
  two-variable regressions with each remaining predictor, adding the lowest RSS. This continues until some stopping rule
  is satisfied.
* **Backward selection**: Now instead the process starts with the model containing all predictors, and remove the
  predictor with the highest p-value, that is the one least statistically significant. Then fit to the new model with
  $p - 1$ predictors, and choose the highest p-value to remove. And so on, until a stopping rule is reached, such as
  when all remaining p-values are below a threshold.
* **Mixed selection**: A combination of forward and backward selection. Predictors are added starting from the null
  model as in forward selection, but when the p-value for any model rises above a threshold, it is removed and the model
  recalculated. Then other predictors can be added and removed in similar steps until a stopping rule is satisfied, such
  as when all in-model predictors' p-values are below a threshold, and any out-model predictors would have a large
  p-value if added.

Note that backward selection cannot be used if $p > n$, but forward selection can always be used. Forward selection is
a greedy approach, and could end up including predictors that end up being redundant, which can be remedied by using
mixed selection to remove such predictors.

Assessing Model Fit in Multiple Linear Regression
-------------------------------------------------

Much like simple linear regression, two common numeric measures of model fit are the RSE and $R^2$ values.

In multiple linear regression, $R^2 = \mathrm{Cor}(Y, \hat{Y})^2$, that is, the square of the correlation between the
response and fitted linear model. In fact, the fitted linear model maximizes this correlation among all possible linear
models. The closer to 1 $R^2$ is implies that the model explains a large portion of variance in the response model.

One should be careful that adding predictors will increase the $R^2$ regardless of the significance of the predictor
because it causes a decrease in residual sum of squares on the training data. However, adding weak predictors will tend
to have a much smaller increase to $R^2$ than stronger predictors. This similarly applies to RSE.

RSE has a slightly different formula for multiple linear regression:

$$
RSE = \sqrt{\frac{1}{n - p - 1}\mathrm{RSS}}
$$

which is equivalent to the earlier formula for simple linear regression, when $p$ is 1.

Beyond these numeric measures, plotting the data can be very helpful, particularly for noticing non-linear patterns that
may hint at a **synergy** or **interaction** effect between predictors, where predictors impact each other. Extensions
to the linear model to account for these effects will be discussed in a later section.

Confidence and Prediction Intervals in Multiple Linear Regression
-----------------------------------------------------------------

Once a regression model has been fitted, we can use our coefficient estimates to find a formula for the least squares
plane, $\hat{Y} = \hat{\beta}_{0} + \hat{\beta}_{1}X_{1} + \cdots \hat{\beta}_{p}X_{p}$, to make predictions. Of course,
as the hats on everything implies, this is just an estimate for the true population regression plane,
$f(x) = \beta_{0} + \beta_{1}X_{1} + \cdots \beta_{p}X_{p}$, and the difference between these two is a result of
reducible error. A **confidence interval** can be calculated to give a range of how close $\hat{Y}$ is to $f(x)$.

Even if we somehow knew the true value for $f(x)$, we would still have to accept the effects of our random error term,
$\epsilon$ which is our irreducible error representing the difference between our predicted value $\hat{Y}$ and the
true value $Y$, which is not equivalent to $f(x)$ due to the irreducible error. For quantifying that, a wider
**prediction interval** can be calculated.

We would also have to account for model bias from choosing a linear model for a possibly non-linear relationship, but
for now, we are treating the linear model as if it is correct for this part of the discussion.

A confidence interval describes the uncertainty around the *average* value of the response variable. That is, it gives
an interval that if collected for many data sets of the same population would contain the actual value of $f(x)$ 95% of
the time. Comparatively, a prediction interval describes the uncertainty around a *particular* response variable value
for a *particular* observation. Because random error will be more "smoothed" out by looking at the average value
compared a particular value, a confidence interval is narrower than a prediction interval.

Coding Qualitative Predictors for Regression Models
---------------------------------------------------

While much of our discussion has assumed quantitative predictors, data sets often include qualitative predictors as
well. Qualitative predictors are also referred to as **factors** and have a certain number of **levels** of possible
values. For example, let's say we are working with a data set of people who may or may not own a car, spread out between
three regions (East, West, or South, perhaps) with the response variable being their annual salary. One qualitative
predictor would be car ownership, which would have two levels, and another would be which region the observation comes
from, which would have three levels.

### Two-Level Factors

For two-level factors, like car ownership, we can create a **dummy variable** that can be one of two numeric values,
perhaps:

<!-- markdownlint-disable MD049 -->
\begin{equation*}
x_{i} =
    \begin{cases}
        1 & \text{if the } i \text{th person owns a car}\\
        0 & \text{if the } i \text{th person does not own a car}\\
    \end{cases}
\end{equation*}
<!-- markdownlint-enable MD049 -->

and then use this variable as a predictor in our model:

<!-- markdownlint-disable MD049 -->
\begin{equation*}
y_{i} = \beta_{0} + \beta_{1}x_{i} + \epsilon_{i} =
    \begin{cases}
        \beta_{0} + \beta_{1} + \epsilon_{i} & \text{if the } i \text{th person owns a car}\\
        \beta_{0} + \epsilon_{i} & \text{if the } i \text{th person does not own a car}.\\
    \end{cases}
\end{equation*}
<!-- markdownlint-enable MD049 -->

Then $\beta_{0}$ is the average salary for non-owners, while $\beta_{0} + \beta_{1}$ is the average salary for owners,
and $\beta_{1}$ as the average difference in salary between owners and non-owners.

Note that this choice of which coding to use is arbitrary, though different codings will change how the coefficients
are interpreted. If instead 1 was used for non-owners, then $\beta_{0}$ would be the average salary for *owners* and
so on, though once interpreted and calculated, the same predictions should be reached.

Another alternative is a coding using 1 and -1:

<!-- markdownlint-disable MD049 -->
\begin{equation*}
x_{i} =
    \begin{cases}
        1 & \text{if the } i \text{th person owns a car}\\
        -1 & \text{if the } i \text{th person does not own a car}\\
    \end{cases}
\end{equation*}
<!-- markdownlint-enable MD049 -->

which would give us the model:

<!-- markdownlint-disable MD049 -->
\begin{equation*}
y_{i} = \beta_{0} + \beta_{1}x_{i} + \epsilon_{i} =
    \begin{cases}
        \beta_{0} + \beta_{1} + \epsilon_{i} & \text{if the } i \text{th person owns a car}\\
        \beta_{0} - \beta_{1} + \epsilon_{i} & \text{if the } i \text{th person does not own a car}.\\
    \end{cases}
\end{equation*}
<!-- markdownlint-enable MD049 -->

This makes $\beta_{0}$ the average salary regardless of car ownership, and $\beta_{1}$ the amount which owners are above
that average, and non-owners are below that average. Regardless, the final predictions of salary for each class will
be the same. Only the interpretation of coefficients changes as coding schemes change.

### Factors with More than Two Levels

Turning to the region factor which has the three levels of East, South, and West, a single dummy variable is not enough.
Instead, you must use multiple dummy variables, one less than the number of levels, so two dummy variables in our
example. For example, the first variable might be:

<!-- markdownlint-disable MD049 -->
\begin{equation*}
x_{i1} =
    \begin{cases}
        1 & \text{if the } i \text{th person is from the South}\\
        0 & \text{if the } i \text{th person is not from the South}\\
    \end{cases}
\end{equation*}
<!-- markdownlint-enable MD049 -->

and the second:

<!-- markdownlint-disable MD049 -->
\begin{equation*}
x_{i2} =
    \begin{cases}
        1 & \text{if the } i \text{th person is from the West}\\
        0 & \text{if the } i \text{th person is not from the West}.\\
    \end{cases}
\end{equation*}
<!-- markdownlint-enable MD049 -->

Using this in the regression equation to get a model:

<!-- markdownlint-disable MD049 -->
\begin{equation*}
y_{i} = \beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2} + \epsilon_{i} =
    \begin{cases}
        \beta_{0} + \beta_{1} + \epsilon_{i} & \text{if the } i \text{th person is from the South}\\
        \beta_{0} + \beta_{2} + \epsilon_{i} & \text{if the } i \text{th person is from the West}\\
        \beta_{0} + \epsilon_{i} & \text{if the } i \text{th person is from the East}.\\
    \end{cases}
\end{equation*}
<!-- markdownlint-enable MD049 -->

This makes $\beta_{0}$ the average salary for people from the East, $\beta_{1}$ the difference in average salary of
people from the South against people from the East, and $\beta_{2}$ the difference in average salary of people from the
West against people from the East. The level that has no dummy variable is thus called the **baseline**.

A choice of baseline is arbitrary, and final predictions should not differ based on choice of coding scheme. However,
that does not apply to the values of the coefficients and their associated p-values, which *do* depend on coding scheme.
Instead of relying on the individual coefficients, an F-test to test $H_{0}: \beta_{1} = \beta_{2} = 0$ can be used as
it is not affected by choice of coding scheme.

Different coding schemes mostly provide different interpretations, and the possibilities are wide, each choice measuring
particular **contrasts**. However, a full discussion of these possibilities is beyond the scope of these notes.

Also note that these dummy variables for qualitative predictors can be used alongside regular quantitative variables in
a regression with no difficulties.

Accounting for Non-Additive Effects through Interaction Terms
-------------------------------------------------------------

A linear regression model carries certain assumptions about the relationship can be often violated by real-world data.
However, the model can be extended to account for these violations. The first assumption we'll deal with is the
assumption that the relationship is **additive**, that is, the relationship between any one predictor and the response
depend only on that predictor, and not any others.

Returning to our earlier example of drug cocktails for HRT, our model assumed resulting hormone levels depended on the
dosages of estradiol and spironolactone given. However, as both of these drugs target the same systems in the body (and
the systems of the body are highly interconnected regardless), it is likely the dosage of spironolactone's effects on
suppressing testosterone may affect the dosage of estradiol's effect on raising estrogen, which is why, of course, both
of these are often prescribed as a cocktail.

This would make our basic linear regression model with two variables:

$$
Y = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + \epsilon
$$

incorrect due to its assumption of additivity. That is, if $X_{1}$ represents the estradiol dosage, the model predicts
an increase in final estrogen levels of $\beta_{1}$ IU\L for each mg of estradiol. It does not account for any effect
that the spironolactone dosage, $X_{2}$ may have on the ability for the estradiol dosage to affect estrogen levels.

This can be addressed by adding a third predictor, an **interaction term**, which is constructed from the product of
$X_{1}$ and $X_{2}$. This gives us a model of:

$$
Y = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} +\beta{3}X_{1}X_{2} + \epsilon
$$

where $\beta_{3}$ is our interaction term. By rewriting this, we get:

<!-- markdownlint-disable MD049 -->
\begin{eqnarray*}
    Y & = & \beta_{0} + (\beta_{1} + \beta_{3}X_{2})X_{1} + \beta_{2}X_{2} + \epsilon \\
    & = & \beta_{0} + \tilde{\beta_{1}}X_{1} + \beta_{2}X_{2} + \epsilon \\
\end{eqnarray*}
<!-- markdownlint-enable MD049 -->

where $\tilde{\beta_{1}} = \beta_{1} + \beta_{3}X_{2}$. This makes $\tilde{\beta_{1}}$ a function of $X_{2}$ rather than
a constant, thus changes in $X_{2}$ causing changes of the association of $X_{1}$ and $Y$. This argument can be reversed
to show the same thing for how $X_{1}$ affects $X_{2}$'s relationship with $Y$.

Adding these interaction terms can often improve the model compared to one that only terms for the **main effects**.
However, when one examines the p-values for each term it is possible that the interaction term will have a statistically
significant p-value, while a main effect term does not. We should not take this as a sign that the main effect term
should be removed from the model due to the **hierarchical principle**, stating that the main effects for any
interaction term must be included regardless of the significance of the p-value for them.

The rationale is that if the interaction term of $X_{1} \times X_{2}$ is indeed related to the response and thus we can
conclude that its coefficient is non-zero, then it's generally immaterial if the coefficients for the main effects are
zero or non-zero. Their importance is implied due to the interaction term's correlation with $X_{1}$ and $X_{2}$, and
excluding the main terms can alter the meaning of the interaction.

Interaction terms can also be useful for qualitative variables as well as a combination of qualitative and quantitative
variables. In the case of the combination, we can think of the model with no interaction terms as describing multiple
parallel lines, one for each level of the qualitative variable. The lines will be parallel because the slope will only
be based on the coefficient for the quantitative predictor, with the qualitative predictor determining the different
intercepts for the factor levels. That is, the slope is the same for all levels of the factor since they do not affect
the coefficient. If an interaction term with the quantitative predictor applied to the quantitative predictor is added,
then the slopes of the lines can now differ between factor levels, possibly leading to non-parallel lines.

Accounting for Non-Linear Relationships through Polynomial Regression
---------------------------------------------------------------------

Another assumption of linear regression models is, of course, that the relationship between the response and the
predictors are linear. However, this may not be the case, obviously. Luckily, there is a very simple way to extend the
linear model to account for this, **polynomial regression**. Future sections will present more complex ways to perform
non-linear fits in more general situations.

For example, let's imagine fitting a model of the gas mileage of cars in response ($Y$) to the horsepower of cars as the
predictor ($X_{1}). If we were to look at the plotted data, we might notice a clear curve to the relationship, implying
that it is not a linear relationship. We might conclude that it may fit a **quadratic** model better:

$$
Y = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{1}^{2} + \epsilon.
$$

This formula uses a non-linear function of $X_{1}$, the horsepower, but still is a linear model, just a multiple linear
regression model with $X_{1}$ and $X_{2} = X_{1}^{2}$. This may indeed provide a better fit especially if the quadratic
term has high statistical significance.

One might then consider adding higher-order polynomial terms such as ones based on $X_{1}^{3}$ and $X_{1}^{4}$ and so
on. However, that may not necessarily make for a better fitting model for much the same reason that adding a quadratic
term to a model that *does* have a linear relationship would not improve it.

Other Potential Problems with Fitting Linear Regression Models
--------------------------------------------------------------

While we have discussed working around a few assumptions, there are many other issues that can be encountered when
fitting linear regression models to a data set. Let us go over a few common ones.

### Non-Linearity of the Data

We have already discussed working around non-linearity using polynomial regression, but it's worth talking about ways
to identify non-linearity. For simple linear regression, it can be useful to make a **residual plot** of the residuals
$e_{i} = y_{i} - \hat{y}_{i}$ versus the predictor $x_{i}$. For multiple linear regression, instead the residuals should
be plotted against the predicted or **fitted** values $\hat{y}_{i}$.

When looking at such plots, it is hoped that there is no discernible pattern to these plots. If there is such a pattern,
that may mean the data has non-linearity. As mentioned before, adding non-linear transformations of the predictors as
we did with polynomial regression may address these problems.

### Correlation of Error Terms

While we have been generally treating error terms as a constant completely independent of observations, it is more
accurate to consider that each observation as having its own error term,
$\epsilon_{1}, \epsilon_{2}, \ldots, \epsilon_{n}$. Further, an assumption of the linear regression model is that these
error terms are uncorrelated. That is, the sign of error term $\epsilon_{i}$ should not provide information about the
sign of the next error term $\epsilon_{i + 1}$. The standard errors that are computed are based on this assumption, and
if the assumption is violated, then the calculated standard errors will be smaller than the true standard errors. This
can lead to incorrectly narrower confidence intervals as well, or even too-low p-values implying significance.

As an example of how this might arise, imagine that data was duplicated so each observation was listed twice. Our
calculations of the standard errors will be based on a sample size of $2n$, but we only have $n$ samples. The estimated
parameters will be the same, but confidence intervals will be narrower by a factor of $\sqrt{2}$.

A more likely cause of correlated error terms could come from **time series** data where the measurements of the
observations are collected at discrete time points. Observations from adjacent time points will tend to be highly
positively correlated. This can be determined with a plot of the residuals of each observation over time. If we see
adjacent residuals tending to have similar values, this is **tracking** and a sign of correlated error terms.

Another example where correlated error terms could arise is from say, measuring the heights and weights of individuals.
If any of the individuals live together, perhaps eat the same diet, or have been exposed to the same environmental
factors, then the error terms for the observation of their weights may be highly correlated.

This assumption of uncorrelated errors is a very basic assumption of linear regression and other statistical methods,
and is best mitigated with good experimental design.

### Non-Constant Variance of Error Terms

Another assumption of linear regression models centering around error terms is that the variance of the error terms
are constant. If this is not true, it will effect calculating the standard errors, calculating confidence intervals, and
performing hypothesis testing. A possible example of a non-constant variance could be seen if the variance increases
as the response value increases, that is, higher response values have more variance. Such non-constant variances are
referred to as **heteroscedasticity**. This heteroscedasticity can be noticed on a residual plot showing the presence of
a funnel shape, where the magnitudes of the residuals increase or decrease with the fitted values.

One possible solution to this is to transform the response $Y$ using a concave function such as $\log{Y}$ or $\sqrt{Y}$
to reduce the heteroscedastic effects. Another possible solution exists if we have an idea of the variance of each
response. If we know the $i$th response was an average of $n_{i}$ raw observations and each of these raw observations
are uncorrelated with variance $\sigma^{2}$ then their average has variance $\sigma_{i}^{2} = \sigma^{2}/n_{i}$. We
can in this case fit the model with **weighted least squares**, where the observations are weighted with the inverse of
the variances. This is possible in most linear regression software.

### Outliers

When the measured response $y_{i}$ for an observation greatly differs from the predicted response of the model, it is
called an **outlier**. There are many reasons why such outliers can occur, such as error during data collection. Even
if an outlier does not affect the least squares fit predictions of the model, it can have a large effect on the standard
error, causing all related values such as confidence intervals and p-values to change.

Outliers will usually be clearly visible on a residual plot, but there is of course the question of how far away from
the other residuals a point needs to be to be considered an outlier. One way to make this easier to answer is to divide
each residual $e_{i}$ by its estimated standard error to get the **studentized residuals**. If the studentized residuals
have an absolute value greater than 3, it is a possible outlier.

Once an outlier is determined, if it is indeed due to error, it may suffice to remove it from the dataset. However,
sometimes such an outlier can point to a deficiency with the model such as a missing predictor. Care must be taken.

### High Leverage Points

Very similar to outliers, **high leverage** points refer to data points with unusual predictor values. Perhaps one
observation has a measurement that is far larger than the measurements for other observations. These high leverage
points tend to have a great effect on the least squares fit even to the degree of invalidating the fit.

While it is pretty straightforward to identify high leverage observations for any one predictor, in multiple linear
regression there may be an observation that has expected per-predictor measurements, but the combination of these
particular predictor measurements as a single observation could be unusual compared to other observations.

To make this easier we can calculate the **leverage statistic** for an observation, where a high value for this
statistic indicating a high leverage observation. For simple linear regression, the formula is:

$$
h_{i} = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum_{j = 1}^{n}(x_j - \bar{x})^2}.
$$

The further $x_i$ gets from $\bar{x}$, the higher the value of $h_i$. This formula can be extended to multiple
predictors, but it is not provided here. The leverage statistic is always between $1/n$ and $1$ with an average leverage
for all observations being equal to $(p + 1)/n$. If the leverage statistic for an observation exceeds $(p + 1)/n$, then
it is likely a high leverage point.

A point that is both a high leverage observation *and* an outlier can be especially dangerous.

### Collinearity

If two or more predictors are closely related, they have **collinearity**. For example, if we were trying to predict
how long employees may stay at a company based on their salary and seniority, then the predictors of salary and
seniority are likely highly related, and likely collinear.

This collinearity makes it hard to separate out the effects of each predictor in a linear regression model since they
tend to increase and decrease together. This can be demonstrated by comparing contour plots for different RSS values
plotted as a function of the coefficients. That is, for a given RSS value, what possible coefficients would work to
make a model with that RSS? For predictors without collinearity, these contour plots should cover a wide circle of
possible coefficients centered around the minimum RSS that we would find with least squares and disjoint from the
coefficients for other RSS values.

If instead the predictors *are* collinear, then such a contour plot will be "squashed" and very different coefficients
can be used to generate the same RSS values. This will cause the standard error to grow, which can cause the t-statistic
to decline making it more likely we would erroneously fail to reject the null hypothesis. This reduces the **power** of
the test, that is, the probability of correctly detecting a non-zero coefficient.

One simple way to identify collinearity is looking at the correlation matrix for all the predictors. If there is a high
absolute value in this matrix for any two predictors, this indicates highly correlated variables and thus a collinearity
problem. However, there may be cases where no pair of predictors are so correlated with each other, but that three or
more predictors may all be collinear *together*. This is called **multicollinearity**.

Multicollinearity may be better assessed by calculating the **variance inflation factor** (VIF). This is the ratio of
the variance of the coefficient found by fitting to the full model over the variance of the coefficient of fitting a
model only to the predictor. If the VIF is 1, its smallest value, then there is *no* collinearity. Normally, there is
some small amount of collinearity, so a good rule of thumb is to consider VIF values above 5 or 10 as a problematic
amount.

The formula for the VIF of each variable is:

$$
\mathrm{VIF}(\hat{\beta}_{j}) = \frac{1}{1 - R_{X_{j} \mid X_{-j}}^{2}},
$$

with $R_{X_{j} \mid X_{-j}}^{2}$ is the $R^2$ from a regression of $X_j$ onto all other predictors. The closer this
value is to 1, the more collinearity that is present, leading to a large VIF.

Once a problematic amount of collinearity is identified, there are a two simple solutions. The first would be to drop
one of the collinear predictors from the regression, which is usually not a problem for the fit because the very
presence of collinearity implies that one of the predictors is redundant. The second solution is to combine the
collinear predictors together into a single predictor, creating a new variable.

A Comparison of Linear Regression with K-Nearest Neighbors
----------------------------------------------------------

Due to its clear definition of a functional form, linear regression is a parametric approach. This gives some benefits.
There are a low number of parameters making it easy to fit. The parametric form for linear regression gives clear
interpretations of the coefficients. Tests of statistical significance are also easy to perform. However, there is a
very clear disadvantage: since there is a strong assumption about the form of the relationship, if that relationship
actually does not closely adhere to that assumption, the fit will be poor. Where parametric methods are disadvantaged
due to their assumptions, non-parametric forms become advantaged due to their lack of assumption about the form of the
relationship.

One simple non-parametric method that can serve as a useful comparison is **K-nearest neighbors regression** (KNN
regression). This is very close to the KNN classifier discussed earlier. Accepting a certain $K$ and a taking a point
to predict, $x_0$, we find the $K$ training observations closest to $x_0$, represented by $\mathcal{N}_0$. The estimate
for $f(x_0)$ is the average of all responses of $\mathcal{N}_0$, which in notation is:

$$
\hat{f}(x_0) = \frac{1}{K}\sum_{x_i \in \mathcal{N}_0}y_i.
$$

When $K$ is one, this will form a step-wise fit where the function is constant around each observation. As $K$
increases, the steps will be smoother but further away from the closest observation due to the effect of other
neighbors. Put another way, small values of $K$ give the most flexible fit to the observations with high variance
between the fit, but larger values for $K$ provide a smoothing effect that can have the downside of adding bias if the
structure of $f(X)$ is masked. This is part of the bias-variance tradeoff as discussed earlier. Methods for estimating
test error rates can be used to select an optimal value of $K$ and will be introduced in a later section.

Having looked at KNN regression, an obvious question is when would a non-parametric approach like KNN outperform a
parametric approach like linear regression? Essentially, if the parametric form selected is close to the true form, it
will do better than a non-parametric approach. For example, if data was taken from a clear linear relationship, then
linear regression will fit better. But when taken from a non-linear relationship, KNN regression will fit better.

Given that many interesting relationships are non-linear it may be hard to see why linear regression would ever be
preferred, but there is another issue with non-parametric methods like KNN regression, the **curse of dimensionality**.
This occurs as more predictors are added. As more dimensions are added, observations may end up being very far away from
other observations overall due to being very far away in a greater number of dimensions. This increased distance between
observations will lead to worse fit. This is especially true if the number of observations are much smaller than the
predictors.

Beyond that, linear regression might still be preferred for interpretability and the smaller number of parameters with
p-values.
