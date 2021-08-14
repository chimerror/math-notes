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
one out of a non-ranked set of **classes**.

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
    \end{array} \right)
$$

Each row of the matrix is one observation, and each column is a variable.

What Is Statistical Learning?
-----------------------------

Let's say we have an $n \times p$ matrix of $n$ observations $\mathbf{X}$. Each row can be thought of as a vector of $p$
variables each of which is $X_{i}$. The group of these variables, $X$ makes up our **input variables**. Let's also say
we have an **output variable** (also called a **response variable**), $Y$.

Now, let us assume there is some fixed but unknown function $f(X)$, that forms a relationship between $X$ and $Y$, such that:

\begin{equation}
    Y = f(X) + \epsilon\label{eq:1}
\end{equation}

Here $\epsilon$ is an **error term**, a random value independent of $X$, with mean zero. While $f$ is the **systematic**
information that $X$ provides about $Y$. Because $f$ takes arguments from the entire group of variables, $X$, it will
likely be a function of many variables. Thus, **statistical learning**, is a method for estimating what $f$ might be
based using $X$ as **training data**. That is we want to find $\hat{f}$, an estimate of $f$. Since the error term
$\epsilon$ has mean zero, we can use $\hat{Y}$ for a prediction of $Y$:

$$
    \hat{Y} = \hat{f}(X)
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
                   & = & |f(X) - \hat{f}(X)|^2 + \mathrm{Var}(\epsilon)
\end{eqnarray*}

Here $|f(X) - \hat{f}(X)|^2$ is the reducible error, while the **variance** of $\epsilon$, $\mathrm{Var}(\epsilon)$, is
the irreducible error. It is important to remember that while the reducible error can be minimized (which will be our
goal), the irreducible error will provide an upper bound for predictions of $Y$. What this bound actually is usually
unknown.

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
    \hat{f}(X) = \beta_{0} + \beta_{1}X_{1}+ \beta_{2}X_{2} + \ldots + \beta_{p}X_{p}
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
variables. Of course input variables can be either qualitative or quantitative, too. However in general, most methods
can be applied regardless of the input variable, as long as qualitative input variables are properly **coded**.
