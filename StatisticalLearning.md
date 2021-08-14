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

A data set useful for statistical learning is made of several *observations* of several *variables*.

### Observations

An *observation* is an individual row of a data set, which could represent a single subject or a single survey
response or a single month of data. Its exact meaning will vary based on the data set in question.

### Variables

Observations are made up of one or more *variables*, the columns of the data set, which could represent the
measurements or values for that particular observation. The exact meaning of variables will vary based on the data
set in question.

### Quantitative and Qualitative Variables

As a computational process, all variables are represented as numbers, either integers or real numbers. However,
variables are not just *quantitative*, where the number itself represents the variable. They can also be
*qualitative*, representing a non-numeric value. Examples of qualitative variables include: which neighborhood of a
city the observation is from, sex, and race. Any such variable that represents one out of a non-ranked set.

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
variables each of which is $X_{i}$. The group of these variables, $X$ makes up our *input variables*. Let's also say we
have an *output variable*, $Y$.

Now, let us assume there is some fixed but unknown function $f(X)$, that forms a relationship between $X$ and $Y$, such that:

\begin{equation}
    Y = f(X) + \epsilon\label{eq:1}
\end{equation}

Here $\epsilon$ is an *error term*, a random value independent of $X$, with mean zero. While $f$ is the *systematic*
information that $X$ provides about $Y$. Because $f$ takes arguments from the entire group of variables, $X$, it will
likely be a function of many variables. *Statistical learning*, then, is a method for estimating what $f$ might be based
from $X$.

Prediction and Inference
------------------------

There are two competing reasons to use statistical learning, prediction and inference. These goals often require
different methods to be reached.

### Prediction

With *prediction*, our major concern is generating the values of $\hat{Y}$, an estimate of $Y$, where $\hat{Y}$ is
defined as:

$$
    \hat{Y} = \hat{f}(X)
$$

Where $\hat{f}$ is our estimate for $f$. And $\hat{Y}$ should be an useful estimate of $Y$ since $\epsilon$ has mean
zero.  In this case we're not concerned with the exact form of $\hat{f}$, just that its predictions of $Y$ are accurate,
making $\hat{f}$ a "black box".

In this case, there are two quantities that affect the accuracy of $\hat{Y}$ as a prediction. The first,
*reducible error* is due to the differences between $\hat{f}$ and $f$. Using statistical learning, we can attempt to
reduce this error by finding more accurate estimates of $\hat{f}$ for $f$.

However, as noted in equation $\eqref{eq:1}$, $Y$ also depends on $\epsilon$. $\epsilon$ cannot be reduced by improving
$\hat{f}$ to better estimate $f$, thus it is *irreducible error*. Despite $\epsilon$ having mean zero, $\epsilon$ may
contain unmeasured or even unmeasurable variables that affect $Y$. Thus, irreducible error is not guaranteed to be zero.

Assuming $\hat{f}$ and $X$ are fixed, we can calculate the average, or *expected value* of the squared difference
between $Y$ and $\hat{Y}$:

\begin{eqnarray*}
    E(Y-\hat{Y})^2 & = & E|f(X) + \epsilon - \hat{f}(X)|^2 \\
                   & = & |f(X) - \hat{f}(X)|^2 + \mathrm{Var}(\epsilon)
\end{eqnarray*}

Here $|f(X) - \hat{f}(X)|^2$ is the reducible error, while the *variance* of $\epsilon$, $\mathrm{Var}(\epsilon)$, is
the irreducible error. It is important to remember that while the reducible error can be minimized (which will be our
goal), the irreducible error will provide an upper bound for predictions of $Y$. What this bound actually is usually
unknown.

### Inference

Comparatively, with *inference*, our major concern is the relationship between the individual variables of $X$,
$X_{1},\ldots,X_{p}$, and $Y$. That is how does $Y$ change as $X$ changes. In this case $\hat{f}$ cannot be treated as a
black box because its exact form is exactly what we're looking for.

Inference can be concerned with the existences of relationships, the directions and sizes of relationships, or even if
the relationships can be summarized with a linear equation. This means inference tends to favor more interpretable
models for $\hat{f}$ even if that leads to less accurate predictions for $Y$.
