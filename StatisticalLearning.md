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
