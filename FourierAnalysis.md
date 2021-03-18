Notes on Fourier Analysis
=========================

Introduction
------------

This is a series of notes on Fourier Analysis originally created for my own edification. However, it may be useful for
others. It assumes very little pre-knowledge, but also does not go too deeply into areas that I feel are reasonably
well known. These topics will be covered in sections with the title "generally", to signal that it is a very brief
overview.


Numbers, Generally
------------------

For our purposes, we are most concerned with the set of real numbers, $\mathbb{R}$. This includes all integers, all
rational numbers, all irrational numbers. The strict definition of $\mathbb{R}$ is the set of numbers that can be
represented as a possibly infinite, non repeating, decimal expansion. Another way to think of it is that it represents
signed distance along the number line.

Sets, Generally
---------------

A set is a possibly empty collection of objects, usually for our purposes, numbers. That is rather than using the entire
set defined by $\mathbb{R}$, we use a subset of them. This subset does not have to be contiguous:

$$
S
$$

Functions, Generally
--------------------

A *function* is a mapping from its *domain* to its *range*. The domain and range are sets of numbers, for our purposes, drawn from $\mathbb{R}$, the set of all real numbers.

A function may defined by specifying the domain, along with a *formula* that defines the range given an element of the domain. For example:

$$
f(x)=x\cdot\sin(x), \forall x\in\mathbb{R}
$$

In this case, the domain is $\mathbb{R}$, as defined following the comma. Usually if the domain is the entirety of $\mathbb{R}$ like this, it will not be explicitly mentioned. This is also the case for well-known functions, such as $\tan(x)$ and other trigonometric functions.

The range, then, is the set of possible values when a member of the domain is substituted into the formula. In the above case, this is also the entirety of $\mathbb{R}$. However if instead we had:

$$
g(x)=5\sin(x)
$$

As the possible values for $\sin(x)$ vary from 1 at the most, and -1 at the least, its range is $[-1,1]$. Multiplying by 5 widens the range of $g(x)$ to $[-5,5]$.