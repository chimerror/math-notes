Notes on Linear Algebra
=======================

Introduction
------------

This is a series of notes on linear algebra originally created for my own edification. However, it may be
useful for others. It assumes very little pre-knowledge, but also does not go too deeply into areas that I feel are
reasonably well known. These topics will be covered in sections with the title "generally", to signal that it is a
very brief overview.

Systems of Linear Equations
---------------------------

Given $n$ variables, $x_{1}, \ldots, x_{n}$, a **linear combination** of the variables can be formed:

$$
a_{1}x_{1} + a_{2}x_{2} + a_{3}x_{3} + \cdots + a_{n}x_{n}
$$

where $a_{1}, \ldots, a_{n} \in \mathbb{R}$ are the **coefficients** of the combination. When a linear combination is
set equal to some **constant** $d \in \mathbb{R}$:

$$
a_{1}x_{1} + a_{2}x_{2} + a_{3}x_{3} + \cdots + a_{n}x_{n} = d
$$

a single **linear equation** is formed. An $n$-tuple $(s_{1}, s_{2}, \ldots, s_{n})$ **satisfies** a linear equation if
replacing the variables ($x_{1}, \ldots, x_{n}$) with the corresponding $s_{1}, \ldots, s_{n}$ results in a true
statement. That is if:

$$
a_{1}s_{1} + a_{2}s_{2} + a_{3}s_{3} + \cdots + a_{n}s_{n} = d
$$

is a true statement, the $n$-tuple is a **solution** of the linear equation. Furthermore, given $m$ linear equations:

\begin{eqnarray*}
    a_{1,1}x_{1} + a_{1,2}x_{2} + a_{1,3}x_{3} + \cdots + a_{1,n}x_{n} & = & d_{1} \\
    a_{2,1}x_{1} + a_{2,2}x_{2} + a_{2,3}x_{3} + \cdots + a_{2,n}x_{n} & = & d_{2} \\
    a_{3,1}x_{1} + a_{3,2}x_{2} + a_{3,3}x_{3} + \cdots + a_{3,n}x_{n} & = & d_{3} \\
     & \vdots & \\
    a_{m,1}x_{1} + a_{m,2}x_{2} + a_{m,3}x_{3} + \cdots + a_{m,n}x_{n} & = & d_{m} \\
\end{eqnarray*}

a **system of linear equations** is formed. If an $n$-tuple satisfies all equations in the system, that is if:

\begin{eqnarray*}
    a_{1,1}s_{1} + a_{1,2}s_{2} + a_{1,3}s_{3} + \cdots + a_{1,n}s_{n} & = & d_{1} \\
    a_{2,1}s_{1} + a_{2,2}s_{2} + a_{2,3}s_{3} + \cdots + a_{2,n}s_{n} & = & d_{2} \\
    a_{3,1}s_{1} + a_{3,2}s_{2} + a_{3,3}s_{3} + \cdots + a_{3,n}s_{n} & = & d_{3} \\
     & \vdots & \\
    a_{m,1}s_{1} + a_{m,2}s_{2} + a_{m,3}s_{3} + \cdots + a_{m,n}s_{n} & = & d_{m} \\
\end{eqnarray*}

is a set of true statements, then it is a **solution** of the system (in addition to being a solution for each
individual equation). Finding the (possibly empty or infinite) **solution set** of all possible solutions of a system is
referred to as **solving** the system of equations.

Gauss's Method
--------------

While there are many methods and algorithms for solving systems of linear equations these notes focus on
**Gaussian elimination** or **Gauss's Method**. Gaussian elimination continually applies one of three operations to
transform one linear system to another, until a system that can be trivially solved is found.

The Gaussian elimination operators are:

1. Swapping Equations: swapping an equation with another equation
2. Scaling Equations: multiplying an equation by a non-zero constant on both sides
3. Combining Equations: combining an equation with a scaled version of another equation through summation

Obviously $1$ is a possible scaling factor for the third equation combining operator, as well as the second equation
scaling operator (though perhaps equally obviously, scaling an equation by $1$ does not actually change the system).

For this to work we need to choose operations that have been proven to preserve the solution set. That is, given a
linear system and its solution set, transforming the linear system through an operator should produce a new linear
system with the exact same solution set.

### Theorem: Swapping Equations Preserves the Solution Set

> *Transforming a system of equations by swapping two equations for each other produces a new system of equations with
> the same solution set.*

For a given linear system:

\begin{eqnarray*}
    a_{1,1}x_{1} + a_{1,2}x_{2} + a_{1,3}x_{3} + \cdots + a_{1,n}x_{n} & = & d_{1} \\
     & \vdots & \\
    a_{m,1}x_{1} + a_{m,2}x_{2} + a_{m,3}x_{3} + \cdots + a_{m,n}x_{n} & = & d_{m} \\
\end{eqnarray*}

its solution set is the set of all $n$-tuples $(s_{1}, s_{2}, \ldots, s_{n})$ such that substituting the $s$ values for
the system's $x$ values produces a set of true statements:

\begin{eqnarray*}
    a_{1,1}s_{1} + a_{1,2}s_{2} + a_{1,3}s_{3} + \cdots + a_{1,n}s_{n} & = & d_{1} \\
     & \vdots & \\
    a_{m,1}s_{1} + a_{m,2}s_{2} + a_{m,3}s_{3} + \cdots + a_{m,n}s_{n} & = & d_{m} \\
\end{eqnarray*}

Note how each equation is paired with the corresponding statement.

To perform the swap, choose some $i,j \in [1, m]$ such that $i > j$ where the $i$th and $j$th equations are:

\begin{eqnarray*}
     & \vdots & \\
    a_{i,1}x_{1} + a_{i,2}x_{2} + a_{i,3}x_{3} + \cdots + a_{i,n}x_{n} & = & d_{i} \\
     & \vdots & \\
    a_{j,1}x_{1} + a_{j,2}x_{2} + a_{j,3}x_{3} + \cdots + a_{j,n}x_{n} & = & d_{j} \\
     & \vdots & \\
\end{eqnarray*}

These are the equations that will be swapped, with corresponding true statements:

\begin{eqnarray*}
     & \vdots & \\
    a_{i,1}s_{1} + a_{i,2}s_{2} + a_{i,3}s_{3} + \cdots + a_{i,n}s_{n} & = & d_{i} \\
     & \vdots & \\
    a_{j,1}s_{1} + a_{j,2}s_{2} + a_{j,3}s_{3} + \cdots + a_{j,n}s_{n} & = & d_{j} \\
     & \vdots & \\
\end{eqnarray*}

Note that assuming $i > j$ is sufficient because we do not allow swapping an equation with itself (the case of $i = j$)
and if $j > i$, we can just instead treat $j$ as $i$ and vice-versa. When the equations are swapped:

\begin{eqnarray*}
     & \vdots & \\
    a_{j,1}x_{1} + a_{j,2}x_{2} + a_{j,3}x_{3} + \cdots + a_{j,n}x_{n} & = & d_{j} \\
     & \vdots & \\
    a_{i,1}x_{1} + a_{i,2}x_{2} + a_{i,3}x_{3} + \cdots + a_{i,n}x_{n} & = & d_{i} \\
     & \vdots & \\
\end{eqnarray*}

the corresponding statements will also be swapped:

\begin{eqnarray*}
     & \vdots & \\
    a_{j,1}s_{1} + a_{j,2}s_{2} + a_{j,3}s_{3} + \cdots + a_{j,n}s_{n} & = & d_{j} \\
     & \vdots & \\
    a_{i,1}s_{1} + a_{i,2}s_{2} + a_{i,3}s_{3} + \cdots + a_{i,n}s_{n} & = & d_{i} \\
     & \vdots & \\
\end{eqnarray*}

While the order has changed, this does not produce a different set of statements that must be true of a solution. Each
original statement matches one-to-one with a statement in the swapped set. Thus, any solution that satisfies one set of
statements must satisfy the other and thus be a solution for both systems. As this is the case for all solutions in the
solution set of the original system and all solutions in the transformed system, they must be the same
set.$\blacksquare$
