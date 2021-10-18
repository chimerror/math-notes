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
individual equation).
