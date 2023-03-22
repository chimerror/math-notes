---
documentclass: report
fontfamily: heuristica
---

Notes on Mathematical Structure and Proof
=========================================

Introduction
------------

This is a series of notes on mathematical structures and proofs originally created for my own edification. However, it
may be useful for others. It assumes very little pre-knowledge, but also does not go too deeply into areas that I feel
are reasonably well known. These topics will be covered in sections with the title "generally", to signal that it is a
very brief overview.

Mathematical Statements
-----------------------

As practiced by mathematicians, mathematics is ultimately about making **mathematical statements** and demonstrating
their validity. That is, a declarative sentence about mathematical concepts. These statements may be either true (often
signified by $\top$) or false (often signified by $\bot$). For example:

> All even numbers greater than $2$ are not prime numbers.

is a true statement, due to all such numbers being a multiple of $2$. However:

> All odd numbers are prime numbers.

is a false statement, as shown by the counterexample of $9$, which is odd, but is a multiple of $3$ so it is not prime.

Connectives
-----------

Statements can be combined using the familiar natural language **connectives** of "and" and "or". However, there are
some technicalities. Using "and" in a statement with statements $P$ and $Q$:

> $P$ and $Q$.

means that both $P$ and $Q$ must be true for the entire statement to be true. This matches the use of "and" in natural
language. However when using "or" with statements $P$ and $Q$:

> $P$ or $Q$.

means that either *or both* of $P$ and $Q$ must be true. This type of use of "or" is referred to as the
**inclusive or**. By allowing both $P$ and $Q$ to be true, the inclusive or is distinguished from the **exclusive or**
as it is some times used in natural language, that is, allowing *either* $P$ or $Q$ to be true, but not *both*. This
truth table summarizes the difference:

| Type of "or" | $P$    | $Q$    | $P$ or $Q$ |
|--------------|--------|--------|------------|
| inclusive    | $\top$ | $\top$ | $\top$     |
| exclusive    | $\top$ | $\top$ | $\bot$     |
| inclusive    | $\top$ | $\bot$ | $\top$     |
| exclusive    | $\top$ | $\bot$ | $\top$     |
| inclusive    | $\bot$ | $\top$ | $\top$     |
| exclusive    | $\bot$ | $\top$ | $\top$     |
| inclusive    | $\bot$ | $\bot$ | $\bot$     |
| exclusive    | $\bot$ | $\bot$ | $\bot$     |

Note that except for $P$ and $Q$ both being true (the first two rows), inclusive and exclusive or have the same results.
With inclusive or $P$ and $Q$ being true is also true, but with exclusive or, the entire statement is false.
