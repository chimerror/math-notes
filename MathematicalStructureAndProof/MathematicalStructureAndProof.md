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

means that either or *both* of $P$ and $Q$ must be true. This type of use of "or" is referred to as the
**inclusive or**. By allowing both $P$ and $Q$ to be true, the inclusive or is distinguished from the **exclusive or**
as it is some times used in natural language, that is, allowing *either* $P$ or $Q$ to be true, but *not* both. This
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

Sets, Generally
---------------

Rather than formally define set theory, let's use an intuitive description of a **set** as a collection of mathematical
objects, made up of **elements**. That is, given a set $X$, and an object $x$ in $X$, we can say $x$ is an **element**
of $X$. This can be written shorter as $x \in X$. Likewise, an object $x$ that is *not* an element of $X$ could be
written as $x \notin X$.

Defining Sets, Generally
------------------------

### Set Definition through Listing

Sets can be defined in many different ways. For example, a particularly small set could be defined by listing:

$$
X = \{ 2, 3, 5, 7, 11 \}.
$$

A set defined through a listing may skip some implied elements using $\ldots$:

$$
X = \{ 1, 3, 5, 7, \ldots, 97, 99 \}
$$

thus defining $X$ as the set of odd integers less than 100. This may even be used to define an infinite set:

$$
\mathbb{N} = \{ 0, 1, 2, 3, 4, \ldots \}.
$$

(This definition is actually one we shall adopt as our first well-known set, the natural numbers, $\mathbb{N}$.)

### Set Definition through Conditions or Properties

Sets can also be defined in relation to other sets, particularly the well-known sets of numbers defined in the
next section. Most precisely, from a given set $X$ and a given condition or property on $X$, $P(x)$, we can define a set
$Y$:

$$
Y = \{ x \in X \mid P(x) \}.
$$

Often, $P(x)$ is defined as a mathematical formula that involves $x$, its variable from the domain $X$. For example, if
$P(x)$ was the formula "$x^3 = 27$", then the membership of members of $X$ in the defined set $Y$ can be determined by
substituting them into this formula. $P(3)$ thus corresponds with the true statement $3^3 = 27$ after substitution of
$3$ for $x$. In this case $P(x)$ is said to "hold" at $3$ and thus, $3$ would also be a member of $Y$. If instead, we
consider $P(2)$, for which substitution of $x$ produces the false statement $2^3 = 27$, then $P(x)$
does not "hold" at $2$ and thus $2$ would *not* be a member of $Y$.

### Other Set Definition Notations

Two more types of notation for defining sets will be useful for these notes.

First, we can also define the set of positive members of a given set $X$ using the notation $X^+$:

$$
X^+ = \{ x \in X \mid x > 0 \}.
$$

(Note that the term "positive" in regard to numbers is defined below.)

Second, we can define the set of all natural numbers ($\mathbb{N}$, defined in the later section on well-known sets)
that are lower than a given natural number $n$:

$$
\lceil n \rceil = \{ x \in \mathbb{N} \mid x \geq 0 \text{ and } x < n \},
$$

which can be also be defined through listing:

$$
\lceil n \rceil = \{ 0, 1, 2, 3, \ldots, n - 1 \}.
$$

(Further methods of defining sets through set operations will be given later in the section on set operations.)

Set Equality through Extensionality
-----------------------------------

The nature of equality within mathematics can be understood as implying that two descriptions of a mathematical object
describe the same object. This is worth mentioning because it is important to regard equality not as a relation between
*objects* but between ways to express that object, that is, *expressions*.

Let us take this simple equality:

$$
8 - 6 = 2.
$$

On each side of the equals sign $=$, is an expression that describes a mathematical object. So for these to be
equal, we are claiming that the number represented by $8 - 6$, an arithmetic expression, is the same number represented
by the numeral $2$, which is also an expression. And in this case, this is a true claim given the common definitions of
simple arithmetic.

But note that our definitions and the context of these statements of equality make a big difference on the validity of
the equality. For example, let's say we have the set equation:

$$
\{ \text{ Bruce Wayne}, \text{ Batman } \} = \{ \text{ Batman } \}.
$$

If by "$\text{Bruce Wayne}$" and "$\text{Batman}$" we mean the fictional character who patrols Gotham City fighting
crime, this equality is true, as those two names refer to the same person: that famous caped crusader. Even though it
looks like there are two elements in the set, they both refer to the same person, so there's really only one. And that
one person is in both sets, so both sets are equal.

However, we can also consider what if "$\text{Bruce Wayne}$" and "$\text{Batman}$" instead meant the strings of text
of those characters, that is, as just the letters in them, not who they might refer to. In that case, this equality
would *not* be true as the string "$\text{Bruce Wayne}$"  does not appear in the second set, even though the string
"$\text{Batman}$" *does* appear in both sets.

This notion of equality, particularly with sets, is referred to as **extensionality**, and thus we can more formally
define **set equality**. That is, given sets $X$ and $Y$, $X = Y$ if and only if that every $x \in X$ is also an element
of $Y$, *and* that every $y \in Y$ is also an element of $X$.

The Empty Set
-------------

An important set with what may seem like a trivial definition is the set with no elements, that is, the
**empty set**, which is denoted by $\emptyset$. This is due to the emptiness of particular sets often being a very
important mathematical question.

Set Relations
-------------

If we were limited to equality and membership, there would not be many statements we could make about sets. So it will
be useful to define ways that sets may relate to each other.

### Subsets and Supersets

Let us begin with the definition of **subsets**, where given sets $X$ and $Y$, $X$ is a subset of $Y$ if every element
of $X$ is also an element of $Y$. Note that there may be elements of $Y$ that are *not* elements of $X$. The subset
relation is written in symbols as $X \subseteq Y$.

We can further define the inverse **superset** relation, where once again given sets $X$ and $Y$ such that
$X \subseteq Y$, $Y$ is a superset of $X$, which is written as $Y \supseteq X$.

Note that a particular consequence of this definition of subset is that $\emptyset \subseteq X$ for any set $X$. As
$\emptyset$ has no elements, it trivially satisfies the requirement that all elements are an element of the set it is a
subset of.

### Set Equality through Subsets

These relations allow an alternate definition of **set equality**, in that $X = Y$ holds for sets $X$ and $Y$ if both
$X \subseteq Y$ and $Y \subseteq X$. This is particularly useful as it is often easier to verify the separate claims of
$X \subseteq Y$ and $Y \subseteq X$ rather than demonstrating that $X = Y$ all at once.

### Proper Subsets and Supersets

We can further define **proper subsets**, where given sets $X$ and $Y$, $X$ is a proper subset of $Y$ if
$X \subseteq Y$ and $X \neq Y$. That is, that there exist some elements of $Y$ that are *not* elements of $X$. This also
defines $Y$ as a **proper superset** of $X$. These relations can be written as $X \subsetneq Y$ and $Y \subsetneq X$.

Set Operations
--------------

Another way to generate new sets from existing sets (for example, from the well-known sets we will define below), is
through set operations, which describe a resulting set from two (or more) other sets.

First, we shall define the union of two sets. Given sets $X$ and $Y$, the **union** of $X$ and $Y$, or $X \cup Y$ is
defined by:

$$
X \cup Y = \{ x \mid x \in X \text{ or } x \in Y \}.
$$

Note, as per our discussion of the mathematical use of the word "or" to signify inclusive or, this includes $x$ such
that $x \in X$ *and* $x \in Y$.

If we wish to talk about only the shared members of $X$ and $Y$, we can refer to the intersection of the two sets. That
is, given sets $X$ and $Y$, the **intersection** of $X$ and $Y$, or $X \cap Y$ is defined by:

$$
X \cap Y = \{ x \mid x \in X \text{ and } x \in Y \}.
$$

Building off this definition, if the intersection of two sets $X$ and $Y$ is the empty set, that is,
$X \cap Y = \emptyset$, then the two sets are said to be **disjoint**.

We can further talk about the members of a set that are unique to it compared to another set by the set difference of
two sets. Given sets $X$ and $Y$, the **set difference** between them, or $X \setminus Y$ is defined as:

$$
X \setminus Y = \{ x \in X \mid x \notin Y \}.
$$

When in the context of a set which is the subset of another set, the set difference between the superset and the set
which is a subset is called the complement of that set. That is, given set $X$ and $U$ such that $X \subseteq U$, the
**complement** is defined as the set $U \setminus X$. When the set $U$ is understood by context, the complement of $X$
may be written as $X^c$.

Set operations, unlike their arithmetic counterparts do not have a standard order of operations. That is, when given
$5 \cdot 6 + 4$, the standard of order of operations requires the answer to be $34$, not $50$. However, given disjoint
non-empty sets $X$ and $Y$, $X \cup Y \cap X$ could be read as either $(X \cup Y) \cap X$, which is equivalent to
$Y \cap X$ or as $X \cup (X \cap Y)$ which is equivalent to $X$.

As such, it is very important that parentheses be used to denote the intended order of operations.

Ordered Tuplets, Generally
--------------------------

As a quick aside in preparation for the next set operation, let us describe another type of collection of mathematical
objects. An **ordered pair** of mathematical objects can be described as exactly that, the two objects combined in a
particular order. One example of such a pair would be $(3,4)$, which combines $3$ as the first element and $4$ as the
second. Note, that due to the pair being ordered, the elements of the pair are compared *in order* (differing from sets,
which are non-ordered). That is, $(3,4) \neq (4,3)$.

More than two objects can be combined in such way expanding the idea into **ordered tuplets**. For example, you may have
an **ordered triplet** of $(9, 8, 7)$.

The Direct Product of Sets
-----------------------------

We'll use this new collection to define an operation on sets which returns a set of ordered pairs. The
**direct product** or **Cartesian product** of two given sets, $X$ and $Y$ is represented by $X \times Y$ and is defined
as:

$$
X \times Y = \{ (x, y) \mid x \in X \text{ and } x \in Y \}.
$$

So for example if we let set $X$ be defined as:

$$
X = \{ 4, 5, 6 \}
$$

and set $Y$ be defined as:

$$
Y = \{ 4, 5 \}.
$$

Then:

$$
X \times Y = \{ (4, 4), (4, 5) (5, 4), (5, 5) (6, 4), (6, 5) \}.
$$

It is, of course, possible to apply the above formula of direct product to a third set. That is given sets $X$, $Y$, and
$Z$, then:

$$
(X \times Y) \times Z = \{ ((x, y), z) \mid x \in X, x \in Y, x \in Z \}.
$$

Note that by our use of parentheses, $(X \times Y) \times Z \neq X \times (Y \times Z)$, formally, as
$((x, y), z) \neq (x, (y, z))$. However, for most applications this difference is not important, and thus we, following
most mathematicians, can instead disregard the parentheses here and write $X \times Y \times Z$, interpreting
$(X \times Y) \times Z$ and $X \times (Y \times Z)$ as equivalent definitions of the same set of ordered triplets (such
as $(x, y, z)$).

This can be further generalized to produce a definition of the direct product of an arbitrary amount of sets. That is,
given $n$ sets, $X_{1}, X_{2}, \ldots, X_{n}$, the **direct product** of these sets, $\prod_{i=1}^{n}X_{i}$, is the set

$$
\prod_{i=1}^{n}X_{i} = \{ (x_{1}, x_{2}, \ldots, x_{n}) \mid x_{i} \in X_{i}, 1 \leq i \leq n \}.
$$

(Note the slight difference in how the direct product is written when inline and displayed.)

The direct product of the same set $X$ with itself $n$ times can be abbreviated as $X^{n}$, matching the representation
of exponentiation of numbers.

Numbers, Generally
------------------

A formal definition of a number besides a member of one of the well-known sets of numbers defined later is out of scope
for these notes. However, it will help to define some basic properties of numbers.

In particular numbers can be classified based on their value compared to 0. Let's say we have number $x$. If $x > 0$
then $x$ is defined as **positive**, and if, instead, $x < 0$ then $x$ is defined as **negative**. Note that $0$ itself
is neither positive nor negative.

Extending from that, that if $x \geq 0$ then $x$ is defined as **non-negative** and if, instead, $x \leq 0$ then $x$ is
defined as **non-positive**.

Well-Known Sets of Numbers
--------------------------

As part of avoiding the formalism of set theory, we shall work from a few well-known sets of numbers. For example, the
set of **natural numbers**, $\mathbb{N}$, is defined by:

$$
\mathbb{N} = \{ 0, 1, 2, 3, 4, \ldots \}.
$$

Note that $\mathbb{N}$ *does* include $0$ as a member. If instead we want to refer to the set of **positive integers**
we will use $\mathbb{N}^+$, defined as:

$$
\mathbb{N}^+ = \{ 1, 2, 3, 4, \ldots \}.
$$

When $\mathbb{N}^+$ is combined with $0$ and the negative integers, we get $\mathbb{Z}$, all of the **integers**:

$$
\mathbb{Z} = \{ \ldots, -4, -3, -2, -1, 0, 1, 2, 3, 4, \ldots \}.
$$

From $\mathbb{Z}$, we can further define the **rational numbers**, $\mathbb{Q}$:

$$
\mathbb{Q} = \{ \frac{p}{q} \text{ where } p,q \in \mathbb{Z} \text{ and } q \neq 0 \}.
$$

Our last well-known set of numbers we will work is known as the **real numbers** and is represented by $\mathbb{R}$. A
formal definition of this set requires a bit more mathematical development than we currently have. Later sections of
this document will provide a full definition of the real numbers starting from the natural numbers, so we will defer
the definition until then.

As an intuitive description the real numbers, $\mathbb{R}$, can be seen as the combination of the rational numbers,
$\mathbb{Q}$, with another set of numbers that we will defer a formal definition of, the **irrational numbers**. A
rough circular intuitive description of the irrational numbers are those numbers of the real numbers that *cannot* be
expressed as a ratio, and therefore are *not* a member of $\mathbb{Q}$. That is:

$$
\mathbb{R} \setminus \mathbb{Q} = \{ x \in \mathbb{R} \mid x \notin \mathbb{Q} \}.
$$

Intervals
---------

When working with the real numbers, $\mathbb{R}$, many non-empty subsets can be described as **intervals**. That is,
given a **bounded interval** $I$ with endpoints $a, b \in \mathbb{R}$ such that $a < b$, $c \in \mathbb{R}$ is in the
interval $I$ if and only if $a < c < b$. There are four forms of bounded intervals, which differ based on whether
neither, either, or both of their endpoints $a$ and $b$ are also members of the interval:

\begin{align*}
    (a, b) & = \{ x \in \mathbb{R} \mid a < x < b \} \\
    [a, b) & = \{ x \in \mathbb{R} \mid a \leq x < b \} \\
    (a, b] & = \{ x \in \mathbb{R} \mid a < x \leq b \} \\
    [a, b] & = \{ x \in \mathbb{R} \mid a \leq x \leq b \}.
\end{align*}

Note in the last case, rather than requiring that $a < b$, we also allow for $a$ and $b$ being equal, that is
$a \leq b$.

When we allow either or both of the endpoints to be $-\infty$ or $\infty$, we can describe **unbounded intervals**,
which also have four different forms based on an endpoint $b \in \mathbb{R}$:

\begin{align*}
    (-\infty, b) & = \{ x \in \mathbb{R} \mid x < b \} \\
    (-\infty, b] & = \{ x \in \mathbb{R} \mid x \leq b \} \\
    (b, \infty) & = \{ x \in \mathbb{R} \mid x > b \} \\
    [b, \infty) & = \{ x \in \mathbb{R} \mid x \geq b \}.
\end{align*}

We can further define an interval as **closed** if it contains all of its endpoints. Given the examples above, $[a, b]$,
$(-\infty, b]$, and $[b, \infty)$ all are closed intervals. Likewise, an interval is **open** if it contains none of its
endpoints. For our examples, $(a, b)$, $(-\infty, b)$, and $(b, \infty)$ all are open intervals.

Quite curiously, when regarded as an interval, $\mathbb{R}$ (or $(-\infty, \infty)$) is *both* an open and closed
interval. It has zero endpoints, so all endpoints are included, since there are none to include. And none of the
endpoints are included, since there are no endpoints to include.
