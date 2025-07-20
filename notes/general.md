# Notes

## Non-locality

### LHVM (Local Hidden Variable Model)
LHVM (Local Hidden Variable Model) is a model that describes the behavior of a system in which the outcomes of measurements are determined by hidden variables that are local to each measurement device.
 In the simplest bipartite case with measurement settings $x,y$ and outcomes $a,b$, an LHVM assumes the existence of a shared “hidden” variable $\lambda$, distributed according to some probability density $\rho(\lambda)$, such that the joint outcome probabilities factorize as

$$
P(a,b\mid x,y)
=\int d\lambda\;\rho(\lambda)\;P_A(a\mid x,\lambda)\;P_B(b\mid y,\lambda)\,.
$$

Here

* $\rho(\lambda)\ge0$ and $\int d\lambda\,\rho(\lambda)=1$,
* $P_A(a\mid x,\lambda)$ is Alice’s response function (the probability she outputs $a$ when measuring $x$, given $\lambda$),
* $P_B(b\mid y,\lambda)$ is Bob’s response function,
  and crucially neither $P_A$ nor $P_B$ depends on the *other* party’s choice of measurement, enforcing **locality**.

In the general $n$-party scenario, this extends to

$$
P(a_1,\dots,a_n\mid x_1,\dots,x_n)
=\int d\lambda\;\rho(\lambda)\;\prod_{i=1}^n P_i\bigl(a_i\mid x_i,\lambda\bigr)\,,
$$

which defines the **local polytope** of all correlations admitting an LHVM.  Any observed correlations lying outside this polytope—i.e.\ violating a Bell inequality—are thus **non-local**, in that they cannot be decomposed in the above form.

This **factorization**—often termed the *locality* condition—**forbids** any dependence of Alice’s outcome distribution on Bob’s setting (and vice versa), and is precisely what rules out any possibility of signalling (even in principle) faster than light.  In particular, it implies the no-signalling constraints

$$
P(a\mid x,y)\;=\;\sum_bP(a,b\mid x,y)\;=\;\sum_bP(a,b\mid x,y')\;=\;P(a\mid x)
\quad\forall\,y,y'
$$

(and similarly for $P(b\mid x,y)$), so that neither party can influence the other’s marginals by choice of measurement .


### Violations of LHVM
A Local Hidden-Variable Model (LHVM) can only reproduce correlations of the form

$$
P(a,b\mid x,y)
=\!\int d\lambda\,\rho(\lambda)\,P_A(a\mid x,\lambda)\,P_B(b\mid y,\lambda)\,,
$$

with each party’s response function independent of the other’s setting (the “locality” or no-signalling condition) .


One can perform a **Bell test** to **violate** an LHVM, you must observe measurement statistics $P(a,b\mid x,y)$ that **cannot** be written in the above factorized form—i.e.\ that lie **outside** the so-called *local polytope*.  Concretely, one picks a particular Bell inequality (a linear combination of the $P(a,b\mid x,y)$ with a known *classical bound*), measures on an entangled quantum state, and finds a value **above** that bound.  The canonical example is the **CHSH** inequality, which in LHVMs satisfies

$$
S \;=\; E_{00} + E_{01} + E_{10} - E_{11}
\;\le\; 2,
$$

where
$\displaystyle E_{xy} = \sum_{a,b=\pm1} a\,b\,P(a,b\mid x,y)$.  Quantum mechanics—e.g.\ measuring a singlet state at settings $0°,45°$ for Alice and $22.5°,67.5°$ for Bob—yields

$$
S = 2\sqrt2 \;>\; 2,
$$

thus **violating** the LHVM bound and demonstrating non-locality .


Within the tropical-tensor-network, such a Bell inequality is encoded as a cost function $H(x)=\sum_I f_I(x_I)$ whose **tropical** contraction in LHVM (min-sum) algebra yields the **classical bound** $\beta$ .  A quantum violation means one experimentally observes correlators whose linear combination exceeds this $\beta$, proving that no assignment of deterministic strategies (no choice of hidden variable $\lambda$) can reproduce the data.

---

So, to **violate** an LHVM you:

1. **Choose** a Bell inequality and compute its classical bound $\beta$ via the LHVM (e.g.\ $\beta_{\rm CHSH}=2$).
2. **Prepare** an entangled quantum state (e.g.\ a singlet).
3. **Measure** in settings that maximize the Bell expression (e.g.\ the 0°/45° vs.\ 22.5°/67.5° CHSH angles).
4. **Observe** a Bell parameter (e.g.\ $S=2\sqrt2$) **above** the LHVM bound.

That empirical excess over $\beta$ is a direct refutation of any LHVM description, i.e.\ a demonstration of non-locality.

### The local polytope

The **local polytope** is the convex set of all correlations that admit a local hidden–variable model (LHVM).  Concretely, in an $n$-party Bell scenario one considers all joint probability distributions

$$
P(a_1,\dots,a_n\mid x_1,\dots,x_n)
=\int d\lambda\,\rho(\lambda)\,\prod_{i=1}^nP_i(a_i\mid x_i,\lambda)\,,
$$

and takes the convex hull of the **deterministic** assignments $\{P_i(a_i\mid x_i,\lambda)\in\{0,1\}\}$.  This yields a polytope in the space of correlators (or conditional probabilities) with the following key properties:

1. **Vertices = deterministic strategies.**
   Every vertex of the local polytope corresponds to a fixed assignment of outcomes to every measurement setting (one element of $S^n$, where $S$ is the set of local deterministic strategies).  By the fundamental theorem of linear programming, any linear functional (i.e.\ Bell expression) attains its maximum over the polytope at one of these vertices .

2. **Facets = tight Bell inequalities.**
   A **facet** (a maximal $(d-1)$-dimensional face) of the polytope is supported by a hyperplane

   $$
   \sum_{a,x}c_{a,x}\,P(a\mid x)\;=\;\beta_{\rm LHV}\,,
   $$

   which is exactly a **tight** Bell inequality: you cannot move that hyperplane inward without cutting off part of the polytope .

3. **Dimension and structure.**
   The dimension of the polytope is determined by the number of independent correlators (after imposing normalization and no-signalling).  Its full H-representation (list of facet inequalities) is exponentially large in the number of parties/settings, which is why finding all facets becomes intractable beyond the simplest scenarios.


The **local polytope** for an N-party Bell scenario (whtih each party $i$ having $m_i$ measurement settings and $d_i$ possible outcomes) is the convex hull of all deterministic boxes of which there are:
$$
    \prod_{i=1}^N d_i^{m_i}
$$

For binary outcomes the number grows exponentially:
$$
    2^{nm}
$$

As for facets, they exploode super-exponentially:
$$
    \text{\#facets} \gtrsim 2^{nm} 
$$
### Why finding a face (facet) matters

* **Optimality.**  A facet–defining inequality gives the **smallest** classical bound $\beta_{\rm LHV}$ for that linear combination of correlators.  Any **non-facet** (redundant) inequality is strictly weaker—it cuts off a larger region than necessary, and so may fail to detect nonlocality in some cases.

* **Completeness.**  The set of all facets is a complete description of the local polytope: a correlation is local **if and only if** it satisfies **every** facet inequality.  In practice, one often seeks a small family of facets tailored to the physical scenario at hand.

* **Detection power.**  Violating any facet‐defining Bell inequality unambiguously proves non-locality (and hence entanglement), and typically gives the **largest** gap between quantum and classical predictions.  Using non-facets can still detect non-locality, but may be suboptimal.

* **Geometric and algorithmic insights.**  In the tropical-tensor-network framework, one finds that the **tropical eigenvectors** of the local cost matrix $F$ encode exactly the **facet structure** of the local polytope (they correspond to closed loops on the De Bruijn graph) .  Thus, locating faces via tropical methods both recovers known tight inequalities and suggests new ones in more exotic geometries.

---

* **All tight Bell inequalities** correspond exactly to the **non-trivial facets** of the local polytope.
* **Not all facets** of the polytope (taken literally) give rise to “interesting” Bell inequalities—only the subset beyond the trivial positivity/normalization ones.

So:

> **Every non-trivial facet is a tight Bell inequality, and every tight Bell inequality defines a non-trivial facet—but there exist other (trivial) facets that we do not regard as Bell inequalities.**


## Tropical algebra

### Tropical semiring
A **min(max)-tropical semiring** is the set $\mathbb{R}\cup\{-\infty\}$ (or $\mathbb{R}\cup\{+\infty\}$) with the operations of **tropical addition** ($\oplus$) and **tropical multiplication** ($\odot$):
$$
\begin{align*}
\text{tropical addition: } & x \oplus y = \min[\max](x,y) \\
\text{tropical multiplication: } & x \odot y = x + y
\end{align*}
$$

### Properties
 - Idempotent addition: $x \oplus x = x$.
 - Distributive: $x \odot (y \oplus z) = (x \odot y) \oplus (x \odot z)$ <span style="display:inline-block; width:0.2in;"></span>$(y \oplus z) \odot x = (y \odot x) \oplus (z \odot x)$.
 - Absorbing: $-(+)\infty\;\odot\;x \;=\;-(+)\infty\quad\forall\, x$
 - Additive identity: $-(+) \infty \oplus x = -(+) \infty $.
 - Multiplicative Identity: $0\;\odot\;x \;=\; x\;\odot\;0 \;=\; x$.
 - Inverse: $x \odot (-x) = 0$.

### Tropical algebra
- Tropical Matrix Product: $(F\odot G)_{ij} = \bigoplus_{k=1}^n \bigl(F_{ik}\odot G_{kj}\bigr) = \min_{k}\bigl\{F_{ik}+G_{kj}\bigr\}$
- Tropical Trace: $ \operatorname{tropTr}(Z) = \bigoplus_{i=1}^n Z_{ii} = \min_{1\le i\le n} Z_{ii} $

- Tropical Kronecker Delta: $ \delta_{ij} = \begin{cases}
    0 & \text{if } i=j \\
    -(+) \infty & \text{if } i\neq j
\end{cases}$

- Tropical Identity: $(I_{\rm trop})_{ij} = \delta_{\rm trop}(i,j) = \begin{cases} 0, & i=j,\\ +\infty, & i\neq j. \end{cases} $


### Some notes on tropical determinants:
Displaying the tropical determinant can be more than just a neat algebraic curiosity—here’s what you can *use* it for:

1. **Optimal assignment / maximum‐weight matching**
   Think of an $n\times n$ matrix $A=(a_{ij})$ as the weights on edges of a complete bipartite graph between “rows” and “columns.”

   $$
   \det_{\rm trop}(A)
   = \max_{\sigma\in S_n}\sum_{i=1}^n a_{i,\sigma(i)}
   $$

   is precisely the *weight* of the best one‐to‐one assignment (the optimal matching).  If you also record which permutation $\sigma$ attains that maximum, you’ve solved an instance of the classic assignment problem in $O(n!\,n)$ time (or faster with Hungarian‐style algorithms in the classical semiring).

2. **Tropical (non‑)singularity and rank**
   In the tropical world a matrix is called **tropically nonsingular** if its maximum (or minimum) in the permutation‐sum is *unique*.  If two different permutations give the same optimal sum, the matrix is **singular**.  So the tropical determinant tests for uniqueness of the optimal assignment and induces a notion of “rank” via the size of the largest tropically nonsingular minor.

3. **Cramer’s rule analogues**
   There are tropical versions of Cramer’s rule: once you know $\det_{\rm trop}(A)$ and the determinants of the minors, you can describe all tropical solutions of linear systems $A\otimes x=b$.  This is widely used in tropical geometry and in solving “max‐plus linear equations.”

4. **Polytope volume & Newton polytope**
   The support of the tropical determinant (i.e. which permutations achieve the max) is exactly the set of vertices of the *Newton polytope* of the classical permanent.  In combinatorial geometry, this tells you which extreme points of a polytope are active in the convex hull, and the determinant value gives a tropical “volume” measure.

5. **Sensitivity & perturbation analysis**
   Because the tropical determinant is piecewise‐linear in the entries $a_{ij}$, examining how it changes when you tweak individual $a_{ij}$ reveals which assignments are robust (i.e. remain optimal under small perturbations).  That’s useful in scheduling and network‐flow applications.


In short, displaying the tropical permanent/determinant—and ideally the achieving permutation—lets you solve and *explain* an entire assignment problem, test tropical nonsingularity, and feed into geometry or linear‐system solvers downstream.
