---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "A Large-D Closure for Asymmetric Kinetic Vector-Spin Systems"
subtitle: "An exercise in high-dimensional wobbling"
summary: "An exercise in high-dimensional wobbling"
authors:
  - me
tags: ["Artificial Intelligence", "Attention", "Closure", "Deep Learning", "Dynamical Systems", "Many-Body Systems", "Mean-Field Theory", "Neural Networks", "Nonequilibrium Dynamics", "Statistical Physics", "Steady State", "Stochastic Thermodynamics", "Transformers", "Vector-Spin Models"]
categories: []
date: 2026-10-20T22:28:17+01:00
lastmod: 2026-10-20T22:28:17+01:00
featured: false
draft: false
toc: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: true

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

# Introduction

> **✨ Code: [`mcbal/large-d-closure`](https://github.com/mcbal/large-d-closure)**

In [previous](https://mcbal.github.io/post/spin-model-transformers/) [posts](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/), we introduced transformer-like modules built from driven stochastic vector-spin systems. The blocks [worked in a toy language model](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/#training-a-toy-autoregressive-language-model), but comparing naive mean field with exact sampling [exposed persistent errors](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/#mean-field-approximation-and-proxy-fidelity), especially in delayed correlations and at lower temperatures.

Here we ask whether increasing the vector-spin dimension gives a better approximation for a fixed realization of couplings and drives. At large $D$, noise gives the field a predictable extra length, yielding a closure for means and correlations whose errors fall approximately as $1/D$ in our numerical checks.

# Background

## Stochastic model dynamics

Consider $N$ spins $\mathbf s_i\in\mathbb R^D$ of length $R$, using the convention[^fn:radiusconvention] $R^2=D/2-1$ with $D>2$. Fixed scalar couplings $J_{ij}$ mix the spins, and fixed fields $\mathbf x_i$ drive them. All sites are resampled in parallel according to

\begin{equation}
P(\mathbf S'\mid\mathbf S)=\prod_i\frac{e^{\beta\,\mathbf s_i'\cdot\mathbf h_i}}{Z(\mathbf h_i)},
\qquad \mathbf h_i=\mathbf x_i+\sum_jJ_{ij}\mathbf s_j,
\end{equation}

where $\beta$ denotes the inverse temperature. Each conditional distribution is a [von Mises–Fisher (vMF) distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution), with mean direction $\hat{\mathbf h}_i$ and concentration parameter $\kappa_i=\beta R\lVert\mathbf h_i\rVert$.

We send $D\to\infty$ at fixed $N$, $J$, $\beta>0$ and $\beta\lVert\mathbf x_i\rVert/R$. Drives of length $O(\sqrt D)$ can then compete with the interaction field, and conditional concentrations $\kappa_i$ are $O(D)$. Spins start independently across sites, each in a vMF distribution with a prescribed mean.


## High-dimensional vMF geometry

In units of the spin radius, the geometry stays simple as $D\to\infty$. With $\kappa\propto D$, typical vMF samples occupy a thin ring at a fixed latitude around the field direction, with the mean at its center (Figure 1, [Appendix A](#appendix-a-the-geometry-of-a-high-dimensional-vmf)).

> 💡 **Why a ring?** The field favors alignment, but the sphere offers far more room to wiggle sideways. With $\kappa \propto D$, these effects balance out at a fixed latitude. Increasing $D$ narrows that latitude without eliminating the transverse wobble: the noise stays large, but its length becomes predictable.

The fluctuation $\delta\mathbf s=\mathbf s-\mathbf m$ has squared length $O(D)$, spread over $D-1$ transverse directions, while its variance along any unit direction is $O(1)$. It is nearly perpendicular to the mean, so $\lVert\mathbf m\rVert^2+\lVert\delta\mathbf s\rVert^2\simeq R^2$. Independent transverse fluctuations are nearly orthogonal, so _their squared lengths add when summed_. Increasing $D$ sharpens the ring without shrinking its radius.

{{< lightbox
  src="ring.png"
  alt="Left: a sphere with a thin ring at a fixed latitude around the field direction; the mean sits at the ring's center, and mean, fluctuation and spin form a right triangle. Right: alignment densities for D = 8, 64 and 512, narrowing around the norm of the mean."
  caption="**Figure 1.** (a) Typical samples occupy a thin ring around the field direction. Mean, fluctuation and spin form a right triangle to leading order. (b) Alignment densities at $\beta\lVert\mathbf h\rVert/R=2$ narrow as $D^{-1/2}$ around $\lVert\mathbf m\rVert/R$. The point marked ‘mode’ maximizes the density on the sphere, not the alignment marginal."
>}}

## Single-spin response

The exact conditional mean is $\boldsymbol\mu_D(\mathbf h)=R A_D(\kappa)\hat{\mathbf h}$, where $A_D(\kappa)=I_{D/2}(\kappa)/I_{D/2-1}(\kappa)$ is a ratio of modified Bessel functions. Its leading large-$D$ form is

\begin{equation}
\varphi_\beta(\mathbf h)=g(\lVert\mathbf h\rVert^2)\,\mathbf h,
\label{eq:response}
\end{equation}

where

\begin{equation}
g(r^2)=\frac{\beta}{1+\gamma(r^2)},
\qquad \gamma(r^2)=\sqrt{1+\beta^2r^2/R^2}.
\end{equation}

This is the response map of [Spin-Model Transformers](https://mcbal.github.io/post/spin-model-transformers/). Its gain differs from the exact one by a relative $O(1/D)$. Two properties matter for the closure: the nonlinearity $g$ depends only on the field norm, and the sphere fixes the conditional variance,

\begin{equation}
\mathbb E\!\left[\lVert\mathbf s'-\boldsymbol\mu_D(\mathbf h)\rVert^2\mid\mathbf h\right]
=R^2-\lVert\boldsymbol\mu_D(\mathbf h)\rVert^2
\simeq\frac{2R^2}{1+\gamma}.
\end{equation}

[Appendix B](#appendix-b-single-spin-moments) gives the exact directional covariances.

# The large-D closure

The response depends on the field's length. To average it over the dynamics, we therefore need to know how much length the input noise adds.

Write $\mathbf s_i=\mathbf m_i+\delta\mathbf s_i$, with $\mathbf m_i=\mathbb E\mathbf s_i$, and define the covariance traces

\begin{equation}
C_{ij}=\operatorname{tr}\,\mathbb E[\delta\mathbf s_i\delta\mathbf s_j^\top]=\mathbb E[\delta\mathbf s_i\cdot\delta\mathbf s_j].
\end{equation}

The trace sums over the $D$ vector components, leaving one scalar per pair of sites. Expectations are over the spin dynamics at fixed $J$ and $\mathbf x$, so $\mathbf m$ and $C$ are deterministic. The field splits into its mean and fluctuation,

\begin{equation}
\mathbf h_i=\bar{\mathbf h}_i+\boldsymbol\eta_i,
\qquad \bar{\mathbf h}_i=\mathbf x_i+\sum_jJ_{ij}\mathbf m_j,
\qquad \boldsymbol\eta_i=\sum_jJ_{ij}\delta\mathbf s_j.
\end{equation}

Naive mean field drops $\boldsymbol\eta_i$ and applies the response to the average field:

\begin{equation}
\mathbf m_i'=\varphi_\beta(\bar{\mathbf h}_i).
\label{eq:mf}
\end{equation}

But the noise can have as much squared length as the mean field, even at large $D$.

## The field's typical length

Since $\mathbb E\boldsymbol\eta_i=0$, the expected squared field norm is exactly

\begin{equation}
r_i^2\equiv\mathbb E\lVert\mathbf h_i\rVert^2
=\lVert\bar{\mathbf h}_i\rVert^2+v_i,
\qquad v_i=(JCJ^\top)_{ii}.
\end{equation}

Both terms can be $O(D)$. The useful large-$D$ fact is that the squared norm concentrates around their sum:

\begin{equation}
\operatorname{std}\!\left(\frac{\lVert\mathbf h_i\rVert^2}{R^2}\right)=O(D^{-1/2}).
\end{equation}

> 💡 **Predictable geometry.** Field lengths depend only on spin–spin and drive–spin overlaps. These start concentrated. Each update changes their typical values smoothly, while fresh noise adds only $O(D^{-1/2})$ fluctuations to normalized overlaps. Concentration therefore carries forward for any fixed number of updates, even as shared inputs correlate the spins ([Appendix C](#appendix-c-concentration-and-closure)).

This works at fixed couplings and site count, without requiring many weak inputs. Correlated spins can have predictable overlaps too.

## A deterministic gain

We now evaluate the same gain function at the typical squared length, $a_i\equiv g(r_i^2)$:

\begin{equation}
\mathbf s_i'\simeq a_i\mathbf h_i+\boldsymbol\epsilon_i,
\qquad a_i=\frac{\beta}{1+\gamma_i},
\qquad \gamma_i=\sqrt{1+\frac{\beta^2(\lVert\bar{\mathbf h}_i\rVert^2+v_i)}{R^2}}.
\label{eq:linear}
\end{equation}

> 💡 **Normalize by the full length.** At strong fields, $\varphi_\beta(\mathbf h)\simeq R\mathbf h/\lVert\mathbf h\rVert$. The denominator concentrates at $r_i$, while the numerator averages to $\bar{\mathbf h}_i$. Thus $\mathbf m_i'\simeq R\bar{\mathbf h}_i/r_i$. Mean field divides by $\lVert\bar{\mathbf h}_i\rVert$ instead, missing the extra length supplied by noise.

Here $\boldsymbol\epsilon_i=\mathbf s_i'-\boldsymbol\mu_D(\mathbf h_i)$ is the fresh sampling noise. Its conditional mean is zero, so it is uncorrelated with the past; conditional independence across sites also makes distinct sites' fresh noises uncorrelated. Its expected squared length is $2R^2/(1+\gamma_i)$ to leading order.

With a deterministic gain, the update is linear in the old fluctuations. Taking expectations and expected dot products of the fluctuations gives, with $A=\mathrm{diag}(a)$,

\begin{equation}
\boxed{
\begin{aligned}
\mathbf m_i'&=a_i\bar{\mathbf h}_i,\\
C'&=AJCJ^\top A+\mathrm{diag}\!\left(\frac{2R^2}{1+\gamma}\right),\\
D_{ij}&=a_i(JC)_{ij}.
\end{aligned}}
\label{eq:closure}
\end{equation}

Here $D_{ij}=\mathbb E[\delta\mathbf s_i'\cdot\delta\mathbf s_j]$ is the one-step delayed covariance. Old noise passes through $AJ$; fresh noise supplies the diagonal term in $C'$ and contributes nothing to $D_{ij}$. These are leading-order moment equations, requiring no Gaussian assumption on the field.

The recursion preserves the sphere identity exactly:

\begin{equation}
\lVert\mathbf m_i'\rVert^2+C_{ii}'
=a_i^2r_i^2+\frac{2R^2}{1+\gamma_i}=R^2.
\end{equation}

Whatever the mean loses in squared length is gained by the noise. Scalar couplings let us follow this balance using traces alone, without $D\times D$ covariance blocks. In units of $R$, the equations have no explicit $D$; a dense step costs $O(N^2D+N^3)$.

## What mean field misses

Because the gain decreases with squared field norm, including $v_i$ lowers the gain. At the same current mean field, the closure therefore predicts a shorter next mean in the same direction. Over later updates, the changed balance between drive and feedback can also change the direction.

From a product start, only the individual input variances contribute:

\begin{equation}
v_i=\sum_jJ_{ij}^2\left(R^2-\lVert\mathbf m_{0,j}\rVert^2\right)
\leq\frac{R^2}{N_{\mathrm{eff},i}},
\qquad N_{\mathrm{eff},i}=\frac{1}{\sum_jJ_{ij}^2}.
\label{eq:first}
\end{equation}

For nonnegative rows summing to one, $N_{\mathrm{eff}}$ ranges from 1 for a single input to $N$ for uniform weights. Diffuse weights suppress independent input noise; concentrated weights retain it.

After an update, sites reading the same inputs acquire covariance. This shared noise feeds back into $v_i$ and contributes to delayed correlations:

\begin{equation}
D_{ij}=\underbrace{a_iJ_{ij}C_{jj}}_{\text{direct}}
+\underbrace{a_i\sum_{l\neq j}J_{il}C_{lj}}_{\text{shared}}.
\end{equation}

The shared term can matter even when $J_{ij}$ is small. With nonnegative $J$ and a product start, the closure generates nonnegative entries of $C$, so the two contributions reinforce each other.

> 💡 **What averaging removes.** Averaging $n$ independent noises divides their expected squared length by $n$. Averaging $n$ identical noises removes nothing. Shared inputs make fluctuations partly common, allowing noise to survive even diffuse averaging. Large $D$ makes its length predictable; suppressing it needs enough effectively independent inputs.


# Numerical checks

We compare exact vMF sampling with naive mean field, the closure with $C$ kept diagonal, and the full closure. The diagonal version keeps norm inflation but drops shared noise, separating the two corrections.[^fn:numerics]

Spins start independently in vMF initializer fields. Drives have random directions and strength $\beta\lVert\mathbf x_i\rVert/R=u$. Couplings are a row-wise $\mathrm{softmax}(sG)$ of a fixed Gaussian matrix: increasing $s$ concentrates the weights onto fewer inputs. Unless stated otherwise, $\beta=2$ and $u=1$.

## Convergence in D

We first fix $N=256$ and $s=3$, giving $N_{\mathrm{eff}}\approx7.5$ on average, and increase $D$ from 16 to 1024.

{{< lightbox
  src="convergence.png"
  alt="Relative error against exact sampling versus D after one and four updates. Mean-field errors level off; resolved closure errors fall roughly as 1/D."
  caption="**Figure 2.** Errors after $K=1$ (top) and $K=4$ (bottom) updates. From a product start, the diagonal and [sampled](#carrying-noise-instead-of-a-matrix) closures give the same means and $D_{ij}$ as the full closure at $K=1$. Open markers: the error interval includes zero. Dotted: $1/D$."
>}}

The closure's resolved errors fall roughly as $1/D$. After four updates, its delayed-correlation error drops from 7% at $D=16$ to 0.25% at $D=512$; at $D=1024$ sampling no longer resolves it. There is no visible change as $D$ passes $N$.

Mean field instead levels off at about 23% for mean overlaps and 20% for delayed correlations. Keeping only diagonal $C$ fixes most of the mean error but leaves a 17% delayed-correlation error. Increasing the vector dimension makes the field length predictable; it does not remove the noise contributing to that length, or the correlations between sites.

{{< lightbox
  src="fieldnorm.png"
  alt="Squared field norms at one site for D = 64 and 1024. The distribution narrows around the closure's prediction, well above the mean-field value."
  caption="**Figure 3.** Squared field norm entering the first update at the most concentrated site. Mean field retains only $\lVert\bar{\mathbf h}_i\rVert^2$, about a third of the typical value. Adding $v_i$ locates the center of the distribution, whose width scales as $D^{-1/2}$."
>}}

## Concentrated weights and shared inputs

A causal mask lets us vary $N_{\mathrm{eff}}$ across sites in one run. After one update at $D=256$, the mean-field error falls from 29% for single-input rows to 3% at $N_{\mathrm{eff}}\approx20$, while the closure stays near 0.4%. The deterministic gap between the two predictions accounts for nearly all of the mean-field error: the missing field length is the main effect.

{{< lightbox
  src="mechanisms.png"
  alt="Per-site mean error falls with N_eff. For fixed row weights, placing large weights on common inputs increases errors in mean field and the diagonal closure."
  caption="**Figure 4.** (a) Per-site mean error after one update under a causal mask ($D=256$). Crosses: the gap between mean field and the closure. (b) Errors after four updates with identical row weights, scattered over columns or placed on common inputs ($D=64$)."
>}}

For a one-step module with a product initializer, the correction needs only one extra pass with squared weights over the initializer's per-site variance, with `gamma0` evaluated in the initializer fields:

```python
H = X + J @ M
slack = 2 * R2 / (1 + gamma0)  # leading initializer variance
v = (J * J) @ slack
gamma = np.sqrt(1 + beta**2 * ((H * H).sum(-1) + v) / R2)
M_next = beta * H / (1 + gamma)[:, None]
```

One can think of this as a variance-aware `RMSNorm`: the normalizer includes the field's noise variance.

After several updates, shared inputs matter too. Keeping each row's weights fixed but moving the large weights onto common columns raises the cross-site contribution to $v_i$ from 1.5% to 37%. Mean field then misses mean overlaps by 76% and delayed correlations by 96%. The diagonal closure reduces the former to 19% but leaves the latter unchanged; the full closure keeps both errors near 5%. The same ordering holds when varying $\beta$ or $u$ separately, with the largest gaps at weak drive.

## The stationary state

At a fixed point, write $B=AJ$. If $\rho(B)<1$, the covariance is

\begin{equation}
C^\star=\sum_{k\geq0}B^k\,\mathrm{diag}\!\left(\frac{2R^2}{1+\gamma}\right)(B^k)^\top.
\end{equation}

Each term is fresh noise propagated through earlier updates. To check whether the closure remains accurate after relaxation, we use $N=64$ with diffuse ($N_{\mathrm{eff}}\approx28$) and concentrated ($N_{\mathrm{eff}}\approx4$) couplings. Stationary references are time averages after burn-in; these runs use one initialization and check drift in the means.

{{< lightbox
  src="relaxation.png"
  alt="Errors along relaxation for diffuse and concentrated couplings at D = 64 and 256, settling near their stationary values by K of about 16."
  caption="**Figure 5.** Errors from $K=1$ to 100, with stationary estimates at $\infty$. Solid: $D=256$. Dashed: $D=64$."
>}}

In these runs, errors settle by $K\approx16$ rather than growing with time. The stationary error still decreases with $D$: for diffuse couplings, the closure's delayed-correlation error falls from 5.1% at $D=64$ to 1.2% at $D=256$.

This also explains the previous post's roughly 11% delayed-correlation gap at $(u,\beta)=(1,1)$. Keeping norm inflation alone leaves the gap; carrying shared covariance makes it shrink with $D$:

| $D$ | mean field | diagonal $C$ | full closure |
|---|---|---|---|
| 16 | 13% | 7.5% | 8.8% |
| 64 | 9.4% | 10% | 2.2% |
| 256 | 11% | 12% | 0.53% |
| 1024 | 12% | 12% | 0.13% |

Large $D$ can still mean quite large. At $(u,\beta)=(0.5,2)$, the closure's mean-overlap error is 89%, 20% and 1.2% at $D=16$, 64 and 1024. The corresponding errors in the mean vectors are 30%, 7% and 0.4%.

## Carrying noise instead of a matrix

The dense covariance update costs $O(N^3)$. We can replace it by $k$ random noise channels per site, keeping fresh variance analytic:

\begin{equation}
\hat C=\mathrm{diag}(\mathbf b)+\frac1kZZ^\top,\qquad Z\in\mathbb R^{N\times k}.
\end{equation}

Set $Y=JZ$ and $v_i=\sum_jJ_{ij}^2b_j+\lVert Y_i\rVert^2/k$, compute the gains as before, and update

\begin{equation}
Z'=A\big(Y+J\,\mathrm{diag}(\sqrt{\mathbf b})\,\Xi\big),\qquad
b_i'=\frac{2R^2}{1+\gamma_i},
\end{equation}

where $\Xi$ has independent standard Gaussian entries. Fresh noise stays diagonal until the couplings mix it across sites. Conditional on the current sketch, this reproduces the closure's covariance update in expectation; later gains depend on sampled noise, so the full trajectory is not unbiased ([Appendix D](#appendix-d-the-sampled-closure)). Evolution costs $O(N^2D+kN^2)$ without storing a dense covariance matrix.

The entropy-production panel uses $\beta\sum_{ij}(J_{ij}-J_{ji})D_{ij}$. At stationarity this is housekeeping entropy production; during relaxation it serves only as a diagnostic.

{{< lightbox
  src="sketch.png"
  alt="Sampled-closure errors against the full closure versus channel count k, for mean overlaps, delayed correlations and entropy production, with and without stationary time averaging."
  caption="**Figure 6.** Sampled-closure errors at $D=64$, $s=3$. Solid: $N=1024$. Dashed: $N=256$. Dotted: the full closure's error against exact sampling."
>}}

Four channels give mean-overlap errors 3–6 times smaller than the closure's own error. For entropy production, four channels suffice at $N=1024$, about three times below the closure's error; at $N=256$ their error is comparable to the closure's, and sixteen bring it below. Errors fall roughly as $k^{-1/2}$.

Individual delayed correlations are harder: their relative error is about $2/\sqrt{k}$, requiring $k\sim10^4$ to match the closure here. At stationarity, averaging over 200 updates cuts this error about fourteenfold in these runs, bringing $k\approx64$–$256$ into reach of the closure's accuracy.


# Relation to other closures

For Ising spins $s_i\in\{-1,1\}$, [Mézard and Sakellariou (2011)](https://arxiv.org/abs/1103.3433) average the single-spin response over a Gaussian local field:

\begin{equation}
\begin{aligned}
m_i'&=\int\!\mathcal Dz\,\tanh\!\left[\beta\left(\bar h_i+\sqrt{\Delta_i}\,z\right)\right],\\
\bar h_i&=H_i+\sum_jJ_{ij}m_j,
\qquad \Delta_i=\sum_jJ_{ij}^2(1-m_j^2),
\end{aligned}
\end{equation}

where $H_i$ is the external field and $\mathcal Dz=e^{-z^2/2}\,dz/\sqrt{2\pi}$. Their delayed-covariance relation is

\begin{equation}
D=\mathrm{diag}(\chi)JC,
\qquad \chi_i=\beta\int\!\mathcal Dz\,\left\{1-\tanh^2\!\left[\beta\left(\bar h_i+\sqrt{\Delta_i}\,z\right)\right]\right\}.
\end{equation}

This has the same matrix form as our delayed covariance, with a Gaussian-averaged susceptibility $\chi_i$ replacing the radial gain $a_i$. Their Gaussian-field argument uses the large-$N$, fully asymmetric setting. Here concentration across vector components makes the radial response reduce to a gain at $\lVert\bar{\mathbf h}_i\rVert^2+v_i$, at fixed $N$ and $J$, without requiring Gaussian fields.

[Huang and Kabashima (2014)](https://arxiv.org/abs/1310.5003) retain correlations between Gaussian local fields, using their full covariance $JCJ^\top$. The variance above becomes $\Delta_i=(JCJ^\top)_{ii}$: the same shared-noise contribution that enters $v_i$ here. Their Gaussian treatment also recovers the delayed-covariance relation without expanding in small cross-field correlations.

[Aguilera et al. (2021)](https://arxiv.org/abs/2002.04309) place Gaussian-field and TAP ([Roudi and Hertz, 2011](https://arxiv.org/abs/1103.1044)) approximations within a common Plefka framework: choose tractable reference dynamics and expand around them. Increasing the vector dimension at fixed couplings gives a different limit; a direct comparison with second-order Plefka for vector spins remains to be done.

The length constraint also connects this construction to [equilibrium large-component models](https://arxiv.org/abs/cond-mat/0408073), where local multipliers enforce each spin's length. Both descriptions follow correlations under local constraints. Here the dynamics are synchronous and the couplings may be asymmetric; [Appendix C](#appendix-c-concentration-and-closure) develops the connection and distinguishes local constraints from a single spherical constraint.


# Conclusion and outlook

Large $D$ makes the field's length predictable while retaining the noise that contributes to it. Evaluating the gain at that length closes the means and covariance traces, capturing both norm inflation and shared inputs. The argument applies at fixed finite time; the numerical checks show roughly $1/D$ errors through relaxation and at the tested stationary states.

For a one-step module, the correction costs one extra pass with squared weights. For several steps, sampled noise channels make the covariance practical to carry, with few channels sufficient for means and the entropy-production proxy in these runs. The immediate learning question is whether this changes trained behavior, or whether training absorbs the mean-field error.

Stacking raises a further choice. Layers driven by the preceding layer's means can reuse this closure. Passing sampled spins onward also passes their noise: vector maps reshape it, and queries and keys can make the couplings fluctuate. [Appendix E](#appendix-e-the-signal-subspace) shows why traces alone then lose useful information about directions. The next step is to test how much directional information a learned stack actually needs.


# Acknowledgements

The author (M.B.) would like to thank Á. Poc-López and M. Aguilera for fruitful discussions. Claude Opus 5.5 and GPT-6 Astra were used for brainstorming, research design, coding, writing assistance, figure generation, and adversarial reviewing. The author (M.B.) takes full responsibility for any remaining slop.

# References

If you happen to find this work useful, please consider citing it as:

```
@article{bal2026largedclosure,
  title   = {A Large-D Closure for Asymmetric Kinetic Vector-Spin Systems},
  author  = {Bal, Matthias},
  year    = {2026},
  month   = {October},
  url     = {https://mcbal.github.io/post/a-large-d-closure-for-asymmetric-kinetic-vector-spin-systems/}
}
```

- M. Bal, [Spin-Model Transformers](https://mcbal.github.io/post/spin-model-transformers/) and [Nonequilibrium Dynamics in Spin-Model Transformers](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/), with [mcbal/neqnn](https://github.com/mcbal/neqnn).
- M. Aguilera, S. A. Moosavi, H. Shimazaki, [A unifying framework for mean-field theories of asymmetric kinetic Ising systems](https://www.nature.com/articles/s41467-021-20890-5), Nat. Commun. 12, 1197 (2021); [arXiv:2002.04309](https://arxiv.org/abs/2002.04309).
- M. Mézard, J. Sakellariou, [Exact mean-field inference in asymmetric kinetic Ising systems](https://arxiv.org/abs/1103.3433), J. Stat. Mech. L07001 (2011).
- Y. Roudi, J. Hertz, [Dynamical TAP equations for non-equilibrium Ising spin glasses](https://arxiv.org/abs/1103.1044), J. Stat. Mech. P03031 (2011).
- H. Huang, Y. Kabashima, [Dynamics of asymmetric kinetic Ising systems revisited](https://arxiv.org/abs/1310.5003), J. Stat. Mech. P05020 (2014).
- H. E. Stanley, [Spherical model as the limit of infinite spin dimensionality](https://link.aps.org/doi/10.1103/PhysRev.176.718), Phys. Rev. 176, 718 (1968).
- L. F. Cugliandolo, D. S. Dean, [Full dynamical solution for a spherical spin-glass model](https://arxiv.org/abs/cond-mat/9502075), J. Phys. A 28, 4213 (1995).
- L. W. Lee, A. Dhar, A. P. Young, [Spin glasses in the limit of an infinite number of spin components](https://arxiv.org/abs/cond-mat/0408073) (2004).
- M. B. Hastings, [Ground State and Spin Glass Phase of the Large N Infinite Range Spin Glass Via Supersymmetry](https://arxiv.org/abs/cond-mat/9906186) (1999).
- A. T. A. Wood, [Simulation of the von Mises–Fisher distribution](https://doi.org/10.1080/03610919408813161), Commun. Stat. Simul. Comput. 23, 157–164 (1994).
- Wikipedia contributors, [Von Mises–Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution).
- M. Bal, [Code and numerical experiments for this post](https://github.com/mcbal/large-d-closure).


# Appendices

## Appendix A: The geometry of a high-dimensional vMF

Write a unit sample as $\mathbf u=t\hat{\mathbf h}+\sqrt{1-t^2}\,\boldsymbol\xi$, where $\boldsymbol\xi$ is uniform on the transverse unit sphere and independent of $t$. The alignment density is

\begin{equation}
p(t)\propto\underbrace{e^{\kappa t}}_{\text{alignment}}\;\underbrace{(1-t^2)^{(D-3)/2}}_{\text{room to fluctuate}},\qquad -1\lt t\lt1.
\end{equation}

At fixed $\kappa/D$, maximizing the leading log density gives the typical latitude; its curvature gives the width:

\begin{equation}
t_\star=\frac{\kappa}{D/2+\sqrt{D^2/4+\kappa^2}},
\qquad \operatorname{std}(t)\simeq\frac{1-t_\star^2}{\sqrt{D(1+t_\star^2)}}.
\end{equation}

The exact mean alignment is $A_D(\kappa)=t_\star+O(1/D)$. With $\kappa=\beta R\lVert\mathbf h\rVert$ and $R^2=D/2-1$, this recovers $t_\star\simeq\lVert\varphi_\beta(\mathbf h)\rVert/R$. The most likely latitude stays away from the pole, even though the pole maximizes the density per unit area.

The transverse length remains finite, but its projection onto any fixed unit vector has standard deviation at most $1/\sqrt{D-1}$. For $n$ independent samples,

\begin{equation}
\mathbb E\lVert\bar{\mathbf u}\rVert^2=A_D(\kappa)^2+\frac{1-A_D(\kappa)^2}{n}.
\end{equation}

Averaging divides the noise's expected squared length by $n$. Increasing $D$ instead makes that length predictable.

Conditioning $\mathcal N(\boldsymbol\nu,\sigma^2I)$ on radius $R$ gives a vMF law with $\kappa=R\lVert\boldsymbol\nu\rVert/\sigma^2$. Normalizing a Gaussian sample generally gives a different law.

## Appendix B: Single-spin moments

For $\kappa=\beta R\lVert\mathbf h\rVert$, the exact mean is $\boldsymbol\mu_D=R A_D(\kappa)\hat{\mathbf h}$. The covariance separates into transverse and longitudinal parts:

\begin{equation}
\Sigma(\mathbf h)=\tau(I-\hat{\mathbf h}\hat{\mathbf h}^\top)+\rho\hat{\mathbf h}\hat{\mathbf h}^\top,
\end{equation}

\begin{equation}
\tau=\frac{R^2A_D(\kappa)}{\kappa},
\qquad
\rho=R^2\left(1-\frac{(D-1)A_D(\kappa)}{\kappa}-A_D(\kappa)^2\right).
\end{equation}

At zero field, $\boldsymbol\mu_D=0$ and $\Sigma=(R^2/D)I$. Two useful exact identities are $\boldsymbol\mu_D=\beta\tau\mathbf h$ and $\partial\boldsymbol\mu_D/\partial\mathbf h=\beta\Sigma$: fluctuations determine the response to a small change in field.

At fixed $\beta\lVert\mathbf h\rVert/R$,

\begin{equation}
\tau=\frac{1}{1+\gamma}+O(1/D),
\qquad
\rho=\frac{1}{\gamma(1+\gamma)}+O(1/D),
\qquad \gamma=\sqrt{1+\beta^2\lVert\mathbf h\rVert^2/R^2}.
\end{equation}

Every directional variance stays $O(1)$. The longitudinal variance is smaller by $1/\gamma$ to leading order, but occupies only one direction. Summing all $D$ directions gives the fresh variance used in the closure:

\begin{equation}
\operatorname{tr}\Sigma=R^2\bigl[1-A_D(\kappa)^2\bigr]\simeq\frac{2R^2}{1+\gamma}.
\end{equation}

## Appendix C: Concentration and closure

### Why the field norm concentrates

Fix $N$, $J$, $\beta$ and a finite number of updates, with drives of length $O(R)$ and a product vMF initialization. Define

\begin{equation}
Q_{ij}=\frac{\mathbf s_i\cdot\mathbf s_j}{R^2},
\qquad U_{ai}=\frac{\mathbf x_a\cdot\mathbf s_i}{R^2}.
\end{equation}

Initially these overlaps have variance $O(1/D)$, with $Q_{ii}=1$ exactly. They determine every field length:

\begin{equation}
\frac{\lVert\mathbf h_i\rVert^2}{R^2}
=\frac{\lVert\mathbf x_i\rVert^2}{R^2}
+2\sum_jJ_{ij}U_{ij}
+\sum_{jl}J_{ij}J_{il}Q_{jl}.
\end{equation}

> 💡 **Follow the geometry.** We need lengths and angles to become predictable, not spins to become independent. Shared inputs change the typical overlaps. Fresh noise only adds small fluctuations around them.

Condition on the current spins. Each fresh covariance $\Sigma_i$ has operator norm $O(1)$ and trace $O(D)$. For a past-measurable vector $\mathbf w$ of length $O(R)$, and for distinct sites $i,j$,

\begin{align}
\mathbb E[(\mathbf w\cdot\boldsymbol\epsilon_i)^2\mid\mathbf S]
&=\mathbf w^\top\Sigma_i\mathbf w=O(D),\\
\mathbb E[(\boldsymbol\epsilon_i\cdot\boldsymbol\epsilon_j)^2\mid\mathbf S]
&=\operatorname{tr}(\Sigma_i\Sigma_j)=O(D).
\end{align}

After dividing by $R^2$, these fluctuations are $O(D^{-1/2})$. The conditional means of the next overlaps are smooth functions of the current ones: for example, $\mathbb E[Q'_{ij}\mid\mathbf S]=\boldsymbol\mu_D(\mathbf h_i)\cdot\boldsymbol\mu_D(\mathbf h_j)/R^2$ for $i\neq j$. Their derivatives stay bounded on the accessible range. The law of total variance then carries the $O(1/D)$ variance bound from one step to the next.

This establishes concentration at fixed finite time. The constants can grow with time, $N$ and the couplings; stationary accuracy needs a separate check.

### From the exact update to the closure

Separate the deterministic gain from its residual:

\begin{equation}
\boldsymbol\mu_D(\mathbf h_i)=a_i\mathbf h_i+\mathbf e_i,
\qquad a_i=g(r_i^2),\qquad r_i^2=\mathbb E\lVert\mathbf h_i\rVert^2.
\end{equation}

The residual includes the finite-$D$ response correction and the fluctuating gain. Since $g'=O(D^{-1})$, concentration gives $\mathbb E\lVert\mathbf e_i\rVert^2=O(1)$ against a spin squared length of $O(D)$. The exact update becomes

\begin{align}
\mathbf m_i'&=a_i\bar{\mathbf h}_i+\mathbb E\mathbf e_i,\\
\delta\mathbf s_i'&=a_i\boldsymbol\eta_i+\boldsymbol\epsilon_i
+\mathbf e_i-\mathbb E\mathbf e_i.
\end{align}

Fresh noise has zero conditional mean and is conditionally independent across sites. Its variance is

\begin{equation}
\mathbb E\lVert\boldsymbol\epsilon_i\rVert^2
=R^2-\mathbb E\lVert\boldsymbol\mu_D(\mathbf h_i)\rVert^2
\simeq\frac{2R^2}{1+\gamma_i}.
\end{equation}

Dropping $\mathbf e_i$ gives Eq. \eqref{eq:closure}. These bounds give errors of order $D^{-1/2}$ in normalized moments. The faster $1/D$ bias seen numerically needs cancellations in expectations.

> 💡 **Fluctuation size is not bias.** The squared field norm fluctuates by $O(\sqrt D)$, but its centered fluctuation has zero mean. This cancels the gain's linear correction along the mean field. Correlations with the field noise and curvature of the gain determine the remaining bias.

For example, let $\Delta_i=\lVert\mathbf h_i\rVert^2-r_i^2$. Since $\mathbb E\Delta_i=0$, expanding the approximate response gives

\begin{align}
\mathbb E\varphi_\beta(\mathbf h_i)
={}&a_i\bar{\mathbf h}_i
+g'(r_i^2)\mathbb E[\Delta_i\boldsymbol\eta_i]\\
&+\frac12g''(r_i^2)\left(
\mathbb E[\Delta_i^2]\bar{\mathbf h}_i
+\mathbb E[\Delta_i^2\boldsymbol\eta_i]\right)+\cdots.
\end{align}

Here $g''=O(D^{-2})$ and $\mathbb E\Delta_i^2=O(D)$. The field covariance has bounded operator norm over finite time: bounded single-spin covariances give a bounded field covariance at fixed $J$, and $\boldsymbol\mu_D=a\mathbf h+\mathbf e$ preserves this bound because $\mathbb E\lVert\mathbf e\rVert^2=O(1)$. Cauchy–Schwarz then gives $\lVert\mathbb E[\Delta_i\boldsymbol\eta_i]\rVert=O(\sqrt D)$. The first correction and the quadratic term along $\bar{\mathbf h}_i$ are thus $O(1/D)$ in units of $R$. Bounding the remaining terms and the covariance bias is still needed for a full $1/D$ estimate.

### A recursion for overlaps

Set $G_{ab}=\mathbf x_a\cdot\mathbf x_b/R^2$. The block Gram matrix

\begin{equation}
\mathcal G=\begin{pmatrix}G&U\\U^\top&Q\end{pmatrix}
\end{equation}

specifies the joint spin–drive geometry up to a common rotation or reflection. Write $\bar Q=\mathbb E Q$ and $\bar U=\mathbb E U$. The expected field overlaps are exactly

\begin{equation}
F_{ij}=\frac{\mathbb E[\mathbf h_i\cdot\mathbf h_j]}{R^2},
\qquad F=G+\bar UJ^\top+J\bar U^\top+J\bar QJ^\top.
\end{equation}

Since $F_{ii}=r_i^2/R^2$, the leading closure becomes

\begin{equation}
\boxed{
\begin{aligned}
\gamma_i&=\sqrt{1+\beta^2F_{ii}},
\qquad a_i=\frac{\beta}{1+\gamma_i},\\
\bar Q'&=AFA+\mathrm{diag}\!\left(\frac{2}{1+\gamma}\right),\\
\bar U'&=(G+\bar UJ^\top)A.
\end{aligned}}
\label{eq:overlapclosure}
\end{equation}

The diagonal stays $\bar Q'_{ii}=1$. Substituting $\bar Q_{ij}=(\mathbf m_i\cdot\mathbf m_j+C_{ij})/R^2$ and $\bar U_{ai}=\mathbf x_a\cdot\mathbf m_i/R^2$ recovers the moment closure. Vector means can be propagated from their initial values using the same gains; the overlap matrices alone do not specify their coordinates.

### Connection to large-component spin glasses

For equilibrium vector spins with $\lVert\mathbf S_i\rVert^2=m$, the $m\to\infty$ limit at fixed site count gives

\begin{equation}
K_{ij}=\frac{\langle\mathbf S_i\cdot\mathbf S_j\rangle}{m},
\qquad K=T(\Lambda-J)^{-1},
\qquad K_{ii}=1,
\end{equation}

where $\Lambda$ contains one multiplier per site enforcing its length constraint ([Lee, Dhar and Young, 2004](https://arxiv.org/abs/cond-mat/0408073)). Our recursion likewise follows a correlation matrix under local length constraints, but describes synchronous dynamics with possibly asymmetric couplings.

Local constraints also distinguish this limit from a spherical model with one global constraint ([Hastings, 1999](https://arxiv.org/abs/cond-mat/9906186)), whose Langevin dynamics can be solved exactly ([Cugliandolo and Dean, 1995](https://arxiv.org/abs/cond-mat/9502075)). The component and thermodynamic limits need not commute. Here the overlaps compare physical sites, rather than replicas, and the finite-time argument makes no claim about a glass phase.

### Units and initialization

Set $\mathbf q_i=\mathbf m_i/R$, $c=C/R^2$ and $\bar{\mathbf b}_i=\bar{\mathbf h}_i/R$. Then

\begin{align}
\gamma_i&=\sqrt{1+\beta^2\big(\lVert\bar{\mathbf b}_i\rVert^2+(JcJ^\top)_{ii}\big)},\\
\mathbf q_i'&=a_i\bar{\mathbf b}_i,\\
c'&=AJcJ^\top A+\mathrm{diag}\!\left(\frac{2}{1+\gamma}\right),
\qquad D_{ij}/R^2=a_i(Jc)_{ij}.
\end{align}

The recursion has no explicit $D$. A product start has $C_0=\mathrm{diag}(R^2-\lVert\mathbf m_{0,i}\rVert^2)$; matching vMF laws maximize entropy for prescribed means strictly inside the sphere.

For nonnegative row-stochastic $J$, positive semidefiniteness and $C_{jj}\leq R^2$ give $v_i\leq R^2$. Shared noise can exceed the independent-input bound $R^2/N_{\mathrm{eff},i}$ while respecting this bound.

In low precision, compute the initializer's leading variance as $2R^2/(1+\gamma_0)$ to avoid subtracting nearly equal numbers. The updated $C_{ii}'=a_i^2v_i+2R^2/(1+\gamma_i)$ is already a sum of nonnegative terms.

## Appendix D: The sampled closure

Split $C=\mathrm{diag}(\mathbf b)+S$ into the last update's fresh variance and older noise. Fresh variance is diagonal; older noise has passed through $J$ and is correlated across sites. Represent it by $S\approx ZZ^\top/k$, with $k$ channels per site.

> 💡 **Sample noise when it mixes.** Fresh variance needs only one number per site. Once the couplings spread it across sites, random channels carry the resulting correlations through the same matrix multiplication as the means.

Given the current state, draw independent standard Gaussian entries in $\Xi\in\mathbb R^{N\times k}$ and set

\begin{equation}
Z'=AJ\big(Z+\mathrm{diag}(\sqrt{\mathbf b})\,\Xi\big),
\qquad b_i'=\frac{2R^2}{1+\gamma_i}.
\end{equation}

Since $\mathbb E\Xi=0$ and $\mathbb E\Xi\Xi^\top=kI$,

\begin{equation}
\mathbb E\!\left[\tfrac1kZ'Z'^\top\,\middle|\,\mathbf m,Z,\mathbf b\right]
=AJ\Big(\tfrac1kZZ^\top+\mathrm{diag}(\mathbf b)\Big)J^\top A
=AJ\hat CJ^\top A.
\end{equation}

Adding $\mathrm{diag}(\mathbf b')$ reproduces the closure's update applied to the current $\hat C$. From a product start, $Z=0$, so the first gains, means and delayed correlations match the full closure. The new off-diagonal covariance is already sampled.

With $Y=JZ$, the required quantities are

\begin{equation}
v_i=\sum_jJ_{ij}^2b_j+\frac{\lVert Y_i\rVert^2}{k},
\qquad \hat D=A\left(J\,\mathrm{diag}(\mathbf b)+\frac{YZ^\top}{k}\right).
\end{equation}

Evolution costs $O(N^2D+kN^2)$ and adds $O(Nk)$ covariance storage. Dense $C$ or $D_{ij}$ readouts can be formed when needed; the stationary entropy-production estimate $\hat\sigma_{\mathrm{hk}}^\star=\beta\sum_{ij}(J-J^\top)_{ij}\hat D_{ij}^\star$ can be computed directly from the factors. In attention terms, $Z$ adds $k$ value channels under the same weights.

Channel fluctuations scale roughly as $k^{-1/2}$. Gains and aggregate observables can average over many sites as well; an individual delayed correlation remains one sampled inner product. This explains their different accuracies in Figure 6.

Later gains depend on sampled $v_i$, so the full trajectory is not exactly unbiased. An $O(1/k)$ bias is plausible from a second-order expansion, but has not been established. The sphere identity holds conditionally in expectation at each update. In Figure 6, the per-site variance misses $R^2-\lVert\mathbf m_i\rVert^2$ by $3$–$5\%$ of $R^2$ at $k=4$ and under $1\%$ at $k=256$, without renormalization.

At stationarity, fresh channel draws can make time averaging useful, but successive states still share propagated noise. The gain depends on autocorrelation. Averaging 200 updates reduced delayed-correlation error about fourteenfold in these runs, mean-overlap error by 11–14 times, and entropy-production error by 7–23 times. These are empirical reductions, not a general effective-sample-size rule.

## Appendix E: The signal subspace

A trace records how much noise there is, but not where it points. For an orthonormal basis $P\in\mathbb R^{D\times r}$, the variance in its subspace is

\begin{equation}
V_i=\operatorname{tr}\!\left[P^\top\mathrm{Cov}(\mathbf s_i)P\right].
\end{equation}

An isotropic estimate assigns it $(r/D)C_{ii}$. This works well for a random subspace, but a signal subspace can align with the directions of suppressed noise. If drives and initializer fields lie in it, symmetry keeps the means and mean fields there too. Realized fields still fluctuate outside it.

We test ranks $r=1,8$ at $N=64$, $s=3$, $\beta=2$, $u=1$ and $K=4$. The control has generic drives and a random subspace. The isotropic comparison uses the sampled trace to isolate the assumption about directions.

{{< lightbox
  src="subspace.png"
  alt="Signal-subspace variance stays below the isotropic share as D grows. A partial directional recursion greatly reduces the discrepancy, while trace errors also decrease."
  caption="**Figure 7.** Left: subspace variance relative to the isotropic share at $D=1024$; crosses show the recursion below. Right: errors for the isotropic share (solid), the directional recursion (dashed) and the closure's $D_{ij}$ (dotted). Open markers indicate unresolved errors."
>}}

At $D=1024$, the mean per-site ratios and relative errors are:

| | exact / isotropic | isotropic error | resolved error | closure $D_{ij}$ error |
|---|---|---|---|---|
| rank-1 drives | 0.73 | 36% | 0.19 ± 0.08% | 0.04 ± 0.14% |
| rank-8 drives | 0.96 | 3.9% | 0.11 ± 0.03% | 0.11 ± 0.05% |
| generic, 8-dim | 1.00 | 0.03 ± 0.08% | — | 0.10 ± 0.03% |

The signal-subspace discrepancy persists as $D$ grows, while mean and delayed-correlation errors decrease. The generic control agrees with isotropy within sampling resolution; the directional recursion below assumes the mean fields lie in the chosen subspace.

> 💡 **Small in the trace, large in a direction.** A fixed-rank subspace contains $O(1)$ of the total $O(D)$ variance. An $O(1)$ mistake there vanishes in a normalized trace but remains a finite relative error after projection.

To follow those directions, carry $\Gamma_{ij}=P^\top\mathrm{Cov}(\mathbf s_i,\mathbf s_j)P$. Write $\mathbf p_i=P^\top\bar{\mathbf h}_i$ and $\mathbf q_j=P^\top\mathbf m_j$. Fresh noise is suppressed along each sample's field, which is tilted away from $\mathbf p_i$ by the transverse noise. Averaging this tilt gives the leading projected fresh covariance

\begin{equation}
\bar\Sigma_i=\frac{1}{1+\gamma_i}\left(I_r-\Big(1-\frac1{\gamma_i}\Big)\frac{\mathbf p_i\mathbf p_i^\top}{r_i^2}\right).
\end{equation}

The factor $\lVert\mathbf p_i\rVert^2/r_i^2$ is the typical squared cosine of the tilt. Using only the mean-field direction would overestimate the suppression.

The propagated noise also changes the gain through $2\bar{\mathbf h}_i\cdot\boldsymbol\eta_i$. A second linear contribution comes from the sphere identity

\begin{equation}
\lVert\delta\mathbf s_j\rVert^2=R^2-\lVert\mathbf m_j\rVert^2-2\mathbf m_j\cdot\delta\mathbf s_j.
\end{equation}

It puts $-2\sum_jJ_{ij}^2\mathbf m_j\cdot\delta\mathbf s_j$ inside $\lVert\boldsymbol\eta_i\rVert^2-v_i$. Keeping these two linear contributions gives

\begin{equation}
L_{ij}=a_iJ_{ij}I_r+2g'(r_i^2)\,\mathbf p_i\big(J_{ij}\mathbf p_i-J_{ij}^2\mathbf q_j\big)^\top,
\qquad
g'(r_i^2)=-\frac{a_i\beta^2}{2R^2\gamma_i(1+\gamma_i)},
\end{equation}

\begin{equation}
\Gamma'_{ik}=\sum_{jl}L_{ij}\Gamma_{jl}L_{kl}^\top+\delta_{ik}\bar\Sigma_i.
\end{equation}

This is a partial extension. It drops the bilinear term $\sum_{j\neq l}J_{ij}J_{il}(\delta\mathbf s_j\cdot\delta\mathbf s_l-C_{jl})$ in the field-norm fluctuation. That term can fluctuate by $O(\sqrt D)$ and contribute $O(1)$ after projection, so it is not parametrically smaller. If mean fields extend outside the subspace, their coupling to noise also requires more directions.

Putting single-spin anisotropy along the mean field and propagating with the plain gain leaves 25% error at rank 1. Including tilt and gain response brings this to 1.7%; adding the sphere term gives errors of 1.4%, 0.36% and 0.19% at $D=64$, 256 and 1024. The sphere term barely changes the rank-8 result. These checks support the correction but do not establish that the omitted bilinear term is negligible.

For stacked layers with $D\times D$ maps, this suggests carrying covariance blocks for a small signal subspace alongside traces for the remaining noise.


# Footnotes

[^fn:radiusconvention]: Scaling $R^2\propto D$ follows the large-component convention for vector spins ([Stanley 1968](https://link.aps.org/doi/10.1103/PhysRev.176.718)). The choice $R^2=D/2-1$ matches the Bessel order and simplifies the leading formulas. In units of $R$, the leading response depends only on $\beta\lVert\mathbf h\rVert/R$; exact finite-$D$ moments still depend on $D$.

[^fn:numerics]: Reference moments use exact vMF sampling ([Wood, 1994](https://doi.org/10.1080/03610919408813161)). Errors are relative Frobenius errors corrected for Monte Carlo noise; intervals show two jackknife standard errors. Mean errors compare $\mathbf m_i\cdot\mathbf m_j/R^2$ and $\mathbf x_i\cdot\mathbf m_j/R^2$, except where vector errors are stated, including Figure 4(a). Mean-field delayed correlations use the previous post's linear-response approximation.
