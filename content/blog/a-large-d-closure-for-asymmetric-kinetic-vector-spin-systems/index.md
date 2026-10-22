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
lastmod: 2026-10-22T8:54:17+01:00
featured: false
draft: false
toc: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

# Introduction

> **✨ Code: [`mcbal/large-d-closure`](https://github.com/mcbal/large-d-closure)**

In [previous](https://mcbal.github.io/post/spin-model-transformers/) [posts](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/), we introduced transformer-like modules built around dynamical mean-field descriptions of driven stochastic vector spins wobbling about on hyperspheres. Even though the resulting parallel transformer blocks [behaved well as part of a toy language model](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/#training-a-toy-autoregressive-language-model), naive mean-field observables showed [persistent errors compared to exact sampling of the stochastic system](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/#mean-field-approximation-and-proxy-fidelity), especially in delayed correlations and at lower temperatures, even as the vector dimension grew. Rather than going to higher-order mean-field approximations, we ask whether just increasing the vector-spin dimension can on its own yield a more faithful approximation for a fixed realization of couplings and drives.

Physically, each vector spin responds to a field made up of a local drive plus its neighbors selected by the couplings. Mean field averages the neighbors before applying the response, which ignores important fluctuations on the hypersphere. We find that the large-$D$ limit makes these noise contributions predictable, and that taking them into account yields a _closure_: a set of recurrence relations for means and covariance traces at equal and consecutive times. We find numerically that errors fall off roughly as $1/D$, and situate our closure as a large-$D$ vector-spin version of [Mézard and Sakellariou (2011)](https://arxiv.org/abs/1103.3433)'s binary-spin approach.

# Background

## Stochastic model dynamics

Consider $N$ spins $\mathbf s_i\in\mathbb R^D$ of length $R$, with $R^2=D/2-1$[^fn:radiusconvention] and $D>2$. Fixed scalar couplings $J_{ij}$ mix the spins, and fixed fields $\mathbf x_i$ drive them. The couplings may be asymmetric, and we do not assume anything about their statistics. All sites are updated in parallel, independently:

\begin{equation}
P(\mathbf S'\mid\mathbf S)=\prod_i\frac{e^{\beta\,\mathbf s_i'\cdot\mathbf h_i}}{\mathcal Z(\mathbf h_i)},
\qquad \mathbf h_i=\mathbf x_i+\sum_jJ_{ij}\mathbf s_j,
\end{equation}

where $\beta > 0$ denotes the inverse temperature. The local conditional distribution at each site is a [von Mises–Fisher (vMF) distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution), with mean direction $\hat{\mathbf h}_i$ and concentration parameter $\kappa_i=\beta R\lVert\mathbf h_i\rVert$. We will send $D\to\infty$ at fixed $N$, $J_{ij}$, and $\beta$, holding the ratio $\beta\lVert\mathbf x_i\rVert/R$ approximately fixed. Drives of length $O(\sqrt D)$ can then compete with the interaction field, and conditional concentrations $\kappa_i$ are $O(D)$. We will find out soon why these scale choices are important.

It is important to remember that, across stochastic trajectories, the spins, and therefore the fields, fluctuate. The intuition for a single trajectory is that the update happens in two steps. Spins start independently across sites, each in a vMF distribution with a prescribed mean. First the current spins determine the fields, which are then used to condition the vMF distributions to sample the next spins from, and so on.

## High-dimensional vMF geometry

The conditional vMF distributions are tuned so that $\kappa\propto D$, which means typical vMF samples occupy a thin ring at a fixed latitude around the field direction, with the mean at its center, see Figure 1(a) and [Appendix A](#appendix-a-the-geometry-of-a-high-dimensional-vmf). Why a ring? The field favors alignment, but the hypersphere in high dimensions always offers more room to wobble sideways. At $\kappa\propto D$, these effects balance at a fixed latitude. Increasing $D$ then sharpens the ring without shrinking it, see Figure 1(b).

{{< lightbox
  src="ring.png"
  alt="Left: a sphere with a thin ring at a fixed latitude around the field direction; the mean sits at the ring's center, and mean, fluctuation and spin form a right triangle. Right: alignment densities for D = 8, 64 and 512, narrowing around the norm of the mean."
  caption="**Figure 1.** (a) Typical samples occupy a thin ring around the field direction. Mean, fluctuation and spin form a right triangle to leading order. (b) Alignment densities at $\beta\lVert\mathbf h\rVert/R=2$ narrow as $D^{-1/2}$ around $\lVert\mathbf m\rVert/R$. The point marked 'mode' maximizes the density on the sphere, not the alignment marginal."
>}}

A fluctuation $\delta\mathbf s=\mathbf s-\mathbf m$ has squared length $O(D)$, spread out mainly over $D-1$ transverse directions, with $O(1)$ variance along any unit direction. It is nearly perpendicular to the mean, so $\lVert\mathbf m\rVert^2+\lVert\delta\mathbf s\rVert^2\simeq R^2$. Even though the low-dimensional cartoon above suggests otherwise, independent transverse fluctuations along the ring are actually nearly orthogonal, which means _their squared lengths approximately add up_.

## Single-spin response

For a given field $\mathbf h$, the exact conditional vMF mean of the next spin is $\boldsymbol\mu_D(\mathbf h)=R A_D(\kappa)\hat{\mathbf h}$, where $A_D(\kappa)=I_{D/2}(\kappa)/I_{D/2-1}(\kappa)$ is a ratio of modified Bessel functions. To get rid of Bessel functions, we consider its leading large-$D$ form

\begin{equation}
\boldsymbol\mu_\infty(\mathbf h)=g(\lVert\mathbf h\rVert^2)\,\mathbf h,
\label{eq:response}
\end{equation}

where the non-linear _gain_ function is given by

\begin{equation}
g(r^2)=\frac{\beta}{1+\gamma(r^2)},
\qquad \gamma(r^2)=\sqrt{1+\beta^2r^2/R^2}.
\end{equation}

This coincides with the response map $\varphi_\beta$ used in [Spin-Model Transformers](https://mcbal.github.io/post/spin-model-transformers/). Its gain differs from the exact one $\boldsymbol\mu_D(\mathbf h)$ by a relative $O(1/D)$.

We will use the following two properties of the single-spin response in what follows: (1) the gain function depends only on the field norm, and (2) the sphere geometry fixes the conditional variance,

\begin{equation}
\mathbb E\!\left[\lVert\mathbf s'-\boldsymbol\mu_D(\mathbf h)\rVert^2\mid\mathbf h\right]
=R^2-\lVert\boldsymbol\mu_D(\mathbf h)\rVert^2
\simeq\frac{2R^2}{1+\gamma}.
\end{equation}

We refer to [Appendix B](#appendix-b-single-spin-moments) for exact directional covariances.


# The large-D closure

We now have all ingredients to start building towards the closure. Let us first decompose the fluctuating spin into $\mathbf s_i=\mathbf m_i+\delta\mathbf s_i$ where $\mathbf m_i=\mathbb E\left[\mathbf s_i\right]$ with expectations are over spin dynamics at fixed $J$ and $\mathbf x$. The field splits similarly into its mean and fluctuation,

\begin{equation}
\mathbf h_i=\bar{\mathbf h}_i+\boldsymbol\eta_i,
\qquad \bar{\mathbf h}_i=\mathbf x_i+\sum_jJ_{ij}\mathbf m_j,
\qquad \boldsymbol\eta_i=\sum_jJ_{ij}\delta\mathbf s_j.
\end{equation}

Naive mean field drops $\boldsymbol\eta_i$ and replaces the average of the response

\begin{equation}
\mathbf m_i'=\mathbb E[\mathbf s_i']=\mathbb E_{\mathbf h_i} \big[\boldsymbol\mu_D(\mathbf h_i)\big]
\end{equation}

with the response to the average field

\begin{equation}
\mathbf m_i'=\boldsymbol\mu_\infty(\bar{\mathbf h}_i).
\label{eq:mf}
\end{equation}

But as we have learned from high-dimensional vMF geometry, these fluctuations can take up as much squared length as the mean field.

## Typical field length

So what is the length of the typical field? Let us split off the noise,

\begin{equation}
\lVert\mathbf h_i\rVert^2=\lVert\bar{\mathbf h}_i\rVert^2+2\,\bar{\mathbf h}_i\cdot\boldsymbol\eta_i+\lVert\boldsymbol\eta_i\rVert^2.
\end{equation}

The noise is spread over all $D$ directions, with $O(1)$ variance along any one of them. The cross term therefore averages to zero and fluctuates by $O(\sqrt D)$. The squared noise length, which we assume is a sum of $D$ comparable contributions, has a mean of order $D$ and fluctuates around it by $O(\sqrt D)$. Mean field and noise add like Pythagoras, just like in Figure 1:

\begin{equation}
\mathbb E\lVert\mathbf h_i\rVert^2
=\lVert\bar{\mathbf h}_i\rVert^2+v_i\equiv r_i^2,
\qquad v_i=\mathbb E\lVert\boldsymbol\eta_i\rVert^2=(JCJ^\top)_{ii}.
\end{equation}

Even though both $\lVert\bar{\mathbf h}_i\rVert^2$ and $v_i$ can be $O(D)$, the fluctuations around their sum are only $O(\sqrt D)$, so that[^fn:app3]

\begin{equation}
\operatorname{std}\!\left(\frac{\lVert\mathbf h_i\rVert^2}{R^2}\right)=O(D^{-1/2}),
\end{equation}

Figure 2 checks this numerically with samples of the squared field norm for a site in a simulated $N=256$ system. The field-norm distribution clearly narrows around $r_i^2$, and _not_ around the mean-field value $\lVert\bar{\mathbf h}_i\rVert^2$.

{{< lightbox
  src="fieldnorm.png"
  alt="Squared field norms at one site for D = 64 and 1024. The distribution narrows around the closure's prediction, well above the mean-field value."
  caption="**Figure 2.** Squared field norm entering the first update at the site with the sharpest attention ($N_{\mathrm{eff},i}=1/\sum_jJ_{ij}^2\approx1$), from the same system simulated in [Figure 3](#convergence-in-d). Mean field retains only $\lVert\bar{\mathbf h}_i\rVert^2$, which boils down to only about a third of the typical value at this site. Adding $v_i$ locates the center of the distribution, whose width scales as $D^{-1/2}$."
>}}


## A deterministic gain...

Recall that each spin is drawn from a ring around the conditional mean that points along its field:

\begin{equation}
\boldsymbol\mu_D(\mathbf h_i)\simeq g(\lVert\mathbf h_i\rVert^2)\,\mathbf h_i.
\end{equation}

The field's direction fluctuates, but its squared length concentrates around $r_i^2$, so the gain (which only depends on the norm) becomes effectively deterministic in the large $D$ regime. We approximate the gain by the same number across realizations,

\begin{equation}
a_i=g(r_i^2)=\frac{\beta}{1+\gamma_i},
\qquad \gamma_i=\sqrt{1+\beta^2r_i^2/R^2}.
\end{equation}

This makes the mean response a simple deterministic rescaling of the field.

## ... and a lot of noise

The next spin is approximately[^fn:residual]

\begin{equation}
\mathbf s_i'\simeq
\underbrace{a_i\bar{\mathbf h}_i}_{\text{next mean}}
+\underbrace{a_i\boldsymbol\eta_i}_{\text{propagated noise}}
+\underbrace{\boldsymbol\epsilon_i}_{\text{fresh noise}}.
\label{eq:linear}
\end{equation}

Here $\boldsymbol\eta_i=\sum_jJ_{ij}\delta\mathbf s_j$ is the noise already present in the inputs. The fresh noise $\boldsymbol\epsilon_i=\mathbf s_i'-\boldsymbol\mu_D(\mathbf h_i)$ comes from drawing a spin around its conditional mean. Both noise terms average to zero, giving our first closure relation $\mathbf m_i'\simeq a_i\bar{\mathbf h}_i$.

What remains is the next fluctuation, $\delta\mathbf s_i'\simeq a_i\boldsymbol\eta_i+\boldsymbol\epsilon_i$. Pairing two of these gives the next covariance traces, and pairing one with a current fluctuation gives the one-step delayed covariance $\mathcal D_{ij}=\mathbb E[\delta\mathbf s_i'\cdot\delta\mathbf s_j]$:

\begin{align}
C_{ij}'&\simeq a_ia_j\,\mathbb E[\boldsymbol\eta_i\cdot\boldsymbol\eta_j]
+a_i\,\mathbb E[\boldsymbol\eta_i\cdot\boldsymbol\epsilon_j]
+a_j\,\mathbb E[\boldsymbol\epsilon_i\cdot\boldsymbol\eta_j]
+\mathbb E[\boldsymbol\epsilon_i\cdot\boldsymbol\epsilon_j],\\
\mathcal D_{ij}&\simeq a_i\,\mathbb E[\boldsymbol\eta_i\cdot\delta\mathbf s_j]
+\mathbb E[\boldsymbol\epsilon_i\cdot\delta\mathbf s_j].
\end{align}

The two kinds of noise contribute differently.

**Propagated noise.** Mixing the old fluctuations through $J$ gives $\mathbb E[\boldsymbol\eta_i\cdot\boldsymbol\eta_j]=\sum_{kl}J_{ik}J_{jl}C_{kl}=(JCJ^\top)_{ij}$ and $\mathbb E[\boldsymbol\eta_i\cdot\delta\mathbf s_j]=(JC)_{ij}$. The gains then rescale these by $a_ia_j$ and $a_i$.

**Fresh noise.** Given the old spins, $\boldsymbol\epsilon_i$ has zero mean, while $\boldsymbol\eta_j$ and $\delta\mathbf s_j$ are fixed. Every term pairing fresh noise with old fluctuations therefore vanishes, and only propagated noise correlates with the current state. Each site also draws independently, so $\mathbb E[\boldsymbol\epsilon_i\cdot\boldsymbol\epsilon_j]=0$ for $i\neq j$: fresh noise adds only to the diagonal.

The sphere fixes how much fresh noise to add. Every new spin has squared length $R^2$, so $\mathbb E\lVert\boldsymbol\epsilon_i\rVert^2=R^2-\mathbb E\lVert\boldsymbol\mu_D(\mathbf h_i)\rVert^2$ exactly. The conditional mean has typical squared length $a_i^2r_i^2$, so we approximate the remaining variance by

\begin{equation}
b_i\equiv R^2-a_i^2r_i^2=\frac{2R^2}{1+\gamma_i}.
\end{equation}

This estimates the variance of the fresh draw only. The total variance at a site also includes the propagated input noise:

\begin{equation}
C_{ii}'\simeq\underbrace{a_i^2v_i}_{\text{propagated noise variance}}+\underbrace{b_i}_{\text{fresh noise variance}}.
\end{equation}

## Closure equations

Combining these pieces, with $A=\mathrm{diag}(a)$, gives the update equations

\begin{equation}
\boxed{
\begin{aligned}
\mathbf m_i'&=a_i\bar{\mathbf h}_i,\\
C'&=AJCJ^\top A+\mathrm{diag}(b),\\
\mathcal D&=AJC.
\end{aligned}}
\label{eq:closure}
\end{equation}

These equations preserve the spin's squared length:

\begin{equation}
\lVert\mathbf m_i'\rVert^2+C_{ii}'
=a_i^2\bigl(\lVert\bar{\mathbf h}_i\rVert^2+v_i\bigr)+b_i=R^2.
\end{equation}


## What mean field misses

**Norm inflation.** Including $v_i$ lowers the gain, shortening the next mean at a given mean field. Later updates can also change its direction. From a product start, only individual input variances contribute:

\begin{equation}
v_i=\sum_jJ_{ij}^2\left(R^2-\lVert\mathbf m_{0,j}\rVert^2\right)
\leq\frac{R^2}{N_{\mathrm{eff},i}},
\qquad N_{\mathrm{eff},i}=\frac{1}{\sum_jJ_{ij}^2}.
\label{eq:first}
\end{equation}

For couplings with nonnegative rows summing to one (like `softmax`), $N_{\mathrm{eff}}$ ranges from 1 for a single input to $N$ for uniform weights. Diffuse attention suppresses independent input noise and sharp attention retains it.

**Shared inputs.** After an update, sites reading the same inputs acquire equal-time covariance. This shared noise feeds back into $v_i$ and contributes to delayed correlations:

\begin{equation}
\mathcal D_{ij}=\underbrace{a_iJ_{ij}C_{jj}}_{\text{direct}}
+\underbrace{a_i\sum_{l\neq j}J_{il}C_{lj}}_{\text{shared}}.
\end{equation}

The shared term can matter even when $J_{ij}$ is small. With nonnegative $J$ and a product start, the closure generates nonnegative entries of $C$, so the two contributions reinforce each other.


# Numerical checks

We compare exact vMF sampling with naive mean field, diagonal-$C$ closure and full closure. The diagonal version retains norm inflation while dropping shared noise. Mean-field delayed correlations use the previous post's linear-response approximation.

Spins start independently in vMF initializer fields. Drives have random directions and strength $\beta\lVert\mathbf x_i\rVert/R=u$. Couplings are a row-wise $\mathrm{softmax}(\alpha G)$ of a fixed Gaussian matrix: increasing $\alpha$ focuses the weights on fewer inputs. Unless stated otherwise, $\beta=2$ and $u=1$.

Errors are relative Frobenius errors corrected for Monte Carlo noise.[^fn:numerics] Mean errors compare overlaps $\mathbf m_i\cdot\mathbf m_j/R^2$ and $\mathbf x_i\cdot\mathbf m_j/R^2$, except where vector errors are stated, including Figure 4(a). Delayed-correlation errors compare $\mathcal D_{ij}/R^2$. Open markers indicate errors unresolved by sampling.

## Convergence in D

We first fix $N=256$ and $\alpha=3$, giving $N_{\mathrm{eff}}\approx7.5$ on average, and increase $D$ from 16 to 1024.

{{< lightbox
  src="convergence.png"
  alt="Relative error against exact sampling versus D after one and four updates. Mean-field errors level off; resolved closure errors fall roughly as 1/D."
  caption="**Figure 3.** Errors after $K=1$ (top) and $K=4$ (bottom) updates. The [sampled closure](#carrying-noise-instead-of-a-matrix) replaces the dense covariance by $k=16$ random noise channels. From a product start, the diagonal and sampled closures give the same means and $\mathcal D_{ij}$ as the full closure at $K=1$. The equal-time column compares off-diagonal $C_{ij}/R^2$, which mean field and the diagonal closure set to zero. Open markers: the error interval includes zero. Dotted: $1/D$."
>}}

The closure's resolved errors fall roughly as $1/D$. After four updates, its delayed-correlation error drops from 7% at $D=16$ to 0.25% at $D=512$; at $D=1024$ sampling no longer resolves it.

At four updates, mean-field errors plateau near 23% for mean overlaps and 20% for delayed correlations. Diagonal $C$ fixes most of the mean error but leaves a 17% delayed-correlation error from shared noise.

## Norm inflation and shared inputs

A causal mask varies $N_{\mathrm{eff}}$ across sites. After one update at $D=256$, the mean-field error falls from 29% for single-input rows to 3% at $N_{\mathrm{eff}}\approx20$, while the closure stays near 0.4%. The gap between the predictions accounts for nearly all of the mean-field error.

{{< lightbox
  src="mechanisms.png"
  alt="Per-site mean error falls with N_eff. For fixed row weights, placing large weights on common inputs increases errors in mean field and the diagonal closure."
  caption="**Figure 4.** (a) Per-site mean error after one update under a causal mask ($D=256$). Upper series: $\beta=2$; lower, faded: $\beta=1$. Crosses: the gap between mean field and the closure. (b) Errors after four updates with identical row weights, scattered over columns (faded) or placed on common inputs (solid) ($D=64$)."
>}}

After several updates, shared inputs matter too. Keeping each row's weights fixed but moving the large weights onto common columns raises the cross-site contribution to $v_i$ from 1.5% to 37%. Mean field then misses mean overlaps by 76% and delayed correlations by 96%. The diagonal closure reduces the former to 19% but leaves the latter unchanged; the full closure keeps both errors near 5%. The same ordering holds when varying $\beta$ or $u$ separately, with the largest gaps at weak drive.

## The stationary state

At a fixed point, write $B=AJ$. If $\rho(B)<1$, the covariance is

\begin{equation}
C^\star=\sum_{n\geq0}B^n\,\mathrm{diag}\!\left(\frac{2R^2}{1+\gamma}\right)(B^n)^\top.
\end{equation}

Each term is propagated fresh noise. The series assumes self-consistent fixed-point gains; convergence of the coupled recursion requires a separate check. We test relaxation at $N=64$ with diffuse ($N_{\mathrm{eff}}\approx28$) and sharp ($N_{\mathrm{eff}}\approx4$) couplings. Stationary references use time averages after burn-in from one initialization. Means from the two halves of each averaging window agree within sampling noise.

{{< lightbox
  src="relaxation.png"
  alt="Errors along relaxation for diffuse and sharp couplings at D = 64 and 256, settling near their stationary values by K of about 16."
  caption="**Figure 5.** Errors from $K=1$ to 100, with stationary estimates at $\infty$. Solid: $D=256$. Dashed: $D=64$."
>}}

In these runs, errors settle by $K\approx16$ rather than growing with time. The stationary error still decreases with $D$: for diffuse couplings, the closure's delayed-correlation error falls from 5.1% at $D=64$ to 1.2% at $D=256$.

For the previous post's diffuse-coupling ensemble at $(u,\beta)=(1,1)$, shared covariance resolves [the persistent 11% delayed-correlation gap](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/#mean-field-approximation-and-proxy-fidelity):

| $D$ | mean field | diagonal $C$ | full closure |
|---|---|---|---|
| 16 | 13% | 7.5% | 8.8% |
| 64 | 9.4% | 10% | 2.2% |
| 256 | 11% | 12% | 0.53% |
| 1024 | 12% | 12% | 0.13% |

Full closure is slightly worse than diagonal $C$ at $D=16$, but improves with $D$.

Accuracy depends strongly on drive strength. At $(u,\beta)=(0.5,2)$, the closure's mean-overlap errors are 89%, 20% and 1.2% at $D=16$, 64 and 1024; mean-vector errors are 30%, 7% and 0.4%. These tests use random couplings; trained modules need checks at their chosen dimension and operating regime.

# Practical versions

## A one-step correction

From a product initializer, the correction needs one extra pass with squared weights over per-site variances (Eq. \eqref{eq:first}). For drives `X` and initializer fields `F0` of shape `(N, D)`, couplings `J` of shape `(N, N)` and `R2` $=R^2$:

```python
gamma0 = np.sqrt(1 + beta**2 * (F0 * F0).sum(-1) / R2)
M = beta * F0 / (1 + gamma0)[:, None]  # initializer means
var0 = 2 * R2 / (1 + gamma0)  # leading initializer variance, R2 - |M|^2
H = X + J @ M
v = (J * J) @ var0
gamma = np.sqrt(1 + beta**2 * ((H * H).sum(-1) + v) / R2)
M_next = beta * H / (1 + gamma)[:, None]
```

This resembles `RMSNorm` with the field's noise variance included in the normalizer.

## Carrying noise instead of a matrix

The dense update costs $O(N^2D+N^3)$. Carrying $k$ random noise channels per site reduces this to $O(N^2D+kN^2)$. Represent the covariance as

\begin{equation}
\hat C=\mathrm{diag}(\mathbf b)+\frac1kZZ^\top,\qquad Z\in\mathbb R^{N\times k},
\end{equation}

with the fresh variance $\mathbf b$ kept analytic. Set $Y=JZ$ and $v_i=\sum_jJ_{ij}^2b_j+\lVert Y_i\rVert^2/k$, compute the gains as before, and update

\begin{equation}
Z'=A\big(Y+J\,\mathrm{diag}(\sqrt{\mathbf b})\,\Xi\big),\qquad
b_i'=\frac{2R^2}{1+\gamma_i},
\end{equation}

where $\Xi$ has independent standard Gaussian entries. Fresh noise stays diagonal until the couplings mix it across sites. Conditional on the current state, this reproduces the closure's covariance update in expectation; later gains depend on sampled noise, so the full trajectory is not unbiased ([Appendix D](#appendix-d-the-sampled-closure)).

We also test $\sigma=\beta\sum_{ij}(J_{ij}-J_{ji})\mathcal D_{ij}$, an aggregate sensitive to asymmetric delayed correlations. At stationarity this is housekeeping entropy production; during relaxation it is only a diagnostic.

{{< lightbox
  src="sketch.png"
  alt="Sampled-closure errors against the full closure versus channel count k, for mean overlaps, delayed correlations and entropy production, with and without stationary time averaging."
  caption="**Figure 6.** Errors of the sampled closure against the full closure, at $D=64$ and $\alpha=3$. Solid: $N=1024$. Dashed: $N=256$. Dotted, for reference: the full closure's own error against exact sampling."
>}}

Four channels give mean-overlap errors 3–6 times smaller than the closure's own error against exact sampling. For entropy production $\sigma$, four channels suffice at $N=1024$; at $N=256$, sixteen are needed to fall below the closure's error. Errors fall roughly as $k^{-1/2}$.

Individual delayed correlations are harder: their relative error is about $2/\sqrt{k}$, requiring $k\sim10^4$ to match the closure here. At stationarity, averaging over 200 updates cuts this error about fourteenfold in these runs, bringing $k\approx64$–$256$ into reach of the closure's accuracy.


# Relation to other closures

For Ising spins $s_i\in\{-1,1\}$, [Mézard and Sakellariou (2011)](https://arxiv.org/abs/1103.3433) average the single-spin response over a Gaussian local field:

\begin{equation}
\begin{aligned}
m_i'&=\mathbb E_z\tanh\!\left[\beta\left(\bar h_i+\sqrt{\Delta_i}\,z\right)\right],\\
\bar h_i&=H_i+\sum_jJ_{ij}m_j,
\qquad \Delta_i=\sum_jJ_{ij}^2(1-m_j^2),
\end{aligned}
\end{equation}

where $H_i$ is the external field and $z\sim\mathcal N(0,1)$. Their delayed-covariance relation is

\begin{equation}
\mathcal D=\mathrm{diag}(\chi)JC,
\qquad \chi_i=\beta\,\mathbb E_z\left\{1-\tanh^2\!\left[\beta\left(\bar h_i+\sqrt{\Delta_i}\,z\right)\right]\right\}.
\end{equation}

The matrix form matches ours, with a Gaussian-averaged susceptibility $\chi_i$ replacing the radial gain $a_i$. Their argument uses large $N$ and fully asymmetric couplings; ours uses component concentration at fixed $N$ and $J$.

[Huang and Kabashima (2014)](https://arxiv.org/abs/1310.5003) keep the correlations between Gaussian local fields, so the variance becomes $\Delta_i=(JCJ^\top)_{ii}$: the same shared-noise contribution that enters $v_i$ here.

[Aguilera et al. (2021)](https://arxiv.org/abs/2002.04309) place Gaussian-field and TAP ([Roudi and Hertz, 2011](https://arxiv.org/abs/1103.1044)) approximations within a common Plefka framework, expanding around tractable reference dynamics. A large vector dimension at fixed couplings is a different limit; a direct comparison with second-order Plefka for vector spins remains to be done.

The length constraint also links this construction to [equilibrium large-component spin glasses](https://arxiv.org/abs/cond-mat/0408073), where one multiplier per site enforces each spin's length.


# Conclusion and outlook

Noise adds field length that naive mean field discards. At large $D$, that length becomes predictable, closing the equations for the means and covariance traces. The finite-time argument gives a leading approximation; numerical errors fall roughly as $1/D$, but can remain substantial at small $D$ or weak drive.

The one-step correction costs one pass with squared weights. Repeated updates need shared covariance; a few sampled channels suffice for means and entropy production in these runs. Whether this improves trained behavior or training absorbs the mean-field error remains open.

Layers driven by preceding means can reuse the closure, including after learned mixing of heads. Passing sampled spins onward requires tracking how projections reshape their noise; sample-dependent queries and keys also make the couplings fluctuate.


# Acknowledgements

The author (M.B.) would like to thank Á. Poc-López and M. Aguilera for stimulating discussions. This project was a personal experiment in agentic research with substantial human steering. Claude Opus 5.5 and GPT-6 Astra participated in brainstorming, experimental design, coding, writing assistance, figure generation, and multiple rounds of adversarial reviewing. Tactical decisions were impressive, strategic manoeuvres proved harder and required significant steering. The author (M.B.) takes full responsibility for any remaining slop.

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

The exact mean alignment is $A_D(\kappa)=t_\star+O(1/D)$. With $\kappa=\beta R\lVert\mathbf h\rVert$ and $R^2=D/2-1$, this recovers $t_\star\simeq\lVert\boldsymbol\mu_\infty(\mathbf h)\rVert/R$. The most likely latitude stays away from the pole, even though the pole maximizes the density per unit area.

The transverse length remains finite, but its projection onto any fixed unit vector has standard deviation at most $1/\sqrt{D-1}$. For $n$ independent samples,

\begin{equation}
\mathbb E\lVert\bar{\mathbf u}\rVert^2=A_D(\kappa)^2+\frac{1-A_D(\kappa)^2}{n}.
\end{equation}

Averaging divides the noise's expected squared length by $n$; increasing $D$ at fixed $n$ leaves it unchanged.

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

## Appendix C: Why the field norm concentrates

Fix $N$, $J$, $\beta$ and the number of updates, take drives of length $O(R)$, and start from independent vMF spins. Every field length is a linear combination of the overlaps

\begin{equation}
Q_{ij}=\frac{\mathbf s_i\cdot\mathbf s_j}{R^2},
\qquad U_{ij}=\frac{\mathbf x_i\cdot\mathbf s_j}{R^2},
\end{equation}

namely

\begin{equation}
\frac{\lVert\mathbf h_i\rVert^2}{R^2}
=\frac{\lVert\mathbf x_i\rVert^2}{R^2}
+2\sum_jJ_{ij}U_{ij}
+\sum_{jl}J_{ij}J_{il}Q_{jl}.
\end{equation}

It is enough to show that every overlap has variance $O(1/D)$, so that its fluctuations are $O(D^{-1/2})$. Since $Q_{ii}=1$ exactly, only overlaps between distinct sites, and with drives, need checking. We proceed one update at a time.

**Independent spins.** Write $\mathbf s_i=\mathbf m_i+\delta\mathbf s_i$ with independent fluctuations of covariance $\Sigma_i$. A vMF spin has $O(1)$ variance along any direction, while its total variance is at most $R^2$. For distinct sites, the fluctuating parts of $R^2Q_{ij}$ are $\mathbf m_i\cdot\delta\mathbf s_j$, $\delta\mathbf s_i\cdot\mathbf m_j$ and $\delta\mathbf s_i\cdot\delta\mathbf s_j$, with variances

\begin{equation}
\mathbf m_i^\top\Sigma_j\mathbf m_i=O(D),
\qquad \mathbf m_j^\top\Sigma_i\mathbf m_j=O(D),
\qquad \operatorname{tr}(\Sigma_i\Sigma_j)=O(D).
\end{equation}

Dividing by $R^4$ gives variance $O(1/D)$. The drive overlaps $U_{ij}$ only involve the first kind of term.

**One update.** Given the current spins, the next spins are $\mathbf s_i'=\boldsymbol\mu_D(\mathbf h_i)+\boldsymbol\epsilon_i$, where the fresh noise is independent across sites, has zero mean, and again has $O(1)$ variance along any direction. The next overlap between distinct sites splits into a conditional mean and noise:

\begin{equation}
Q_{ij}'=\frac{\boldsymbol\mu_D(\mathbf h_i)\cdot\boldsymbol\mu_D(\mathbf h_j)}{R^2}
+\frac{\boldsymbol\mu_D(\mathbf h_i)\cdot\boldsymbol\epsilon_j+\boldsymbol\epsilon_i\cdot\boldsymbol\mu_D(\mathbf h_j)+\boldsymbol\epsilon_i\cdot\boldsymbol\epsilon_j}{R^2}.
\end{equation}

The noise term has conditional variance $O(1/D)$, by the same calculation as for independent spins, with the conditional means in place of $\mathbf m_i$. The conditional mean is a function of the current overlaps: the response is a field-norm-dependent gain times the field, and $\mathbf h_i\cdot\mathbf h_j/R^2$ is a linear combination of overlaps, like the field length above. Overlaps are bounded, and on this bounded range the function has bounded derivatives, uniformly in $D$. A function with bounded derivatives of variables with variance $O(1/D)$ also has variance $O(1/D)$. By the law of total variance,

\begin{equation}
\operatorname{Var}Q_{ij}'
=\mathbb E\operatorname{Var}(Q_{ij}'\mid\mathbf S)
+\operatorname{Var}\mathbb E(Q_{ij}'\mid\mathbf S)
=O(1/D).
\end{equation}

The same argument applies to $U_{ij}'$, which completes the step. Spins correlated by shared inputs are no obstacle: only the fresh noise needs to be independent, and it is, given the current spins.

This establishes concentration at any fixed number of updates. The constants can compound from step to step and grow with $N$ and the couplings, so stationary accuracy needs a separate check.

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
\qquad \hat{\mathcal D}=A\left(J\,\mathrm{diag}(\mathbf b)+\frac{YZ^\top}{k}\right).
\end{equation}

Evolution costs $O(N^2D+kN^2)$ and adds $O(Nk)$ covariance storage. Dense $C$ or $\mathcal D_{ij}$ readouts can be formed when needed; the stationary entropy-production estimate $\hat\sigma_{\mathrm{hk}}^\star=\beta\sum_{ij}(J-J^\top)_{ij}\hat{\mathcal D}_{ij}^\star$ can be computed directly from the factors. In attention terms, $Z$ adds $k$ value channels under the same weights.

Channel fluctuations scale roughly as $k^{-1/2}$. Gains and aggregate observables can average over many sites as well; an individual delayed correlation remains one sampled inner product. This explains their different accuracies in Figure 6.

Later gains depend on sampled $v_i$, so the full trajectory is not exactly unbiased. An $O(1/k)$ bias is plausible from a second-order expansion, but has not been established. The sphere identity holds conditionally in expectation at each update. In Figure 6, the per-site variance misses $R^2-\lVert\mathbf m_i\rVert^2$ by $3$–$5\%$ of $R^2$ at $k=4$ and under $1\%$ at $k=256$, without renormalization.

At stationarity, fresh channel draws can make time averaging useful, but successive states still share propagated noise. The gain depends on autocorrelation. Averaging 200 updates reduced delayed-correlation error about fourteenfold in these runs, mean-overlap error by 11–14 times, and entropy-production error by 7–23 times. These are empirical reductions, not a general effective-sample-size rule.


# Footnotes

[^fn:radiusconvention]: Scaling $R^2\propto D$ follows the large-component convention for vector spins ([Stanley 1968](https://link.aps.org/doi/10.1103/PhysRev.176.718)). The choice $R^2=D/2-1$ also matches the Bessel order appearing in vMF means and simplifies their leading-order formulas. In units of $R$, the leading response depends only on $\beta\lVert\mathbf h\rVert/R$.

[^fn:app3]: [Appendix C](#appendix-c-why-the-field-norm-concentrates) tries to make this precise over several updates, where shared inputs can correlate the spins.

[^fn:residual]: Eq. \eqref{eq:linear} becomes exact once we add the residual $\mathbf e_i=\boldsymbol\mu_D(\mathbf h_i)-a_i\mathbf h_i$. It has two parts:

    - The finite-$D$ response, $\boldsymbol\mu_D(\mathbf h_i)-\boldsymbol\mu_\infty(\mathbf h_i)$: a relative $O(1/D)$ correction to a vector of length $O(\sqrt D)$, so its length is $O(D^{-1/2})$.
    - The fluctuating gain, $\bigl[g(\lVert\mathbf h_i\rVert^2)-g(r_i^2)\bigr]\mathbf h_i\approx g'(r_i^2)\,\Delta_i\mathbf h_i$ with $\Delta_i=\lVert\mathbf h_i\rVert^2-r_i^2$. Since $g'=O(1/D)$, $\Delta_i=O(\sqrt D)$ and $\lVert\mathbf h_i\rVert=O(\sqrt D)$, its length is $O(1)$.

    The residual's mean shifts $\mathbf m_i'$ and its fluctuation adds to the noise, both by $O(1)$ against a spin length $R=O(\sqrt D)$. At a fixed number of updates, normalized moments are thus accurate to $O(D^{-1/2})$. The $1/D$ decay we observe numerically is faster than this bound.

[^fn:numerics]: Reference moments use exact vMF sampling ([Wood, 1994](https://doi.org/10.1080/03610919408813161)) over independent replicates. Squared errors are formed from products across distinct replicates, which cancels Monte Carlo variance in expectation. Intervals show two jackknife standard errors, leaving out one replicate at a time.
