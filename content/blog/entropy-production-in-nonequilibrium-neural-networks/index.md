---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Entropy Production in Nonequilibrium Neural Networks"
subtitle: "A nonequilibrium thermodynamics perspective on transformers"
summary: "A nonequilibrium thermodynamics perspective on transformers"
aliases:
  - /post/entropy-production-in-non-equilibrium-neural-networks/
authors:
  - me
tags: ["Artificial Intelligence", "Associative Memories", "Attention", "Cybernetics", "Deep Learning", "Dynamical Systems", "Entropy Production", "Ising Models", "Many-Body Systems", "Mean-Field Theory", "Neural Networks", "Near-Equilibrium Dynamics", "Nonequilibrium Dynamics", "Quench Dynamics", "Relaxation", "Self-Organizing Computational Stability", "Statistical Physics", "Steady State", "Stochastic Thermodynamics", "Transformers", "Vector-Spin Models"]
categories: []
date: 2026-02-02T09:28:17+01:00
lastmod: 2026-07-16T08:30:41+01:00
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

<a title="Walter Baxter / A murmuration of starlings at Gretna" href="https://commons.wikimedia.org/wiki/File:Starling_murmuration.jpg"><img width="512" alt="A murmuration of starlings at Gretna" src="https://upload.wikimedia.org/wikipedia/commons/8/8d/Starling_murmuration.jpg?20150218191823"></a>

---

# Introduction

> **✨ GitHub repository:  [`mcbal/neqnn`](https://github.com/mcbal/neqnn) (work in progress)**

Transformers are powerful driven dynamical systems, yet their internal computation is rarely discussed in terms of nonequilibrium thermodynamics. Building on dynamical mean-field theory for vector-spin models introduced in [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/), we design a minimal parallel transformer-like module whose forward pass implements a controllable quench-and-relax process.

We characterize its dynamical regimes to elucidate a design space of stateless and stateful variations of transformer-like and deep-equilibrium-like architectures and leverage its physics-based architecture to compute differentiable proxies for housekeeping entropy production and post-quench relaxation mismatch. In this way, a _spin-transformer module_ separates fast state relaxation, changing external drive, and slow parameter learning, making the trade-off between memory, responsiveness, stationary circulation, and adaptation measurable. The mean-field spin-model framework provides both a scalable laboratory for nonequilibrium many-body dynamics on modern accelerators and a testable learning hypothesis.

We explore whether a self-supervised protocol of (1) predicting the next drive-conditioned steady state before the new drive arrives, (2) using the actually observed drive to generate a detached corrected target, and (3) minimizing the resulting predictive mismatch can produce structure-sensitive dynamics without global end-to-end credit assignment. The risk is that the system finds shortcuts leading to representational collapse, inertia, flattened dynamics, and other failure modes. The bet is that there is sufficient signal in learning to reduce relaxation after a quench to learn to predict. We readily admit that the main motivation for this bet is aesthetic. To move beyond aesthetics, we run numerical experiments.


# Driving a spin-transformer module

In this section we design a minimal spin-transformer module whose forward pass implements a controllable nonequilibrium quench-and-relax process. We identify three timescales and two dynamical regimes, leading to a natural categorization of the design space into stateless and stateful variations of transformer-like and deep-equilibrium-like architectures.

## A minimal controllable drive-conditioned system

In [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/) we showed how to apply dynamical mean-field theory to approximate the time-dependent behavior of asymmetric vector-spin models. We started from a spin system of $N$ vector spins $\mathbf{s}_{i,t} \in \mathbb{R}^{D}$ talking to each other via an $N \times N$ pairwise coupling matrix $J_{ij}$ with the underlying parallel-updates stochastic dynamics characterized by a discrete-time Markov chain transition probability $P(\mathbf{s}_{t} | \mathbf{s}_{t-1})$. External magnetic fields $\mathbf{x}_{i,t} \in \mathbb{R}^{D}$ bias the vector spins and act as local drives. We use the notation $\mathbf{x}_{t} = \{ \mathbf{x}_{i,t} \}^{N}_{i=1}$ to refer to the full-drive matrix instead of the local drive at a single position.

<img src="vector_spins.png" alt="Random Ising model configuration with vector spins" width="250px"/>

Using a simple first-order `Plefka[t-1,t]` mean-field approximation, we calculated a closed expression for updating the spin expectation values in the large-vector-dimension limit,

\begin{equation}
\mathbf{m}_{i,t} = \frac{\beta \left( \mathbf{x}_{i,t} + \sum_{j} J_{ij} \mathbf{m}_{j,t-1} \right)}{1+\sqrt{1+\beta^2 \lVert \mathbf{x}_{i,t} + \sum_{j} J_{ij} \mathbf{m}_{j,t-1} \rVert^2 / R^2 }},
\end{equation}

where the magnetization vectors $\mathbf{m}_{i,t} \in \mathbb{R}^{D}$ capture the mean-field influence of the spins on each other. Additionally, $\beta$ denotes the inverse temperature and $R=\sqrt{D/2 -1}$ is the natural hyperspherical length scale resulting from the large-vector-dimension approximation[^fn:largedlim].

If we now consider some kind of _parameterized drive-dependent couplings_

\begin{equation}
  \mathbf{J} (\mathbf{x}_{t}) = \mathrm{softmax}\left( \mathbf{x}_{t} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{T} \mathbf{x}_{t}^{T} \right), \label{eq:softmax}
\end{equation}

then we turn the fixed-size $N \times N$ coupling matrix into a parameterized rule that supports variable system size, drive-dependent routing, and a way to scale system size without learning new explicit parameters[^fn:couplings]. If we also augment the local drives with some kind of _parameterized non-linear drive-dependent field_,

\begin{equation}
  \mathbf{x}_{i,t} \to \mathbf{x}_{i,t} + \mathbf{FFN}\left( \mathbf{x}_{i,t} \right),
\end{equation}

where $\mathbf{FFN}$ denotes a position-wise feed-forward network, then our earlier recurrence relation becomes

\begin{equation}
  \mathbf{m}_{i,t,k+1} = \frac{\beta \left( \mathbf{x}_{i,t} + \mathbf{FFN}\left( \mathbf{x}_{i,t} \right) + \sum_{j} J_{ij} (\mathbf{x}_{t}) \mathbf{m}_{j,t,k} \right)}{1+\sqrt{1+\beta^2 \lVert \mathbf{x}_{i,t} + \mathbf{FFN}\left( \mathbf{x}_{i,t} \right) + \sum_{j} J_{ij} (\mathbf{x}_{t}) \mathbf{m}_{j,t,k} \rVert^2 / R^2 }}, \label{eq:paralleltransformer}
\end{equation}

where we have pushed down the recurrence into an internal relaxation index $k$ as we reserve $t$ for changes in the external drive. By making the effective drive as well as the couplings depend on the drive $\mathbf{x}_{t}$, a sudden shift $\mathbf{x}_{t} \to \mathbf{x}_{t+1}$ changes both the local fields as well as the interactions and quenches the module into a new instantaneous dynamics[^fn:protocol]. During internal relaxation updates $k$, the drive $\mathbf{x}_{t}$ and the parameters $\boldsymbol{\theta} = \{ \mathbf{W}_{Q}, \mathbf{W}_{K}, \mathbf{FFN} \}$, and therefore the transition rule, are held fixed. (At the stochastic level, the process looks something like $P_{\boldsymbol{\theta}, \mathbf{x}_{t}}(\mathbf{s}_{t, k+1} | \mathbf{s}_{t, k})$.)

We end up with a _highly reconfigurable system_ that is _dynamically shaped by the drive_. It is these dynamic drive dependencies that enable encoding subtle correlational structures, even though the system is treated only at mean-field level. Each vector spin effectively experiences a local mean-field that is the sum of a residual stream drive, a feed-forward-like drive, and attention-like couplings. Importantly, these terms contain _parameters_ that can be _shaped through training_: we can _control how the system responds_, fluctuates, and relaxes after getting quenched. Slow parameter updates then make this responsive system adaptive.

> Note that all results in this post are based on the transformer-like architecture derived from a simple first-order `Plefka[t-1,t]` mean-field approximation built around independent models at times $t−1$ and $t$. This is of course not the only possible approximation choice. Different mean-field ansätze will lead to different architectures.


## Building modules: three clocks, slow plasticity, and two relaxation limits

Looking at Eq. \eqref{eq:paralleltransformer} we already notice its close resemblance to the forward pass of a [parallel transformer block](https://xn--rss.to/parallel-transformer-blocks.html). To make this more precise, we need to specify _how to implement_ spin-transformer modules in practice. What is up with this weird internal state and internal relaxation dimension? How can this system even serve as a neural network module?

Let us begin by writing the forward pass Eq. \eqref{eq:paralleltransformer} more generally as

\begin{equation}
  \mathbf{m}^{(l)}_{t, k+1} = F_{\boldsymbol{\theta}^{(l)}_{n}} \left( \mathbf{x}^{(l)}_{t}, \mathbf{m}^{(l)}_{t, k} \right)
\end{equation}

and clearly state the clocks involved:

- $k$ indexes fast internal relaxation within the module
- $t$ indexes changes in the environmental drive or input context
- $n$ indexes slow parameter updates $\boldsymbol{\theta}^{(l)}_{n+1} = \boldsymbol{\theta}^{(l)}_{n} + \eta \nabla_{\boldsymbol{\theta}^{(l)}} \mathcal{L}^{l}_{t}$ for some learning rate $\eta \ll 1$ and (potentially layer-dependent) loss function $\mathcal{L}^{l}_{t}$ where slow here actually means small parameter changes since the optimizer clock $n$ often tracks the drive clock $t$ in practice
- $l$ indexes network depth (number of stacked layers)

> A deep network is a stack of parameterized modules, which, in our framework, make up a collective of _different_ driven spin systems driving each other sequentially. The layer index does not (have to) correspond to external time nor to internal relaxation time; it is an additional axis labeling the simple feed-forward topology (depth) of the computational graph. For clarity, we drop it in the remainder of this section.

The three clocks define three characteristic rates: $\tau_{\mathrm{relax}}$, $\tau_{\mathrm{drive}}$, and $\tau_{\mathrm{learn}}$. Throughout, we assume slow plasticity $\tau_{\mathrm{drive}} \ll \tau_{\mathrm{learn}}$. The relative size of $\tau_{\mathrm{relax}}$ and $\tau_{\mathrm{drive}}$, or, equivalently, the number of internal updates allocated before the next quench, determines the computational regime.

We frame our module-design intuition around a _quench-and-relax scenario_: when the input drive switches, _i.e._ $\mathbf{x}_{t-1} \to \mathbf{x}_{t}$, the spin system has to adapt to the sudden change. A general post-quench module then looks like

\begin{equation}
  \mathbf{m}_{t, K} = F^{K}_{\boldsymbol{\theta}_{n}} \left( \mathbf{x}_{t}, \mathbf{m}_{t, 0} \right),
\end{equation}

with two independent design choices: the number of internal relaxation steps $K$ (relaxation horizon) and the choice of $\mathbf{m}_{t, 0}$ (initialization policy), leading to the following design space:

| | Reset or amortized initialization           | Carried initialization                        |
| -- | --------------------------------- | ----------------------------------------- |
| **Finite-step regime with $K < \infty$** | finite-depth, stateless, **transformer-like** module | recurrent stateful module |
| **Fixed-point regime where $K \to \infty$** | implicit or **deep-equilibrium-like** (DEQ) module   | identical if the fixed point is unique; path-dependent if mean-field branches coexist                   |

### Finite-step regime

In this regime, only $K < \infty$ internal updates are allocated before the next quench, which may reflect genuine competition between relaxation and drive timescales, or simply deliberate computational truncation as in recent looped and recursive-reasoning approaches. The intuition here is that the system tracks a moving family of instantaneous stationary marginals with potentially nonzero lag. In this regime, there can be no strong separation between drive and relaxation if approaching the steady state takes more than $K$ steps. But, since the module is controllable, the outer loop can nudge the system's parameters $\boldsymbol{\theta}_{n}$ towards more efficient and useful relaxation.

The initialization $\mathbf{m}_{t, 0} = \mathbf{m}_{t-1, K}$ makes the module architecture genuinely recurrent and stateful, but with a full context window of hidden states, situating it somewhere in between recurrent neural networks and transformers. Another option is to warm-start with a learned amortized initializer $\mathbf{m}_{t, 0} = \mathbf{x}_{t}\mathbf{W}_{V}$ for the post-quench relaxation, which estimates the drive-conditioned response to which the module should relax. If we call these initializations _values_, then, for $K=1$, the forward pass pretty much matches that of a parallel transformer block.

### Fixed-point regime

In this regime, $\tau_{\mathrm{relax}} \ll \tau_{\mathrm{drive}}$ so we consider $\mathbf{x}_{t}$ clamped and let $K \to \infty$ until the deterministic mean-field equations converge to fixed-point magnetizations $\mathbf{m}^{\star}_{t}(\mathbf{x}_{t})$ compatible with the frozen drive $\mathbf{x}_{t}$. These values approximate the stationary marginals of an underlying instantaneous frozen-drive nonequilibrium steady state (NESS). The intuition here is that the clamped input fixes an instantaneous stochastic transition rule. Although its one-point marginals become stationary, asymmetric couplings can sustain probability currents and positive entropy production beneath those stationary marginals.

In case of a unique fixed point, the initial values $\mathbf{m}_{t, 0}$ are erased, and the module is stateless. But the deterministic mean-field equations may admit multiple stable fixed-point branches or basins. Warm-starting with $\mathbf{m}_{t, 0} = \mathbf{m}^{\star}_{t-1}$ can then produce path-dependent branch selection and hysteresis behavior.

## On the connection to transformers

Let us step back for a bit and emphasize that this close resemblance between forward passes acts as a _plausibility bridge_ at this point. It is _not evidence_ that trained transformers literally implement the approximated nonequilibrium thermodynamics scenarios we will cover in the next sections. But the proximity in module architecture space of a minimal spin-transformer to a class of transformers known to scale does at least suggest that transformers may also admit module-level nonequilibrium interpretations.

> **A nonequilibrium picture of autoregressive inference from a quench-and-relax perspective:** a freshly generated token changes the context window and therefore quenches the stack of modules beginning at the bottom, the module relaxes and then drives the next module consecutively all the way to the top where the final magnetizations get mapped to a probability distribution to sample the next token from. The parameters of the stack of modules have been carefully optimized during successive stages of training to implement useful finite-step relaxation.

Even on their own, spin-transformer modules have merit since they turn transformer-like neural networks into computational laboratories for nonequilibrium dynamics that can be executed on modern accelerators at scale. This makes it possible to study large, high-dimensional systems with structured input-dependent couplings, nonstationary data streams, and slowly adapting parameters rather than staying close to analytically tractable toy models. The resulting observables remain mean-field approximations, and must be calibrated against exact stochastic systems at small scale. But once calibrated, the framework offers a route to computational experiments on collective adaptation and irreversible organisation in regimes that are otherwise difficult to access.

We end this section with a cheat sheet mapping concepts between spin-transformer modules and transformer modules.

| Spin-transformer module           | Transformer module                        |
| --------------------------------- | ----------------------------------------- |
| Local drives $\mathbf{x}_{i,t}$   | Input embeddings and latent embeddings    |
| "The drive" $\mathbf{x}_{t}$      | Current context window                    |
| Parameterized couplings $J_{ij}$  | Attention matrix                          |
| Magnetizations $\mathbf{m}_{i,t}$ | Latent embeddings and output embeddings   |


# Computing differentiable entropy-production proxies

We could stop here, pretrain a spin-transformer model using next-token prediction on chunks of text, and compare evaluation metrics to those of compute-matched vanilla transformers. Let us focus instead on what our framework enables that feels hard to come up with _without_ having access to an underlying nonequilibrium spin-model perspective. In this section, we show how the quench-and-relax process behind the forward pass of a spin-transformer module relates to notions of _irreversibility_. We add just[^fn:lit] enough physical context to introduce differentiable entropy production proxies that we can compute at the same mean-field level as the spin system.

## Two ways to be irreversible

During a quench-and-relax process we hold the input drive $\mathbf{x}_{t}$ fixed and let the spin system settle. Its average magnetizations $\mathbf{m}_{t}$ may stop changing, but its microscopic dynamics need not become reversible. Asymmetric couplings can sustain circulating probability currents when forward sequences of spin configurations remain more likely than their backward step reversals. We call this source of irreversibility _steady-state irreversibility_. Its entropy-production rate is the running cost of maintaining a nonequilibrium steady state under the current input drive. For our system, we can estimate this "housekeeping" entropy production from asymmetric couplings and delayed correlations,

\begin{equation}
  \sigma^{\star}_{\mathrm{hk},t} = \beta \sum_{ij} \left(J_{ij}(\mathbf{x}_{t}) - J_{ji}(\mathbf{x}_{t})\right) D^{\star}_{ij,t} , \label{eq:sigma_hk}
\end{equation}

where $D^{\star}_{ij,t}$ denote the one-step delayed correlations evaluated under the stationary one-step joint. Intuitively, this is like

\begin{equation}
  \sigma^{\star}_{\mathrm{hk},t} = \sum_{ij} \left[\operatorname{directionality}\right]_{ij} \times \left[\operatorname{delayed\ flow}\right]_{ij,t}.
\end{equation}

The couplings provide a directional bias; the delayed correlations report whether fluctuations actually propagate along that direction. A fixed point of the magnetizations therefore does not imply equilibrium: it describes stationary averages, not the absence of microscopic currents.

The system is snuggly settled and at peace. Time to quench again! After changing the input drive abruptly, $\mathbf{x}_{t} \to \mathbf{x}_{t+1}$, the system is still distributed approximately according to its old steady state $\pi_{t}$, while the new transition rule $P_{\boldsymbol{\theta}, \mathbf{x}_{t+1}}(\mathbf{s}_{t+1, k+1} | \mathbf{s}_{t+1, k})$ induced by the new input drive actually favors another steady state $\pi_{t+1}$. The relative-entropy distance to the frozen-drive steady state
\begin{equation}
M_{t+1, k} = D_{\mathrm{KL}}\left(p_{t+1,k} \lVert \pi_{t+1} \right),\label{eq:vmfkl}
\end{equation}
where $p_{t+1,0} = \pi_{t}$ for a fully relaxed carried-state quench, is a relaxation-irreversibility proxy measuring post-quench relaxation mismatch. During relaxation under a fixed transition rule, its decrease is related to nonadiabatic entropy production. The new housekeeping part remains after relaxation while $M_{t+1, k} \to 0$ as the actual distribution relaxes to the new stationary distribution.

> A driven spin-transformer module with asymmetric couplings can thus be irreversible in two ways in the quench-and-relax process: the cost of **"running"** a nonequilibrium steady state after relaxation and the **"catching up"** during relaxation after its input drive changes.

## Mean-field proxy for housekeeping entropy production

To evaluate Eq. \eqref{eq:sigma_hk}, we need stationary one-step delayed correlations $D^{\star}_{ij,t}$. To this end, let us first compute a first-order `Plefka[t-1,t]` mean-field approximation of the _transient_ one-step delayed correlations $D_{ij,t,k}$,

\begin{equation}
  D_{ij,t,k} = \mathbb{E}_{(\mathbf{s}, \mathbf{s}') \sim p_{t,k-1}(\mathbf{s}) P_{\boldsymbol{\theta}, \mathbf{x}_{t}}(\mathbf{s}' | \mathbf{s})} \left[ \left( \mathbf{s}'_{i} - \mathbf{m}_{i,t,k} \right) \cdot \left( \mathbf{s}_{j} - \mathbf{m}_{j,t,k-1}\right) \right] ,
\end{equation}

Evaluating this expression as we did for the magnetizations in [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/), we end up with the mean-field approximation

\begin{align}
  D_{ij,t,k} \approx \beta J_{ij}(\mathbf{x}_{t}) \operatorname{Tr} \left( \Sigma_{i,t,k} \Sigma_{j,t,k-1} \right),
\end{align}

where $\Sigma_{i,t,k} = \operatorname{Cov} \left[ s_{i,t,k} \right]$ denotes the single-site covariance / susceptibility. The trace captures which directions on the vector-spin sphere are still available to fluctuate. If a spin is weakly magnetized, it has many soft directions. If it is strongly magnetized, many directions are suppressed because the spin is pinned close to its mean direction.

At stationarity, $p_{t,k-1} \to \pi_{t}$ and $\mathbf{m}_{i,t,k} \to \mathbf{m}^{\star}_{i,t}$ so that

\begin{align}
  D^{\star}_{ij,t} = \lim_{k\to\infty} D_{ij,t,k} = \mathbb{E}_{\mathbf{s} \sim \pi_{t}, \mathbf{s'} \sim P_{\boldsymbol{\theta}, \mathbf{x}_{t}}(\cdot | \mathbf{s})} \left[ \left( \mathbf{s'}_{i} - \mathbf{m}^{\star}_{i,t} \right) \cdot \left( \mathbf{s}_{j} - \mathbf{m}^{\star}_{j,t}\right) \right]
\end{align}

and the mean-field approximation becomes

\begin{align}
  D^{\star}_{ij,t} \approx \beta J_{ij}(\mathbf{x}_{t}) \operatorname{Tr} \left( \Sigma^{\star}_{i,t} \Sigma^{\star}_{j,t} \right),
\end{align}

Substituting the large-$D$ approximation from [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/),

\begin{align}
  \Sigma_{i,t} \approx \frac{\mathbf{1}_{D}}{1+\gamma_{i,t}} - \frac{\mathbf{m}_{i,t} \mathbf{m}_{i,t}^{T}}{R^2 \gamma_{i,t}},
\end{align}

we end up with the $(i \leftrightarrow j)$-symmetric explicit expression

\begin{align}
  C^{\star}_{ij,t} &= \operatorname{Tr} \left( \Sigma^{\star}_{i,t} \Sigma^{\star}_{j,t} \right) \geq 0 \nonumber \\
  &= \frac{D}{(1+\gamma^{\star}_{i,t})(1+\gamma^{\star}_{j,t})} \nonumber\\
  &- \frac{\lVert \mathbf{m}^{\star}_{j,t} \rVert^2}{R^2 \gamma^{\star}_{j,t} \left( 1 + \gamma^{\star}_{i,t} \right)} \nonumber\\
  &- \frac{\lVert \mathbf{m}^{\star}_{i,t} \rVert^2}{R^2 \gamma^{\star}_{i,t} \left( 1 + \gamma^{\star}_{j,t} \right)} \nonumber\\
  &+ \frac{\left( \mathbf{m}^{\star}_{i,t} \cdot \mathbf{m}^{\star}_{j,t} \right)^2}{R^4 \gamma^{\star}_{i,t} \gamma^{\star}_{j,t}},
\end{align}

where

\begin{align}
  \gamma^{\star}_{i,t} &= \sqrt{1 + \beta^2 \lVert \mathbf{h}^{\star}_{i,t} \rVert^2 / R^2 },\\
  \mathbf{h}^{\star}_{i,t} &= \mathbf{x}_{i,t} + \mathbf{FFN}\left( \mathbf{x}_{i,t} \right) + \sum_{l} J_{il} (\mathbf{x}_{t}) \mathbf{m}^{\star}_{l,t}.
\end{align}

Since $\gamma^{\star} \sim O(1)$, the one-step delayed correlations $D^{\star}_{ij,t}$ are dominated by a large angle-independent isotropic contribution $O(D)$ with the other three terms being $O(1)$. The housekeeping entropy production proxy becomes

\begin{equation}
  \sigma^{\star}_{\mathrm{hk},t} \approx \beta^2 \sum_{ij} \left(J_{ij}(\mathbf{x}_{t}) - J_{ji}(\mathbf{x}_{t})\right) J_{ij}(\mathbf{x}_{t}) C^{\star}_{ij,t} ,
\end{equation}

or, using the symmetry of $C^{\star}_{ij,t}$,

\begin{equation}
  \sigma^{\star}_{\mathrm{hk},t} \approx \frac{\beta^2}{2} \sum_{ij} \left(J_{ij}(\mathbf{x}_{t}) - J_{ji}(\mathbf{x}_{t})\right)^2 C^{\star}_{ij,t} .
\end{equation}

For unconstrained couplings, the first-order stationary proxy measures squared coupling nonreciprocity, weighted by how strongly the fluctuation spaces of the two sites overlap. Under a strict causal mask, reciprocal off-diagonal pairs are forbidden by construction, and the same expression reduces to a susceptibility-weighted squared norm of the causal attention weights, closely related to attention concentration rather than learned antisymmetry.


## Mean-field proxy for post-quench relaxation mismatch

To evaluate Eq. \eqref{eq:vmfkl}, we use the fact that the mean-field vector-spin model machinery is built on the von Mises-Fischer distribution (see Appendix A).

Explicitly, the single-site contribution comparing the current state $q_{i,t,k}$ with the frozen-drive fixed-point state $q_{i,t}^{\star}$ is given by

\begin{equation}D_{\mathrm{KL}}\left(q_{i,t,k}\Vert q_{i,t}^{\star}\right)=\log\frac{C_D(\kappa_{i,t,k})}{C_D(\kappa_{i,t}^{\star})}+A_D(\kappa_{i,t,k})\left(\kappa_{i,t,k}-\kappa_{i,t}^{\star}(\boldsymbol\mu_{i,t,k})^\top\boldsymbol\mu_{i,t}^{\star}\right).\end{equation}

Substituting $\boldsymbol\mu_{\mathbf h}=\mathbf h/\lVert\mathbf h\rVert$ and $\kappa_{\mathbf h}=\beta R\lVert\mathbf h\rVert$ gives the equivalent effective-field expression

\begin{equation}D_{\mathrm{KL}}\left(q_{\mathbf h_a}\Vert q_{\mathbf h_b}\right)=\log\frac{C_D(\beta R\lVert\mathbf h_a\rVert)}{C_D(\beta R\lVert\mathbf h_b\rVert)}+\beta R A_D(\beta R\lVert\mathbf h_a\rVert)\left(\lVert\mathbf h_a\rVert-\frac{\mathbf h_a^\top\mathbf h_b}{\lVert\mathbf h_a\rVert}\right).\end{equation}

Here $\mathbf h_a$ parameterizes the distribution in the first argument of the KL divergence and $\mathbf h_b$ parameterizes the reference distribution in the second argument. The original vMF form above remains well defined when $\mathbf h_a=\mathbf 0$, in which case $\kappa_a=0$ and its mean direction is irrelevant.

In the large-vector-dimension limit, we again set $R^2=D/2-1$ and introduce

\begin{equation}\gamma_{\mathbf h}=\sqrt{1+\frac{\beta^2\lVert\mathbf h\rVert^2}{R^2}}.\end{equation}

Using the leading large-$D$ magnetization response $\mathbf m_{\mathbf h}\approx\beta\mathbf h/(1+\gamma_{\mathbf h})$, the KL divergence becomes

\begin{equation}D_{\mathrm{KL}}^{D\to\infty}\left(q_{\mathbf h_a}\Vert q_{\mathbf h_b}\right)\approx\frac{\beta^2\left(\lVert\mathbf h_a\rVert^2-\mathbf h_a^\top\mathbf h_b\right)}{1+\gamma_{\mathbf h_a}}+R^2\left(\gamma_{\mathbf h_b}-\gamma_{\mathbf h_a}-\log\frac{1+\gamma_{\mathbf h_b}}{1+\gamma_{\mathbf h_a}}\right).\end{equation}

The same expression can be written entirely in terms of the corresponding magnetizations. The large-$D$ response can be inverted as

\begin{equation}
\mathbf{h}\approx\frac{2R^2}{\beta\left(R^2-\lVert\mathbf{m}\rVert^2\right)}\mathbf{m},\qquad \lVert\mathbf{m}\rVert < R. \end{equation}

Substitution gives the magnetization-only form

\begin{equation}D_{\mathrm{KL}}^{D\to\infty}\left(q_{\mathbf m_a}\Vert q_{\mathbf m_b}\right)\approx R^2\log\frac{R^2-\lVert\mathbf m_b\rVert^2}{R^2-\lVert\mathbf m_a\rVert^2}+\frac{2R^2\left(\lVert\mathbf m_b\rVert^2-\mathbf m_a^\top\mathbf m_b\right)}{R^2-\lVert\mathbf m_b\rVert^2}.\end{equation}

For the post-quench relaxation considered here, $\mathbf m_a=\mathbf m_{i,t,k}$ is the current magnetization and $\mathbf m_b=\mathbf m_{i,t}^{\star}$ is the instantaneous fixed-point magnetization. The full factorized mean-field mismatch is therefore

\begin{equation}M_{t,k}^{\mathrm{MF}}\approx\sum_i\left[R^2\log\frac{R^2-\lVert\mathbf m_{i,t}^{\star}\rVert^2}{R^2-\lVert\mathbf m_{i,t,k}\rVert^2}+\frac{2R^2\left(\lVert\mathbf m_{i,t}^{\star}\rVert^2-\mathbf m_{i,t,k}^\top\mathbf m_{i,t}^{\star}\right)}{R^2-\lVert\mathbf m_{i,t}^{\star}\rVert^2}\right].\end{equation}

This expression is asymmetric, as required for a KL divergence. It vanishes when $\mathbf m_{i,t,k}=\mathbf m_{i,t}^{\star}$ at every site and penalizes both differences in magnetization norm and angular misalignment with the frozen-drive fixed-point response.


# A local self-supervised learning rule

In this section, we show how the quench-and-relax intuition points to a local self-supervised learning rule. Instead of adding a separate prediction network, we use the same relaxation dynamics of the spin-transformer module forward pass to both anticipate the next response and to incorporate the next observation. Prediction and then correcting-the-prediction are two runs of the same spin dynamics under different clamping patterns where only the source and timing of the drives change.


## A module-local prediction-and-relaxation protocol

Assume that the module has relaxed under the current drive \(\mathbf{x}_t\), producing fixed-point magnetizations

\begin{equation}\mathbf{m}^{\star}_{t}=\lim_{K\rightarrow\infty}F^{K}_{\boldsymbol{\theta}_{n}}\left(\mathbf{x}_{t},\mathbf{m}_{t,0}\right).\end{equation}

Before the next observation is available, we construct a predictive drive

\begin{equation}\mathbf{x}^{-}_{t+1\mid t},\end{equation}

built from information available at time \(t\). For a rolling context window, this could be the shifted current context with the not-yet-observed positions masked, nulled, or left unclamped. In a closed-loop setting, it may additionally contain the action applied at time \(t\). Constructing this drive is an _interface protocol_, not a separate learned predictor.

Starting from the current response \(\mathbf{m}^{\star}_{t}\), we apply the same module dynamics for a finite prediction horizon \(K_{\mathrm{pred}}\):

\begin{equation}\mathbf{m}^{-}_{t+1}=F^{K_{\mathrm{pred}}}_{\boldsymbol{\theta}_{n}}\left(\mathbf{x}^{-}_{t+1\mid t},\mathbf{m}^{\star}_{t}\right).\end{equation}

The superscript \({-}\) indicates that this state is computed before observing the new drive \(\mathbf{x}_{t+1}\). It is the response that the existing spin dynamics predict from the information available at time \(t\).

When the next observation arrives, it supplies the actual drive \(\mathbf{x}_{t+1}\) and quenches the system again. Starting from the predictive state, the module relaxes according to

\begin{align}\mathbf{m}^{+}_{t+1,k+1}&=F_{\boldsymbol{\theta}_{n}}\left(\mathbf{x}_{t+1},\mathbf{m}^{+}_{t+1,k}\right),\\\mathbf{m}^{+}_{t+1,0}&=\mathbf{m}^{-}_{t+1},\end{align}

until it reaches

\begin{equation}\mathbf{m}^{\star}_{t+1}=\lim_{K\rightarrow\infty}F^{K}_{\boldsymbol{\theta}_{n}}\left(\mathbf{x}_{t+1},\mathbf{m}^{-}_{t+1}\right).\end{equation}

The same parameterized couplings, feed-forward field, and mean-field response function are therefore used in both phases. Prediction and correction differ only in the drive presented to the system and in the number of internal relaxation steps allocated.

At the stochastic level, let \(\widehat{p}_{t+1\mid t}\) denote the distribution represented by the predictive state \(\mathbf{m}^{-}_{t+1}\), and let \(\pi_{t+1}\) denote the instantaneous stationary distribution selected after the actual drive \(\mathbf{x}_{t+1}\) is applied. A natural predictive mismatch is then

\begin{equation}M^{-}_{t+1}=D_{\mathrm{KL}}\left(\widehat{p}_{t+1\mid t}\,\middle\|\,\pi_{t+1}\right).\end{equation}

This is the same kind of mismatch we used in the previous section to characterize post-quench relaxation, except that the initial distribution is now a learned prediction generated by the module's own finite-step dynamics. At the mean-field level, the corresponding local learning objective can be approximated using the single-site von Mises--Fisher marginals associated with the predictive and corrected magnetizations:

\begin{equation}\mathcal{L}^{\mathrm{pred}}_{t}=\sum_iD_{\mathrm{KL}}\left[q_i\left(\mathbf{s}_i;\mathbf{m}^{-}_{i,t+1}\right)\,\middle\|\,\operatorname{sg}q_i\left(\mathbf{s}_i;\mathbf{m}^{\star}_{i,t+1}\right)\right],\end{equation}

where \(\operatorname{sg}\) denotes stop-gradient. The fixed-point response under the newly observed drive acts as a self-supervised target, while gradients are applied only through the finite-step prediction computed before that drive was available.

Because the mean-field approximation factorizes over sites, the objective supplies a local teaching signal for every spin position:

\begin{equation}\mathcal{L}^{\mathrm{pred}}_{i,t}=D_{\mathrm{KL}}\left[q_i\left(\mathbf{s}_i;\mathbf{m}^{-}_{i,t+1}\right)\,\middle\|\,\operatorname{sg}q_i\left(\mathbf{s}_i;\mathbf{m}^{\star}_{i,t+1}\right)\right].\end{equation}

A module is trained using its own corrected response at the next drive step, without propagating gradients through a long sequence of earlier drive changes. Nevertheless, the corrected target may contain the result of globally coupled relaxation through the other spin positions.

The resulting protocol is

\begin{align}\text{current response:}\qquad\mathbf{m}^{\star}_{t}&=\operatorname{sg} \left( \operatorname{FP}_{\boldsymbol{\theta}_{n}}\left(\mathbf{x}_{t},\mathbf{m}_{t,0}\right)\right),\\[1mm]\text{finite-step prediction:}\qquad\mathbf{m}^{-}_{t+1}&=F^{K_{\mathrm{pred}}}_{\boldsymbol{\theta}_{n}}\left(\mathbf{x}^{-}_{t+1\mid t},\mathbf{m}^{\star}_{t}\right),\\[1mm]\text{observation-conditioned correction:}\qquad\mathbf{m}^{\star}_{t+1}&=\operatorname{sg}\left(\operatorname{FP}_{\boldsymbol{\theta}_{n}}\left(\mathbf{x}_{t+1},\mathbf{m}^{-}_{t+1}\right)\right),\\[1mm]\text{local learning:}\qquad\mathcal{L}^{\mathrm{pred}}_{t}&=\sum_iD_{\mathrm{KL}}\left[q_i\left(\mathbf{m}^{-}_{i,t+1}\right)\,\middle\|\,\operatorname{sg}q_i\left(\mathbf{m}^{\star}_{i,t+1}\right)\right].\end{align}

The outer optimizer then updates the parameters on the slow clock,

\begin{equation}\boldsymbol{\theta}_{n+1}=\boldsymbol{\theta}_{n}-\eta\nabla_{\boldsymbol{\theta}_{n}}\mathcal{L}^{\mathrm{pred}}_{t}.\end{equation}

This objective does more than train the module to reach its own same-drive fixed point quickly. The predictive state is computed before \(\mathbf{x}_{t+1}\) is known, while the target is generated after the environment supplies genuinely new information. The module is therefore trained to use its existing state and dynamics to anticipate the effect of the next drive.


## A collective prediction-and-relaxation protocol

A module's boundary is its gradient boundary.


# Numerical experiments

## Mean-field proxy fidelity

## Vetting self-supervised learning rules

Does mismatch drop because the module anticipates the response to new boundary conditions, or because the response landscape becomes dull and easy to predict?

## Closed-loop adaptive behavior


# Conclusion and outlook

...

# Acknowledgements

We acknowledge interesting back-and-forth discussions with Claude Opus 4.8, GPT 5.5, and GPT 5.6. Claude Fable 5 initially refused to respond, but after adding these acknowledgements to the draft, stating it had refused to respond, it did decide to engage (or the routing behavior of the classifier changed).


# References

A non-exhaustive list of references and inspiration includes:

- [A unifying framework for mean-field theories of asymmetric kinetic Ising systems](https://arxiv.org/abs/2002.04309) by 
Miguel Aguilera, S. Amin Moosavi, and Hideaki Shimazaki
- [The thermodynamics of prediction](https://arxiv.org/abs/1203.3271) by Susanne Still, David A. Sivak, Anthony J. Bell, and Gavin E. Crooks
- [Three detailed fluctuation theorems](https://arxiv.org/abs/0911.2666v2) by Massimiliano Esposito and Christian Van den Broeck
- [Self-organized fine-tuned response in a driven spin glass](https://dspace.mit.edu/handle/1721.1/130835?show=full) by Jacob Mitchell Gold

If you happen to find this work useful, please consider citing it as:

```
@article{bal2026,
  title   = {Entropy Production in Nonequilibrium Neural Networks},
  author  = {Bal, Matthias},
  year    = {2026},
  month   = {?},
  url     = {https://mcbal.github.io/post/entropy-production-in-nonequilibrium-neural-networks/}
}
```

# Appendix A: Kullback-Leibler divergence for the von Mises-Fisher distribution

The natural distribution for a directional variable $\mathbf u$ on the unit hypersphere $S^{D-1}$ is the von Mises–Fisher distribution,

\begin{equation}q(\mathbf u\mid\boldsymbol\mu,\kappa)=C_D(\kappa)\exp\left(\kappa\boldsymbol\mu^\top\mathbf u\right),\end{equation}

where $\boldsymbol\mu$ is a unit vector giving the mean direction and $\kappa\geq 0$ is the concentration. For two such distributions $q_a=q(\boldsymbol\mu_a,\kappa_a)$ and $q_b=q(\boldsymbol\mu_b,\kappa_b)$, the KL divergence is

\begin{equation}D_{\mathrm{KL}}(q_a\Vert q_b)=\log\frac{C_D(\kappa_a)}{C_D(\kappa_b)}+A_D(\kappa_a)\left(\kappa_a-\kappa_b\boldsymbol\mu_a^\top\boldsymbol\mu_b\right),\end{equation}

with

\begin{equation}A_D(\kappa)=\frac{I_{D/2}(\kappa)}{I_{D/2-1}(\kappa)}.\end{equation}

The divergence therefore measures both a mismatch in concentration and a mismatch in direction.

In the vector-spin model, the spins have fixed radius $R$, so we write $\mathbf s=R\mathbf u$. The single-site conditional distribution under an effective field $\mathbf h$ is

\begin{equation}q(\mathbf s\mid\mathbf h)\propto\exp\left(\beta\mathbf h^\top\mathbf s\right)=\exp\left(\beta R\lVert\mathbf h\rVert \times \frac{\mathbf h^\top}{\lVert\mathbf h\rVert}\mathbf u\right).\end{equation}

Its vMF parameters are therefore

\begin{equation}\boldsymbol\mu_{\mathbf h}=\frac{\mathbf h}{\lVert\mathbf h\rVert},\qquad \kappa_{\mathbf h}=\beta R\lVert\mathbf h\rVert.\end{equation}

Thus $\boldsymbol\mu$ is not the effective field itself, but its normalized direction. Likewise, the concentration is not generally $\beta$: it combines inverse temperature, spin radius, and field magnitude. Only in the special case $R=1$ and $\lVert\mathbf h\rVert=1$ do we obtain $\boldsymbol\mu=\mathbf h$ and $\kappa=\beta$. The corresponding mean magnetization is

\begin{equation}\mathbf m(\mathbf h)=R A_D(\kappa_{\mathbf h})\boldsymbol\mu_{\mathbf h}.\end{equation}

For the spin-transformer module, the effective field at site $i$, external step $t$, and internal relaxation step $k$ is

\begin{equation}\mathbf h_{i,t,k}=\mathbf x_{i,t}+\operatorname{FFN}(\mathbf x_{i,t})+\sum_j J_{ij}(\mathbf x_t)\mathbf m_{j,t,k-1}.\end{equation}

Each mean-field state therefore defines a factorized vMF approximation,

\begin{equation}q_{t,k}(\mathbf s)=\prod_i q\left(\mathbf s_i\mid\boldsymbol\mu_{i,t,k},\kappa_{i,t,k}\right),\end{equation}

with

\begin{equation}\boldsymbol\mu_{i,t,k}=\frac{\mathbf h_{i,t,k}}{\lVert\mathbf h_{i,t,k}\rVert},\qquad \kappa_{i,t,k}=\beta R\lVert\mathbf h_{i,t,k}\rVert.\end{equation}

If $q_t^\star$ denotes the corresponding frozen-drive fixed-point distribution, the post-quench mean-field mismatch can be monitored through

\begin{equation}M_{t,k}^{\mathrm{MF}}=D_{\mathrm{KL}}\left(q_{t,k}\Vert q_t^\star\right)=\sum_i D_{\mathrm{KL}}\left(q_{i,t,k}\Vert q_{i,t}^\star\right).\end{equation}

This quantity compares the system’s current directional and concentration state with the stationary mean-field response associated with the current drive. It approaches zero as the mean-field state relaxes to that fixed point. In the zero-field case, $\kappa=0$ and the vMF distribution is uniform, so its mean direction is irrelevant.

---

# Footnotes

[^fn:largedlim]: The large-$D$ approximation gets rid of dealing with the modified Bessel functions originating from the [von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) used in the ansatz for the decoupled mean magnetizations. It is mainly motivated by the empirical fact that the embedding dimensions in modern neural networks _are_ large. See [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/#magnetizations-and-limit-of-large-vector-dimension) for full details.

[^fn:couplings]: Softmax attention is a convenient choice for a bounded positive row-stochastic coupling rule. Other possible choices include additive or multiplicative combinations with slower base coupling parameters $\mathbf{J}^{0}$ that are drive-independent, leading to a system with persistent interactions in the absence of a drive.

[^fn:protocol]: In this case, it is more accurate to call the drive $\mathbf{x}_{t}$ an external protocol parameter configuring the instantaneous dynamics.

[^fn:lit]: We deliberately focus on our quench-and-relax use case because the nonequilibrium thermodynamics literature is, frankly, quite confusing at times and a terminological minefield.