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
lastmod: 2026-07-13T08:30:41+01:00
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

Transformers are powerful driven dynamical systems, yet their internal computation is rarely discussed in terms of nonequilibrium thermodynamics. Building on [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/), we design a minimal parallel transformer-like module whose forward pass implements a controllable quench-and-relax process. We characterize its dynamical regimes to elucidate a design space of stateless and stateful variations of transformer-like and deep-equilibrium-like architectures and leverage its physics-based architecture to compute differentiable proxies for entropy production. This provides both a scalable laboratory for nonequilibrium many-body dynamics on hardware accelerators and a testable learning hypothesis.

We ask whether module-local ascent on a housekeeping entropy-production proxy, under bounded dynamics and structured input streams, can lead a system to acquire structure-sensitive, predictive dynamics without an externally supplied task loss or end-to-end credit assignment. The risk is that the system finds local dissipative shortcuts: asymmetric attention collapse, self-exciting cycles, or coupling to noise. The bet is that, once trivial dissipative shortcuts are bounded or exhausted, latching onto persistent temporal structure provides the most reliable support for continuing asymmetric delayed flow. We readily admit that the main motivation for this bet is aesthetic. To move beyond aesthetics, we run numerical experiments.


# Driving a spin-transformer module

In this section we design a minimal spin-transformer module whose forward pass implements a controllable nonequilibrium quench-and-relax process. We identify three timescales and two dynamical regimes, leading to a natural categorization of the design space into stateless and stateful variations of transformer-like and deep-equilibrium-like architectures.

## A minimal controllable drive-conditioned system

In [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/) we showed how to apply dynamical mean-field theory to approximate the time-dependent behavior of asymmetric vector-spin models. We started from a spin system of $N$ vector spins $\mathbf{s}_{i,t} \in \mathbb{R}^{D}$ talking to each other via an $N \times N$ pairwise coupling matrix $J_{ij}$ with the underlying parallel-updates stochastic dynamics characterized by a discrete-time Markov chain transition probability $P(\mathbf{s}_{t} | \mathbf{s}_{t-1})$. External magnetic fields $\mathbf{x}_{i,t} \in \mathbb{R}^{D}$ bias the vector spins and act as local drives. We will use the notation $\mathbf{x}_{t} = \{ \mathbf{x}_{i,t} \}^{N}_{i=1}$ to refer to the full-drive matrix instead of the local drive at a single position.

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

where we have pushed down the recurrence into an internal relaxation index $k$. By making the effective drive as well as the couplings depend on the drive $\mathbf{x}_{t}$, a sudden shift $\mathbf{x}_{t} \to \mathbf{x}_{t+1}$ changes both the local fields as well as the interactions and quenches the module into a new instantaneous dynamics[^fn:protocol]. During internal relaxation updates $k$, the drive $\mathbf{x}_{t}$ and the parameters $\boldsymbol{\theta} = \{ \mathbf{W}_{Q}, \mathbf{W}_{K}, \mathbf{FFN} \}$, and therefore the transition rule, are held fixed. (At the stochastic level, the process looks something like $P_{\boldsymbol{\theta}, \mathbf{x}_{t}}(\mathbf{s}_{t, k+1} | \mathbf{s}_{t, k})$.)

We end up with a _highly reconfigurable system_ that is _dynamically shaped by the drive_. Each vector spin effectively experiences a local mean-field that is the sum of a residual stream drive, a feed-forward-like drive, and attention-like couplings. Importantly, these terms are _parameterized_ and can be _shaped through training_: we can control how the system responds, fluctuates, and relaxes after getting quenched. Slow parameter updates then make this responsive system adaptive.


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

We frame our module-design intuition around a _quench-and-relax scenario_: when the input drive switches, _i.e._ $\mathbf{x}_{t} \to \mathbf{x}_{t+1}$, the spin system has to adapt to the sudden change. A general post-quench module then looks like

\begin{equation}
  \mathbf{m}_{t, K} = F^{K}_{\boldsymbol{\theta}_{n}} \left( \mathbf{x}_{t}, \mathbf{m}_{t, 0} \right),
\end{equation}

with two independent design choices: the number of internal relaxation steps $K$ (relaxation horizon) and the choice of $\mathbf{m}_{t, 0}$ (initialization policy), leading to the following design space:

| | Reset or amortized initialization           | Carried initialization                        |
| -- | --------------------------------- | ----------------------------------------- |
| **Finite-step regime with $K < \infty$** | finite-depth, stateless, transformer-like module | recurrent stateful module |
| **Fixed-point regime where $K \to \infty$** | implicit or deep-equilibrium-like (DEQ) module   | identical if the fixed point is unique; path-dependent if mean-field branches coexist                   |

### Finite-step regime

In this regime, only $K < \infty$ internal updates are allocated before readout or the next quench. This may reflect genuine competition between relaxation and drive timescales, or simply deliberate computational truncation. The intuition here is that the system tracks a moving family of instantaneous stationary marginals with potentially nonzero lag. There is no strong separation between drive and relaxation, but, since the module is controllable, we can shape the system's behavior through nudging the parameters $\boldsymbol{\theta}_{n}$ in the outer loop.

The initialization $\mathbf{m}_{t, 0} = \mathbf{m}_{t-1, K}$ makes the module architecture genuinely recurrent and stateful, but with a full context window of hidden states, situating it somewhere in between recurrent neural networks and transformers. Another option is to learn an amortized initializer $\mathbf{m}_{t, 0} = \mathbf{x}_{t}\mathbf{W}_{V}$ for the post-quench relaxation from the drive, which estimates the drive-conditioned response to which the module should relax. If we recognize this state initialization as _values_, then, for $K=1$, the forward pass pretty much matches that of a parallel transformer block.

### Fixed-point regime

In this regime, $\tau_{\mathrm{relax}} \ll \tau_{\mathrm{drive}}$ so we consider $\mathbf{x}_{t}$ clamped and let $K \to \infty$ until the deterministic mean-field equations converge to fixed-point magnetizations $\mathbf{m}^{*}_{t}(\mathbf{x}_{t})$ compatible with the frozen drive $\mathbf{x}_{t}$. These values approximate the stationary marginals of an underlying instantaneous frozen-drive nonequilibrium steady state (NESS). The intuition here is that the clamped input fixes an instantaneous stochastic transition rule. Although its one-point marginals become stationary, asymmetric couplings can sustain probability currents and positive entropy production beneath those stationary marginals.

In case of a unique fixed point, the initial values $\mathbf{m}_{t, 0}$ are erased, and the module is stateless. But the deterministic mean-field equations may admit multiple stable fixed-point branches or basins. Warm-starting with $\mathbf{m}_{t, 0} = \mathbf{m}^{*}_{t-1}$ can then produce path-dependent branch selection and hysteresis behavior.

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

We could stop here, pretrain a spin-transformer model using next-token prediction on chunks of text, and compare evaluation metrics to those of compute-matched vanilla transformers. But let us focus instead on what our framework enables that feels hard to come up with _without_ having access to a nonequilibrium spin-model perspective.

In this section, we show how the quench-and-relax process behind the forward pass of a spin-transformer module relates to notions of _irreversibility_. We add just enough physical context to introduce differentiable entropy production proxies that we can compute at the same mean-field level as the spin system.

## Two ways to be irreversible

During the quench-and-relax process we hold the input drive $\mathbf{x}_{t}$ fixed and let the spin system settle. Its average magnetizations $\mathbf{m}_{t}$ may stop changing, but its microscopic dynamics need not become reversible. Asymmetric couplings can sustain circulating probability currents when forward sequences of spin configurations remain more likely than their backward step reversals. We call this source of irreversibility _steady-state irreversibility_. Its entropy-production rate is the running cost of maintaining a nonequilibrium steady state under the current input drive. We can estimate this "housekeeping" entropy production in our mean-field approximation from asymmetric couplings and delayed correlations,

\begin{equation}
  \sigma^{*}_{\mathrm{hk},t} = \beta \sum_{ij} \left(J_{ij}(\mathbf{x}_{t}) - J_{ji}(\mathbf{x}_{t})\right) D^{*}_{ij,t} , \label{eq:sigma_hk}
\end{equation}

where $D^{*}_{ij,t}$ denote the one-step delayed correlations evaluated at the system's stationary marginals $\mathbf{m}^{*}_{t}(\mathbf{x}_{t})$. Intuitively, this is like

\begin{equation}
  \sigma^{*}_{\mathrm{hk},t} = \sum_{ij} \left[\operatorname{directionality}\right]_{ij} \times \left[\operatorname{delayed\ flow}\right]_{ij,t}.
\end{equation}

The couplings provide a directional bias; the delayed correlations report whether fluctuations actually propagate along that direction. A fixed point of the magnetizations therefore does not imply equilibrium: it describes stationary averages, not the absence of microscopic currents.

Now we quench again! After changing the input drive abruptly $\mathbf{x}_{t} \to \mathbf{x}_{t+1}$, the system is still distributed approximately according to its old steady state $\pi_{t}$, while the new dynamics $P_{\boldsymbol{\theta}, \mathbf{x}_{t+1}}(\mathbf{s}_{t+1, k+1} | \mathbf{s}_{t+1, k})$ induced by the new input drive actually favor another steady state $\pi_{t+1}$. The mismatch $D_{\mathrm{KL}}\left(\pi_{t} \lVert \pi_{t+1} \right)$ is a relaxation-irreversibility proxy measuring how far the old response lies from the response required by the new dynamics. The new housekeeping part remains after relaxation while $D_{\mathrm{KL}}\left(\pi_{t} \lVert \pi_{t+1} \right) \to 0$ once the system has caught up.

> A driven spin-transformer module with asymmetric couplings can thus be irreversible in two ways: by **running a nonequilibrium steady state** and by **catching up after its input drive changes**.

## Mean-field proxy for housekeeping entropy production

> **TODO:** Update to reflection relaxation index $k$

Back to reality. If we write down $D_{ij,t}$ for the vector-spin case,

\begin{equation}
  D_{ij,t} = \int \mathrm{d} \mathbf{s}_{t} \int \mathrm{d} \mathbf{s}_{t-1} \; \left( \mathbf{s}_{i,t} - \mathbf{m}_{i,t} \right) \cdot \left( \mathbf{s}_{j,t-1} - \mathbf{m}_{j,t-1}\right) \; P( \mathbf{s}_{t}, \mathbf{s}_{t-1} ),
\end{equation}

we can compute a first-order `Plefka[t-1,t]` mean-field approximation for the time-delayed correlations, similar to the computations we did previously for the magnetizations in [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/), leading to something like

\begin{align}
  D_{ij,t} = &\beta J_{ij} \operatorname{Tr} \left( \Sigma_{i,t} \Sigma_{j,t-1} \right),
\end{align}

where $\Sigma_{i,t} = \operatorname{Var} \left[ s_{i,t} \right]$ denotes the single-site covariance / susceptibility. The trace captures which directions on the vector-spin sphere are still available to fluctuate. If a spin is weakly magnetized, it has many soft directions. If it is strongly magnetized, many directions are suppressed because the spin is pinned close to its mean direction.

Substituting the large-$D$ approximation

\begin{align}
  \Sigma_{i,t} \approx \frac{\mathbb{1}}{1+\gamma_{i,t}} - \frac{\mathbf{m}_{i,t} \mathbf{m}_{i,t}^{T}}{R^2 \gamma_{i,t}},
\end{align}

we end up with the explicit expression

\begin{align}
  D_{ij,t} = &\frac{\beta J_{ij}}{1+\gamma_{i,t}} \left(R^2 - \mathbf{m}_{j,t-1}^2 \right) \nonumber\\\\
  &- \frac{\beta J_{ij}}{R^2 \gamma_{i,t} \left( 1 + \gamma_{j,t-1} \right)} \mathbf{m}_{i,t}^2 \nonumber\\\\
  &+ \frac{\beta J_{ij}}{R^4 \gamma_{i,t} \gamma_{j,t-1}} \left( \mathbf{m}_{i,t} \cdot \mathbf{m}_{j,t-1} \right)^2,
\end{align}

where

\begin{align}
  \gamma_{i,t} &= \sqrt{1 + \beta^2 \lVert \mathbf{h}_{i,t} \rVert^2 / R^2 } \\\\
  \mathbf{h}_{i,t} &= \mathbf{x}_{i,t} + \sum_{j} J_{ij} \mathbf{m}_{j,t-1}.
\end{align}

The first-order time-delayed correlations $D_{ij,t}$ are a mean-field estimate of how much the fluctuation in one vector spin is transmitted one time step later "into" another spin. Or, put differently, when spin $j$ fluctuates away from its mean at the previous time step $t-1$, how much of that fluctuation shows up as a fluctuation of spin $i$ at the current time step $t$?

> **Waving hands and checking vibes:** Let us try to get a feel for what the entropy production looks like for vector-spin models using some rough back-of-the-envelope estimations. Assume both vectors $\mathbf{m}_{i,t}$ and $\mathbf{m}_{j,t-1}$ have a norm $\mathcal{O}(R)$, then the time-delayed correlations behave approximately like
\begin{align}
  D_{ij,t} \sim \beta J_{ij} \cos^2 \alpha_{(i,t)(j,t-1)},
\end{align}
where $\alpha_{(i,t)(j,t-1)}$ denotes the angle between the magnetization vectors. So the entropy production looks approximately like
\begin{equation}
  \langle \sigma_{t} \rangle \sim \beta^2 \sum_{ij} \left(J_{ij}^2 - J_{ij} J_{ji}\right) \cos^2 \alpha_{(i,t)(j,t-1)},
\end{equation}
which, in general, is minimized for symmetric coupling matrices or orthogonal embeddings and maximized for fully-asymmetric couplings or (anti-)parallel embeddings.
But for the softmax attention matrix Eq. \eqref{eq:softmax}, we have additional constraints $J_{ij} \geq 0$ as well as a Frobenius norm of $\mathcal{O}(\sqrt{N})$ preventing unbounded growth under maximization. Additionally, imposing a causal mask on the couplings to do autoregressive modeling leads to even more constraints since then the upper triangular part of $J_{ij}$ is fixed to zero. So it feels like maximizing entropy production for causal softmax couplings promotes some kind of compromise between _sparse attention_ (intuitively, if the upper-triangular part is zero then it is favorable to push most of the lower-triangular elements close to zero as well) and _clustering of embeddings_ (weighted maximization of cosine similarity).

## Mean-field proxy for nonadiabatic entropy production

Apply von Mises-Fisher KL divergence expressions...

# A learning hypothesis: optimizing entropy production

We could stop here, and use the mean-field entropy-production proxies derived in the previous section as diagnostic evaluation measures or monitoring tools to track the behavior of spin-transformer modules during training and inference. But let us again focus instead on what our framework enables that feels hard to come up with without having access to a nonequilibrium spin-model perspective.

Since these entropy-production proxies are differentiable, we might as well try treating them as module-local loss functions. Sure, you can have an external task steering the optimization process using, for example, a cross-entropy loss. But a truly adapative module should be able to learn to reshape its drive-conditioned steady states online so that its current state lies close to the response required by likely future drives. It should be able to do this locally at the level of module without needing a global backpropagation signal. Once trivial dissipative shortcuts are bounded or exhausted, persistent temporal structure should provide the most reliable support for continuing asymmetric delayed flow.

...

## A module-local learning rule

If we compute the gradient with respect to module parameters of the housekeeping entropy production Eq. \eqref{eq:sigma_hk}, we get

\begin{align}
  \beta &\sum_{ij} \left(J_{ij}(\mathbf{x}_{t}) - J_{ji}(\mathbf{x}_{t})\right) \mathrm{stop\_gradient} \left( D^{*}_{ij,t} \right) \notag \\
  & + \beta \sum_{ij} \mathrm{stop\_gradient} \left(J_{ij}(\mathbf{x}_{t}) - J_{ji}(\mathbf{x}_{t})\right) D^{*}_{ij,t}
\end{align}


A temporally asymmetric Hebbian rule, etc.


# Numerical experiments

## Mean-field proxy fidelity

## Structure-sensitive learned irreversibility

## Closed-loop adaptive behavior


# Conclusion and outlook

...

A spin-transformer module is a driven nonequilibrium response system. It receives drives from other systems or the environment, relaxes to a response, and emits magnetizations that perturb those boundaries. State need not be internal to a module; it may reside in the environment or in the closed-loop configuration of coupled modules.

...

Interfacing multiple modules into collectives. Global coherence from local backpropagation. Collectives, loops, and adaptive systems. Open-ended adaptation.

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

---

# Footnotes

[^fn:largedlim]: The large-$D$ approximation gets rid of dealing with the modified Bessel functions originating from the [von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) used in the ansatz for the decoupled mean magnetizations. It is mainly motivated by the empirical fact that the embedding dimensions in modern neural networks _are_ large. See [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/#magnetizations-and-limit-of-large-vector-dimension) for full details.

[^fn:couplings]: Softmax attention is a convenient choice for a bounded positive row-stochastic coupling rule. Other possible choices include additive or multiplicative combinations with slower base coupling parameters $\mathbf{J}^{0}$ that are drive-independent, leading to a system with persistent interactions in the absence of a drive.

[^fn:protocol]: In this case, it is more accurate to call the drive $\mathbf{x}_{t}$ an external protocol parameter configuring the instantaneous dynamics.