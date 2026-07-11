---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Entropy Production in Nonequilibrium Neural Networks"
subtitle: "A nonequilibrium thermodynamics perspective on transformers"
summary: "A nonequilibrium thermodynamics perspective on transformers"
aliases:
  - /post/entropy-production-in-non-equilibrium-neural-networks/
authors:
  - me
tags: ["Artificial Intelligence", "Associative Memories", "Attention", "Cybernetics", "Deep Learning", "Dynamical Systems", "Entropy Production", "Ising Models", "Many-Body Systems", "Mean-Field Theory", "Neural Networks", "Near-Equilibrium Dynamics", "Nonequilibrium Dynamics", "Self-Organizing Computational Stability", "Statistical Physics", "Steady State", "Stochastic Thermodynamics", "Transformers", "Vector-Spin Models"]
categories: []
date: 2026-02-02T09:28:17+01:00
lastmod: 2026-07-11T16:30:41+01:00
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

In this post, we build on [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/) to design adaptive systems that can reuse a fixed substrate, remain online, and continuously reshape internal dynamics under local constraints. We design a minimal spin-model transformer module whose forward pass implements a controllable nonequilibrium quench-and-relax process. This physics-inspired architecture enables us to measure proxies for [_entropy production_](https://en.wikipedia.org/wiki/Entropy_production#Entropy_production_in_stochastic_processes), a thermodynamic quantity measuring irreversibility by quantifying the asymmetry between forward and backward time steps. 

Since every operation in the computational graph is differentiable, the entropy-production proxies can be made into loss functions to steer irreversible flow through the system. For example, maximizing entropy production incentivizes the system to _lean into the external drive_ by nudging its parameters towards asymmetric delayed responses that absorb and transmit structure in the incoming drive. Internally, we imagine the system reshaping itself into ordered structures to enable more efficient dissipation of the tension caused by the incoming data stream. 

The risk is that the system finds local dissipative shortcuts: asymmetric attention collapse, self-exciting cycles, or coupling to noise. In the interesting regime of bounded driven system, useless dissipation saturates while structure-sensitive flows remain persistent. For this to happen, environments, as well as the boundary interfaces coupling different driven systems, need to be engineered so that the most stable way to increase entropy production when flooded by a structured data stream is to latch onto the latent (temporal) structure. Ideally, individual modules locally amplify asymmetric delayed flows in parallel, while module connectivity and environment feedback collectively constrain which flows remain stable and useful for the system as a whole.

The bet is that, embedded in sufficiently structured streams and with a capability to act on its environment, the cheapest way for a bounded local system to keep dissipating is to become predictive, where prediction is a thermodynamic adaptation to ensure continuing support for asymmetric delayed flows. We readily admit that the main motivation for this bet is aesthetic. To move beyond aesthetics, we run numerical experiments to find out whether local ascent on a computable entropy-production proxy, under bounded dynamics and structured drive, can lead a system to acquire structure-sensitive, predictive dynamics without an externally supplied task loss or end-to-end credit assignment.


# Driving a spin-model transformer module

In [Spin-Model Transformers (2023)](https://mcbal.github.io/post/spin-model-transformers/) we showed how to apply dynamical mean-field theory to approximate the time-dependent behavior of asymmetric vector-spin models. Let us not worry about why for now and jump straight to the results.

> The relevant context is a spin system of $N$ vector spins $\mathbf{s}_{i,t} \in \mathbb{R}^{D}$ talking to each other via an $N \times N$ pairwise coupling matrix $J_{ij}$ with the underlying stochastic dynamics characterized by a discrete-time Markov chain transition probability $P(\mathbf{s}_{t} | \mathbf{s}_{t-1})$ (parallel updates). We drive the system by clamping the applied external fields $\mathbf{x}_{i,t} \in \mathbb{R}^{D}$ to vectors of interest (not random!).
<img src="vector_spins.png" alt="Random Ising model configuration with vector spins" width="250px"/>

## A minimal model

Using the simplest first-order `Plefka[t-1,t]` mean-field approximation, we arrive at a closed expression for updating the spin expectation values in the large-vector-dimension limit,

\begin{equation}
\mathbf{m}_{i,t} = \frac{\beta \left( \mathbf{x}_{i,t} + \sum_{j} J_{ij} \mathbf{m}_{j,t-1} \right)}{1+\sqrt{1+\beta^2 \lVert \mathbf{x}_{i,t} + \sum_{j} J_{ij} \mathbf{m}_{j,t-1} \rVert^2 / R^2 }},
\end{equation}

where the magnetization vectors $\mathbf{m}_{i,t} \in \mathbb{R}^{D}$ at position $i$ and time $t$ capture the mean-field influence of the spins on each other. The local drives $\mathbf{x}_{i,t} \in \mathbb{R}^{D}$ act as applied external fields and $J_{ij}$ defines the (asymmetric) pairwise couplings between sites $i$ and $j$. Additionally, $\beta$ denotes the inverse temperature and $R=\sqrt{D/2 -1}$ is the natural hyperspherical length scale resulting from the large-vector-dimension approximation[^fn:largedlim].

If we now consider some kind of _parametrized drive-dependent couplings_

\begin{equation}
  \mathbf{J} (\mathbf{x}_{t}) = \mathrm{softmax}\left( \mathbf{x}_{t} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{T} \mathbf{x}_{t}^{T} \right), \label{eq:softmax}
\end{equation}

then we turn the fixed-size $N \times N$ coupling matrix into a parametrized rule that supports variable system size, drive-dependent routing, and a way to scale system size without learning new explicit parameters[^fn:couplings]. If we also augment the local drives with some kind of _parametrized non-linear drive-dependent field_,

\begin{equation}
  \mathbf{x}_{i,t} \to \mathbf{x}_{i,t} + \mathrm{FFN}\left( \mathbf{x}_{i,t} \right),
\end{equation}

then our earlier recurrence relation becomes

\begin{equation}
  \mathbf{m}_{i,t} = \frac{\beta \left( \mathbf{x}_{i,t} + \mathrm{FFN}\left( \mathbf{x}_{i,t} \right) + \sum_{j} J_{ij} (\mathbf{x}_{t}) \mathbf{m}_{j,t-1} \right)}{1+\sqrt{1+\beta^2 \lVert \mathbf{x}_{i,t} + \mathrm{FFN}\left( \mathbf{x}_{i,t} \right) + \sum_{j} J_{ij} (\mathbf{x}_{t}) \mathbf{m}_{j,t-1} \rVert^2 / R^2 }}. \label{eq:paralleltransformer}
\end{equation}

Making the effective drives as well as the couplings depend non-linearly on the drive $\mathbf{x}_{t}$ leads to a _highly-adaptive system_ where the interaction landscape and local susceptibilities are _dynamically shaped by the drive_. Each vector spin effectively experiences a local mean-field that is the sum of a residual stream drive, a feed-forward-like drive, and attention-like couplings. Importantly, the interaction landscape and local susceptibilities are parametrized and can be learned: _we can control how the system responds, relaxes, and behaves_.

Interpreting Eq. \eqref{eq:paralleltransformer} as the forward pass of a neural network module (which we have been calling a _spin-transformer module_), we observe its close resemblance to the forward pass of a [parallel transformer block](https://xn--rss.to/parallel-transformer-blocks.html). Though instead of values $\mathbf{x}_{t}\mathbf{W}_{V}$ linearly transformed from the current drive, we find the magnetizations of the previous time step $\mathbf{m}_{j,t-1}$. This particular spin-transformer module, built on the simplest first-order `Plefka[t-1,t]` mean-field approximation, is recurrent even though its update is parallel over its context window. It sits somewhere in between a recurrent neural network and a transformer architecture: its state is persistent, but its routing is recomputed globally from the current drive.

## On the connection to transformers

Let us step back for a bit and emphasize that this close resemblance between forward passes acts as a plausibility bridge at this point. It is _not evidence_ that trained transformers literally implement the approximated nonequilibrium thermodynamics scenarios we will cover in the next sections. But the proximity in module architecture space of a minimal spin-model transformer to a class of transformers known to scale does at least suggest that transformers may also admit module-level nonequilibrium interpretations.

Even on their own, spin-transformer modules have merit. We will see that they can turn transformer-like neural networks into computational laboratories for nonequilibrium dynamics that can be executed on modern accelerators at scale. This makes it possible to study large, high-dimensional systems with structured input-dependent couplings, nonstationary data streams, and slowly adapting parameters rather than staying close to analytically tractable toy models. The resulting observables remain mean-field approximations, and must be calibrated against exact stochastic systems at small scale. But once calibrated, the framework offers a route to computational experiments on collective adaptation and irreversible organisation in regimes that are otherwise difficult to access.

We end this section with a cheat sheet mapping concepts between spin-transformer modules and transformer modules.

| Spin-transformer module    | Transformer module |
| -------- | ------- |
| Local drives $\mathbf{x}_{i,t}$  | Input embedding vectors    |
| "The drive" $\mathbf{x}_{t}$  | Current context window    |
| Parametrized couplings $J_{ij}$    | Attention matrix    |
| Magnetizations $\mathbf{m}_{i,t}$ | Transformed output embedding vectors     |


# Building modules: three timescales and two regimes

Attentive readers will have noticed that we actually did not specify how to _implement_ spin-transformer modules in the previous section. What is up with this weird internal state? Why bother with an explicit time dimension? How can this system even serve its function as a neural network module?

To begin to address these questions, we are forced to grapple with nonequilibrium thermodynamics (_*audience groans and starts rolling their eyes*_). Basically, the system has three clocks and, in some regimes, a hierarchy of effective timescales. Let us rewrite the forward pass Eq. \eqref{eq:paralleltransformer} as follows:

\begin{equation}
  \mathbf{m}^{(l)}_{t, k+1} = F_{\boldsymbol{\theta}^{(l)}_{n}} \left( \mathbf{x}^{(l)}_{t}, \mathbf{m}^{(l)}_{t, k} \right)
\end{equation}

where the clock

- $k$ indexes fast internal relaxation within a module
- $t$ indexes changes in the environmental drive or input context
- $n$ indexes slow parameter updates $\boldsymbol{\theta}^{(l)}_{n+1} = \boldsymbol{\theta}^{(l)}_{n} + \eta \nabla_{\boldsymbol{\theta}^{(l)}} \mathcal{L}^{l}_{t}$ for some learning rate $\eta \ll 1$ and (potentially layer-dependent) loss function $\mathcal{L}^{l}_{t}$ of the parameters $\boldsymbol{\theta} = \{ \mathbf{W}_{Q}, \mathbf{W}_{K}, \mathbf{FFN} \}$ where slow here means small parameter changes since optimizer clock $n$ often tracks fresh-drive clock $t$
- $l$ indexes network depth (number of stacked layers); not a physical time

> A deep network is a stack of (untied) modules, which, in our framework, make up a collective of _different_ driven spin systems driving each other sequentially. The layer index does not (have to) correspond to external time nor to internal relaxation time; it is an additional axis labeling the simple feed-forward topology (depth) of the computational graph. For clarity, we drop it in the remainer of this section.

Given the three relevant clocks for a single module, we have three relevant timescales:

\begin{equation}
\tau_{\mathrm{relax}} \ll \tau_{\mathrm{drive}} \ll \tau_{\mathrm{learn}}
\end{equation}

We build our intuition around a _quench-and-relax scenario_: we probe the system with data and watch it respond. When the input drive changes discretely as $\mathbf{x}_{t-1} \to \mathbf{x}_{t}$, the spin system has to adapt to the change in boundary conditions. Assuming slow plasticity ($\tau_{\mathrm{learn}} \gg \tau_{\mathrm{drive}}$) so we can freeze the outer learning loop, the system fixed at its current parameters can _relax_ in two different ways, leading to different concrete implementations.

## Fixed-point regime

In this regime, $\tau_{\mathrm{relax}} \ll \tau_{\mathrm{drive}}$, so that, for every timestep $t$ we clamp $\mathbf{x}_{t}$ and let $k \to \infty$ until the deterministic mean-field equations converge

\begin{equation}
  \mathbf{m}^{*}_{t} = F_{\boldsymbol{\theta}_{n}} \left( \mathbf{x}_{t}, \mathbf{m}^{*}_{t} \right),
\end{equation}

to fixed-point magnetizations $\mathbf{m}^{*}_{t}$ compatible with the frozen drive $\mathbf{x}_{t}$. These values approximate the stationary marginals of an underlying instantaneous frozen-drive nonequilibrium steady state (NESS). The intuition here is that the "continuous kicking" of the drive "sustains" the magnetizations dynamically. 

> Physically, for a frozen drive $\mathbf{x}_{t}$, imagine an underlying stochastic vector-spin system whose asymmetric couplings may violate detailed balance. Its stationary state can therefore sustain probability currents and positive entropy production. The deterministic mean-field equations do not represent these microscopic currents directly: their fixed point $\mathbf{m}^{*}_{t}$ approximates the stationary one-point magnetizations of the underlying  stochastic system.

In case of a unique fixed point, the initial values $\mathbf{m}_{t, 0}$ are erased, and the module is stateless. But a fixed-point module can also end up being stateful in case of multistability and hysteresis under changing drive. Warm-starting with $\mathbf{m}_{t, 0} = \mathbf{m}^{*}_{t-1}$ then keeps track of the path or acts as a memory for basin selection.

## Finite-step recurrent regime

In this regime, $\tau_{\mathrm{relax}} \sim \tau_{\mathrm{drive}}$, there is no time for full relaxation. Only a small number $K$ of relaxation steps can happen before the drive changes again,

\begin{equation}
  \mathbf{m}_{t, K} = F^{K}_{\boldsymbol{\theta}_{n}} \left( \mathbf{x}_{t}, \mathbf{m}_{t, 0} \right),
\end{equation}

The intuition here is madness: the system is perpetually chasing a moving instantaneous NESS but can never reach it. Warm-starting with $\mathbf{m}_{t, 0} = \mathbf{m}_{t-1, K}$ adds persistent recurrence, carrying dynamical state. In this regime, there is no strong separation between drive and relaxation, _but shaping the system's behavior through nudging the parameters $\boldsymbol{\theta}_{n}$ in the outer loop might help with that_.

> The case $K=1$ most closely resembles the forward pass of a conventional transformer module. A picture of autoregressive inference: a freshly generated token changes the context window and therefore quenches the (stack of) module(s) again, the (stack of) module(s) one-step relaxes to generate a probability distribution to sample the next token from. The parameters of the (stack of) module(s) have been carefully optimized during successive stages of training to target useful one-step relaxation.


## Values, DEQs, and looping

In the previous section, we noted that the forward pass of a spin-transformer module contains magnetizations of the previous time step $\mathbf{m}_{j,t-1}$ where we expect values $\mathbf{x}_{t}\mathbf{W}_{V}$. Can we interpret this now that we know about dynamical regimes? Since relaxation is strongly tied to the current drive $\mathbf{x}_{t}$, perhaps values are amortized feed-forward approximations of internal relaxation steps, like a one-shot learned estimator of a fixed-point relaxation response.

The fixed-point regime is reminiscent of deep equilibrium models (DEQs) and the finite-step regime of looped, recursive reasoning approaches. The iterations here are, arguably, less _ad hoc_ since they are done to solve self-consistent mean-field message-passing equations. Indeed, applying the same module again can be seen as allowing the underlying nonequilibrium system to settle more snuggly into its steady state for that particular configuration of drive and parameters. However, as soon as the drive changes, or the parameters change, the system has to somehow renegotiate a different steady state compatible with what its new configuration dictates the response should be.


# More physics please: on entropy production and irreversibility

Let us now introduce some physics to get a handle on what driving these out-of-equilibrium systems means.

...

Maintaining currents under frozen drive versus catching up when the boundary changes.

...

# An approximation hierarchy: computing differentiable entropy-production proxies

## Mean-field proxy for entropy production

Following [Aguilera et al. (2020)](https://arxiv.org/abs/2002.04309), the housekeeping entropy production for the kinetic Ising model, assuming a nonequilibrium steady state, is given by

\begin{equation}
  \langle \sigma_{t} \rangle = \sum_{ij} \left(J_{ij} - J_{ji}\right) D_{ij,t} \geq 0, \label{eq:sigma_hk}
\end{equation}

where $J_{ij}$ corresponds to the couplings and $D_{ij,t}$ denotes the time-delayed correlations. Intuitively, this is like

\begin{equation}
  \langle \sigma_{t} \rangle = \sum_{ij} \left[\operatorname{directionality}\right]_{ij} \times \left[\operatorname{delayed\ flow}\right]_{ij,t},
\end{equation}

or, even more hand-wavy, $\operatorname{dissipation} \sim \operatorname{force} \times \operatorname{flux}$. The asymmetric part of the couplings says whether that propagation channel is directionally biased. The full sum rewards directed, temporally effective, vector-aligned information flow.

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
  \Sigma_{i,t} \approx \frac{1}{1+\gamma_{i,t}} - \frac{\mathbf{m}_{i,t} \mathbf{m}_{i,t}^{T}}{R^2 \gamma_{i,t}},
\end{align}

we end up with the explicit expression

\begin{align}
  D_{ij,t} = &\frac{\beta J_{ij}}{1+\gamma_{i,t}} \left(R^2 - \mathbf{m}_{j,t-1}^2 \right) \nonumber\\\\
  &- \frac{\beta J_{ij}}{R^2 \gamma_{i,t} \left( 1 + \gamma_{j,t-1} \right)} \mathbf{m}_{i,t}^2 \nonumber\\\\
  &+ \frac{\beta J_{ij}}{R^4 \gamma_{i,t} \gamma_{j,t-1}} \left( \mathbf{m}_{i,t} \cdot \mathbf{m}_{j,t-1} \right)^2,
\end{align}

where

\begin{align}
  \gamma_{i,t} &= \sqrt{1 + \beta^2 \lVert \boldsymbol{\theta}_{i,t} \rVert^2 / R^2 } \\\\
  \boldsymbol{\theta}_{i,t} &= \mathbf{x}_{i,t} + \sum_{j} J_{ij} \mathbf{m}_{j,t-1}.
\end{align}

The first-order time-delayed correlations $D_{ij,t}$ is a mean-field estimate of how much the fluctuation in one vector spin is transmitted one time step later "into" another spin. Or, put differently, when spin $j$ fluctuates away from its mean at the previous time step $t-1$, how much of that fluctuation shows up as a fluctuation of spin $i$ at the current time step $t$?


## Waving hands and checking vibes

Let us try to get a feel for what the entropy production looks like for vector-spin models using some rough back-of-the-envelope estimations. Assume both vectors $\mathbf{m}_{i,t}$ and $\mathbf{m}_{j,t-1}$ have a norm $\mathcal{O}(R)$, then the time-delayed correlations behave approximately like

\begin{align}
  D_{ij,t} \sim J_{ij} \cos^2 \alpha_{(i,t)(j,t-1)},
\end{align}

where $\alpha_{(i,t)(j,t-1)}$ denotes the angle between the magnetization vectors. So the entropy production looks approximately like

\begin{equation}
  \langle \sigma_{t} \rangle \sim \sum_{ij} \left(J_{ij}^2 - J_{ij} J_{ji}\right) \cos^2 \alpha_{(i,t)(j,t-1)},
\end{equation}

which, in general, is minimized for symmetric coupling matrices or orthogonal embeddings and maximized for fully-asymmetric couplings or (anti-)parallel embeddings.

But for the softmax attention matrix Eq. \eqref{eq:softmax}, we have additional constraints $J_{ij} \geq 0$ as well as a Frobenius norm of $\mathcal{O}(\sqrt{N})$ preventing unbounded growth under maximization. Additionally, imposing a causal mask on the couplings to do autoregressive modeling leads to even more constraints since then the upper triangular part of $J_{ij}$ is fixed to zero. So it feels like maximizing entropy production for causal softmax couplings promotes some kind of compromise between _sparse attention_ (intuitively, if the upper-triangular part is zero then it is favorable to push most of the lower-triangular elements close to zero as well) and _clustering of embeddings_ (weighted maximization of cosine similarity).


## Entropy production decompositions

We referred to Eq. \eqref{eq:sigma_hk} as _housekeeping_ entropy production and then never mentioned why we called it that way. Let us think again how we can turn vector-spin models with weird drive-dependent couplings $J_{ij}(\mathbf{x}_{t})$ into a transformer-like neural network. Doing so will force us, once again, to grapple with nonequilibrium thermodynamics (_*audience now visible annoyed and looking for the exit*_). In the fixed-point interpretation, the housekeeping entropy production

\begin{equation}
  \langle \sigma_{t} \rangle = \sum_{ij} \left(J_{ij}(\mathbf{x}_{t}) - J_{ji}(\mathbf{x}_{t})\right) D_{ij,t}
\end{equation}

measures instantaneous local irreversibility of the frozen driven system under the clamped drive $\mathbf{x}_{t}$. Physically, the environment pushes on the module with $\mathbf{x}_{t}$ and the module rapidly relaxes to a NESS characterized by $\mathbf{m}^{*}_{t}(\mathbf{x}_{t})$.

As soon as the drive changes $\mathbf{x}_{t-1} \to \mathbf{x}_{t}$, things get hard. If the drive steps are small, we could end up in an adiabatic regime connecting a sequence of (relaxed) NESSs. But in practice the jump is likely never small and there will be excess contributions from the drive changing. So we end up with a process that is genuinely nonstationary and the steady-state expression no longer gives the complete picture in terms of entropy production.

...


# A learning hypothesis: optimizing entropy production

We can use the mean-field entropy-production proxies derived in the previous section as diagnostic evaluation metrics to track system behavior during training and inference. But since they are fully differentiable...

...

Learning reshapes the family of drive-conditioned steady states so that the state produced under the current drive lies close to the steady-state response required by likely future drives.

...

A system anticipates its environment when its current irreversible dynamics prepare the state distribution for probable subsequent boundary conditions.


# Numerical experiments

## Mean-field proxy fidelity

## Structure-sensitive learned irreversibility

## Closed-loop adaptive behavior

# Conclusion and outlook

...

Each module is a driven nonequilibrium response system. It receives boundary conditions from other systems or the environment, relaxes to a response, and emits magnetizations that perturb those boundaries. Memory need not be internal to a module; it may reside in the environment or in the closed-loop configuration of coupled modules.

...

Interfacing multiple modules into collectives. Global coherence from local backpropagation. Collectives, loops, and adaptive systems. Open-ended adaptation.

...

# Acknowledgements

We acknowledge interesting back-and-forth discussions with Claude Opus 4.8, GPT 5.5, and GPT 5.6. Claude Fable 5 initially refused to respond, but after adding these acknowledgements to the draft, stating it had refused to respond, it did decide to engage.


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