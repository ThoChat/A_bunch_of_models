# Universal Power Law Governing Pedestrian Interactions

**Paper:** Karamouzas, I., Skinner, B., Guy, S. J. (2014). *Universal Power Law Governing Pedestrian Interactions*. Physical Review Letters 113, 238701. DOI: [10.1103/PhysRevLett.113.238701](https://doi.org/10.1103/PhysRevLett.113.238701)

## How it works

Unlike the other models in this collection, this paper's main contribution is not a simulation algorithm but a **data-driven physical law**, derived by directly measuring how strongly real pedestrians repel each other — which the authors then turn into a force model.

1. **Statistical-mechanics measurement:** the authors use the *pair distribution function* g(x) — a standard tool from condensed-matter physics — computed over real recorded pedestrian trajectories. g(x) compares how often two pedestrians are observed at a given relative configuration x versus how often that configuration would occur if pedestrians did not interact at all (estimated by sampling pairs of pedestrians who were never in the scene at the same time).
2. **Key finding — anticipation, not distance, governs interaction strength:** plotting g as a function of raw physical separation r gives inconsistent curves depending on how fast the two pedestrians are closing the distance. But plotting g as a function of the **time to collision τ** (how long until the two would collide if both kept their current velocity) makes all the curves collapse onto a single curve — meaning pedestrians react to an *anticipated future* collision, not to current distance.
3. Treating g(τ) as a Boltzmann-like distribution (`g(τ) ∝ exp[−E(τ)/E₀]`), the interaction "energy" E(τ) can be extracted directly from the data: `E(τ) ∝ ln(1/g(τ))`. Fitting this to the data reveals a clean power law: **E(τ) ∝ k/τ² · e^(−τ/τ₀)** (quadratic falloff with time to collision, cut off beyond a characteristic screening range τ₀ ≈ 3 s).
4. This energy law directly implies a **repulsive force** `F = −∇r[k/τ² · e^(−τ/τ₀)]` that can be plugged into any force-based agent simulation (in the spirit of the Social Force Model) alongside a simple goal-reaching driving force.

## What it's best for

- Providing an **empirically-grounded, minimal-parameter interaction force** for force-based crowd simulations — as an alternative to hand-tuned social-force parameters, since this law was fit directly to human motion data rather than designed by intuition.
- Simulations that need to reproduce known collective crowd phenomena (lane formation, arching, congestion slowdowns, anticipatory avoidance) with a very simple 2-parameter force law (k, τ₀).
- It generalizes across very different crowd conditions (sparse multidirectional pedestrian flow vs. dense unidirectional bottleneck flow) with the *same* underlying law — useful when a single interaction model needs to remain valid across a wide density range.
- Less suited on its own to extremely high densities, where the authors note that finite reaction-time saturation effects and short-range "shock wave" phenomena are not captured by this time-to-collision-only formulation.

## How it was validated

This is the only model in the collection whose core claim *is itself* a direct empirical measurement from real human motion-capture/video data, not a simulation compared post-hoc to reality:

1. **Two real-world pedestrian trajectory datasets** were used to derive and cross-validate the law:
   - An **"Outdoor" dataset** (1,146 trajectories) combining several sparse-to-moderate density outdoor pedestrian datasets, including the "Crowds by Example" dataset (Lerner, Chrysanthou & Lischinski 2007) and the ETH pedestrian dataset (Pellegrini, Ess, Schindler & Van Gool, ICCV 2009).
   - A **"Bottleneck" dataset** (354 trajectories) of pedestrians in dense crowds passing through narrow bottlenecks, from controlled laboratory experiments (Seyfried, Passon, Steffen, Boltes, Rupprecht & Klingsch, *Transportation Science* 43, 2009).
   - Despite these two datasets capturing qualitatively very different motion (multidirectional/sparse vs. unidirectional/dense), **both independently produced the same E(τ) ∝ 1/τ² power law** (R² = 0.92 for Outdoor, R² = 0.94 for Bottleneck) — this cross-dataset consistency is the paper's central validation claim.
2. **Self-consistency check via simulation:** the derived force law (Eq. 3) was used to drive an actual agent-based simulation; the resulting *simulated* trajectories were then run back through the same pair-distribution-function measurement procedure. The simulation reproduced the same E(τ) power law observed in the real data, whereas a control simulation using a purely **distance**-based force (no time-to-collision anticipation) failed to reproduce this dependency at all — directly demonstrating that anticipation (not distance) is the key mechanism.
3. **Reproduction of known qualitative crowd phenomena** in simulation: arching around narrow passages, "clogging"/zipping patterns at bottlenecks, and spontaneous self-organized lane formation in opposing flows were all observed emerging from agents using only this force law.
4. **Match to the fundamental diagram:** the paper states that simulated pedestrians using this law match the well-known real-world fundamental diagram (speed vs. density relationship; Weidmann 1993) of human crowds.
5. A qualitative large-scale synchronization/flocking effect was also observed for goal-less agents propelled forward, compared informally to reported dense non-goal-oriented human crowd behavior.
