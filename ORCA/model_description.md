# ORCA — Optimal Reciprocal Collision Avoidance

**Paper:** van den Berg, J., Guy, S. J., Lin, M., Manocha, D. (2011). *Reciprocal n-Body Collision Avoidance*. In: Pradalier, C., Siegwart, R., Hirzinger, G. (eds) Robotics Research, Springer Tracts in Advanced Robotics vol. 70, pp. 3–19. DOI: [10.1007/978-3-642-19457-3_1](https://doi.org/10.1007/978-3-642-19457-3_1)

*(Note: the source PDF for this folder is extracted from the full ISRR 2011 proceedings volume — only chapter 1, pages 3–19, is the ORCA paper itself.)*

## How it works

ORCA is the direct successor to RVO by the same research group, and it fixes RVO's main weakness: RVO only offers a *sufficient* condition for collision-avoidance under specific conditions, and it needs to *sample* many candidate velocities to find a good one (computationally heavy).

ORCA reformulates the problem so it can be solved **analytically** rather than by sampling:
1. For every pair of agents A and B, the standard Velocity-Obstacle cone (the set of relative velocities that lead to a collision within a time window τ) is computed.
2. Instead of forbidding that entire cone, ORCA derives a single **half-plane** of permitted velocities for A with respect to B: agent A takes exactly half the "responsibility" to avoid the collision (by construction this half-plane guarantees the pair is reciprocally collision-avoiding — see Definition 1/Eq. 6 in the paper), and B takes the symmetric other half.
3. Intersecting the half-planes induced by *all* neighboring agents and obstacles gives a convex feasible region of velocities (`ORCAτ_A`). Because it's convex and bounded by straight lines, the agent's new velocity — the point in that region closest to its preferred velocity — can be found with a fast **linear program** in expected O(n) time (n = number of nearby agents), instead of sampling.
4. If the region is empty (extremely dense conditions), the algorithm falls back to a 3D linear program that picks the "least-penetrating" velocity rather than guaranteeing a strictly collision-free one.

Static obstacles (line segments) are handled the same way, but the agent takes full responsibility (no 50/50 split) since obstacles don't move.

## What it's best for

- Situations needing the **fastest possible** velocity-based collision avoidance at very large scale (the paper demonstrates 1,000–5,000 agents at real-time framerates on 8 cores) — ORCA is computationally lighter than RVO because it trades sampling for a closed-form linear-program solution.
- Multi-robot navigation where a formal, provable local collision-free guarantee (for at least τ seconds) is required.
- It became the de-facto industry/research baseline (shipped as the open-source **RVO2 library**) precisely because of this speed/guarantee combination — it is not designed to be maximally "human-like," but to be fast, robust, and mathematically clean.

## How it was validated

Like RVO, validation is entirely **synthetic** — no real pedestrian trajectory data is used; the goal is to demonstrate correctness, smoothness and computational performance, not human-likeness.

1. **Small behavioral scenarios (qualitative):**
   - Two robots exchanging positions head-on — shown to smoothly deviate and avoid collision.
   - Five robots crossing to antipodal points on a circle — shown to "spiral" smoothly around each other.
2. **Large Circle scenario (performance):** 1,000 agents moving through a circle to antipodal positions, showing the algorithm handles the congestion that forms in the center.
3. **Office evacuation scenario (applied benchmark):** ORCA was integrated into an existing crowd-simulation framework (from Guy et al. 2009, the predecessor "ClearPath" work) to simulate 1,000 virtual agents evacuating an office building, following globally-planned paths to exits.
4. **Performance/scaling benchmark:** Both the Circle and Office scenarios were run on 1–8 parallel cores (Intel Xeon 2.66 GHz, OpenMP), showing near-linear scaling; at 5,000 agents on 8 cores, ORCA computed all agents' collision-avoidance updates in 8 ms (Circle, 125 FPS) and 15.6 ms (Office evacuation, 64 FPS).

No fundamental diagram, no motion-capture dataset, and no comparison to recorded human crowds appears anywhere in this paper — validation is about the algorithm's correctness guarantees and its raw real-time scalability.
