# RVO — Reciprocal Velocity Obstacles

**Paper:** van den Berg, J., Lin, M., Manocha, D. (2008). *Reciprocal Velocity Obstacles for Real-Time Multi-Agent Navigation*. IEEE ICRA 2008. DOI: [10.1109/ROBOT.2008.4543489](https://doi.org/10.1109/ROBOT.2008.4543489)

## How it works

RVO is a velocity-based collision-avoidance algorithm. Each agent perceives its neighbors' current positions and velocities and must pick a new velocity, every simulation step, that keeps it collision-free.

It builds on the older **Velocity Obstacle (VO)** concept (Fiorini & Shiller, 1998): given a neighbor B, the set of velocities that would cause A to collide with B in the future forms a cone-shaped region in A's velocity space. If A simply always picks a velocity outside all such cones, and B does the same treating A as an obstacle, both agents end up **oscillating** — each reacts to the other's *previous* reaction, causing back-and-forth "dancing" motions.

RVO's fix: instead of picking any velocity outside the VO cone, an agent picks the velocity that is the **average of its current velocity and a velocity outside the cone**. The paper proves this "50/50 sharing of the avoidance effort" is both collision-free (Theorem 6) and provably oscillation-free (Theorem 8), as long as both agents use the same rule. Each agent's admissible velocities from all neighbors and obstacles are combined into one region, and a candidate velocity is chosen by sampling many velocities and picking the one closest to the agent's preferred velocity, penalized by how soon it would lead to a collision.

## What it's best for

- Very large populations of homogeneous agents (hundreds to ~1000) that need smooth, real-time, collision-free, non-oscillating motion with no central coordination and no communication between agents.
- Scenarios with both static and independently-moving obstacles (e.g., a car driving through a crowd).
- Situations where full parallelization matters, since every agent's velocity is computed independently.
- It is *not* aimed at reproducing subtle human-like behavior (no comparison to real pedestrian data is made) — it targets algorithmic correctness (no collisions, no oscillation) and raw performance/scalability.

## How it was validated

The validation is purely **synthetic/algorithmic**, not based on real pedestrian data — there is no comparison to recorded human trajectories anywhere in the paper. It consists of:

1. **Circle scenario** — agents placed evenly on a circle must swap to the antipodal position, forcing a dense crowd to form in the middle.
   - First run with 12 agents, directly comparing RVO's smooth paths against the original (oscillating, chaotic) Velocity Obstacle approach — qualitative proof that the reciprocal averaging removes oscillation.
   - Then scaled up to 250 and 1000 agents to test scalability; frame time was shown to scale linearly with the number of agents, achieving >10 FPS with 1000 agents and a 0.25 s timestep on a 2008-era Intel Core2 Duo 1.66 GHz laptop.
2. **Narrow Passage scenario** — four groups of 25 agents in the four corners of an environment, each heading to the opposite corner through square obstacles that create narrow passages, forcing groups with opposing goals to meet and funnel through the same bottleneck. Used to confirm agents don't get stuck (except for a noted failure case with U-shaped obstacles).
3. **Moving Obstacle scenario** — eleven agents cross a street on which a car (treated as a passively-moving, non-reactive obstacle) is driving, testing correct handling of high-speed, non-cooperating obstacles.

No real-world dataset, fundamental diagram, or human-subject experiment is used anywhere in this paper — the "ground truth" being validated against is the mathematical guarantee (proven analytically) of collision-freedom and oscillation-freedom, supported by visual inspection of the resulting agent traces and by frame-rate/scalability benchmarks.
