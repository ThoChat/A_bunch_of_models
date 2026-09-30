# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import dataclass, replace

import numpy as np
from jupedsim.models.custom_model import CustomOperationalModel
from scipy.optimize import linprog

EPS = 1e-9
ES = 2.23  # J/(kg s), standing cost, paper footnote to Eq. 1 [Whi07]
EW = 1.26  # J s/(kg m^2), speed-dependent cost, same footnote


@dataclass(kw_only=True, frozen=True)
class PLEState:
    velocity: tuple[float, float] = (0.0, 0.0)
    # The step only exposes the *direction* to the target, but Eq. 4 needs
    # the distance |G_A - p_A| too. The agent therefore dead-reckons its own
    # position (p += movement, which the simulation applies unchanged) and
    # carries its goal. Runners check this against the engine's position.
    position: tuple[float, float]
    goal: tuple[float, float]
    radius: float = 0.3  # "a hard disk with radius of at least 0.3m" (Sec. 6)
    es: float = ES
    ew: float = EW
    # Look-ahead of the greedy formulation (Eq. 3-4): v_new is assumed to be
    # held for tau seconds. The paper gives no value; see the notebook.
    tau: float = 0.5
    # PV_A is ORCA's permissible set [vdBGLM09]; these are its parameters.
    time_horizon: float = 2.0
    time_horizon_obst: float = 0.5
    max_speed: float = 2.0
    neighbor_dist: float = 5.0
    # Sec. 4.4: "In practice, we may only consider the closest neighboring
    # agents during the computation of PV_A."
    max_neighbors: int = 10
    # Diagnostics, written every step.
    pv_empty: bool = False


def _det(a, b):
    """2-D cross product a x b on (..., 2) arrays."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def energy_rate(v, es=ES, ew=EW):
    """Eq. 1 per unit mass: P/m = es + ew |v|^2 [W/kg]."""
    return es + ew * np.einsum("...i,...i->...", v, v)


def min_energy(length, es=ES, ew=EW):
    """Corollary 1 per unit mass: 2 L sqrt(es ew) [J/kg]."""
    return 2.0 * length * np.sqrt(es * ew)


class PLEdestriansModel(CustomOperationalModel):
    """
    PLEdestrians, as described in: "PLEdestrians: A Least-Effort Approach to
    Crowd Simulation" by Guy, Chhugani, Curtis, Dubey, Lin & Manocha
    (SCA 2010).

    Every step, agent A picks, among its permissible velocities PV_A, the one
    minimising the greedy effort estimate of Eq. 4 (per unit mass):

        f(v) = tau (es + ew |v|^2) + 2 sqrt(es ew) |G_A - p_A - tau v|

    i.e. the energy of holding v for tau seconds plus the least possible
    energy (Corollary 1) of then walking straight to the goal.

    PV_A (Sec. 4.1) is ORCA's permissible set [vdBGLM09]: one half-plane per
    neighbour (the neighbour takes the other half of the avoidance), one per
    nearby wall segment, and the max-speed disc. It is written here from
    the ORCA paper's construction (Eqs. 5-6 and Sec. 5.4 of vdBGLM09).

    Minimisation (Sec. 4.3): f is convex and PV_A is convex, so the minimum
    is either the unconstrained optimum, or lies on the boundary of PV_A.
    We enumerate the boundary exactly as the paper's Steps 1-3 do: clip
    every constraint line to the other constraints and the disc, minimise f
    on each resulting segment, keep the best. The paper solves the 1-D
    problem on a line in closed form (quartic Eq. 7 plus Eq. 8); we find
    the same minimiser by bisection on the derivative, which is monotone
    because f is convex along the line. This gives the same point to 1e-9
    and avoids choosing the right root of the quartic.

    The disc's own boundary is an arc, not a line: on |v| = R, f is smallest
    for v parallel to G_A - p_A, so R * unit(G_A - p_A) is added as a
    candidate when it is permissible.

    Beyond the paper, both needed in practice and flagged in the notebook:

    * Empty PV_A. The paper does not cover dense crowds where the
      half-planes have no common point (it notes overlaps above
      4 agents/m^2). As ORCA's Sec. 5.3 does, we then take the velocity
      that minimises the largest violation of the agent half-planes,
      keeping walls hard, solved as a small LP.
    * Symmetry breaking (`pref_noise`, off by default). In an exactly
      symmetric crossing (the n-Agent Circle) every agent's left and right
      neighbours give mirror-image half-planes, the optimum lies on the
      radial axis, and all agents brake together into a packed ring that
      never resolves. The same group's RVO2 reference code perturbs each
      preferred velocity by a random vector of at most 1e-4 m/s every step
      "to avoid deadlocks due to perfect symmetry". We do the same to the
      unconstrained optimum: the goal vector is shifted by tau * xi with
      |xi| <= pref_noise (m/s), so v_opt moves by at most pref_noise.
    * Already-overlapping pairs and walls (spawn jitter, fallback steps).
      ORCA's construction is undefined there; we use the half-plane that
      separates the pair within one step (time horizon dt), as ORCA's
      reference code does.
    """

    def __init__(self, pref_noise: float = 0.0, seed: int = 0):
        CustomOperationalModel.__init__(self)
        self.pref_noise = pref_noise
        self._rng = np.random.default_rng(seed)
        self.n_pv_empty = 0

    # --------------------------------------------------------- PV_A (ORCA)
    @staticmethod
    def agent_halfplanes(state, v, rel_pos, vel_other, rad_other, dt):
        """ORCA_A|B half-planes, as (point, direction): allowed = left of
        direction, i.e. det(direction, point - v) <= 0."""
        rel_vel = v[None, :] - vel_other
        dist_sq = np.einsum("ij,ij->i", rel_pos, rel_pos)
        comb_r = state.radius + rad_other
        inv_tau = 1.0 / state.time_horizon
        n = len(rel_pos)
        dirs = np.empty((n, 2))
        u = np.empty((n, 2))

        # Vector from the cut-off circle's centre (rel_pos / tau) to rel_vel.
        w = rel_vel - inv_tau * rel_pos
        w_sq = np.einsum("ij,ij->i", w, w)
        dot1 = np.einsum("ij,ij->i", w, rel_pos)
        colliding = dist_sq <= comb_r**2
        on_cutoff = ~colliding & (dot1 < 0.0) & (dot1**2 > comb_r**2 * w_sq)
        on_legs = ~colliding & ~on_cutoff

        if on_cutoff.any():
            w_len = np.sqrt(w_sq[on_cutoff])
            unit_w = w[on_cutoff] / w_len[:, None]
            dirs[on_cutoff] = np.stack([unit_w[:, 1], -unit_w[:, 0]], axis=1)
            u[on_cutoff] = (comb_r[on_cutoff] * inv_tau - w_len)[:, None] * unit_w

        if on_legs.any():
            p, r, d2 = rel_pos[on_legs], comb_r[on_legs], dist_sq[on_legs]
            leg = np.sqrt(d2 - r * r)
            left = _det(p, w[on_legs]) > 0.0
            d_left = np.stack([p[:, 0] * leg - p[:, 1] * r, p[:, 0] * r + p[:, 1] * leg], 1)
            d_right = -np.stack([p[:, 0] * leg + p[:, 1] * r, -p[:, 0] * r + p[:, 1] * leg], 1)
            d = np.where(left[:, None], d_left, d_right) / d2[:, None]
            rv = rel_vel[on_legs]
            dirs[on_legs] = d
            u[on_legs] = np.einsum("ij,ij->i", rv, d)[:, None] * d - rv

        if colliding.any():
            # Beyond the paper (see class docstring): separate within dt.
            wc = rel_vel[colliding] - rel_pos[colliding] / dt
            wc_len = np.maximum(np.linalg.norm(wc, axis=1), 1e-12)
            unit_w = wc / wc_len[:, None]
            dirs[colliding] = np.stack([unit_w[:, 1], -unit_w[:, 0]], axis=1)
            u[colliding] = (comb_r[colliding] / dt - wc_len)[:, None] * unit_w

        return v[None, :] + 0.5 * u, dirs

    @staticmethod
    def wall_halfplanes(state, closest, dist, dt):
        """Walls: v . c_hat <= (d - r) / tau_obst, c_hat = unit vector to the
        segment's closest point. This is the tangent of the truncated VO of
        the segment at its point closest to the origin (vdBGLM09 Sec. 5.4
        with optimisation velocity 0); exact for a straight wall. When
        already touching (d <= r), push out within one step instead."""
        c_hat = closest / np.maximum(dist, 1e-12)[:, None]
        horizon = np.where(dist > state.radius, state.time_horizon_obst, dt)
        points = c_hat * ((dist - state.radius) / horizon)[:, None]
        dirs = np.stack([-c_hat[:, 1], c_hat[:, 0]], axis=1)
        return points, dirs

    # ------------------------------------------------------------ Eq. 4
    @staticmethod
    def effort(v, a, state):
        """Eq. 4 per unit mass, for v of shape (..., 2)."""
        tau = state.tau
        rest = np.linalg.norm(a - tau * v, axis=-1)
        return tau * energy_rate(v, state.es, state.ew) + 2.0 * np.sqrt(
            state.es * state.ew
        ) * rest

    @staticmethod
    def unconstrained_optimum(a, state):
        """Minimiser of Eq. 4 without constraints: v_des = sqrt(es/ew) toward
        the goal (paper's case 1), or a / tau once the goal is closer than
        tau * v_des (then f's kink at tau v = a is the minimum)."""
        v_des = np.sqrt(state.es / state.ew)
        dist = np.linalg.norm(a)
        if dist < 1e-12:
            return np.zeros(2)
        if dist >= state.tau * v_des:
            return v_des * a / dist
        return a / state.tau

    @staticmethod
    def boundary_candidates(points, dirs, radius, a, state):
        """Steps 1-2 of Sec. 4.3: minimiser of Eq. 4 on every boundary
        segment of PV_A. Returns an (m, 2) array (m may be 0)."""
        n = len(points)
        # Disc clip: |p + t d| <= R.
        pd = np.einsum("ij,ij->i", points, dirs)
        disc = pd**2 + radius**2 - np.einsum("ij,ij->i", points, points)
        ok = disc >= 0.0
        sq = np.sqrt(np.clip(disc, 0.0, None))
        lo, hi = -pd - sq, -pd + sq

        # Clip line i by every other line j: det(d_j, p_j - p_i - t d_i) <= 0
        # <=> t * den >= num.
        num = _det(dirs[None, :, :], points[None, :, :] - points[:, None, :])
        den = _det(dirs[None, :, :], dirs[:, None, :])
        np.fill_diagonal(num, -1.0)
        np.fill_diagonal(den, 0.0)
        par = np.abs(den) <= EPS
        ok &= ~np.any(par & (num > EPS), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        lo = np.maximum(lo, np.max(np.where(den > EPS, t, -np.inf), axis=1))
        hi = np.minimum(hi, np.min(np.where(den < -EPS, t, np.inf), axis=1))
        ok &= lo <= hi + 1e-12
        if not ok.any():
            return np.empty((0, 2))
        p, d, lo, hi = points[ok], dirs[ok], lo[ok], hi[ok]

        # Along v = p + t d (|d| = 1), with q = a - tau p, c = det(d, q) and
        # u = tau t - q.d, Eq. 4's stationarity condition divided by tau is
        #   phi(u) = 2 ew (p.d + (u + q.d) / tau) + k u / sqrt(u^2 + c^2) = 0,
        # which is the paper's quartic (Eq. 7) before squaring. phi is
        # strictly increasing, so its root is unique and lies where the
        # bounded term k u / sqrt(.) in [-k, k] allows. g is convex along the
        # line, so the minimiser on the segment is that root clipped to it.
        tau, ew = state.tau, state.ew
        k = 2.0 * np.sqrt(state.es * ew)
        q = a[None, :] - tau * p
        pd, qd, c2 = np.einsum("ij,ij->i", p, d), np.einsum("ij,ij->i", q, d), _det(d, q) ** 2
        alpha, beta = 2.0 * ew / tau, 2.0 * ew * (pd + qd / tau)
        u_lo, u_hi = (-beta - k) / alpha, (-beta + k) / alpha
        u = 0.5 * (u_lo + u_hi)
        for _ in range(60):  # safeguarded Newton; typically 5-10 iterations
            root = np.sqrt(u * u + c2 + 1e-30)
            phi = alpha * u + beta + k * u / root
            u_lo = np.where(phi < 0.0, u, u_lo)
            u_hi = np.where(phi > 0.0, u, u_hi)
            newton = u - phi / (alpha + k * c2 / root**3)
            inside = (newton > u_lo) & (newton < u_hi)
            u_new = np.where(inside, newton, 0.5 * (u_lo + u_hi))
            if np.max(np.abs(u_new - u)) < 1e-12:
                u = u_new
                break
            u = u_new
        t_best = np.clip((u + qd) / tau, lo, hi)
        return p + t_best[:, None] * d

    @staticmethod
    def least_penetration(points, dirs, n_hard, radius):
        """Beyond the paper (see class docstring): ORCA Sec. 5.3. Minimise the
        largest violation s of the agent half-planes, walls hard, inside a
        64-gon inscribed in the speed disc. Variables (vx, vy, s)."""
        # det(d, p - v) = det(d, p) - (d_x v_y - d_y v_x) <= s
        #   <=>  d_y v_x - d_x v_y - s <= -det(d, p)
        rhs = -_det(dirs, points)
        A = np.column_stack([dirs[:, 1], -dirs[:, 0], -np.ones(len(points))])
        A[:n_hard, 2] = 0.0  # walls: no slack
        ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        disc = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(64)])
        A_ub = np.vstack([A, disc])
        b_ub = np.concatenate([rhs, np.full(64, radius)])
        res = linprog(
            [0.0, 0.0, 1.0], A_ub=A_ub, b_ub=b_ub,
            bounds=[(None, None), (None, None), (0.0, None)], method="highs",
        )
        if res.status != 0:
            # Walls alone infeasible (deep in a corner after a fallback
            # step): relax them too.
            A_ub[:n_hard, 2] = -1.0
            res = linprog(
                [0.0, 0.0, 1.0], A_ub=A_ub, b_ub=b_ub,
                bounds=[(None, None), (None, None), (0.0, None)], method="highs",
            )
        return res.x[:2] if res.status == 0 else np.zeros(2)

    # --------------------------------------------------------------- step
    def compute_next_state(self, state: PLEState, step):
        dt = step.dt
        v = np.array(state.velocity, dtype=float)
        a = np.array(state.goal) - np.array(state.position)
        # Around obstacles the goal is not in line of sight. The paper then
        # aims at "the next intermediary node along the guiding path"
        # (Sec. 4.1); the step exposes only the direction to JuPedSim's next
        # path corner, not its distance. So: direction from JuPedSim's
        # router, length = straight-line distance to the goal. In an open
        # scene both directions coincide and this is exactly G_A - p_A.
        route_dir = np.array(step.orientation_to_next_target, dtype=float)
        if route_dir @ route_dir > 0.5:
            a = np.linalg.norm(a) * route_dir
        if self.pref_noise > 0.0 and np.linalg.norm(a) > 1e-6:
            # Beyond the paper: symmetry breaking (see class docstring).
            ang = self._rng.uniform(0.0, 2.0 * np.pi)
            mag = self._rng.uniform(0.0, self.pref_noise)
            a = a + state.tau * mag * np.array([np.cos(ang), np.sin(ang)])

        # Walls first (hard in the fallback), then agents.
        wall_range = state.radius + state.max_speed * state.time_horizon_obst
        walls = step.walls_in_range(wall_range)
        if walls:
            closest = np.array([w.closest_point for w in walls], dtype=float)
            dist = np.array([w.distance for w in walls], dtype=float)
            w_pts, w_dirs = self.wall_halfplanes(state, closest, dist, dt)
        else:
            w_pts = w_dirs = np.empty((0, 2))

        neighbors = step.other_agents_in_range(state.neighbor_dist)
        if neighbors:
            rel_pos = np.array([n.relative_position for n in neighbors], dtype=float)
            order = np.argsort(np.einsum("ij,ij->i", rel_pos, rel_pos), kind="stable")
            order = order[: state.max_neighbors]
            nstates = [neighbors[i].state for i in order]
            a_pts, a_dirs = self.agent_halfplanes(
                state, v, rel_pos[order],
                np.array([s.velocity for s in nstates], dtype=float),
                np.array([s.radius for s in nstates], dtype=float), dt,
            )
        else:
            a_pts = a_dirs = np.empty((0, 2))

        points = np.concatenate([w_pts, a_pts])
        dirs = np.concatenate([w_dirs, a_dirs])
        R = state.max_speed

        def permissible(cand):
            if len(points) == 0:
                return np.linalg.norm(cand, axis=-1) <= R + 1e-9
            viol = _det(dirs[None, :, :], points[None, :, :] - cand[:, None, :])
            return np.all(viol <= 1e-9, axis=1) & (np.linalg.norm(cand, axis=1) <= R + 1e-9)

        v_opt = self.unconstrained_optimum(a, state)
        pv_empty = False
        if permissible(v_opt[None, :])[0]:
            new_v = v_opt
        else:
            cands = [self.boundary_candidates(points, dirs, R, a, state)]
            dist_a = np.linalg.norm(a)
            if dist_a > 1e-12:
                arc = R * a / dist_a
                if permissible(arc[None, :])[0]:
                    cands.append(arc[None, :])
            cands = np.concatenate(cands)
            if len(cands):
                new_v = cands[np.argmin(self.effort(cands, a, state))]
            else:
                pv_empty = True
                self.n_pv_empty += 1
                new_v = self.least_penetration(points, dirs, len(w_pts), R)

        new_velocity = (float(new_v[0]), float(new_v[1]))
        movement = (new_velocity[0] * dt, new_velocity[1] * dt)
        new_state = replace(
            state,
            velocity=new_velocity,
            position=(state.position[0] + movement[0], state.position[1] + movement[1]),
            pv_empty=pv_empty,
        )
        return new_state, movement
