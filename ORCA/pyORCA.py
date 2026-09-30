# SPDX-License-Identifier: LGPL-3.0-or-later
import time
from dataclasses import dataclass, replace

import numpy as np
from jupedsim.models.custom_model import CustomOperationalModel

EPSILON = 1e-5


@dataclass(kw_only=True, frozen=True)
class ORCAState:
    velocity: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.25
    pref_speed: float = 1.5
    max_speed: float = 1.5
    time_horizon: float = 2.0
    time_horizon_obst: float = 0.5
    # None = the paper's own cut-off (Section 5.1): robots farther away than
    # (v_max_A + v_max_B) * tau can never collide within tau, with v_max_B
    # "guessed" equal to A's own, i.e. 2 * max_speed * time_horizon.
    neighbor_dist: float | None = None
    # None = use every neighbour within neighbor_dist (the paper). The
    # authors' RVO2 library caps this at the k nearest; only set it for the
    # timing sweep, where we say so.
    max_neighbors: int | None = None
    # Waypoint-overshoot fix (see ORCAModel docstring). Must be False when
    # agents follow a multi-corner route (the office), where a sharp turn at
    # a corner also reverses the orientation.
    stop_at_goal: bool = True
    last_orientation: tuple[float, float] = (0.0, 0.0)
    arrived: bool = False
    # Diagnostics, written every step: whether the 2-D LP was infeasible and
    # the 3-D "least penetration" program of Section 5.3 had to be used.
    used_3d_lp: bool = False


def _det(a, b):
    """2-D cross product a x b; works on (2,) and (n, 2) arrays."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _linear_program1(points, dirs, k, radius, opt, direction_opt):
    """Optimum on line k subject to lines [0, k) and the speed disc.

    Port of RVO2's linearProgram1 (the 1-D step of the randomized
    incremental LP of de Berg et al., which the paper cites as [3]), with
    the loop over previous lines vectorized. Returns (ok, result).
    """
    p, d = points[k], dirs[k]
    dot = p @ d
    disc = dot * dot + radius * radius - p @ p
    if disc < 0.0:
        return False, None  # the speed disc invalidates line k entirely
    sq = np.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq

    if k > 0:
        denom = _det(d, dirs[:k])
        numer = _det(dirs[:k], p - points[:k])
        parallel = np.abs(denom) <= EPSILON
        if np.any(numer[parallel] < 0.0):
            return False, None  # parallel and on the wrong side
        t = numer[~parallel] / denom[~parallel]
        pos = denom[~parallel] >= 0.0
        if pos.any():
            t_right = min(t_right, t[pos].min())
        if (~pos).any():
            t_left = max(t_left, t[~pos].max())
        if t_left > t_right:
            return False, None

    if direction_opt:
        return True, p + (t_right if opt @ d > 0.0 else t_left) * d
    t = d @ (opt - p)
    t = min(max(t, t_left), t_right)
    return True, p + t * d


def _linear_program2(points, dirs, radius, opt, direction_opt):
    """Point closest to `opt` (or furthest along `opt` if direction_opt)
    inside the speed disc and left of every line. Port of RVO2's
    linearProgram2. Returns (index_of_first_failing_line, result);
    index == len(points) means success."""
    if direction_opt:
        result = opt * radius
    elif opt @ opt > radius * radius:
        result = opt / np.linalg.norm(opt) * radius
    else:
        result = opt.copy()

    n = len(points)
    i = 0
    while i < n:
        violated = _det(dirs[i:], points[i:] - result) > 0.0
        if not violated.any():
            break
        k = i + int(np.argmax(violated))
        ok, new = _linear_program1(points, dirs, k, radius, opt, direction_opt)
        if not ok:
            return k, result
        result = new
        i = k + 1
    return n, result


def _linear_program3(points, dirs, n_obst, begin, radius, result):
    """Section 5.3: minimise the maximum signed distance to the agent
    half-planes (the 3-D LP projected to 2-D), keeping obstacle half-planes
    hard. Port of RVO2's linearProgram3."""
    distance = 0.0
    for i in range(begin, len(points)):
        if _det(dirs[i], points[i] - result) <= distance:
            continue
        # Project agent lines j in [n_obst, i) onto line i.
        pj, dj = points[n_obst:i], dirs[n_obst:i]
        det_ij = _det(dirs[i], dj)
        parallel = np.abs(det_ij) <= EPSILON
        same_dir = parallel & ((dj @ dirs[i]) > 0.0)
        keep = ~same_dir
        proj_p = np.empty_like(pj)
        safe_det = np.where(parallel, 1.0, det_ij)
        proj_p[:] = points[i] + (_det(dj, points[i] - pj) / safe_det)[:, None] * dirs[i]
        proj_p[parallel] = 0.5 * (points[i] + pj[parallel])
        proj_d = dj - dirs[i]
        proj_d /= np.maximum(np.linalg.norm(proj_d, axis=1), 1e-12)[:, None]
        proj_points = np.concatenate([points[:n_obst], proj_p[keep]])
        proj_dirs = np.concatenate([dirs[:n_obst], proj_d[keep]])

        opt = np.array([-dirs[i][1], dirs[i][0]])
        fail, new = _linear_program2(proj_points, proj_dirs, radius, opt, True)
        if fail >= len(proj_points):
            result = new
        # else: can in principle only happen through floating-point error;
        # keep the previous result, exactly as RVO2 does.
        distance = _det(dirs[i], points[i] - result)
    return result


class ORCAModel(CustomOperationalModel):
    """
    Optimal Reciprocal Collision Avoidance, as described in:
    "Reciprocal n-Body Collision Avoidance" by van den Berg, Guy, Lin &
    Manocha (ISRR 2009 / Robotics Research, STAR 70, 2011).

    Every step each agent A:

    1. builds, for each neighbour B, the half-plane ORCA_A|B of Eq. (6):
       with the optimization velocity equal to the current velocity
       (the paper's recommended choice, Section 5.2), u is the smallest
       change to the relative velocity v_A - v_B that takes it to the
       boundary of the truncated velocity obstacle VO^tau_A|B (Eq. 5), and
       A takes half of it: the permitted set is the half-plane through
       v_A + u/2 with outward normal n.
    2. builds, for each wall segment O, the half-plane ORCA_A|O of
       Section 5.4: optimization velocity 0, so the delimiting line is the
       tangent to VO^tau_obst_A|O at its point closest to the origin. For a
       segment at distance d that point is (d - r_A)/tau_obst in the
       direction of the segment's closest point, and the tangent is
       perpendicular to that direction: v . c_hat <= (d - r_A) / tau_obst.
    3. picks the velocity closest to v_pref inside the intersection of all
       half-planes and the max-speed disc (Eqs. 7-8), with the randomized
       incremental LP the paper cites ([3]).
    4. if that LP is infeasible (dense crowds), picks the velocity that
       minimises the maximum penetration of the agent half-planes, keeping
       the obstacle half-planes hard (Eq. 10, Section 5.3).

    The geometry of step 1 and the LP solvers of steps 3-4 are a direct
    port of the authors' reference implementation (RVO2 library,
    Agent.cpp), since the paper states the construction but not code.

    Two things beyond the paper, both needed in practice:

    * Already-overlapping pair (distance < r_A + r_B). Eq. (5) has no
      meaningful "closest boundary point" once the pair overlaps, and the
      paper does not cover it. As RVO2 does, we then use the cut-off
      circle of a VO with time horizon dt, i.e. the half-plane that
      separates the pair within the next step.
    * Waypoint overshoot. JuPedSim keeps pointing an agent at its final
      waypoint after it arrives, so a pure "v_pref = v_max * direction"
      agent flips back and forth across the point forever at full speed,
      and its neighbours see (and reciprocate against) that fake 1.5 m/s
      velocity. The step does not expose the distance to the target, so we
      detect the overshoot instead (the orientation reverses between two
      steps) and set v_pref = 0 from then on. RVO2's examples do the
      equivalent by shrinking v_pref within 1 m of the goal.
    * Symmetry breaking (`pref_noise`). In an exactly symmetric setup (the
      five-robot circle of Fig. 7b) every agent computes the mirror image of
      everyone else's LP, and all of them stop in front of each other for
      good. The authors' own RVO2 example code adds a random vector of at
      most 1e-4 m/s to every preferred velocity each step "to avoid
      deadlocks due to perfect symmetry"; the paper does not mention it.
      Off by default (0.0); the scenarios that need it say so.
    """

    def __init__(self, pref_noise: float = 0.0, seed: int = 0):
        CustomOperationalModel.__init__(self)
        self.pref_noise = pref_noise
        self._rng = np.random.default_rng(seed)
        # Wall-clock time spent inside compute_next_state, for comparing
        # with the paper's "time to solve the LP for every agent" (Fig. 10b).
        self.callback_seconds = 0.0
        self.n_3d_lp = 0

    @staticmethod
    def agent_lines(state, v, rel_pos, vel_other, rad_other, dt):
        """ORCA half-planes (point, direction) for each neighbour, vectorized.

        A half-plane is the set left of `direction` through `point`.
        """
        rel_vel = v[None, :] - vel_other
        dist_sq = np.einsum("ij,ij->i", rel_pos, rel_pos)
        comb_r = state.radius + rad_other
        comb_r_sq = comb_r * comb_r
        inv_tau = 1.0 / state.time_horizon

        n = len(rel_pos)
        dirs = np.empty((n, 2))
        u = np.empty((n, 2))

        # w: from the centre of the cut-off circle to the relative velocity.
        w = rel_vel - inv_tau * rel_pos
        w_len_sq = np.einsum("ij,ij->i", w, w)
        dot1 = np.einsum("ij,ij->i", w, rel_pos)

        colliding = dist_sq <= comb_r_sq
        cutoff = ~colliding & (dot1 < 0.0) & (dot1 * dot1 > comb_r_sq * w_len_sq)
        legs = ~colliding & ~cutoff

        if cutoff.any():
            w_len = np.sqrt(w_len_sq[cutoff])
            unit_w = w[cutoff] / w_len[:, None]
            dirs[cutoff] = np.stack([unit_w[:, 1], -unit_w[:, 0]], axis=1)
            u[cutoff] = ((comb_r[cutoff] * inv_tau - w_len))[:, None] * unit_w

        if legs.any():
            rp = rel_pos[legs]
            r = comb_r[legs]
            dsq = dist_sq[legs]
            leg = np.sqrt(dsq - r * r)
            left = _det(rp, w[legs]) > 0.0
            d_left = np.stack(
                [rp[:, 0] * leg - rp[:, 1] * r, rp[:, 0] * r + rp[:, 1] * leg], axis=1
            ) / dsq[:, None]
            d_right = -np.stack(
                [rp[:, 0] * leg + rp[:, 1] * r, -rp[:, 0] * r + rp[:, 1] * leg], axis=1
            ) / dsq[:, None]
            d = np.where(left[:, None], d_left, d_right)
            rv = rel_vel[legs]
            dirs[legs] = d
            u[legs] = np.einsum("ij,ij->i", rv, d)[:, None] * d - rv

        if colliding.any():
            # Beyond the paper: see the class docstring.
            inv_dt = 1.0 / dt
            wc = rel_vel[colliding] - inv_dt * rel_pos[colliding]
            wc_len = np.maximum(np.linalg.norm(wc, axis=1), 1e-12)
            unit_w = wc / wc_len[:, None]
            dirs[colliding] = np.stack([unit_w[:, 1], -unit_w[:, 0]], axis=1)
            u[colliding] = (comb_r[colliding] * inv_dt - wc_len)[:, None] * unit_w

        points = v[None, :] + 0.5 * u
        return points, dirs

    @staticmethod
    def obstacle_lines(state, closest, dist):
        """Section 5.4 half-planes: v . c_hat <= (d - r) / tau_obst."""
        c_hat = closest / np.maximum(dist, 1e-12)[:, None]
        points = c_hat * ((dist - state.radius) / state.time_horizon_obst)[:, None]
        dirs = np.stack([-c_hat[:, 1], c_hat[:, 0]], axis=1)
        return points, dirs

    def compute_next_state(self, state: ORCAState, step):
        t0 = time.perf_counter()
        dt = step.dt
        v = np.array(state.velocity, dtype=float)

        orientation = np.array(step.orientation_to_next_target, dtype=float)
        arrived = state.stop_at_goal and (
            state.arrived
            or float(orientation @ np.array(state.last_orientation)) < 0.0
        )
        v_pref = np.zeros(2) if arrived else state.pref_speed * orientation
        if self.pref_noise > 0.0 and not arrived:
            # RVO2 example code: uniform angle, uniform magnitude in [0, noise).
            a = self._rng.uniform(0.0, 2.0 * np.pi)
            v_pref = v_pref + self._rng.uniform(0.0, self.pref_noise) * np.array(
                [np.cos(a), np.sin(a)]
            )

        # --- obstacle half-planes (hard constraints, listed first) ---
        obst_range = state.max_speed * state.time_horizon_obst + state.radius
        walls = step.walls_in_range(obst_range)
        if walls:
            closest = np.array([w.closest_point for w in walls], dtype=float)
            dist = np.array([w.distance for w in walls], dtype=float)
            ok = dist > 1e-9
            o_points, o_dirs = self.obstacle_lines(state, closest[ok], dist[ok])
        else:
            o_points = o_dirs = np.empty((0, 2))

        # --- agent half-planes ---
        nd = state.neighbor_dist
        if nd is None:
            nd = 2.0 * state.max_speed * state.time_horizon
        neighbors = step.other_agents_in_range(nd)
        if neighbors:
            rel_pos = np.array([n.relative_position for n in neighbors], dtype=float)
            states = [n.state for n in neighbors]
            vel_other = np.array([s.velocity for s in states], dtype=float)
            rad_other = np.array([s.radius for s in states], dtype=float)
            d2 = np.einsum("ij,ij->i", rel_pos, rel_pos)
            order = np.argsort(d2, kind="stable")  # nearest first, like RVO2
            if state.max_neighbors is not None:
                order = order[: state.max_neighbors]
            a_points, a_dirs = self.agent_lines(
                state, v, rel_pos[order], vel_other[order], rad_other[order], dt
            )
        else:
            a_points = a_dirs = np.empty((0, 2))

        points = np.concatenate([o_points, a_points])
        dirs = np.concatenate([o_dirs, a_dirs])
        n_obst = len(o_points)

        fail, new_v = _linear_program2(points, dirs, state.max_speed, v_pref, False)
        used_3d = fail < len(points)
        if used_3d:
            new_v = _linear_program3(points, dirs, n_obst, fail, state.max_speed, new_v)
            self.n_3d_lp += 1

        new_velocity = (float(new_v[0]), float(new_v[1]))
        movement = (new_velocity[0] * dt, new_velocity[1] * dt)
        new_state = replace(
            state,
            velocity=new_velocity,
            last_orientation=(float(orientation[0]), float(orientation[1])),
            arrived=arrived,
            used_3d_lp=used_3d,
        )
        self.callback_seconds += time.perf_counter() - t0
        return new_state, movement
