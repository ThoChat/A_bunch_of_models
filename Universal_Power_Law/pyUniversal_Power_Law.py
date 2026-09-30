# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import dataclass, replace

import numpy as np
from jupedsim.models.custom_model import CustomOperationalModel


@dataclass(kw_only=True, frozen=True)
class PowerLawState:
    velocity: tuple[float, float]
    radius: float = 0.25
    pref_speed: float = 1.3
    # Interaction law. "ttc" is the paper's anticipatory force (Eq. S2);
    # "distance" is the distance-based control of the paper's Fig. 4 inset /
    # Fig. S5B / Fig. S6A (Helbing, Farkas & Vicsek 2000, the paper's Ref. [9]).
    interaction: str = "ttc"
    # Goal mode. "goal" follows the JuPedSim journey; "self_propelled" is the
    # goal-less walker of Fig. 3(e): propelled along its own current velocity.
    self_propelled: bool = False
    # Anticipatory-force parameters (Supplemental Material and the authors'
    # C++ reference code, Example.cpp): k = 1.5, tau0 = 3 s, exponent m = 2,
    # relaxation time xi = 0.54 s, neighbour range 10 m, max. acceleration 20.
    k: float = 1.5
    tau0: float = 3.0
    exponent: float = 2.0
    xi: float = 0.54
    neighbor_dist: float = 10.0
    max_accel: float = 20.0
    # Distance-based (Helbing et al. 2000) parameters, per unit mass (m = 80 kg).
    sfm_A: float = 2000.0 / 80.0
    sfm_B: float = 0.08
    sfm_k: float = 1.2e5 / 80.0
    sfm_kappa: float = 2.4e5 / 80.0
    sfm_range: float = 3.0


class UniversalPowerLawModel(CustomOperationalModel):
    """
    Anticipatory force model of Karamouzas, Skinner & Guy, "Universal Power
    Law Governing Pedestrian Interactions", PRL 113, 238701 (2014).

    The interaction energy between two pedestrians depends on their time to
    collision tau (Eq. 2), E(tau) = k / tau^2 * exp(-tau / tau0), and the
    force is its spatial gradient (Eq. 3). With x = x_i - x_j, v = v_i - v_j,
    R = r_i + r_j, a = |v|^2, b = -x.v, c = |x|^2 - R^2, d = b^2 - a c, the
    time to collision is tau = (b - sqrt(d)) / a, and the force on i is
    (Supplemental Material, Eq. S2):

        F_ij = -k e^(-tau/tau0) / (a tau^2) * (2/tau + 1/tau0)
               * [ v - (a x - (x.v) v) / sqrt(d) ]

    It is zero when there is no future collision (d <= 0 or tau <= 0).
    Each pedestrian also gets the driving force (v_pref - v) / xi of
    Helbing et al. 2000 (the paper's Ref. [9]), and walls repel with the
    same energy, using the time to collision with the wall segment.

    Everything below follows the authors' own C++ reference implementation
    (`Agent::computeForces`, http://motion.cs.umn.edu/PowerLaw/), which the
    Supplemental Material points to for the complete model:

    * unit mass (force = acceleration) and the total acceleration clamped to
      `max_accel` (20 m/s^2) before the semi-implicit Euler update
      v += a dt, x += v dt;
    * an already-overlapping pair uses R := R - |x| instead of R (the
      reference code's way of keeping tau defined during contact);
    * wall force: the time to collision with the wall segment inflated to a
      capsule of the agent's radius; skipped when the agent moves away from
      the wall.

    One extension that is not in the paper or the reference code is needed
    in JuPedSim: the engine raises an error when a step would cross a wall,
    and the reference model has no contact force at all (a pedestrian
    already touching a wall gets no wall force). `_wall_guard` removes the
    part of the step that would carry the agent closer than `WALL_MARGIN`
    to a wall. It only acts on agents already touching a wall.
    """

    WALL_MARGIN = 0.05

    def __init__(self):
        CustomOperationalModel.__init__(self)

    # ------------------------------------------------------------ agents
    @staticmethod
    def _ttc_agent_force(state, w, v_rel, radii):
        """Eq. S2, vectorized over neighbours.

        w: (n, 2) positions of neighbours relative to the agent (x_j - x_i),
        v_rel: (n, 2) relative velocities v_i - v_j, radii: (n,) r_i + r_j.
        Written with w = -x as in the reference code, so that b = w.v.
        """
        dist_sq = np.einsum("ij,ij->i", w, w)
        radius_sq = radii**2
        keep = dist_sq != radius_sq
        overlap = dist_sq < radius_sq
        # Reference code: overlapping pairs use their overlap depth as radius.
        radius_sq = np.where(overlap, (radii - np.sqrt(dist_sq)) ** 2, radius_sq)
        a = np.einsum("ij,ij->i", v_rel, v_rel)
        b = np.einsum("ij,ij->i", w, v_rel)
        c = dist_sq - radius_sq
        discr = b * b - a * c
        valid = keep & (discr > 0.0) & (np.abs(a) > 1e-5)
        if not valid.any():
            return np.zeros(2)
        w, v_rel, a, b, discr = w[valid], v_rel[valid], a[valid], b[valid], discr[valid]
        sq = np.sqrt(discr)
        tau = (b - sq) / a
        pos = tau > 0.0
        if not pos.any():
            return np.zeros(2)
        w, v_rel, a, b, sq, tau = w[pos], v_rel[pos], a[pos], b[pos], sq[pos], tau[pos]
        m = state.exponent
        mag = state.k * np.exp(-tau / state.tau0) / (a * tau**m) * (m / tau + 1.0 / state.tau0)
        direction = v_rel - (b[:, None] * v_rel - a[:, None] * w) / sq[:, None]
        return -(mag[:, None] * direction).sum(axis=0)

    @staticmethod
    def _distance_agent_force(state, w, v_rel, radii):
        """Helbing, Farkas & Vicsek (2000) pedestrian interaction, per unit mass.

        w: (n, 2) = x_j - x_i. Repulsion A exp((R - d)/B) plus body force
        k (R - d) and sliding friction kappa (R - d) dv_t on contact.
        """
        d = np.linalg.norm(w, axis=1)
        d = np.maximum(d, 1e-6)
        n = -w / d[:, None]  # from neighbour to agent
        overlap = np.clip(radii - d, 0.0, None)
        f_n = state.sfm_A * np.exp((radii - d) / state.sfm_B) + state.sfm_k * overlap
        t = np.stack([-n[:, 1], n[:, 0]], axis=1)
        dv_t = np.einsum("ij,ij->i", -v_rel, t)  # (v_j - v_i) . t
        f_t = state.sfm_kappa * overlap * dv_t
        return (f_n[:, None] * n + f_t[:, None] * t).sum(axis=0)

    # ------------------------------------------------------------ walls
    @staticmethod
    def _ttc_wall_force(state, vel, p1, p2):
        """Reference code's anticipatory wall force; agent sits at the origin."""
        seg = p2 - p1
        seg_len_sq = seg @ seg
        if seg_len_sq < 1e-12:
            closest = p1
        else:
            s = np.clip(-(p1 @ seg) / seg_len_sq, 0.0, 1.0)
            closest = p1 + s * seg
        d_w = closest @ closest
        r = state.radius
        if vel @ closest < 0 or d_w == r * r or d_w > state.neighbor_dist**2:
            return np.zeros(2)
        radius = np.sqrt(d_w) if d_w < r * r else r
        a = vel @ vel
        if a < 1e-5:
            return np.zeros(2)

        t_min = np.inf
        disc_hit = None
        # Time to collision with the two end caps (discs) of the capsule.
        for end in (p1, p2):
            b = end @ vel
            c = end @ end - radius * radius
            discr = b * b - a * c
            if discr > 0.0:
                sq = np.sqrt(discr)
                t = (b - sq) / a
                if 0.0 < t < t_min:
                    t_min, disc_hit = t, (end, b, sq)
        # Time to collision with the two long sides of the capsule.
        seg_hit = None
        if seg_len_sq >= 1e-12:
            normal = np.array([-seg[1], seg[0]]) / np.sqrt(seg_len_sq)
            for sign in (1.0, -1.0):
                o1 = p1 + sign * radius * normal
                det_vo = vel[0] * seg[1] - vel[1] * seg[0]
                if det_vo != 0.0:
                    rel = -o1  # agent position (origin) minus o1
                    t = (seg[0] * rel[1] - seg[1] * rel[0]) / det_vo
                    s = (vel[0] * rel[1] - vel[1] * rel[0]) / det_vo
                    if t > 0.0 and 0.0 <= s <= 1.0 and t < t_min:
                        t_min, seg_hit, disc_hit = t, det_vo, None
        if not np.isfinite(t_min):
            return np.zeros(2)
        m = state.exponent
        scale = state.k * np.exp(-t_min / state.tau0) / t_min**m * (m / t_min + 1.0 / state.tau0)
        if disc_hit is not None:
            end, b, sq = disc_hit
            return -scale * (vel - (b * vel - a * end) / sq) / a
        return scale / seg_hit * np.array([-seg[1], seg[0]])

    @staticmethod
    def _distance_wall_force(state, vel, wall):
        d = wall.distance
        n = np.array(wall.normal)
        overlap = max(state.radius - d, 0.0)
        f_n = state.sfm_A * np.exp((state.radius - d) / state.sfm_B) + state.sfm_k * overlap
        t = np.array([-n[1], n[0]])
        f_t = -state.sfm_kappa * overlap * (vel @ t)
        return f_n * n + f_t * t

    def _wall_guard(self, state, walls, movement):
        """JuPedSim-specific extension (see class docstring): never step
        into a wall. Only the wall-normal component of the step is cut, and
        only for walls closer than radius + step length."""
        for wall in walls:
            n = np.array(wall.normal)
            if not n.any():
                continue
            toward = -(movement @ n)
            if toward <= 0.0:
                continue
            allowed = max(wall.distance - self.WALL_MARGIN, 0.0)
            if toward > allowed:
                movement = movement + (toward - allowed) * n
        return movement

    # ------------------------------------------------------------ step
    def compute_next_state(self, state: PowerLawState, step):
        vel = np.array(state.velocity, dtype=float)
        if state.self_propelled:
            speed = np.hypot(*vel)
            heading = vel / speed if speed > 1e-9 else np.zeros(2)
            v_pref = state.pref_speed * heading
        else:
            v_pref = state.pref_speed * np.array(step.orientation_to_next_target)

        force = (v_pref - vel) / state.xi

        rng = state.neighbor_dist if state.interaction == "ttc" else state.sfm_range
        neighbors = step.other_agents_in_range(rng)
        if neighbors:
            w = np.array([n.relative_position for n in neighbors], dtype=float)
            v_other = np.array([n.state.velocity for n in neighbors], dtype=float)
            radii = state.radius + np.array([n.state.radius for n in neighbors])
            v_rel = vel[None, :] - v_other
            if state.interaction == "ttc":
                force = force + self._ttc_agent_force(state, w, v_rel, radii)
            else:
                force = force + self._distance_agent_force(state, w, v_rel, radii)

        walls = step.walls_in_range(rng)
        for wall in walls:
            if state.interaction == "ttc":
                p1, p2 = (np.array(p, dtype=float) for p in wall.segment)
                force = force + self._ttc_wall_force(state, vel, p1, p2)
            else:
                force = force + self._distance_wall_force(state, vel, wall)

        if state.interaction == "ttc":
            # Reference code clamps the total acceleration (unit mass).
            mag = np.hypot(*force)
            if mag > state.max_accel:
                force = force * (state.max_accel / mag)

        new_vel = vel + force * step.dt
        movement = new_vel * step.dt
        near = [wl for wl in walls if wl.distance < state.radius + np.hypot(*movement)]
        if near:
            guarded = self._wall_guard(state, near, movement)
            if not np.array_equal(guarded, movement):
                movement = guarded
                new_vel = movement / step.dt
        return (
            replace(state, velocity=(float(new_vel[0]), float(new_vel[1]))),
            (float(movement[0]), float(movement[1])),
        )
