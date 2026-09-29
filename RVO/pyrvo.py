# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import dataclass, replace

import numpy as np
from jupedsim.models.custom_model import CustomOperationalModel


@dataclass(kw_only=True, frozen=True)
class RVOState:
    velocity: tuple[float, float]
    radius: float = 0.25
    max_speed: float = 1.5
    reciprocal: bool = True
    is_reactive: bool = True
    neighbor_dist: float = 6.0
    wall_dist: float = 3.0
    n_speed_samples: int = 6
    n_angle_samples: int = 24
    collision_weight: float = 2.0


class ReciprocalVelocityObstacleModel(CustomOperationalModel):
    """
    Implementation of Reciprocal Velocity Obstacles as described in:
    "Reciprocal Velocity Obstacles for Real-Time Multi-Agent Navigation"
    by van den Berg, Lin & Manocha (ICRA 2008).

    Each agent picks, every step, the velocity closest to its preferred
    velocity that avoids every neighbor's Velocity Obstacle (VO): the cone
    of velocities that would eventually put the two agents' disks (radius
    r_i + r_j) into contact, given the neighbor's current velocity.

    Plain VO (apex = neighbor's velocity v_j) makes both sides react to
    each other's *previous* reaction, which oscillates. RVO's fix is to
    require each side to only take half the avoidance responsibility: an
    agent's new velocity v must satisfy `2v - v_i - v_j` outside the raw
    VO cone, i.e. `v` outside a cone with the SAME shape, apex-shifted to
    the *average* `(v_i + v_j) / 2`. Both sides shifting by the same
    average is what removes the oscillation (paper's Theorem 8), while
    each side individually staying outside its half of the cone is what
    keeps the pair collision-free (Theorem 6) as long as both use the
    rule -- against a non-cooperating obstacle (a wall, or another
    agent's `is_reactive=False`, e.g. a car) there is no "other side" to
    share the responsibility with, so the apex must stay at the
    obstacle's own velocity (0 for a wall) instead of the average --
    using the reciprocal average there would silently assume the
    obstacle avoids too, and it doesn't.

    A candidate velocity is chosen by sampling a polar grid of velocities
    (plus the preferred velocity, current velocity, and zero) and picking
    the sample that minimizes distance to the preferred velocity, plus a
    penalty for how soon it would enter a VO (shorter time-to-entry means
    a larger penalty). This sampling-based selection is the same approach
    described in the paper (as opposed to e.g. ORCA's later exact
    half-plane / linear-program formulation).

    A blind polar grid alone deadlocks in near-exact head-on encounters:
    once two agents are almost touching, forward motion is blocked across
    almost the entire cone (its half-angle approaches 90 degrees as the
    gap closes), while every unblocked grid sample lies so far off the
    preferred direction that standing still scores better -- and standing
    still is a fixed point, since two stopped agents keep recomputing the
    exact same blocked cone forever. To keep the *fewest-collisions*
    candidate from ever coinciding with a permanent freeze, every active
    VO's two tangent directions (the exact edge of its blocked cone) are
    also added to the candidate set each step, at a couple of speeds --
    these are the closest a velocity can get to "straight at the goal"
    while still guaranteed to clear that particular obstacle, so a viable
    way around is available even when the blind grid's angular resolution
    would otherwise miss it.
    """

    def __init__(self):
        CustomOperationalModel.__init__(self)

    @staticmethod
    def _candidate_velocities(
        v_pref: np.ndarray,
        v_current: np.ndarray,
        max_speed: float,
        n_speed_samples: int,
        n_angle_samples: int,
    ) -> np.ndarray:
        """Build the (n_candidates, 2) array of velocities to score."""
        speeds = np.linspace(0.0, max_speed, n_speed_samples)
        angles = np.linspace(0.0, 2 * np.pi, n_angle_samples, endpoint=False)
        s, a = np.meshgrid(speeds, angles, indexing="ij")
        grid = np.stack([s * np.cos(a), s * np.sin(a)], axis=-1).reshape(-1, 2)
        extra = np.stack([v_pref, v_current, (0.0, 0.0)])
        return np.concatenate([grid, extra], axis=0)

    @staticmethod
    def _time_to_vo_entry(
        candidates: np.ndarray,
        apex: np.ndarray,
        p_rel: np.ndarray,
        combined_radius: float,
    ) -> np.ndarray:
        """Smallest t>0 at which each candidate enters the (apex-shifted) VO.

        `u = v - apex` is treated as a closing velocity toward the disk of
        radius `combined_radius` centered at `p_rel`: we solve
        `|t*u - p_rel| = combined_radius` for the smallest positive root.
        Returns +inf where the candidate never enters (or the disk is
        already overlapped, handled by the caller).
        """
        u = candidates - apex  # (n, 2)
        a = np.einsum("ij,ij->i", u, u)
        b = -2.0 * (u @ p_rel)
        c = p_rel @ p_rel - combined_radius**2

        t = np.full(candidates.shape[0], np.inf)
        if c < 0.0:
            # Already overlapping (spawn jitter / numerical edge case, not
            # part of the textbook algorithm): no genuine "entry time"
            # exists since we're already inside. Handled by the caller via
            # a direct separation bonus instead of a time penalty.
            return t

        valid = a > 1e-12
        disc = b**2 - 4.0 * a * c
        valid &= disc >= 0.0
        sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
        denom = np.where(valid, 2.0 * a, 1.0)
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        lo = np.minimum(t1, t2)
        hi = np.maximum(t1, t2)
        root = np.where(lo > 1e-9, lo, np.where(hi > 1e-9, hi, np.inf))
        t = np.where(valid, root, np.inf)
        return t

    @staticmethod
    def _tangent_candidates(
        apex: np.ndarray, p_rel: np.ndarray, combined_radius: float, max_speed: float
    ) -> np.ndarray:
        """Velocities right at the edge of the VO cone from `apex`.

        The cone's two tangent lines to the disk of radius `combined_radius`
        centered at `p_rel` sit at angle +-asin(combined_radius/dist) from
        the `p_rel` axis. A handful of speeds along each tangent give
        candidates that pass as close as possible to the obstacle while
        remaining (right at the margin of) collision-free.
        """
        dist = np.linalg.norm(p_rel)
        ratio = np.clip(combined_radius / dist, -1.0, 1.0)
        half_angle = np.arcsin(ratio)
        axis_angle = np.arctan2(p_rel[1], p_rel[0])
        speeds = np.array([0.5, 0.75, 1.0]) * max_speed
        pts = []
        for sign in (1.0, -1.0):
            angle = axis_angle + sign * half_angle
            direction = np.array([np.cos(angle), np.sin(angle)])
            pts.append(apex[None, :] + speeds[:, None] * direction[None, :])
        return np.concatenate(pts, axis=0)

    def compute_next_state(self, state: RVOState, step):
        v_pref = np.array(state.max_speed) * np.array(step.orientation_to_next_target)

        if not state.is_reactive:
            # A non-cooperating obstacle (e.g. the paper's car): drives
            # straight at its own preferred velocity, oblivious to everyone.
            new_velocity = (float(v_pref[0]), float(v_pref[1]))
            movement = (new_velocity[0] * step.dt, new_velocity[1] * step.dt)
            return replace(state, velocity=new_velocity), movement

        v_current = np.array(state.velocity)
        candidates = [
            self._candidate_velocities(
                v_pref,
                v_current,
                state.max_speed,
                state.n_speed_samples,
                state.n_angle_samples,
            )
        ]

        # Pass 1: collect every active constraint (and its tangent escape
        # candidates) before scoring anything.
        constraints = []  # list of (apex, p_rel, combined_radius, overlapping)
        for neighbor in step.other_agents_in_range(state.neighbor_dist):
            p_rel = np.array(neighbor.relative_position)
            neighbor_is_reactive = getattr(neighbor.state, "is_reactive", True)
            combined_radius = state.radius + neighbor.state.radius
            if state.reciprocal and neighbor_is_reactive:
                apex = (v_current + np.array(neighbor.state.velocity)) / 2.0
            else:
                apex = np.array(neighbor.state.velocity)
            dist = np.linalg.norm(p_rel)
            overlapping = dist < combined_radius
            constraints.append((apex, p_rel, combined_radius, overlapping))
            if not overlapping:
                candidates.append(
                    self._tangent_candidates(apex, p_rel, combined_radius, state.max_speed)
                )

        for wall in step.walls_in_range(state.wall_dist):
            p_rel = np.array(wall.closest_point)
            apex = np.zeros(2)
            combined_radius = state.radius
            dist = np.linalg.norm(p_rel)
            overlapping = dist < combined_radius
            constraints.append((apex, p_rel, combined_radius, overlapping))
            if not overlapping:
                candidates.append(
                    self._tangent_candidates(apex, p_rel, combined_radius, state.max_speed)
                )

        candidates = np.concatenate(candidates, axis=0)
        speed = np.linalg.norm(candidates, axis=1)
        over_limit = speed > state.max_speed
        candidates[over_limit] *= (state.max_speed / speed[over_limit])[:, None]

        # Pass 2: score every candidate against every constraint.
        t_min = np.full(candidates.shape[0], np.inf)
        overlap_bonus = np.zeros(candidates.shape[0])
        for apex, p_rel, combined_radius, overlapping in constraints:
            dist = np.linalg.norm(p_rel)
            if overlapping:
                # Spawn jitter / near-miss: no VO cone is well-defined once
                # already overlapping, so directly reward moving away
                # instead of computing an entry time.
                overlap_bonus += candidates @ (-p_rel / max(dist, 1e-6))
                continue
            t = self._time_to_vo_entry(candidates, apex, p_rel, combined_radius)
            t_min = np.minimum(t_min, t)

        pref_cost = np.linalg.norm(candidates - v_pref, axis=1)
        any_overlap = any(overlapping for _, _, _, overlapping in constraints)
        safe = np.isinf(t_min)

        if any_overlap or not safe.any():
            # Either already touching someone (no candidate is safe against
            # that neighbor this step regardless) or every sampled candidate
            # enters some cone: fall back to the soft trade-off between
            # progress and collision urgency.
            collision_cost = np.where(
                safe, 0.0, state.collision_weight / np.maximum(t_min, 0.05)
            )
            total_cost = pref_cost + collision_cost - overlap_bonus
            best = np.argmin(total_cost)
        else:
            # A hard constraint is enforceable this step: among candidates
            # that are provably collision-free against every active VO,
            # take the one closest to the preferred velocity. Soft-only
            # scoring (the branch above) would sometimes still pick a
            # slightly-unsafe-but-more-direct candidate over a fully-safe
            # one whenever its penalty happened to undercut the safe
            # candidate's distance-to-preferred cost -- fine with one
            # neighbor, but with several simultaneous neighbors near a
            # crowded crossing this let real (if shallow) penetrations
            # through. Restricting the choice to the safe set whenever one
            # exists removes that failure mode.
            best = np.flatnonzero(safe)[np.argmin(pref_cost[safe])]
        v_best = candidates[best]
        new_velocity = (float(v_best[0]), float(v_best[1]))
        movement = (new_velocity[0] * step.dt, new_velocity[1] * step.dt)
        return replace(state, velocity=new_velocity), movement
