# SPDX-License-Identifier: LGPL-3.0-or-later
"""Injury-tracking extension of the given PythonSocialForceModel.

Reuses the exact force formulas from ../pysocial_force.py (unmodified) and
adds the bookkeeping needed to reproduce Helbing, Farkas & Vicsek (2000)'s
injury rule for Figure 1:

    "people are injured and become non-moving obstacles for others, if the
    sum of the magnitudes of the radial forces acting on them divided by
    their circumference exceeds a pressure of 1,600 N/m (ref. 5)."

This is computed online (every physics step, at dt=1e-4 -- i.e. at the same
resolution as the continuous-time model in the paper) rather than from
downsampled trajectory output, since the sqlite writer only records every
Nth frame. Once injured, an agent freezes in place (zero movement every
subsequent step) and keeps exerting its normal social/obstacle repulsion on
others, i.e. it becomes a physical obstacle exactly as described in the
paper -- it is not removed from the simulation.

IMPORTANT fix, found after the first full sweep: jupedsim's built-in exit
stage always routes every agent toward the exit polygon's fixed *centroid*
(`Exit::Target()` in libsimulator/src/Stage.cpp ignores the agent argument
entirely), regardless of the agent's own position. For a 1 m-wide door, this
means all 50 agents funnel toward the exact same pixel instead of spreading
naturally across the doorway -- a severe artificial pinch that, combined
with the freeze-on-injury rule, blocked the exit almost immediately even at
v0=0.6 m/s (the paper's "relaxed" baseline, where evacuation should be
smooth). The custom-model API does not expose an agent's absolute position
directly, so we track it ourselves in the state (updated each step by the
same `position + movement` integration jupedsim itself uses internally,
seeded from the agent's spawn position) and steer toward the *nearest point
on the real door segment* instead of `step.orientation_to_next_target`.
"""

from dataclasses import dataclass, replace

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pysocial_force import PythonSocialForceModel, PythonSocialForceModelState

INJURY_PRESSURE_THRESHOLD = 1600.0  # N/m, Smith & Dickie (1993) via Helbing et al. (2000)


@dataclass(kw_only=True, frozen=True)
class InjuryTrackingState(PythonSocialForceModelState):
    injured: bool = False
    peak_pressure: float = 0.0  # N/m, diagnostic: highest pressure ever experienced
    position: tuple[float, float] = (0.0, 0.0)  # self-tracked, seeded from spawn position


class InjuryTrackingSFM(PythonSocialForceModel):
    """PythonSocialForceModel + injury/freeze bookkeeping (Fig. 1 of the paper).

    `immune_distance` is *not* part of the paper's model. It is an
    exploratory variant, added to test a specific hypothesis about why this
    exactly-1m-wide door blocks so much more readily than the paper implies
    (see sfm_injury.py's module docstring / the notebook's Fig. 1 section):
    since the door has zero slack, a single frozen agent anywhere in that
    1 m gap blocks it completely. If we exempt agents from freezing while
    they are within `immune_distance` of the door (default 0.0 = paper's
    literal rule, no exemption), does that prevent the permanent-blockage
    cascade? Pressure is still tracked (`peak_pressure`) inside the immune
    zone for diagnostics -- only the actual freeze is suppressed there.
    """

    def __init__(
        self,
        door_p0: tuple[float, float],
        door_p1: tuple[float, float],
        immune_distance: float = 0.0,
    ):
        super().__init__()
        self._door_p0 = np.array(door_p0, dtype=float)
        self._door_p1 = np.array(door_p1, dtype=float)
        self._door_vec = self._door_p1 - self._door_p0
        self._door_len2 = float(self._door_vec @ self._door_vec)
        self._immune_distance = immune_distance

    def _nearest_door_point(self, pos: tuple[float, float]) -> np.ndarray:
        p = np.array(pos, dtype=float)
        t = float((p - self._door_p0) @ self._door_vec) / self._door_len2
        t = min(1.0, max(0.0, t))
        return self._door_p0 + t * self._door_vec

    def compute_next_state(self, state: InjuryTrackingState, step):
        if state.injured:
            # Injured pedestrians stop and act as stationary obstacles.
            # Must still return a *new* state object (never the same instance),
            # even though nothing about it changes.
            return replace(state), (0.0, 0.0)

        target_point = self._nearest_door_point(state.position)
        to_target = target_point - np.array(state.position)
        dist_to_target = np.linalg.norm(to_target)
        target_dir = tuple(to_target / dist_to_target) if dist_to_target > 1e-6 else (0.0, 0.0)

        acc_x, acc_y = self._desired_force(
            velocity=state.velocity,
            target_direction=target_dir,
            desired_speed=state.desired_speed,
            reaction_time=state.reaction_time,
        )

        # Sum of radial (normal) contact-force magnitudes on this agent this
        # step, for the pressure check -- paper eq.: pressure = sum|f_radial| / circumference.
        radial_force_sum = 0.0

        for neighbor in step.other_agents_in_range(2.0):
            fx, fy = self._social_force(state, neighbor)
            acc_x += fx / state.mass
            acc_y += fy / state.mass

            dx = -neighbor.relative_position[0]
            dy = -neighbor.relative_position[1]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = state.radius + neighbor.state.radius
            if dist < min_dist:
                # Body-compression contribution only (the paper's "radial
                # force" from physical contact), matching k*(r_ij - d_ij).
                radial_force_sum += state.body_force * (min_dist - dist)

        for wall in step.walls_in_range(5.0):
            fx, fy = self._obstacle_force(state, wall)
            acc_x += fx / state.mass
            acc_y += fy / state.mass

            if wall.distance < state.radius:
                radial_force_sum += state.body_force * (state.radius - wall.distance)

        circumference = 2.0 * np.pi * state.radius
        pressure = radial_force_sum / circumference
        new_peak = max(state.peak_pressure, pressure)
        would_injure = pressure > INJURY_PRESSURE_THRESHOLD
        in_immune_zone = dist_to_target < self._immune_distance
        newly_injured = would_injure and not in_immune_zone

        new_velocity = (
            state.velocity[0] + acc_x * step.dt,
            state.velocity[1] + acc_y * step.dt,
        )
        movement = (new_velocity[0] * step.dt, new_velocity[1] * step.dt)
        new_position = (state.position[0] + movement[0], state.position[1] + movement[1])

        new_state = replace(
            state,
            velocity=(0.0, 0.0) if newly_injured else new_velocity,
            position=state.position if newly_injured else new_position,
            injured=newly_injured,
            peak_pressure=new_peak,
        )
        return new_state, ((0.0, 0.0) if newly_injured else movement)
