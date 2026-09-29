# SPDX-License-Identifier: LGPL-3.0-or-later
"""Herding extension of the given PythonSocialForceModel, for Figure 3
(smoky room / individualistic vs. herding escape) of Helbing, Farkas &
Vicsek (2000).

Reuses the exact _social_force / _obstacle_force / _desired_force formulas
from ../pysocial_force.py unmodified. The only thing this subclass changes
is *what direction an agent is trying to walk in* -- everything about
collisions and contact forces is identical to the base model.

Implements the paper's Eq. 4:

    e0_i(t) = Norm[ (1-p_i) e_i + p_i * <e0_j(t)>_{j in radius R_i} ]

where e_i is the agent's own fixed, randomly-chosen "individual" direction
(its private guess of where an exit might be) and <e0_j(t)> is the average
*current* preferred direction of neighbors within radius R. Agents do not
use JuPedSim's routed direction to the exit at all -- exits in this model
are "invisible": the only way an agent finds one is by physically wandering
(individually or via the herd) into its capture zone, exactly as described
in the paper ("If one of the exits is closer than 2 m, the room is left" --
implemented at the geometry level in run_fig3_smoky_room.py via a ~2 m-radius
exit-stage capture polygon around each real door).

Also implements "When reaching a boundary, the direction of a pedestrian is
reflected": when an agent's current preferred direction points into a wall
it is touching, that direction is mirrored about the wall's normal.
"""

from dataclasses import dataclass, replace

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pysocial_force import PythonSocialForceModel, PythonSocialForceModelState


@dataclass(kw_only=True, frozen=True)
class HerdingState(PythonSocialForceModelState):
    individual_direction: tuple[float, float] = (1.0, 0.0)
    current_direction: tuple[float, float] = (1.0, 0.0)
    panic_parameter: float = 0.0  # p_i in the paper, 0 = pure individualist, 1 = pure herd-follower
    herd_radius: float = 5.0  # R_i in the paper


class HerdingSFM(PythonSocialForceModel):
    """PythonSocialForceModel with Eq. 4 herding instead of routed navigation."""

    @staticmethod
    def _reflect_at_walls(
        direction: tuple[float, float], walls, contact_radius: float
    ) -> tuple[float, float]:
        dx, dy = direction
        for wall in walls:
            if wall.distance > contact_radius:
                continue
            nx, ny = wall.normal
            if nx == 0.0 and ny == 0.0:
                continue
            dot = dx * nx + dy * ny
            if dot < 0.0:  # direction points into the wall -> mirror it
                dx = dx - 2.0 * dot * nx
                dy = dy - 2.0 * dot * ny
        return HerdingSFM._normalize_local((dx, dy))

    @staticmethod
    def _normalize_local(vector: tuple[float, float]) -> tuple[float, float]:
        norm = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if norm < 1e-10:
            return (0.0, 0.0)
        return (vector[0] / norm, vector[1] / norm)

    def compute_next_state(self, state: HerdingState, step):
        # --- Eq. 4: blend individual direction with the herd's average current direction
        neighbors = step.other_agents_in_range(state.herd_radius)
        p = state.panic_parameter
        if neighbors and p > 0.0:
            sum_x = sum(n.state.current_direction[0] for n in neighbors)
            sum_y = sum(n.state.current_direction[1] for n in neighbors)
            avg_dir = self._normalize_local((sum_x, sum_y))
            blended = (
                (1.0 - p) * state.individual_direction[0] + p * avg_dir[0],
                (1.0 - p) * state.individual_direction[1] + p * avg_dir[1],
            )
        else:
            blended = state.individual_direction
        new_direction = self._normalize_local(blended)

        # --- reflect off walls the agent is touching (contact-range query only)
        touching_walls = step.walls_in_range(state.radius + 0.05)
        new_direction = self._reflect_at_walls(new_direction, touching_walls, state.radius + 0.05)
        if new_direction == (0.0, 0.0):
            new_direction = state.current_direction

        # --- forces: identical to the base model, but using new_direction instead
        # of step.orientation_to_next_target (exits are "invisible" to this model)
        acc_x, acc_y = self._desired_force(
            velocity=state.velocity,
            target_direction=new_direction,
            desired_speed=state.desired_speed,
            reaction_time=state.reaction_time,
        )

        for neighbor in step.other_agents_in_range(2.0):
            fx, fy = self._social_force(state, neighbor)
            acc_x += fx / state.mass
            acc_y += fy / state.mass

        for wall in step.walls_in_range(5.0):
            fx, fy = self._obstacle_force(state, wall)
            acc_x += fx / state.mass
            acc_y += fy / state.mass

        new_velocity = (
            state.velocity[0] + acc_x * step.dt,
            state.velocity[1] + acc_y * step.dt,
        )
        movement = (new_velocity[0] * step.dt, new_velocity[1] * step.dt)

        return (
            replace(state, velocity=new_velocity, current_direction=new_direction),
            movement,
        )
