# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers for reading JuPedSim sqlite trajectories and computing the
metrics used to compare against van den Berg, Lin & Manocha (2008)."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def load_trajectory(db_path: str | Path) -> tuple[pd.DataFrame, float]:
    """Load a JuPedSim sqlite trajectory into a tidy DataFrame.

    Returns:
        (df, fps) where df has columns [frame, id, pos_x, pos_y, t] and fps
        is frames-per-second of the (possibly downsampled) written trajectory.
    """
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT frame, id, pos_x, pos_y FROM trajectory_data ORDER BY frame, id",
        con,
    )
    fps_row = con.execute(
        "SELECT value FROM metadata WHERE key = 'fps'"
    ).fetchone()
    fps = float(fps_row[0]) if fps_row else 25.0
    con.close()
    df["t"] = df["frame"] / fps
    return df, fps


def min_pairwise_distance(
    df: pd.DataFrame, radii: dict[int, float] | float
) -> tuple[float, float]:
    """Minimum center-to-center distance between any two agents, and the
    minimum *clearance* (distance minus the sum of their radii -- negative
    means an actual overlap/collision).

    `radii` is either a uniform radius for every agent, or a dict mapping
    agent id to its own radius (needed once a car or other differently-sized
    agent is present).
    """
    min_dist = float("inf")
    min_clearance = float("inf")
    uniform = not isinstance(radii, dict)
    for _, frame_df in df.groupby("frame"):
        pos = frame_df[["pos_x", "pos_y"]].to_numpy()
        ids = frame_df["id"].to_numpy()
        if len(pos) < 2:
            continue
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        dist = d[i, j]
        if uniform:
            combined = 2 * radii
        else:
            combined = radii.get(int(ids[i]), 0.25) + radii.get(int(ids[j]), 0.25)
        clearance = dist - combined
        if dist < min_dist:
            min_dist = dist
        if clearance < min_clearance:
            min_clearance = clearance
    return min_dist, min_clearance


def path_tortuosity(df: pd.DataFrame) -> pd.Series:
    """Per-agent path length divided by straight-line start-to-end distance.

    1.0 means a perfectly direct path; larger values mean more wandering.
    Not itself an oscillation measure (a smooth curved detour also scores
    above 1), but a cheap, standard proxy used alongside `turning_metrics`.
    """
    results = {}
    for agent_id, g in df.sort_values("frame").groupby("id"):
        pos = g[["pos_x", "pos_y"]].to_numpy()
        if len(pos) < 2:
            continue
        seg_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        path_length = seg_lengths.sum()
        straight = np.linalg.norm(pos[-1] - pos[0])
        results[agent_id] = path_length / straight if straight > 1e-6 else float("nan")
    return pd.Series(results, name="tortuosity")


def trim_to_active_phase(
    df: pd.DataFrame,
    targets: dict[int, tuple[float, float]],
    tolerance: float,
    buffer: float = 1.0,
) -> pd.DataFrame:
    """Drop each agent's frames after it settles at its target (+ buffer).

    Waypoint stages (used throughout this repo's RVO scenarios) don't
    remove an agent on arrival, so a full trajectory recording is mostly a
    long idle tail once everyone has arrived. Left in, that tail dominates
    metrics like `turning_metrics` with meaningless dithering around a
    fixed point rather than the interesting crossing behaviour. `buffer`
    extra seconds are kept after arrival to capture genuine braking, not
    indefinite idle jitter.
    """
    kept = []
    for agent_id, g in df.groupby("id"):
        g = g.sort_values("frame")
        target = targets.get(agent_id)
        if target is None:
            kept.append(g)
            continue
        d = np.hypot(g["pos_x"] - target[0], g["pos_y"] - target[1])
        arrived = g.loc[d <= tolerance, "t"]
        cutoff = arrived.min() + buffer if not arrived.empty else g["t"].max()
        kept.append(g[g["t"] <= cutoff])
    return pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0]


def turning_metrics(
    df: pd.DataFrame, speed_floor: float = 0.05, resample_dt: float | None = 0.5
) -> pd.DataFrame:
    """Per-agent oscillation metrics based on heading (direction of travel).

    For each agent, estimates heading over time and measures how much it
    swings back and forth -- exactly what RVO's reciprocal averaging is
    meant to suppress (paper's Theorem 8), and what plain VO is prone to
    (each agent reacting to the other's last move, "dancing" back and
    forth).

    `resample_dt`, if given, first resamples each agent's path onto a
    coarser fixed time grid (linear interpolation) before computing
    headings from the resampled steps. This matters because our per-step
    velocity selection re-optimizes a discrete candidate grid completely
    independently every control tick: even smooth macroscopic motion shows
    small (a few degrees), high-frequency heading "chatter" between
    adjacent simulation steps as the argmin hops between near-tied discrete
    candidates -- real but not what the paper's Fig. 2 "dancing" refers to,
    which is a macroscopic, visible back-and-forth over a meaningful
    distance. Resampling at a coarser dt averages out that per-tick
    chatter and isolates genuine path-direction reversals. Pass
    `resample_dt=None` to use the raw per-recorded-frame headings instead.

    Frames (or resampled steps) where the agent is nearly stationary
    (speed < speed_floor) are dropped before computing headings, since a
    near-zero velocity vector's direction is numerical noise, not a real
    heading.

    Returns a DataFrame indexed by agent id with columns:
        total_turning_deg: sum of |heading change| between consecutive
            (non-stationary) steps, in degrees -- higher means more
            back-and-forth steering.
        reversals: count of heading changes whose magnitude exceeds 90
            degrees in a single step -- a simple, discrete "did it
            basically reverse direction" count.
        n_headings: number of heading samples the above is computed over,
            so a fair mean comparison can normalize by it.
    """
    rows = []
    for agent_id, g in df.sort_values("frame").groupby("id"):
        g = g.drop_duplicates("t")
        pos = g[["pos_x", "pos_y"]].to_numpy()
        t = g["t"].to_numpy()
        if len(pos) < 3:
            continue

        if resample_dt is not None:
            t_grid = np.arange(t[0], t[-1], resample_dt)
            if len(t_grid) < 3:
                continue
            pos = np.stack(
                [np.interp(t_grid, t, pos[:, 0]), np.interp(t_grid, t, pos[:, 1])],
                axis=1,
            )
            t = t_grid

        frame_dt = np.diff(t)
        frame_dt[frame_dt <= 0] = np.nan
        vel = np.diff(pos, axis=0) / frame_dt[:, None]  # m/s, not per-step displacement
        speed = np.linalg.norm(vel, axis=1)
        moving = speed >= speed_floor
        vel = vel[moving]
        if len(vel) < 2:
            continue
        headings = np.arctan2(vel[:, 1], vel[:, 0])
        dh = np.diff(headings)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        total_turning_deg = np.abs(np.degrees(dh)).sum()
        reversals = int((np.abs(np.degrees(dh)) > 90).sum())
        rows.append(
            {
                "id": agent_id,
                "total_turning_deg": total_turning_deg,
                "reversals": reversals,
                "n_headings": len(headings),
            }
        )
    return pd.DataFrame(rows).set_index("id")


def reached_target(
    df: pd.DataFrame, targets: dict[int, tuple[float, float]], tolerance: float
) -> pd.Series:
    """Whether each agent's final recorded position is within `tolerance` of
    its own target -- used where agents are not removed on arrival (plain
    waypoint stages), so "did it get there" has to be read off the
    trajectory rather than from a removal event."""
    last = df.sort_values("frame").groupby("id").tail(1).set_index("id")
    results = {}
    for agent_id, (tx, ty) in targets.items():
        if agent_id not in last.index:
            results[agent_id] = False
            continue
        row = last.loc[agent_id]
        d = np.hypot(row["pos_x"] - tx, row["pos_y"] - ty)
        results[agent_id] = bool(d <= tolerance)
    return pd.Series(results, name="reached")
