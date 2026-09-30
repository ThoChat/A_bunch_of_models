# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers for reading JuPedSim sqlite trajectories and computing the
metrics used to compare against van den Berg, Guy, Lin & Manocha (2011)."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import shapely


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
    fps_row = con.execute("SELECT value FROM metadata WHERE key = 'fps'").fetchone()
    fps = float(fps_row[0]) if fps_row else 25.0
    con.close()
    df["t"] = df["frame"] / fps
    return df, fps


def load_geometry(db_path: str | Path) -> shapely.Geometry:
    """Walkable area stored by the trajectory writer (WKT)."""
    con = sqlite3.connect(str(db_path))
    wkt = con.execute("SELECT wkt FROM geometry").fetchone()[0]
    con.close()
    return shapely.from_wkt(wkt)


def clearance_per_frame(df: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Minimum pairwise clearance (centre distance - 2r) in every frame.

    Uses a sort-and-sweep on x so it stays affordable at N=1000: only pairs
    closer than 2r + 1 m in x are compared. Negative clearance = overlap.
    Returns a DataFrame [frame, t, min_clearance, n_overlapping_pairs].
    """
    cutoff = 2 * radius + 1.0
    rows = []
    for frame, g in df.groupby("frame", sort=True):
        pos = g[["pos_x", "pos_y"]].to_numpy()
        if len(pos) < 2:
            continue
        order = np.argsort(pos[:, 0])
        pos = pos[order]
        best = np.inf
        n_over = 0
        for shift in range(1, len(pos)):
            dx = pos[shift:, 0] - pos[:-shift, 0]
            if dx.min() > cutoff:
                break
            d = np.hypot(dx, pos[shift:, 1] - pos[:-shift, 1])
            best = min(best, d.min())
            n_over += int((d < 2 * radius - 1e-9).sum())
        rows.append(
            {
                "frame": frame,
                "t": g["t"].iloc[0],
                "min_clearance": best - 2 * radius,
                "n_overlapping_pairs": n_over,
            }
        )
    return pd.DataFrame(rows)


def wall_clearance(df: pd.DataFrame, walkable: shapely.Geometry, radius: float) -> float:
    """Smallest (distance to the walkable area's boundary - r) over all
    recorded positions. Negative means an agent's disk crossed a wall."""
    pts = shapely.points(df["pos_x"].to_numpy(), df["pos_y"].to_numpy())
    d = shapely.distance(walkable.boundary, pts)
    return float(d.min() - radius)


def reached_target(
    df: pd.DataFrame, targets: dict[int, tuple[float, float]], tolerance: float
) -> pd.Series:
    """Whether each agent's final recorded position is within `tolerance` of
    its OWN target (every agent in these scenarios has a distinct target)."""
    last = df.sort_values("frame").groupby("id").tail(1).set_index("id")
    results = {}
    for agent_id, (tx, ty) in targets.items():
        if agent_id not in last.index:
            results[agent_id] = False
            continue
        row = last.loc[agent_id]
        results[agent_id] = bool(np.hypot(row["pos_x"] - tx, row["pos_y"] - ty) <= tolerance)
    return pd.Series(results, name="reached")


def arrival_times(
    df: pd.DataFrame, targets: dict[int, tuple[float, float]], tolerance: float
) -> pd.Series:
    """First time each agent comes within `tolerance` of its own target
    (NaN if never)."""
    out = {}
    for agent_id, g in df.groupby("id"):
        if agent_id not in targets:
            continue
        tx, ty = targets[agent_id]
        d = np.hypot(g["pos_x"] - tx, g["pos_y"] - ty)
        hit = g.loc[d <= tolerance, "t"]
        out[agent_id] = hit.min() if not hit.empty else np.nan
    return pd.Series(out, name="arrival_t")


def trim_to_active_phase(
    df: pd.DataFrame,
    targets: dict[int, tuple[float, float]],
    tolerance: float,
    buffer: float = 0.5,
) -> pd.DataFrame:
    """Drop each agent's frames after it reaches its target (+ buffer):
    agents are not removed on arrival (plain waypoint stages)."""
    arr = arrival_times(df, targets, tolerance)
    kept = []
    for agent_id, g in df.groupby("id"):
        cutoff = arr.get(agent_id, np.nan)
        kept.append(g if np.isnan(cutoff) else g[g["t"] <= cutoff + buffer])
    return pd.concat(kept, ignore_index=True)


def path_metrics(df: pd.DataFrame, resample_dt: float = 0.2) -> pd.DataFrame:
    """Per-agent smoothness metrics.

    Positions are resampled onto a `resample_dt` grid first. Columns:
      path_length, straight (start->end), tortuosity (= ratio),
      total_turning_deg: sum of |heading change| between resampled steps
          (steps slower than 0.05 m/s are dropped: their heading is noise),
      max_turn_deg: the single largest heading change between two
          consecutive resampled steps, a direct check of "smooth",
      reversals: heading changes > 90 deg in one resampled step.
    """
    rows = []
    for agent_id, g in df.sort_values("frame").groupby("id"):
        g = g.drop_duplicates("t")
        t = g["t"].to_numpy()
        pos = g[["pos_x", "pos_y"]].to_numpy()
        if len(pos) < 3:
            continue
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        path_length = seg.sum()
        straight = np.linalg.norm(pos[-1] - pos[0])
        tg = np.arange(t[0], t[-1], resample_dt)
        if len(tg) < 3:
            continue
        rp = np.stack([np.interp(tg, t, pos[:, 0]), np.interp(tg, t, pos[:, 1])], 1)
        vel = np.diff(rp, axis=0) / resample_dt
        vel = vel[np.linalg.norm(vel, axis=1) >= 0.05]
        if len(vel) < 2:
            turn = np.array([0.0])
        else:
            h = np.arctan2(vel[:, 1], vel[:, 0])
            turn = np.abs(np.degrees((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
        rows.append(
            {
                "id": agent_id,
                "path_length": path_length,
                "straight": straight,
                "tortuosity": path_length / straight if straight > 1e-6 else np.nan,
                "total_turning_deg": turn.sum(),
                "max_turn_deg": turn.max(),
                "reversals": int((turn > 90).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("id")


def speeds(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame speed of every agent from finite differences."""
    out = []
    for agent_id, g in df.sort_values("frame").groupby("id"):
        t = g["t"].to_numpy()
        pos = g[["pos_x", "pos_y"]].to_numpy()
        if len(t) < 2:
            continue
        v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
        out.append(pd.DataFrame({"id": agent_id, "t": t[1:], "speed": v}))
    return pd.concat(out, ignore_index=True)
