# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers for reading JuPedSim sqlite trajectories and computing the
metrics used to compare against Helbing, Farkas & Vicsek (2000)."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def load_trajectory(db_path: str | Path) -> tuple[pd.DataFrame, float]:
    """Load a JuPedSim sqlite trajectory into a tidy DataFrame.

    Returns:
        (df, fps) where df has columns [frame, id, pos_x, pos_y, t] and
        fps is frames-per-second of the *written* (possibly downsampled)
        trajectory (see metadata table).
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


def leaving_times(df: pd.DataFrame, still_inside_ids: set[int] = frozenset()) -> pd.Series:
    """Time at which each agent id genuinely left (reached its exit stage).

    JuPedSim removes an agent from the trajectory once it reaches its exit
    stage, so the last recorded frame per id is normally the moment of
    departure. The exception is a run that was cut off at a step cap while
    agents were still inside: their "last seen" frame is just the cutoff,
    not a real departure. Pass their ids as `still_inside_ids` (from the
    runner script's own `sim.agents()` at cutoff) to exclude them.
    """
    lt = df.groupby("id")["t"].max()
    if still_inside_ids:
        lt = lt.drop(index=[i for i in still_inside_ids if i in lt.index])
    return lt


def time_for_n_to_leave(
    df: pd.DataFrame, n: int, still_inside_ids: set[int] = frozenset()
) -> float | None:
    """Simulated time at which the n-th agent has left, or None if fewer
    than n agents left within the recorded trajectory.
    """
    lt = sorted(leaving_times(df, still_inside_ids).values)
    if len(lt) < n:
        return None
    return lt[n - 1]


def outflow_rate(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    exclude_last: int = 0,
    still_inside_ids: set[int] = frozenset(),
) -> float:
    """Average number of agents leaving per second in [t_start, t_end].

    Mirrors the paper's flow measurement, which starts once the first agent
    leaves and (for Fig. 1) excludes the final `exclude_last` agents whose
    departure is dominated by end-of-queue edge effects.
    """
    lt = np.sort(leaving_times(df, still_inside_ids).values)
    if exclude_last:
        lt = lt[: len(lt) - exclude_last] if len(lt) > exclude_last else lt[:0]
    mask = (lt >= t_start) & (lt <= t_end)
    n = mask.sum()
    duration = t_end - t_start
    return n / duration if duration > 0 else float("nan")


def density_in_window(
    df: pd.DataFrame,
    t: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    fps: float,
    tol: float = 0.5,
) -> tuple[float, int]:
    """Number of agents / area (P/m^2) inside a rectangle at time t."""
    frame = round(t * fps)
    snap = df[(df["frame"] >= frame - tol * fps) & (df["frame"] <= frame + tol * fps)]
    if snap.empty:
        return float("nan"), 0
    # use the frame closest to the target
    nearest_frame = snap.iloc[(snap["frame"] - frame).abs().argsort()[:1]]["frame"].values[0]
    snap = df[df["frame"] == nearest_frame]
    inside = snap[
        (snap["pos_x"] >= xmin)
        & (snap["pos_x"] <= xmax)
        & (snap["pos_y"] >= ymin)
        & (snap["pos_y"] <= ymax)
    ]
    area = (xmax - xmin) * (ymax - ymin)
    return len(inside) / area, len(inside)


def mean_progress_efficiency(
    df: pd.DataFrame,
    fps: float,
    axis: str,
    desired_speed: float,
    t_start: float,
    t_end: float,
) -> float:
    """E = <v . e0> / v0 averaged over agents and the [t_start, t_end] window.

    `axis` is "x" or "y": the corridor's long axis, standing in for the
    paper's e0_i (preferred direction). Velocity is estimated by finite
    difference of consecutive recorded frames per agent.
    """
    col = "pos_x" if axis == "x" else "pos_y"
    sub = df[(df["t"] >= t_start) & (df["t"] <= t_end)].sort_values(["id", "frame"])
    if sub.empty:
        return float("nan")
    g = sub.groupby("id")
    dpos = g[col].diff()
    dt = g["t"].diff()
    v_along = (dpos / dt).dropna()
    v_along = v_along[np.isfinite(v_along)]
    if v_along.empty:
        return float("nan")
    return float(v_along.mean() / desired_speed)
