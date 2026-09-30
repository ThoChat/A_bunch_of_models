# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers for reading JuPedSim sqlite trajectories and computing the
metrics used to compare against van Toll, Chatagnon, Braga, Solenthaler &
Pettré (2021), "SPH crowds"."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_trajectory(db_path: str | Path) -> tuple[pd.DataFrame, float]:
    """Load a JuPedSim sqlite trajectory into a tidy DataFrame.

    Returns (df, fps): df has columns [frame, id, pos_x, pos_y, t], fps is
    frames per second of the (downsampled) written trajectory.
    """
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT frame, id, pos_x, pos_y FROM trajectory_data ORDER BY frame, id", con)
    fps_row = con.execute("SELECT value FROM metadata WHERE key = 'fps'").fetchone()
    fps = float(fps_row[0]) if fps_row else 25.0
    con.close()
    df["t"] = df["frame"] / fps
    return df, fps


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Central-difference velocity per agent from consecutive recorded frames."""
    df = df.sort_values(["id", "frame"]).copy()
    g = df.groupby("id")
    dt = g["t"].shift(-1) - g["t"].shift(1)
    df["vx"] = (g["pos_x"].shift(-1) - g["pos_x"].shift(1)) / dt
    df["vy"] = (g["pos_y"].shift(-1) - g["pos_y"].shift(1)) / dt
    return df


# ---------------------------------------------------------------- paper colours
# Fig. 3 (right): SPH density colour ramp, 0 (blue) ... 8 P/m^2 (purple).
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "paper_density",
    [(0 / 8, "#1010ff"), (1 / 8, "#20a0ff"), (2 / 8, "#20e0c0"), (3 / 8, "#40d040"),
     (4 / 8, "#e0f020"), (5 / 8, "#ff9a10"), (6 / 8, "#ff1010"), (7 / 8, "#ff10c0"),
     (8 / 8, "#8020a0")],
)


def velocity_colors(vx, vy, v_full=0.75):
    """Fig. 3 (left): hue = walking direction (right = red, down = green,
    left = cyan, up = blue), saturation = speed (full at 0.75 m/s), grey at rest."""
    vx = np.asarray(vx, float)
    vy = np.asarray(vy, float)
    ang = np.degrees(np.arctan2(vy, vx))
    hue = ((-ang) % 360.0) / 360.0
    s = np.clip(np.hypot(vx, vy) / v_full, 0.0, 1.0)
    rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1))
    grey = np.array([0.62, 0.62, 0.62])
    return (1 - s)[..., None] * grey + s[..., None] * rgb


# ---------------------------------------------------------------- scenario metrics
def flow_rate(exit_times, n_end=350):
    """Paper Table 1: evacuees per second from the first to the n_end-th."""
    t = np.sort(np.asarray(exit_times))
    if t.size < n_end:
        return float("nan")
    return (n_end - 1) / (t[n_end - 1] - t[0])


def wave_profile(snap: np.ndarray, y_band=(-10.0, 10.0), bins=np.arange(25.0, 70.0, 1.0)):
    """Mean forward speed v_x of agents in a horizontal band, per 1 m x-bin.

    snap rows are (x, y, vx, vy, rho, uid). Returns (bin centres, mean vx).
    """
    x, y, vx = snap[:, 0], snap[:, 1], snap[:, 2]
    sel = (y >= y_band[0]) & (y <= y_band[1])
    idx = np.digitize(x[sel], bins) - 1
    centres = 0.5 * (bins[1:] + bins[:-1])
    mean = np.full(centres.size, np.nan)
    for k in range(centres.size):
        m = idx == k
        if m.sum() >= 3:
            mean[k] = vx[sel][m].mean()
    return centres, mean


def wave_front(centres, mean_vx, threshold=0.2, direction=+1):
    """Wave position = x of the largest |v_x| moving in `direction` (+1 toward
    the stage, -1 back), if it exceeds `threshold` m/s; else nan."""
    v = direction * mean_vx
    if not np.isfinite(v).any() or np.nanmax(v) < threshold:
        return float("nan")
    return float(centres[int(np.nanargmax(v))])


def boundary_jitter(snap_a: np.ndarray, snap_b: np.ndarray, dt: float, rho_max: float, frac=0.6):
    """Mean speed of the crowd's boundary agents (SPH density below
    `frac * rho_max` in snap_a), estimated from the displacement between two
    snapshots dt apart -- the "splashing" of Fig. 8."""
    ia = {int(u): k for k, u in enumerate(snap_a[:, 5])}
    ib = {int(u): k for k, u in enumerate(snap_b[:, 5])}
    common = [u for u in ia if u in ib]
    a = snap_a[[ia[u] for u in common]]
    b = snap_b[[ib[u] for u in common]]
    boundary = a[:, 4] < frac * rho_max
    disp = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]) / dt
    return float(disp[boundary].mean()), float(disp[~boundary].mean()), int(boundary.sum())
