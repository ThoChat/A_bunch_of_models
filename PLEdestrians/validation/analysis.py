# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers for reading JuPedSim sqlite trajectories and computing the
metrics used to compare against Guy et al. (2010), PLEdestrians."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ES = 2.23  # J/(kg s)  paper, footnote to Eq. 1
EW = 1.26  # J s/(kg m^2)


def load_trajectory(db_path: str | Path) -> tuple[pd.DataFrame, float]:
    """Load a JuPedSim sqlite trajectory into a DataFrame with columns
    [frame, id, pos_x, pos_y, t]; also returns the recorded frames/second."""
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT frame, id, pos_x, pos_y FROM trajectory_data ORDER BY frame, id", con
    )
    fps_row = con.execute("SELECT value FROM metadata WHERE key = 'fps'").fetchone()
    fps = float(fps_row[0]) if fps_row else 25.0
    con.close()
    df["t"] = df["frame"] / fps
    return df, fps


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-agent velocity from forward differences of recorded positions
    (the velocity held over [t, t + frame interval])."""
    df = df.sort_values(["id", "frame"]).copy()
    g = df.groupby("id")
    dtf = g["t"].shift(-1) - df["t"]
    df["vx"] = (g["pos_x"].shift(-1) - df["pos_x"]) / dtf
    df["vy"] = (g["pos_y"].shift(-1) - df["pos_y"]) / dtf
    df["dt_frame"] = dtf
    df["speed"] = np.hypot(df["vx"], df["vy"])
    return df


def arrival_times(df: pd.DataFrame, goals: dict[int, tuple[float, float]], tol: float) -> pd.Series:
    """First time each agent is within `tol` of its OWN goal (NaN if never).
    Every agent here has its own goal point, so a per-agent tolerance is a
    meaningful arrival test."""
    out = {}
    for aid, g in df.groupby("id"):
        if aid not in goals:
            continue
        gx, gy = goals[aid]
        d = np.hypot(g["pos_x"].to_numpy() - gx, g["pos_y"].to_numpy() - gy)
        hit = np.flatnonzero(d <= tol)
        out[aid] = g["t"].to_numpy()[hit[0]] if len(hit) else np.nan
    return pd.Series(out, name="t_arrive")


def effort_per_agent(
    df: pd.DataFrame, goals: dict[int, tuple[float, float]], tol: float,
    v_des: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Eq. 2 per unit mass: E/m = sum (es + ew |v|^2) dt from t=0 until the
    agent first reaches its goal (J/kg), plus path length and arrival time.
    Agents that never arrive are integrated to the end of the run and
    flagged `arrived=False`.

    `v_des` (per agent) gives heterogeneous constants with the same
    sqrt(es ew) as the paper's average human: es = c v, ew = c / v,
    c = sqrt(2.23 * 1.26). Omitted: the paper's es, ew for everyone."""
    c = np.sqrt(ES * EW)
    dv = add_velocity(df)
    t_arr = arrival_times(df, goals, tol)
    rows = []
    for aid, g in dv.groupby("id"):
        if aid not in goals:
            continue
        ta = t_arr.get(aid, np.nan)
        arrived = np.isfinite(ta)
        seg = g[(g["t"] < ta) if arrived else g["dt_frame"].notna()]
        seg = seg[seg["dt_frame"].notna()]
        vd = v_des.get(aid, np.sqrt(ES / EW)) if v_des else np.sqrt(ES / EW)
        es, ew = c * vd, c / vd
        e = float(((es + ew * seg["speed"] ** 2) * seg["dt_frame"]).sum())
        length = float((seg["speed"] * seg["dt_frame"]).sum())
        rows.append({"id": aid, "energy": e, "path_length": length,
                     "t_arrive": ta, "arrived": arrived})
    return pd.DataFrame(rows).set_index("id")


def min_clearance(df: pd.DataFrame, radius: float) -> float:
    """Smallest centre distance minus 2*radius over all frames (negative =
    overlap)."""
    best = np.inf
    for _, f in df.groupby("frame"):
        p = f[["pos_x", "pos_y"]].to_numpy()
        if len(p) < 2:
            continue
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        best = min(best, d.min())
    return best - 2 * radius


def voronoi_density(points: np.ndarray, polygon, ) -> np.ndarray:
    """Per-point density 1/area of its Voronoi cell clipped to `polygon`
    (shapely). Standard pedestrian-dynamics measure (e.g. Steffen & Seyfried
    2010); used for the speed-density comparison."""
    import shapely
    from shapely.ops import voronoi_diagram

    mp = shapely.MultiPoint(points)
    cells = voronoi_diagram(mp, envelope=polygon.buffer(50))
    areas = np.full(len(points), np.nan)
    tree = shapely.STRtree([shapely.Point(p) for p in points])
    for cell in cells.geoms:
        idx = tree.query(cell, predicate="contains")
        if len(idx) == 1:
            areas[idx[0]] = cell.intersection(polygon).area
    return 1.0 / areas


# ------------------------------------------------------------------ corridor
def crowd_body(frame: pd.DataFrame, trim: float = 3.0) -> pd.DataFrame:
    """Agents of one frame away from the crowd's free front and back: the
    Long Corridor crowd walks south (-y) into empty corridor, so the first
    and last `trim` metres of the crowd are a thinning boundary layer."""
    y0, y1 = frame["pos_y"].min(), frame["pos_y"].max()
    return frame[(frame["pos_y"] > y0 + trim) & (frame["pos_y"] < y1 - trim)]


def lateral_speed_profile(dv: pd.DataFrame, width: float, t0: float, t1: float,
                          bin_w: float = 1.0) -> pd.DataFrame:
    """Fig. 9: mean speed (and southward speed) of crowd-body agents in
    lateral bins across the corridor, averaged over frames in [t0, t1]."""
    sel = dv[(dv["t"] >= t0) & (dv["t"] <= t1) & dv["speed"].notna()]
    body = pd.concat([crowd_body(f) for _, f in sel.groupby("frame")])
    edges = np.arange(0.0, width + 1e-9, bin_w)
    body = body.assign(xbin=pd.cut(body["pos_x"], edges))
    g = body.groupby("xbin", observed=False)
    return pd.DataFrame({
        "x": 0.5 * (edges[:-1] + edges[1:]),
        "speed": g["speed"].mean().to_numpy(),
        "speed_south": (-g["vy"].mean()).to_numpy(),
        "n": g.size().to_numpy(),
    })


def speed_density_samples(dv: pd.DataFrame, polygon, t0: float, t1: float,
                          every: int = 4) -> pd.DataFrame:
    """Fig. 7: per-agent (Voronoi density, speed) samples from crowd-body
    agents, every `every`-th recorded frame in [t0, t1]."""
    frames = sorted(dv.loc[(dv["t"] >= t0) & (dv["t"] <= t1), "frame"].unique())[::every]
    out = []
    for fr in frames:
        f = dv[(dv["frame"] == fr) & dv["speed"].notna()]
        rho = voronoi_density(f[["pos_x", "pos_y"]].to_numpy(), polygon)
        f = f.assign(density=rho)
        out.append(crowd_body(f)[["t", "id", "pos_x", "pos_y", "speed", "density"]])
    return pd.concat(out, ignore_index=True)


def overtakes(df: pd.DataFrame, t0: float, t1: float, horizon: float = 5.0,
              lane: float = 0.8, step: float = 2.5) -> pd.DataFrame:
    """Overtaking events in a southward (-y) flow: at time t, agent i is up
    to 3 m behind j (north of it) and within `lane` m laterally; at
    t + horizon it is south of j. Returns one row per event with the
    overtaker's lateral position at t. Sampled every `step` seconds."""
    rows = []
    times = np.arange(t0, t1 - horizon + 1e-9, step)
    by_t = {t: f.set_index("id") for t, f in df.groupby(df["t"].round(3))}
    for t in times:
        a, b = by_t.get(round(t, 3)), by_t.get(round(t + horizon, 3))
        if a is None or b is None:
            continue
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        x, y = a["pos_x"].to_numpy(), a["pos_y"].to_numpy()
        yb = b["pos_y"].to_numpy()
        dx = np.abs(x[:, None] - x[None, :])
        behind = (y[:, None] > y[None, :]) & (y[:, None] - y[None, :] < 3.0)
        passed = yb[:, None] < yb[None, :]
        i, j = np.nonzero((dx < lane) & behind & passed)
        rows += [{"t": t, "overtaker": common[k], "overtaken": common[m], "x": x[k]}
                 for k, m in zip(i, j)]
    return pd.DataFrame(rows)


# ----------------------------------------------------------- narrow passage
def crossing_times(df: pd.DataFrame, x_line: float) -> pd.DataFrame:
    """First time each agent's x passes `x_line`, with its y there."""
    rows = []
    for aid, g in df.sort_values("t").groupby("id"):
        x, y, t = g["pos_x"].to_numpy(), g["pos_y"].to_numpy(), g["t"].to_numpy()
        k = np.flatnonzero((x[:-1] < x_line) & (x[1:] >= x_line))
        if len(k):
            k = k[0]
            w = (x_line - x[k]) / (x[k + 1] - x[k])
            rows.append({"id": aid, "t": t[k] + w * (t[k + 1] - t[k]),
                         "y": y[k] + w * (y[k + 1] - y[k])})
    return pd.DataFrame(rows).set_index("id")


def occupancy_map(df: pd.DataFrame, xlim, ylim, cell: float = 0.25,
                  t0: float = 0.0, t1: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-averaged density (agents/m^2) on a grid, from recorded frames
    in [t0, t1]."""
    sel = df[(df["t"] >= t0) & (df["t"] <= t1)]
    xe = np.arange(xlim[0], xlim[1] + 1e-9, cell)
    ye = np.arange(ylim[0], ylim[1] + 1e-9, cell)
    h, _, _ = np.histogram2d(sel["pos_x"], sel["pos_y"], bins=[xe, ye])
    return h.T / (sel["frame"].nunique() * cell * cell), xe, ye
