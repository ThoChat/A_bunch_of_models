# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers for reading JuPedSim sqlite trajectories and computing the
metrics used to compare against Karamouzas, Skinner & Guy (2014):
time to collision, the pair distribution functions g(tau) and g(r), the
inferred interaction energy E(tau) = ln(1/g(tau)), and Edie's speed/density
in 0.5 m cells (Supplemental Material, Fig. S4A)."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Paper's analysis radius: "each pedestrian modeled as a disc ... we used a
# radius of 0.1 m" (Supplemental Material, Experimental Datasets).
ANALYSIS_RADIUS = 0.1


# ------------------------------------------------------------------ loading
def load_trajectory(db_path: str | Path) -> tuple[pd.DataFrame, float]:
    """Load a JuPedSim sqlite trajectory into a DataFrame with columns
    [frame, id, pos_x, pos_y, t, vx, vy, speed]. Velocities are central
    differences of the recorded positions (one-sided at an agent's first
    and last frame), as the paper does with its datasets."""
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT frame, id, pos_x, pos_y FROM trajectory_data ORDER BY id, frame", con
    )
    fps_row = con.execute("SELECT value FROM metadata WHERE key = 'fps'").fetchone()
    con.close()
    fps = float(fps_row[0]) if fps_row else 10.0
    df["t"] = df["frame"] / fps
    vx = np.empty(len(df))
    vy = np.empty(len(df))
    for _, idx in df.groupby("id").indices.items():
        t = df["t"].to_numpy()[idx]
        if len(idx) < 2:
            vx[idx] = vy[idx] = 0.0
            continue
        vx[idx] = np.gradient(df["pos_x"].to_numpy()[idx], t)
        vy[idx] = np.gradient(df["pos_y"].to_numpy()[idx], t)
    df["vx"] = vx
    df["vy"] = vy
    df["speed"] = np.hypot(vx, vy)
    return df.sort_values(["frame", "id"]).reset_index(drop=True), fps


# ------------------------------------------------------------------ pairs
def time_to_collision(dx, dy, dvx, dvy, combined_radius):
    """tau for relative position (x_j - x_i) and relative velocity
    (v_i - v_j); NaN where no future collision exists (Supplemental
    Material: tau = (b - sqrt(d)) / a). Pairs already overlapping get 0."""
    a = dvx * dvx + dvy * dvy
    b = dx * dvx + dy * dvy
    c = dx * dx + dy * dy - combined_radius**2
    d = b * b - a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = (b - np.sqrt(d)) / a
    tau = np.where((d > 0) & (a > 1e-9) & (tau > 0), tau, np.nan)
    return np.where(c < 0, 0.0, tau)


def pair_samples(df: pd.DataFrame, frames=None, frame_col="frame", max_r=8.0,
                 radius=ANALYSIS_RADIUS, exclude_same=None) -> pd.DataFrame:
    """All unordered pairs present in the same frame, with their distance r,
    approach rate v = -dr/dt and time to collision tau.

    `frame_col` lets the same code run on the time-scrambled data set.
    `exclude_same` (a column name) drops pairs sharing that value -- used
    to exclude pairing an agent with itself after scrambling."""
    out = []
    sub = df if frames is None else df[df[frame_col].isin(frames)]
    cols = ["pos_x", "pos_y", "vx", "vy"]
    for _, g in sub.groupby(frame_col):
        if len(g) < 2:
            continue
        arr = g[cols].to_numpy()
        i, j = np.triu_indices(len(g), k=1)
        if exclude_same is not None:
            same = g[exclude_same].to_numpy()
            keep = same[i] != same[j]
            i, j = i[keep], j[keep]
        dx = arr[j, 0] - arr[i, 0]
        dy = arr[j, 1] - arr[i, 1]
        r = np.hypot(dx, dy)
        near = r < max_r
        i, j, dx, dy, r = i[near], j[near], dx[near], dy[near], r[near]
        dvx = arr[i, 2] - arr[j, 2]
        dvy = arr[i, 3] - arr[j, 3]
        # dr/dt = (x_j - x_i).(v_j - v_i) / r ; approach rate v = -dr/dt
        approach = (dx * dvx + dy * dvy) / np.maximum(r, 1e-9)
        tau = time_to_collision(dx, dy, dvx, dvy, 2 * radius)
        out.append(np.stack([r, approach, tau], axis=1))
    if not out:
        return pd.DataFrame(columns=["r", "v", "tau"])
    return pd.DataFrame(np.concatenate(out), columns=["r", "v", "tau"])


def scramble_time(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Supplemental Material: "randomly permuting time information between
    different pedestrians, so that each moment in a pedestrian's trajectory
    has its instantaneous position and velocity preserved, but is assigned
    a randomly permuted time". The number of agents per time is kept."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["frame_scrambled"] = rng.permutation(out["frame"].to_numpy())
    return out


def observed_and_scrambled(df, frame_stride=1, seed=0, radius=ANALYSIS_RADIUS, n_scrambles=1):
    """Pair samples of the real (simulated) data and of the time-scrambled
    reference, over the same frames. `frame_stride` subsamples frames."""
    frames = np.sort(df["frame"].unique())[::frame_stride]
    sub = df[df["frame"].isin(frames)]
    obs = pair_samples(sub, radius=radius)
    scr = pd.concat(
        [pair_samples(scramble_time(sub, seed + s), frame_col="frame_scrambled",
                      radius=radius, exclude_same="id") for s in range(n_scrambles)],
        ignore_index=True,
    )
    return obs, scr


def pair_distribution(obs_values, ref_values, edges):
    """g(x) = P(x) / P_NI(x), both normalized as densities over `edges`."""
    p, _ = np.histogram(obs_values, bins=edges, density=True)
    q, _ = np.histogram(ref_values, bins=edges, density=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(q > 0, p / q, np.nan)
    return g, 0.5 * (edges[1:] + edges[:-1])


def interaction_energy(obs, scr, fine=0.01, coarse=0.05, tau_max=8.0, norm_range=(3.0, 8.0)):
    """E(tau) = ln(1/g(tau)) (paper Eq. 1) on `fine` bins, then mean and
    standard deviation in `coarse` windows, as in Fig. 4 / Fig. S6
    ("average energy values every 0.05 s ... +-1 standard deviation").

    Pairs with undefined tau are excluded; the paper analyses tau < 8 s.
    Normalization: the Supplemental Material defines g so that g -> 1 for
    non-interacting separations. With `norm_range` = (lo, hi), g is scaled
    to average 1 over that tail (default: beyond tau0 = 3 s). With
    `norm_range=None`, both histograms are normalized as densities over
    [0, tau_max] instead; the suppression at small tau then pushes g above
    1 elsewhere and E gets a negative offset."""
    edges = np.arange(0.0, tau_max + fine / 2, fine)
    t_obs = obs["tau"].dropna()
    t_scr = scr["tau"].dropna()
    counts, _ = np.histogram(t_obs, bins=edges)
    if norm_range is None:
        g, centers = pair_distribution(t_obs, t_scr, edges)
    else:
        ref, _ = np.histogram(t_scr, bins=edges)
        centers = 0.5 * (edges[1:] + edges[:-1])
        tail = (centers >= norm_range[0]) & (centers < norm_range[1])
        scale = ref[tail].sum() / max(counts[tail].sum(), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(ref > 0, counts * scale / ref, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_fine = np.log(1.0 / g)
    e_fine[~np.isfinite(e_fine)] = np.nan
    fine_df = pd.DataFrame({"tau": centers, "E": e_fine, "g": g, "count": counts})
    fine_df["window"] = np.floor(fine_df["tau"] / coarse) * coarse + coarse / 2
    coarse_df = fine_df.groupby("window").agg(
        E_mean=("E", "mean"), E_std=("E", "std"), count=("count", "sum")
    ).reset_index().rename(columns={"window": "tau"})
    return fine_df, coarse_df


def fit_power_law(tau, energy, tau_lo, tau_hi):
    """Two fits over [tau_lo, tau_hi]:
    * the paper's law with fixed exponent 2, E = c / tau^2 (least squares
      for c; R^2 on the linear scale, as in Fig. 2);
    * a free exponent, from a linear fit of log E vs. log tau on the
      points with E > 0 (the paper: 2.05 Outdoor, 2.02 Bottleneck)."""
    tau = np.asarray(tau)
    energy = np.asarray(energy)
    m = (tau >= tau_lo) & (tau <= tau_hi) & np.isfinite(energy)
    t, e = tau[m], energy[m]
    basis = t**-2.0
    c = float(basis @ e / (basis @ basis))
    resid = e - c * basis
    r2 = 1.0 - resid @ resid / ((e - e.mean()) @ (e - e.mean()))
    pos = e > 0
    slope, intercept = np.polyfit(np.log(t[pos]), np.log(e[pos]), 1)
    return {"c": c, "r2_fixed_exponent": float(r2), "free_exponent": float(-slope),
            "n_points": int(m.sum())}


# ------------------------------------------------------------------ Edie
def edie_cells(df, cell=0.5, interval=4.0, region=None):
    """Supplemental Material, Fig. S4A: "Each simulation area was divided
    into two-dimensional cells measuring 0.5 m x 0.5 m and for each cell we
    determined its average density and speed over 4 s intervals", with
    Edie's generalized definitions: density = total time spent in the cell /
    (cell area x interval), speed = total distance / total time.

    With a fixed sampling step, total time = n_samples * dt and total
    distance = sum(speed) * dt, so speed = mean sample speed. `region` =
    (xmin, xmax, ymin, ymax) limits the analysis to part of the scene."""
    d = df
    if region is not None:
        x0, x1, y0, y1 = region
        d = d[(d.pos_x >= x0) & (d.pos_x < x1) & (d.pos_y >= y0) & (d.pos_y < y1)]
    dt = np.median(np.diff(np.sort(df["t"].unique())))
    key = pd.DataFrame({
        "cx": np.floor(d["pos_x"] / cell).astype(int),
        "cy": np.floor(d["pos_y"] / cell).astype(int),
        "ti": np.floor(d["t"] / interval).astype(int),
        "speed": d["speed"],
    })
    agg = key.groupby(["cx", "cy", "ti"]).agg(n=("speed", "size"), speed=("speed", "mean")).reset_index()
    agg["density"] = agg["n"] * dt / (cell * cell * interval)
    return agg


def binned_fundamental_diagram(cells, bin_width=0.1):
    """Average speed of all cells in each density bin of `bin_width`
    ("clustering the data into bins of 0.1 ppl/m^2")."""
    b = np.floor(cells["density"] / bin_width) * bin_width + bin_width / 2
    return cells.assign(rho=b).groupby("rho").agg(
        speed=("speed", "mean"), n_cells=("speed", "size")
    ).reset_index()


def weidmann_speed(rho, v0=1.34, gamma=1.913, rho_max=5.4):
    """Weidmann (1993) speed-density relation, the paper's reference [10]."""
    rho = np.asarray(rho, dtype=float)
    with np.errstate(divide="ignore"):
        v = v0 * (1.0 - np.exp(-gamma * (1.0 / rho - 1.0 / rho_max)))
    return np.clip(v, 0.0, None)


# ------------------------------------------------------------------ misc
def min_clearance(df, radius):
    """Minimum over frames of (centre distance - 2 radius); negative = overlap."""
    best = np.inf
    for _, g in df.groupby("frame"):
        if len(g) < 2:
            continue
        p = g[["pos_x", "pos_y"]].to_numpy()
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        best = min(best, d.min())
    return best - 2 * radius


# ------------------------------------------------------------------ crossing
def crossing_agent_metrics(df, summary):
    """Per agent of the Crossing scenario: how much of its delay comes from
    slowing down vs. from deviating (paper: pedestrians "prefer to slow down
    and let others pass rather than deviate from their planned courses").

    delay       = time spent before reaching the exit line - free walking time
    detour_time = (path length - straight distance) / preferred speed
    max_lateral = largest sideways offset from the agent's start line.

    The line is 0.5 m before the exit zone (which starts 3 m before the
    wall): agents are removed as soon as they enter the zone, so their last
    recorded position (10 frames/s) can fall just short of it."""
    exit_line = summary["arena_half"] - 3.5
    group_a = set(summary["group_a"])
    rows = []
    for aid, g in df.groupby("id"):
        g = g.sort_values("t")
        along, across = ("pos_x", "pos_y") if aid in group_a else ("pos_y", "pos_x")
        s_along = g[along].to_numpy()
        reached = np.flatnonzero(s_along >= exit_line)
        end = reached[0] if len(reached) else len(g) - 1
        g = g.iloc[: end + 1]
        v0 = summary["pref_speed"][str(aid)]
        straight = exit_line - s_along[0]
        path = np.hypot(np.diff(g["pos_x"]), np.diff(g["pos_y"])).sum()
        t_travel = g["t"].iloc[-1] - g["t"].iloc[0]
        rows.append({
            "id": aid, "group": "A (+x)" if aid in group_a else "B (+y)",
            "delay": t_travel - straight / v0,
            "detour_time": (path - straight) / v0,
            "max_lateral": np.abs(g[across] - g[across].iloc[0]).max(),
            "min_speed_ratio": g["speed"].iloc[1:-1].min() / v0 if len(g) > 2 else np.nan,
            "reached": bool(len(reached)),
        })
    return pd.DataFrame(rows).set_index("id")


def stripe_orientation(df, summary, zone=6.0, r_max=1.2, min_each=8):
    """Orientation of same-group neighbour links inside the mixing zone
    (|x|, |y| < zone) while both groups are there. Stripes of one group
    make those links line up; the nematic average of exp(2i phi) gives the
    dominant axis (degrees, in [-90, 90)) and its strength S in [0, 1].
    The initial square blocks give links along 0 and 90 deg in equal
    numbers, which cancel in exp(2i phi). Returns per-frame values too."""
    group_a = set(summary["group_a"])
    total = 0j
    n_total = 0
    per_frame = []
    inzone = df[(df.pos_x.abs() < zone) & (df.pos_y.abs() < zone)]
    for frame, g in inzone.groupby("frame"):
        is_a = g["id"].isin(group_a).to_numpy()
        if is_a.sum() < min_each or (~is_a).sum() < min_each:
            continue
        p = g[["pos_x", "pos_y"]].to_numpy()
        i, j = np.triu_indices(len(g), k=1)
        d = p[j] - p[i]
        r = np.hypot(d[:, 0], d[:, 1])
        same = (is_a[i] == is_a[j]) & (r < r_max)
        phi = np.arctan2(d[same, 1], d[same, 0])
        z = np.exp(2j * phi).sum()
        total += z
        n_total += same.sum()
        if same.sum():
            per_frame.append({"t": g["t"].iloc[0], "angle": np.degrees(np.angle(z) / 2),
                              "S": abs(z) / same.sum(), "n_links": int(same.sum())})
    angle = float(np.degrees(np.angle(total) / 2)) if n_total else np.nan
    strength = float(abs(total) / n_total) if n_total else np.nan
    return angle, strength, pd.DataFrame(per_frame)


# ------------------------------------------------------------------ passages
def passage_times(df, x_line, y_range=None):
    """Time at which each agent first crosses the vertical line x = x_line
    (door / bottleneck exit), sorted. `y_range` restricts to crossings
    within that band."""
    times = []
    for _, g in df.groupby("id"):
        g = g.sort_values("t")
        x = g["pos_x"].to_numpy()
        idx = np.flatnonzero((x[:-1] < x_line) & (x[1:] >= x_line))
        if len(idx) == 0:
            continue
        k = idx[0]
        if y_range is not None and not (y_range[0] <= g["pos_y"].iloc[k] <= y_range[1]):
            continue
        frac = (x_line - x[k]) / (x[k + 1] - x[k])
        times.append(g["t"].iloc[k] + frac * (g["t"].iloc[k + 1] - g["t"].iloc[k]))
    return np.sort(np.array(times))


def steady_flow(times, drop=0.1):
    """Flow [1/s] from passage times, dropping the first and last `drop`
    fraction of passages (start-up and emptying)."""
    n = len(times)
    lo, hi = int(n * drop), int(n * (1 - drop)) - 1
    if hi <= lo:
        return np.nan
    return (hi - lo) / (times[hi] - times[lo])


def jam_front(snapshot, door, angles_deg=np.arange(-75, 76, 15), half_bin=7.5,
              neighbor_r=0.8, min_neighbors=3):
    """Arch shape in front of a door at `door` = (x, y), room at x < door_x.
    For each angular bin (0 deg = straight into the room, away from the
    door), the jam front is the largest distance to the door among agents
    that are part of the dense jam (>= `min_neighbors` others within
    `neighbor_r`). A semicircular arch has a front that does not depend on
    angle. Returns a DataFrame [angle, front]."""
    p = snapshot[["pos_x", "pos_y"]].to_numpy()
    inside = p[:, 0] < door[0]
    p = p[inside]
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    dense = (d < neighbor_r).sum(axis=1) - 1 >= min_neighbors
    rel = p[dense] - np.array(door)
    r = np.hypot(rel[:, 0], rel[:, 1])
    ang = np.degrees(np.arctan2(rel[:, 1], -rel[:, 0]))
    rows = []
    for a in angles_deg:
        sel = np.abs(ang - a) <= half_bin
        rows.append({"angle": a, "front": r[sel].max() if sel.any() else np.nan,
                     "n": int(sel.sum())})
    return pd.DataFrame(rows)


def lateral_layers(df, x_range, width, bin_width=0.04, smooth=2, prominence=0.15):
    """Lateral position profile inside a bottleneck (x in `x_range`) and its
    peaks ("layers"). Returns (centers, density, peak positions)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    sel = df[(df.pos_x >= x_range[0]) & (df.pos_x <= x_range[1])]
    edges = np.arange(-width / 2, width / 2 + bin_width / 2, bin_width)
    h, _ = np.histogram(sel["pos_y"], bins=edges, density=True)
    hs = gaussian_filter1d(h, smooth)
    centers = 0.5 * (edges[1:] + edges[:-1])
    peaks, _ = find_peaks(hs, prominence=prominence * hs.max())
    return centers, hs, centers[peaks]


# ------------------------------------------------------------------ collective
def collective_order(df, times, center=(0.0, 0.0), k=1.5, tau0=3.0, radius=0.25, r_max=10.0):
    """For each time in `times`: polarization |<v_hat>|, milling (vortex)
    order |<r_hat x v_hat>| about `center`, and the mean interaction energy
    per agent, sum_j k/tau^2 exp(-tau/tau0) over neighbours within `r_max`
    (the paper's Eq. 2 with the model's own radii and parameters)."""
    rows = []
    frames = df["frame"].unique()
    fps = 1.0 / np.median(np.diff(np.sort(df["t"].unique())))
    for t in times:
        fr = frames[np.argmin(np.abs(frames / fps - t))]
        g = df[df["frame"] == fr]
        p = g[["pos_x", "pos_y"]].to_numpy() - np.array(center)
        v = g[["vx", "vy"]].to_numpy()
        s = np.linalg.norm(v, axis=1)
        ok = s > 1e-6
        vh = v[ok] / s[ok, None]
        rh = p[ok] / np.maximum(np.linalg.norm(p[ok], axis=1), 1e-9)[:, None]
        pol = np.linalg.norm(vh.mean(axis=0))
        mill = abs(np.mean(rh[:, 0] * vh[:, 1] - rh[:, 1] * vh[:, 0]))
        i, j = np.triu_indices(len(g), k=1)
        dx = p[j, 0] - p[i, 0]
        dy = p[j, 1] - p[i, 1]
        near = np.hypot(dx, dy) < r_max
        tau = time_to_collision(dx[near], dy[near], v[i[near], 0] - v[j[near], 0],
                                v[i[near], 1] - v[j[near], 1], 2 * radius)
        tau = tau[np.isfinite(tau) & (tau > 0)]
        energy = (k / tau**2 * np.exp(-tau / tau0)).sum() * 2 / len(g)
        rows.append({"t": fr / fps, "polarization": pol, "milling": mill,
                     "energy_per_agent": energy, "n": len(g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ lanes
def lane_order(df, east_ids, times, band=0.4, reach=3.0, x_range=None, seed=0):
    """Lane order parameter (Rex & Loewen 2007 style) for a bidirectional
    corridor. For each agent, the neighbours ahead or behind within a
    lateral band |dy| < `band` and |dx| < `reach` are counted as same- or
    opposite-direction; phi_i = ((n_same - n_opp) / (n_same + n_opp))^2
    and phi = <phi_i>. phi = 1 means perfect lanes. The reference value
    `phi_mixed` recomputes phi after shuffling the direction labels among
    the agents of that frame (same positions, no lane structure)."""
    east = set(east_ids)
    rng = np.random.default_rng(seed)
    frames = df["frame"].unique()
    fps = 1.0 / np.median(np.diff(np.sort(df["t"].unique())))

    def phi_of(p, lab):
        dx = p[None, :, 0] - p[:, None, 0]
        dy = p[None, :, 1] - p[:, None, 1]
        nb = (np.abs(dy) < band) & (np.abs(dx) < reach)
        np.fill_diagonal(nb, False)
        same = (nb & (lab[None, :] == lab[:, None])).sum(axis=1)
        tot = nb.sum(axis=1)
        ok = tot > 0
        return np.mean(((2 * same[ok] - tot[ok]) / tot[ok]) ** 2) if ok.any() else np.nan

    rows = []
    for t in times:
        g = df[df["frame"] == frames[np.argmin(np.abs(frames / fps - t))]]
        if x_range is not None:
            g = g[(g.pos_x >= x_range[0]) & (g.pos_x <= x_range[1])]
        lab = g["id"].isin(east).to_numpy()
        if lab.sum() < 5 or (~lab).sum() < 5:
            continue
        p = g[["pos_x", "pos_y"]].to_numpy()
        rows.append({"t": t, "phi": phi_of(p, lab), "phi_mixed": phi_of(p, rng.permutation(lab)),
                     "n": len(g)})
    return pd.DataFrame(rows)
