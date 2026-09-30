# SPDX-License-Identifier: LGPL-3.0-or-later
"""Digitize the numeric plots of Guy et al. (2010): Figs. 7, 9 and 10.

The PDF pages were rendered with `pdftoppm -r 300` and each figure cropped
into results/paper_fig*.png. For every curve we mask its line colour, take
the median pixel row per pixel column, and convert pixels to data units
using the gridline rows (detected as long grey rows) and the tick-label
columns (read by eye from the crop). The printed values are pasted into the
notebook's reference cell; run this script to regenerate them.

Accuracy: gridlines are located to +-1 px, the lines are ~3 px thick, so
values are good to about +-0.01 m/s (Fig. 7, 9), +-0.03 m^2 in area/person
(Fig. 7), +-0.15 m in x (Fig. 9) and +-15 J/kg (Fig. 10).
"""

from pathlib import Path

import numpy as np
from PIL import Image

RES = Path(__file__).resolve().parent / "results"


def curve(path, mask_fn, col_range, row_range):
    """Track one coloured line column by column inside the plot area: in
    each column, split the masked pixels into runs and keep the run closest
    to the previous column's row (so text, legend swatches and crossing
    curves of a similar colour are not picked up)."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    mask = mask_fn(im)
    mask[: row_range[0]] = False
    mask[row_range[1]:] = False
    cols, rows, prev = [], [], None
    for c in range(*col_range):
        r = np.flatnonzero(mask[:, c])
        if not len(r):
            continue
        runs = np.split(r, np.flatnonzero(np.diff(r) > 2) + 1)
        centres = np.array([run.mean() for run in runs])
        if prev is None:
            pick = centres[np.argmax([len(run) for run in runs])]
        else:
            pick = centres[np.argmin(np.abs(centres - prev))]
            if abs(pick - prev) > 25:  # a jump: not the same line
                continue
        cols.append(c)
        rows.append(pick)
        prev = pick
    return np.array(cols, float), np.array(rows, float)


BLUE = lambda im: (im[..., 2] > 150) & (im[..., 0] < 120) & (im[..., 1] < 160)  # noqa: E731
RED = lambda im: (im[..., 0] > 150) & (im[..., 1] < 110) & (im[..., 2] < 110)  # noqa: E731
GREEN = lambda im: (  # noqa: E731
    (im[..., 1] > 150) & (im[..., 0] < 190) & (im[..., 2] < 110) & (im[..., 0] > 90)
)


def sample(cols, rows, xs_px):
    """Row at each requested pixel column (NaN outside the curve's extent)."""
    return np.where(
        (xs_px >= cols.min()) & (xs_px <= cols.max()), np.interp(xs_px, cols, rows), np.nan
    )


def fig7():
    # x: y-axis at col 110 = 0 m^2; tick "5" centred at col 608 -> 99.6 px/unit.
    # y: gridline rows 333 (0) ... 60 (1.4) -> 195 px per m/s.
    x_of = lambda c: (c - 110) / 99.6  # noqa: E731
    y_of = lambda r: (333 - r) / 195.0  # noqa: E731
    out = {}
    for name, fn in [("ple", BLUE), ("nelson", RED), ("fruin", GREEN)]:
        cols, rows = curve(RES / "paper_fig7_speed_density.png", fn, (111, 607), (55, 331))
        # Area/person grid: dense where the curves bend.
        area = np.array([0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4,
                         1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        v = y_of(sample(cols, rows, 110 + 99.6 * area))
        ok = np.isfinite(v)
        out[name] = {"area_per_person": area[ok].round(2).tolist(), "speed": v[ok].round(3).tolist()}
    return out


def fig9():
    # x: col 95 = 0 m, tick "25" at col 674 -> 23.16 px/m.
    # y: gridline rows 261 (0.8) ... 61 (1.4) -> 333.3 px per m/s.
    cols, rows = curve(RES / "paper_fig9_edge_effect.png", BLUE, (96, 672), (55, 259))
    xs = np.arange(1.0, 25.5, 1.0)
    v = 0.8 + (261 - sample(cols, rows, 95 + 23.16 * xs)) / 333.3
    ok = np.isfinite(v)
    return {"x": xs[ok].tolist(), "speed": v[ok].round(3).tolist()}


def fig10():
    # x: tick "700" at col 131.5, "1300" at col 571 -> 0.7325 px/agent.
    # y: gridline rows 318 (0) ... 79.5 (2000) -> 0.11925 px per J/kg.
    out = {}
    n = np.arange(700, 1401, 50)
    for name, fn in [("clearpath", RED), ("rvo", GREEN), ("ple", BLUE)]:
        cols, rows = curve(RES / "paper_fig10_tradeshow.png", fn, (132, 655), (75, 316))
        rows = np.where(rows < 81, np.nan, rows)  # ClearPath leaves the axis at 2000
        e = (318 - sample(cols[np.isfinite(rows)], rows[np.isfinite(rows)],
                          131.5 + 0.7325 * (n - 700))) / 0.11925
        ok = np.isfinite(e)
        out[name] = {"n_agents": n[ok].tolist(), "energy": e[ok].round(0).tolist()}
    return out


if __name__ == "__main__":
    import json

    print("PAPER_FIG7 =", json.dumps(fig7()))
    print("PAPER_FIG9 =", json.dumps(fig9()))
    print("PAPER_FIG10 =", json.dumps(fig10()))
