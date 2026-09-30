# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import dataclass, replace

import numpy as np
from jupedsim.models.custom_model import CustomOperationalModel

# Navigation profiles of Section 5.5 of the paper.
PROFILES = ("SF", "RVO", "SPH", "SF+SPH", "RVO+SPH", "SF->SPH", "RVO->SPH")

_COS_FOV = np.cos(np.radians(100.0))  # Sec. 5.3.1: forces from behind 100 deg are halved


@dataclass(kw_only=True, frozen=True)
class SPHCrowdState:
    """Per-agent state. Defaults are the paper's Section 5.5 / 5.6 settings."""

    velocity: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.24  # D_i, disk radius used by contact forces and RVO
    mass: float = 1.0  # m_i = (D_i / 0.24)^2 (Sec. 5.6)
    uid: int = -1  # scenario-assigned id (the callback does not know the jupedsim id)
    profile: str = "SPH"

    # Agent parameters (Sec. 5.6)
    pref_speed: float = 1.4
    max_speed: float = 1.8

    # SPH (Sec. 5.2 / 5.5)
    h: float = 1.0
    k_gas: float = 200.0
    mu: float = 0.0
    rho0_min: float = 0.0
    rho0_max: float = 5.0
    t_rho: float = 0.1

    # Contact forces (Sec. 5.4). "nav" = profiles SF/RVO, "sph" = every profile with SPH.
    k_ag_nav: float = 1000.0
    k_obs_nav: float = 1000.0
    k_ag_sph: float = 50.0
    k_obs_sph: float = 200.0

    # Social force (Sec. 5.3.1)
    k_goal: float = 1.0
    tau: float = 0.5
    sf_v0: float = 2.1
    sf_sigma: float = 0.3
    sf_T: float = 2.0
    sf_u0: float = 2.1
    sf_r: float = 0.1

    # RVO (Sec. 5.3.2)
    rvo_w: float = 1.0
    rvo_samples: int = 100

    avoid_range: float = 5.0  # "all neighboring agents and walls within a radius of 5 m"
    dt_coarse: float = 0.1
    blend_low: float = 2.0  # {(nav, 2), (SPH, 4)} profile sequence of Sec. 5.5
    blend_high: float = 4.0

    # Dynamic quantities
    rho_hat: float = -1.0  # EMA of the SPH density (Eq. 8); < 0 means "not yet initialised"
    density: float = 0.0  # last SPH density rho_i (stored for measurement only)
    step_count: int = 0
    nav_acc: tuple[float, float] = (0.0, 0.0)  # cached coarse-step RVO acceleration


def _poly6(r2, h):
    """2D Poly6 density kernel (Eq. 13), argument = squared distance."""
    return np.where(r2 < h * h, 4.0 / (np.pi * h**8) * (h * h - r2) ** 3, 0.0)


def _spiky_grad_mag(r, h):
    """|grad W_p| of the 2D spiky kernel (Eq. 14); direction is -r_hat."""
    return np.where(r < h, 30.0 / (np.pi * h**5) * (h - r) ** 2, 0.0)


def _visc_lap(r, h):
    """Laplacian of the 2D viscosity kernel (Eq. 15)."""
    return np.where(r < h, 360.0 / (29.0 * np.pi * h**5) * (h - r), 0.0)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class SPHCrowdModel(CustomOperationalModel):
    """
    SPH-enhanced agent-based crowd simulation, as described in:
    "SPH crowds: Agent-based crowd simulation up to extreme densities using
    fluid dynamics" by van Toll, Chatagnon, Braga, Solenthaler & Pettré
    (Computers & Graphics 98, 2021).

    Every agent is an SPH particle. Per step, agent i:

    1. computes its SPH density rho_i (Eq. 9: Poly6 kernel over agents plus
       one representative point per visible wall, Eq. 10),
    2. updates its personal rest density rho0_i = clamp(EMA of rho_i) (Eq. 8),
    3. computes the pressure p_i = k (rho_i - rho0_i) (Eq. 5) and the SPH
       acceleration (-grad p_i + mu lap v_i) / rho_i (Eqs. 6, 7, 11, 12),
       with the pressure force set to 0 when rho_i < rho0_i (Sec. 5.2),
    4. adds a navigation acceleration (social forces, Eqs. 16-20, or RVO,
       Eq. 21) and Helbing contact forces (Eqs. 22, 23),
    5. blends a navigation profile and the SPH profile linearly in rho_i
       for the "->" profiles (Sec. 4.4),
    6. integrates with Euler: v += a dt, |v| <= s_max, x += v dt.

    JuPedSim computes every agent's update from the same snapshot of all
    states (compute-then-apply), so the SPH loop's "compute every rho and p
    before any force" ordering is honoured *exactly*: the pressure force on
    i needs rho_j and p_j of each neighbour j within h, and all of j's own
    neighbours (and walls) within h lie within 2h of i. So agent i recomputes
    rho_j, and j's rest-density update, from its own 2h neighbourhood --
    the same numbers j computes for itself this step, not a lagged copy.

    The model object holds `push_uids`: agents whose `uid` is in it perform
    the concert "push" of Sec. 6.3 (goal force with K_goal = 1 only,
    ignoring contact and SPH forces). The scenario script sets and clears it.
    """

    N_RAYS = 64  # angular resolution of the wall-shadow area a(P_k)

    def __init__(self, seed: int = 0):
        CustomOperationalModel.__init__(self)
        self.rng = np.random.default_rng(seed)
        self.push_uids: frozenset[int] = frozenset()
        self.push_k_goal = 1.0
        angles = np.linspace(0.0, 2 * np.pi, self.N_RAYS, endpoint=False)
        self._rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self._dtheta = 2 * np.pi / self.N_RAYS

    # ------------------------------------------------------------------ walls
    def _wall_shadows(self, origins, seg_a, seg_b, h):
        """Area a(P_k) and representative point r_k for every (origin, wall).

        P_k is the part of the kernel disk (radius h around the origin) hidden
        behind wall k (Sec. 4.3.1). We integrate it in polar coordinates:
        a ray at angle theta that first hits wall k at distance t < h adds
        (h^2 - t^2)/2 * dtheta to a(P_k). Taking only the *first* hit on each
        ray handles occlusion: the back face of a thick wall, or a wall
        hidden behind another, gets no area, so it is not counted twice.
        (The paper just says a(P_k) "can be computed via simple geometric
        operations"; for a single unoccluded wall this is that same area.)

        The representative point is the point halfway between the nearest
        wall point r*_k and the kernel edge, along the direction to r*_k
        (Fig. 2).

        Returns (area (S, W), rep (S, W, 2) relative to each origin).
        """
        e = seg_b - seg_a  # (W, 2)
        rel_a = seg_a[None, :, :] - origins[:, None, :]  # (S, W, 2)
        d = self._rays  # (R, 2)
        denom = _cross(d[:, None, :], e[None, :, :])  # (R, W)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(denom) > 1e-12, 1.0 / denom, np.nan)
            t = _cross(rel_a[:, None, :, :], e[None, None, :, :]) * inv[None]  # (S, R, W)
            u = _cross(rel_a[:, None, :, :], d[None, :, None, :]) * inv[None]
        hit = (t >= 0.0) & (t < h) & (u >= 0.0) & (u <= 1.0)
        t = np.where(hit, t, np.inf)
        owner = np.argmin(t, axis=2)  # (S, R)
        t_first = np.take_along_axis(t, owner[..., None], axis=2)[..., 0]
        contrib = np.where(np.isfinite(t_first), 0.5 * (h * h - t_first**2) * self._dtheta, 0.0)
        area = np.zeros((origins.shape[0], e.shape[0]))
        np.add.at(area, (np.arange(origins.shape[0])[:, None], owner), contrib)

        # Nearest point on each wall to each origin.
        ee = np.maximum(np.einsum("ij,ij->i", e, e), 1e-12)
        s = np.clip(-np.einsum("swk,wk->sw", rel_a, e) / ee[None], 0.0, 1.0)
        nearest = rel_a + s[..., None] * e[None]  # (S, W, 2)
        dist = np.linalg.norm(nearest, axis=2)
        direction = nearest / np.maximum(dist, 1e-9)[..., None]
        rep = direction * (0.5 * (dist + h))[..., None]
        return area, rep

    # ------------------------------------------------------------------ SPH
    def _sph(self, state, dt, x, m, v_nb, rho_hat_nb, seg_a, seg_b):
        """SPH density of i and of its neighbours within h, and i's SPH force.

        x: (M, 2) neighbour positions relative to i (within 2h), m: masses.
        Returns (rho_i, rho0_i, rho_hat_i_new, acc_sph (2,), mask_h, rho_h, p_h).
        """
        h = state.h
        pos = np.vstack([np.zeros((1, 2)), x])  # index 0 = agent i
        mass = np.concatenate([[state.mass], m])
        r_i = np.linalg.norm(x, axis=1)
        in_h = np.concatenate([[True], r_i < h])  # agents whose density we need
        src = pos[in_h]  # (S, 2)

        d2 = ((src[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)
        rho_agents = (_poly6(d2, h) * mass[None, :]).sum(axis=1)  # includes self term

        # Previous rest density of each of these agents (Eq. 8 input).
        rho_hat_old = np.concatenate([[state.rho_hat], rho_hat_nb])[in_h]
        rho_hat_old = np.where(rho_hat_old < 0.0, rho_agents, rho_hat_old)
        rho0_old = np.clip(rho_hat_old, state.rho0_min, state.rho0_max)

        rho = rho_agents
        area = rep = None
        if seg_a.shape[0] > 0:
            area, rep = self._wall_shadows(src, seg_a, seg_b, h)
            w_rep = _poly6((rep**2).sum(axis=2), h)  # (S, W)
            rho = rho + rho0_old * (area * w_rep).sum(axis=1)  # Eqs. 9, 10

        alpha = min(1.0, dt / state.t_rho)
        rho_hat_new = (1.0 - alpha) * rho_hat_old + alpha * rho  # Eq. 8
        rho0 = np.clip(rho_hat_new, state.rho0_min, state.rho0_max)
        p = state.k_gas * (rho - rho0)  # Eq. 5

        rho_i, p_i = rho[0], p[0]
        acc = np.zeros(2)
        if state.k_gas > 0.0 or state.mu > 0.0:
            xj = src[1:]
            rj = np.maximum(np.linalg.norm(xj, axis=1), 1e-9)
            mj = mass[in_h][1:]
            rho_j = rho[1:]
            force = np.zeros(2)
            if rho_i >= rho0[0]:  # negative pressure ignored (Sec. 5.2)
                coef = mj * (p_i + p[1:]) / (2.0 * rho_j) * _spiky_grad_mag(rj, h)
                force += (coef[:, None] * (-xj / rj[:, None])).sum(axis=0)  # Eq. 6
                if area is not None:
                    rk = rep[0]
                    dk = np.maximum(np.linalg.norm(rk, axis=1), 1e-9)
                    coef_w = p_i * area[0] * _spiky_grad_mag(dk, h)
                    force += (coef_w[:, None] * (-rk / dk[:, None])).sum(axis=0)  # Eq. 12
            if state.mu > 0.0 and xj.shape[0] > 0:
                vj = v_nb[r_i < h]
                lap = (mj / rho_j * _visc_lap(rj, h))[:, None] * (vj - np.asarray(state.velocity))
                force += state.mu * lap.sum(axis=0)  # Eq. 7
            acc = force / rho_i  # Eq. 2
        return rho_i, rho0[0], rho_hat_new[0], acc

    # ------------------------------------------------------------------ contact
    @staticmethod
    def _contact(state, x, radii, walls_d, walls_n, k_ag, k_obs):
        """Helbing contact forces (Eqs. 22, 23), divided by the agent's mass."""
        f = np.zeros(2)
        if k_ag > 0.0 and x.shape[0] > 0:
            r = np.maximum(np.linalg.norm(x, axis=1), 1e-9)
            overlap = np.maximum(0.0, state.radius + radii - r)
            f += (k_ag * overlap[:, None] * (-x / r[:, None])).sum(axis=0)
        if k_obs > 0.0 and walls_d.shape[0] > 0:
            overlap = np.maximum(0.0, state.radius - walls_d)
            f += (k_obs * overlap[:, None] * walls_n).sum(axis=0)
        return f / state.mass

    # ------------------------------------------------------------------ social force
    @staticmethod
    def _social_force(state, v, e_dir, x, v_nb, walls_d, walls_n):
        """Avoidance forces of the social force model (Eqs. 18-20)."""
        acc = np.zeros(2)
        if x.shape[0] > 0:
            r = -x  # r_ij = r_i - r_j
            y = (v[None, :] - v_nb) * state.sf_T  # v_ij T
            q = r + y
            a = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
            c = np.maximum(np.linalg.norm(q, axis=1), 1e-9)
            yn2 = (y**2).sum(axis=1)
            b = 0.5 * np.sqrt(np.maximum((a + c) ** 2 - yn2, 0.0))
            b = np.maximum(b, 1e-3)
            grad_b = ((a + c) / (4.0 * b))[:, None] * (r / a[:, None] + q / c[:, None])
            f = (state.sf_v0 / state.sf_sigma * np.exp(-b / state.sf_sigma))[:, None] * grad_b
            # Field of view: the paper's text says "angle between v_i and r_ij";
            # we use the direction *towards* j (-r_ij), as in Helbing & Molnar's
            # intent (people react less to what is behind them).
            in_view = (x / a[:, None]) @ e_dir >= _COS_FOV
            acc += (np.where(in_view, 1.0, 0.5)[:, None] * f).sum(axis=0)
        if walls_d.shape[0] > 0:
            f = (state.sf_u0 / state.sf_r * np.exp(-walls_d / state.sf_r))[:, None] * walls_n
            in_view = (-walls_n) @ e_dir >= _COS_FOV
            acc += (np.where(in_view, 1.0, 0.5)[:, None] * f).sum(axis=0)
        return acc

    # ------------------------------------------------------------------ RVO
    def _rvo(self, state, v, v_pref, x, radii, v_nb, seg_a, seg_b, walls_p):
        """RVO velocity selection (Eq. 21) -> acceleration (v* - v)/dt_coarse."""
        n = state.rvo_samples
        ang = self.rng.uniform(0.0, 2 * np.pi, n)
        rad = state.max_speed * np.sqrt(self.rng.uniform(0.0, 1.0, n))
        cand = np.vstack([v_pref[None, :], np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)])
        u = 2.0 * cand - v[None, :]  # RVO: TTC is evaluated for 2v - v_i
        ttc = np.full(cand.shape[0], np.inf)

        if x.shape[0] > 0:
            w = u[:, None, :] - v_nb[None, :, :]  # (C, M, 2) velocity relative to j
            R = state.radius + radii
            c0 = (x**2).sum(axis=1) - R**2  # (M,)
            A = (w**2).sum(axis=2)
            B = -2.0 * np.einsum("cmk,mk->cm", w, x)
            disc = B**2 - 4.0 * A * c0[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
            ok = (disc >= 0.0) & (A > 1e-12) & (t1 >= 0.0)
            t_nb = np.where(ok, t1, np.inf)
            # Already overlapping (c0 < 0): not covered by the paper. A pair
            # that is closing gets TTC = 0 (maximal penalty); one that is
            # separating gets none, so overlapping agents prefer to part.
            overl = c0 < 0.0
            if overl.any():
                closing = np.einsum("cmk,mk->cm", w, x) > 0.0
                t_nb = np.where(overl[None, :], np.where(closing, 0.0, np.inf), t_nb)
            ttc = np.minimum(ttc, t_nb.min(axis=1))

        if seg_a.shape[0] > 0:
            ttc = np.minimum(ttc, self._ttc_walls(u, seg_a, seg_b, walls_p, state.radius))

        with np.errstate(divide="ignore"):
            penalty = np.where(ttc > 0.0, state.rvo_w / ttc, 1e6)
        cost = np.linalg.norm(cand - v_pref[None, :], axis=1) + penalty
        v_star = cand[int(np.argmin(cost))]
        return (v_star - v) / state.dt_coarse

    @staticmethod
    def _ttc_walls(u, seg_a, seg_b, closest, radius):
        """Time for a disk at the origin moving with u to touch each wall."""
        C = u.shape[0]
        t_best = np.full(C, np.inf)
        e = seg_b - seg_a
        L = np.maximum(np.linalg.norm(e, axis=1), 1e-9)
        tdir = e / L[:, None]
        nrm = np.stack([-tdir[:, 1], tdir[:, 0]], axis=1)
        side = np.einsum("wk,wk->w", -seg_a, nrm)  # signed distance of the origin
        nrm = nrm * np.where(side >= 0.0, 1.0, -1.0)[:, None]  # normal pointing to agent
        dist = np.abs(side)
        # Offset line: n.(t u - a) = radius  ->  t = (radius - dist) / (u.n)  with u.n < 0
        un = u @ nrm.T  # (C, W)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_line = (dist - radius)[None, :] / (-un)
        p = t_line[..., None] * u[:, None, :] - seg_a[None, :, :]
        along = np.einsum("cwk,wk->cw", p, tdir)
        ok = (un < 0) & (t_line >= 0) & (along >= 0) & (along <= L[None, :])
        t_best = np.minimum(t_best, np.where(ok, t_line, np.inf).min(axis=1))
        # End points (disk of radius `radius`)
        for P in (seg_a, seg_b):
            A = (u**2).sum(axis=1)[:, None]
            B = -2.0 * (u @ P.T)
            c0 = (P**2).sum(axis=1)[None, :] - radius**2
            disc = B**2 - 4 * A * c0
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2 * A)
            ok = (disc >= 0) & (A > 1e-12) & (t1 >= 0) & (c0 >= 0)
            t_best = np.minimum(t_best, np.where(ok, t1, np.inf).min(axis=1))
        # Already touching a wall: moving further into it is a collision now.
        cd = np.linalg.norm(closest, axis=1)
        touching = cd < radius
        if touching.any():
            into = (u @ closest[touching].T) > 0.0
            t_best = np.where(into.any(axis=1), 0.0, t_best)
        return t_best

    # ------------------------------------------------------------------ main
    def compute_next_state(self, state: SPHCrowdState, step):
        dt = step.dt
        v = np.asarray(state.velocity, dtype=float)
        e = np.asarray(step.orientation_to_next_target, dtype=float)
        v_pref = state.pref_speed * e
        profile = state.profile
        h = state.h

        # --- neighbourhood for SPH (2h) and contacts
        nbs = step.other_agents_in_range(2.0 * h)
        M = len(nbs)
        if M:
            x = np.array([nb.relative_position for nb in nbs], dtype=float)
            sts = [nb.state for nb in nbs]
            m = np.array([s.mass for s in sts])
            radii = np.array([s.radius for s in sts])
            v_nb = np.array([s.velocity for s in sts], dtype=float)
            rho_hat_nb = np.array([s.rho_hat for s in sts])
        else:
            x = np.zeros((0, 2)); m = radii = rho_hat_nb = np.zeros(0); v_nb = np.zeros((0, 2))
        walls = step.walls_in_range(2.0 * h)
        if walls:
            seg = np.array([w.segment for w in walls], dtype=float)
            seg_a, seg_b = seg[:, 0, :], seg[:, 1, :]
            walls_d = np.array([w.distance for w in walls])
            walls_n = np.array([w.normal for w in walls], dtype=float)
        else:
            seg_a = seg_b = np.zeros((0, 2)); walls_d = np.zeros(0); walls_n = np.zeros((0, 2))

        rho_i, rho0_i, rho_hat_i, acc_sph = self._sph(state, dt, x, m, v_nb, rho_hat_nb, seg_a, seg_b)

        # --- concert push (Sec. 6.3): goal force only, K_goal = 1
        if state.uid in self.push_uids:
            acc = self.push_k_goal * (v_pref - v) / state.tau
            return self._integrate(state, step, v, acc, walls_d, walls_n, rho_i, rho_hat_i, state.nav_acc)

        close = np.linalg.norm(x, axis=1) < (state.radius + 0.3) if M else np.zeros(0, bool)
        c_sph = lambda: self._contact(state, x[close], radii[close], walls_d, walls_n, state.k_ag_sph, state.k_obs_sph)
        c_nav = lambda: self._contact(state, x[close], radii[close], walls_d, walls_n, state.k_ag_nav, state.k_obs_nav)

        # Blend weight of the SPH profile (Sec. 4.4)
        nav_kind = "SF" if profile.startswith("SF") else ("RVO" if profile.startswith("RVO") else None)
        if profile in ("SF->SPH", "RVO->SPH"):
            w_sph = float(np.clip((rho_i - state.blend_low) / (state.blend_high - state.blend_low), 0.0, 1.0))
        elif profile == "SPH":
            w_sph = 1.0
        else:
            w_sph = 0.0

        # --- navigation acceleration a^0 of the low-density profile
        # SF is evaluated at every fine step; RVO only at coarse steps, its
        # acceleration being held in between (Sec. 5.1). For the blend to be
        # the same operation for both, a^0 must always be the *current* value
        # of that profile: RVO is therefore refreshed at every coarse step even
        # while its blend weight is 0 (rho_i >= 4). Otherwise an agent leaving
        # the dense crowd would reuse an RVO result cached seconds earlier.
        # SF has no state, so skipping it while its weight is 0 is exact.
        nav_acc = np.asarray(state.nav_acc, dtype=float)
        a_nav = np.zeros(2)
        coarse = state.step_count % max(1, int(round(state.dt_coarse / dt))) == 0
        need_nav = nav_kind is not None and (w_sph < 1.0 or (nav_kind == "RVO" and coarse))
        if need_nav:
            if nav_kind == "SF" or coarse:
                far = step.other_agents_in_range(state.avoid_range)
                if far:
                    xf = np.array([nb.relative_position for nb in far], dtype=float)
                    stf = [nb.state for nb in far]
                    vf = np.array([s.velocity for s in stf], dtype=float)
                    rf = np.array([s.radius for s in stf])
                else:
                    xf = np.zeros((0, 2)); vf = np.zeros((0, 2)); rf = np.zeros(0)
                wf = step.walls_in_range(state.avoid_range)
                if wf:
                    segf = np.array([w.segment for w in wf], dtype=float)
                    wd = np.array([w.distance for w in wf])
                    wn = np.array([w.normal for w in wf], dtype=float)
                    wp = np.array([w.closest_point for w in wf], dtype=float)
                else:
                    segf = np.zeros((0, 2, 2)); wd = np.zeros(0); wn = np.zeros((0, 2)); wp = np.zeros((0, 2))
            if nav_kind == "SF":
                speed = np.linalg.norm(v)
                e_dir = v / speed if speed > 1e-6 else e
                a_nav = state.k_goal * (v_pref - v) / state.tau + self._social_force(state, v, e_dir, xf, vf, wd, wn)
            else:
                if coarse:  # RVO runs at dt_coarse; its result is reused in between (Sec. 5.1)
                    nav_acc = self._rvo(state, v, v_pref, xf, rf, vf, segf[:, 0, :], segf[:, 1, :], wp)
                a_nav = nav_acc

        goal = state.k_goal * (v_pref - v) / state.tau
        if profile == "SF" or profile == "RVO":
            acc = a_nav + c_nav()
        elif profile == "SPH":
            acc = goal + acc_sph + c_sph()
        elif profile in ("SF+SPH", "RVO+SPH"):
            acc = a_nav + acc_sph + c_sph()
        else:  # blended
            a_low = a_nav + c_nav() if w_sph < 1.0 else np.zeros(2)
            a_high = goal + acc_sph + c_sph() if w_sph > 0.0 else np.zeros(2)
            acc = (1.0 - w_sph) * a_low + w_sph * a_high

        return self._integrate(state, step, v, acc, walls_d, walls_n, rho_i, rho_hat_i, nav_acc)

    @staticmethod
    def _integrate(state, step, v, acc, walls_d, walls_n, rho_i, rho_hat_i, nav_acc):
        dt = step.dt
        v_new = v + acc * dt
        speed = np.linalg.norm(v_new)
        if speed > state.max_speed:
            v_new *= state.max_speed / speed
        move = v_new * dt
        # JuPedSim aborts the whole run if a move leaves the walkable area
        # ("move_on_surface(): path hit a wall"). The paper's engine has no
        # such hard constraint (weak wall contact forces just push agents
        # back). Practical extension: if the step would cross a wall, drop
        # its component into the nearest wall (slide along it); if that is
        # still blocked, stand still for this step.
        if not step.no_geometry_between((float(move[0]), float(move[1]))):
            if walls_d.shape[0]:
                n = walls_n[int(np.argmin(walls_d))]
                into = min(0.0, float(move @ n))
                move = move - into * n
                v_new = v_new - min(0.0, float(v_new @ n)) * n
            if not step.no_geometry_between((float(move[0]), float(move[1]))):
                move = np.zeros(2)
                v_new = np.zeros(2)
        new_state = replace(
            state,
            velocity=(float(v_new[0]), float(v_new[1])),
            rho_hat=float(rho_hat_i),
            density=float(rho_i),
            step_count=state.step_count + 1,
            nav_acc=(float(nav_acc[0]), float(nav_acc[1])),
        )
        return new_state, (float(move[0]), float(move[1]))


def make_state(rng, profile, uid, **overrides) -> SPHCrowdState:
    """Agent with the paper's random radius D in [0.215, 0.265], m = (D/0.24)^2."""
    radius = float(rng.uniform(0.215, 0.265))
    return SPHCrowdState(radius=radius, mass=(radius / 0.24) ** 2, uid=uid, profile=profile, **overrides)
