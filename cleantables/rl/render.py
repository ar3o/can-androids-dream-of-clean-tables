"""Labelled videos of recorded episodes: a kinematic replay of a Recorder pickle (`cleantables.rl.chain_env.Recorder`; the physics
ran in Isaac / PhysX during recording, this only draws the recorded states), rendered with matplotlib.
  left    3-D view: table, objects (boxes; moving objects coloured, static ones grey), both Franka arms (FK of the recorded joint angles;
          link chain + collision spheres), both hands (palm + fingers), the goal rings (green = met, orange = not), the active carry path
          (solid) and hand route (dashed; the zero-edit route dotted grey when the learned edit moved it > 1 cm)
  right   top-down view of the same + object names
  bottom  caption strip: time, met count, per gripper the last decision (type / object / grasp yaw / result) and current events
Highlighted when they happen (recomputed from the recorded arm poses, as ArmTracker checks them): arm-arm, arm-object and arm-table
overlaps, hand-hand overlap, fingers below the table top, IK failures, take aborts, pushes while held (> 1 cm) and disturbances (drop /
knock) with whether the object was back at its goal at the episode end.
  $SIM_PY -m cleantables.rl.render REC.pkl OUT_DIR [--label trained] [--speed 2]
"""
import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                     # noqa: E402
from matplotlib.patches import Circle, Polygon                                       # noqa: E402

from cleantables.robots import franka_kinematics as K                                      # noqa: E402
from cleantables.rl.gripper import FL, FT, FW, PH, PL, PW                                          # noqa: E402

COLS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#ff7f0e", "#7f7f7f", "#d62728", "#aec7e8", "#98df8a"]
GCOL = ["#0055cc", "#cc6600"]                                                       # gripper 0 / 1
HALF = [np.array([PL, PW, PH]) / 2, np.array([FT, FW, FL]) / 2, np.array([FT, FW, FL]) / 2]
TYPES = {"take": "TAKE", "move": "FREE MOVE", "wait": "WAIT", "done": "DONE"}


def short(n):
    return n.split("__")[0]


def box_corners(c, half, yaw):
    """(8, 3) corners of a box centre c, half sizes, yaw about z."""
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    cz, sz_ = math.cos(yaw), math.sin(yaw)
    R = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
    return s @ R.T + c


EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box3d(ax, cr, col, lw=1.0, alpha=1.0):
    for a, b in EDGES:
        ax.plot(*zip(cr[a], cr[b]), color=col, lw=lw, alpha=alpha)


def foot(cr):
    """the footprint polygon (4, 2) of a box's corners (bottom face, ordered)."""
    return cr[[0, 2, 6, 4], :2]


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Clip:
    def __init__(self, R, e, label):
        self.R, self.e, self.label = R, e, label
        self.env = R["env"][e]; self.dt = R["dt"]
        ep = [x for x in R["eps"] if x["env"] == e]
        self.ep = ep[0] if ep else None
        s0 = self.ep["s0"] if self.ep else 0; s1 = (self.ep["s1"] if self.ep else R["frames"][-1]["step"]) + int(1.0 / self.dt)
        self.F = [f for f in R["frames"] if s0 <= f["step"] <= s1]
        self.s0 = s0
        self.ids = self.env["ids"]
        self.n = min(len(self.ids), R["frames"][0]["P"].shape[1])                                    # the sim's slots (K) may be fewer
        self.ext = np.asarray(self.env["ext"])[:self.n]                                              # padded to the run's K
        self.mov = R["mov"][e][:self.n] & ~R["static"][e][:self.n]
        self.B = self.env["bases"]
        self.dec = [d for d in R["decisions"] if d["env"] == e and s0 <= d["step"] <= s1]
        self.routes = [r for r in R["routes"] if r["env"] == e and s0 <= r["step"] <= s1]
        self.ev = [d for d in R["ev"] if d["env"] == e and s0 <= d["s0"] <= s1]
        self.takes = [t for t in R["takes"] if t["env"] == e and s0 <= t["step"] <= s1]
        self.dist = [d for d in R["dist"] if d["env"] == e and s0 <= d["step"] <= s1]
        self.plans = [p for p in R.get("plans", []) if p["env"] == e and s0 <= p["step"] <= s1]   # the WM's plan for every mover
        self._abort_steps()
        last = self.F[-1] if self.F else None
        self.end_met = last["met"][e][:self.n] if last is not None else np.zeros(self.n, bool)

    def _abort_steps(self):
        """The step each aborted take ended (approach aborts: the retraction's start; carry aborts: its release)."""
        for t in self.takes:
            if t.get("abort") and t.get("abort_step") is None:
                nxt = [r["step"] for r in self.routes if r["g"] == t["g"] and r["after"] == 0 and r["step"] > t["step"]]
                if t["abort"].endswith("carry") and t.get("ev") is not None and t["ev"] < len(self.R["ev"]) and self.R["ev"][t["ev"]].get("rel") is not None:
                    t["abort_step"] = self.R["ev"][t["ev"]]["rel"]
                else:
                    t["abort_step"] = min(nxt) if nxt else t["step"]

    # ---------------------------------------------------------------------------------------------------------------------------------
    def collisions(self, f):
        """per frame: arm-arm / arm-object / arm-table overlapping spheres per arm, hand-hand overlap, fingers below the table."""
        import torch
        e = self.e; out = dict(aa=[[], []], ao=[[], []], at=[[], []], ao_obj=set(), sph=None)
        if f["q"] is None:
            return out
        q = torch.as_tensor(f["q"][e], dtype=torch.float32); B = torch.as_tensor(self.B, dtype=torch.float32)
        p, r = K.spheres(q, B)                                                              # (2, S, 3), (S,)
        out["sph"] = (p.numpy(), r.numpy())
        d = (p[0][:, None] - p[1][None]).norm(dim=-1) - r[:, None] - r[None]
        ia, ib = np.nonzero(d.numpy() < 0); out["aa"] = [sorted(set(ia.tolist())), sorted(set(ib.tolist()))]
        P = f["P"][e][:self.n]; ex = self.ext[:self.n]
        ok_ = np.isfinite(P).all(-1)
        ctr = np.stack([P[:, 0], P[:, 1], P[:, 2] + 0.5 * ex[:, 2]], -1); rad = 0.5 * np.minimum(ex[:, :2].min(-1), ex[:, 2])
        skip_o = np.array([b in K.TABLE_SKIP for b in K.S_BODY]); skip_t = np.array([b in K.CONTACT_SKIP for b in K.S_BODY])
        for g in (0, 1):
            pg, rg = p[g].numpy(), r.numpy()
            for k in range(self.n):
                if not ok_[k] or k == f["excl"][e, g] or k == f["excl"][e, 1 - g] and f["phase"][e, 1 - g] in (1, 2):
                    continue
                dd = np.linalg.norm(pg - ctr[k], axis=-1) - rg - rad[k]
                hit = np.nonzero((dd < 0) & ~skip_o)[0]
                if len(hit):
                    out["ao"][g] += hit.tolist(); out["ao_obj"].add(k)
            dt_ = pg[:, 2] - rg
            out["at"][g] = np.nonzero((dt_ < -0.005) & ~skip_t)[0].tolist()
        return out

    def pushed_now(self, f):
        """{object: cm} moved > 1 cm since the start of an active held carry (not the carried object, the other gripper's, or riders)."""
        e = self.e; st = f["step"]; P = f["P"][e][:self.n]; ok_ = np.isfinite(P).all(-1); out = {}
        if not hasattr(self, "_cp0"):
            self._cp0 = {}
        for g in (0, 1):
            if f["phase"][e, g] != 1:
                continue
            evs = [d for d in self.ev if d["g"] == g and d["s0"] <= st and (d.get("end") is None or d["end"] >= st)]
            if not evs:
                continue
            d = evs[-1]; a = int(d["carried"]); key = (g, d["s0"])
            if key not in self._cp0:
                self._cp0[key] = min(self.F, key=lambda ff: abs(ff["step"] - d["s0"]))["P"][e][:self.n].copy()
            P0 = self._cp0[key]; oth = f["excl"][e, 1 - g]
            for k in range(self.n):
                if k == a or k == oth or not (ok_[k] and np.isfinite(P0[k]).all()):
                    continue
                if np.linalg.norm(P0[k, :2] - P0[a, :2]) < 0.5 * self.ext[a, :2].max() and P0[k, 2] > P0[a, 2] + 0.005:
                    continue
                mvd = 100 * np.linalg.norm(P[k, :3] - P0[k, :3])
                if mvd > 1.0:
                    out[k] = max(out.get(k, 0), mvd)
        return out

    def hand_flags(self, f):
        e = self.e; pl = f["palm"][e]
        rr = 0.5 * math.sqrt(PL ** 2 + PW ** 2 + PH ** 2) * 0.6
        hh = np.linalg.norm(pl[0] - pl[1]) < 2 * rr
        under = [False, False]
        if f["parts"] is not None:
            for g in (0, 1):
                hz = np.array([HALF[i][2] for i in range(f["parts"].shape[2])])
                under[g] = bool(((f["parts"][e, g, :, 2] - hz) < -0.002).any())
        return hh, under

    # ---------------------------------------------------------------------------------------------------------------------------------
    def render(self, out, speed=2.0, fps=20):
        from matplotlib.animation import FFMpegWriter
        if not self.F:
            return None
        tb = self.env["table"]; x0, x1 = tb["x"]; y0, y1 = tb["y"]
        bx = self.B[:, :2, 3]
        lo = np.minimum([x0, y0], bx.min(0)) - 0.15; hi = np.maximum([x1, y1], bx.max(0)) + 0.15
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax3 = fig.add_axes([0.0, 0.2, 0.56, 0.78], projection="3d"); ax2 = fig.add_axes([0.58, 0.24, 0.40, 0.72]); axc = fig.add_axes([0.01, 0.0, 0.98, 0.2])
        wr = FFMpegWriter(fps=fps, metadata=dict(title=f"{self.label} {self.env['name']}"), bitrate=4000)
        # push-while-held bookkeeping: the pose of every object at each carry's start
        carry_P0 = {}
        hold_frames = int(round(speed * 0))                                           # one frame per recorded frame; speed via fps
        msgs = []                                                                     # (until_step, text, colour)
        with wr.saving(fig, out, dpi=100):
            for fi, f in enumerate(self.F):
                st = f["step"]; t = (st - self.s0) * self.dt; e = self.e
                ax3.cla(); ax2.cla(); axc.cla(); axc.axis("off")
                P = f["P"][e][:self.n]; ok_ = np.isfinite(P).all(-1)
                col = self.collisions(f); hh, under = self.hand_flags(f)
                # --- table + bases
                for ax in (ax3,):
                    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [0] * 5, color="k", lw=1.2)
                ax2.add_patch(Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], closed=True, fill=True, fc="#f3efe6", ec="k", lw=1.2))
                # --- the world model's latest plan for every mover, faint (from the latest decision); the chosen carry is drawn bold below
                if self.plans:
                    pl_ = [p for p in self.plans if p["step"] <= st]
                    if pl_:
                        for k, pth in pl_[-1]["paths"].items():
                            k = int(k)
                            if k >= self.n:
                                continue
                            pth = np.asarray(pth); c_ = COLS[k % len(COLS)]
                            ax3.plot(pth[:, 0], pth[:, 1], pth[:, 2], color=c_, lw=1.6, alpha=0.55)
                            ax2.plot(pth[:, 0], pth[:, 1], color=c_, lw=1.6, alpha=0.55)
                # --- active carries (phase 1 / 2) and pushes while held
                pushed = {}
                for g in (0, 1):
                    if f["phase"][e, g] in (1, 2):
                        evs = [d for d in self.ev if d["g"] == g and d["s0"] <= st and (d.get("end") is None or d["end"] >= st)]
                        if evs:
                            d = evs[-1]; pth = np.asarray(d["path"]); a = int(d["carried"])
                            c_ = COLS[a % len(COLS)]
                            ax3.plot(pth[:, 0], pth[:, 1], pth[:, 2], color=c_, lw=2.2); ax2.plot(pth[:, 0], pth[:, 1], color=c_, lw=2.2)
                            key = (g, d["s0"])
                            if key not in carry_P0:
                                carry_P0[key] = min(self.F, key=lambda ff: abs(ff["step"] - d["s0"]))["P"][e][:self.n].copy()
                            P0 = carry_P0[key]
                            oth = f["excl"][e, 1 - g]
                            if f["phase"][e, g] == 1:
                                for k in range(self.n):
                                    if k == a or k == oth or not (ok_[k] and np.isfinite(P0[k]).all()):
                                        continue
                                    if np.linalg.norm(P0[k, :2] - P0[a, :2]) < 0.5 * self.ext[a, :2].max() and P0[k, 2] > P0[a, 2] + 0.005:
                                        continue                                                     # rides on the carried object
                                    mvd = 100 * np.linalg.norm(P[k, :3] - P0[k, :3])
                                    if mvd > 1.0:
                                        pushed[k] = max(pushed.get(k, 0), mvd)
                # --- routes of moving hands
                for g in (0, 1):
                    if f["phase"][e, g] == 4:
                        rs = [r for r in self.routes if r["g"] == g and r["step"] <= st]
                        if rs:
                            r = rs[-1]; rt, r0 = r["route"], r["route0"]
                            ret = f["retract"] is not None and f["retract"][e, g]
                            cc = "#d62728" if ret else GCOL[g]
                            ax3.plot(rt[:, 0], rt[:, 1], rt[:, 2], ls="--", color=cc, lw=1.6); ax2.plot(rt[:, 0], rt[:, 1], ls="--", color=cc, lw=1.6)
                            if np.abs(rt - r0).max() > 0.01:
                                ax3.plot(r0[:, 0], r0[:, 1], r0[:, 2], ls=":", color="#888", lw=1.2); ax2.plot(r0[:, 0], r0[:, 1], ls=":", color="#888", lw=1.2)
                # --- objects + goals
                for k in range(self.n):
                    if not ok_[k]:
                        continue
                    c_ = COLS[k % len(COLS)] if self.mov[k] else "#9a9a9a"
                    yw = yaw_of(P[k, 3:])
                    cr = box_corners(P[k, :3] + [0, 0, self.ext[k, 2] / 2], self.ext[k] / 2, yw)
                    ec = "#ff0000" if (k in pushed or k in col["ao_obj"]) else c_
                    draw_box3d(ax3, cr, ec, lw=2.2 if ec == "#ff0000" else 1.2)
                    ax2.add_patch(Polygon(foot(cr), closed=True, fc=c_, ec=ec, lw=2.5 if ec == "#ff0000" else 0.8, alpha=0.55))
                    lab = short(self.ids[k]) + (f"  pushed {pushed[k]:.1f} cm" if k in pushed else "") + ("  (off the table)" if P[k, 2] < -0.05 else "")
                    ax2.text(P[k, 0], P[k, 1], lab, fontsize=6.5, ha="center", va="center", color="#b00" if k in pushed else "k")
                    if self.mov[k]:
                        gl = f["goal"][e][k]; met = bool(f["met"][e][k])
                        rad = 0.5 * self.ext[k, :2].max()
                        gc = "#2ca02c" if met else "#ff7f0e"
                        th = np.linspace(0, 2 * np.pi, 40)
                        ax3.plot(gl[0] + rad * np.cos(th), gl[1] + rad * np.sin(th), np.full(40, gl[2]), color=gc, lw=1.8)
                        ax2.add_patch(Circle(gl[:2], rad, fill=False, ec=gc, lw=2.0, ls="-"))
                        ax2.plot([P[k, 0], gl[0]], [P[k, 1], gl[1]], color=gc, lw=0.5, alpha=0.5)
                # --- arms
                if col["sph"] is not None:
                    import torch
                    p_, r_ = col["sph"]
                    for g in (0, 1):
                        Fk = K.fk(torch.as_tensor(f["q"][e, g], dtype=torch.float32)).numpy()
                        pts = (self.B[g] @ np.concatenate([Fk[:, :3, 3], np.ones((len(Fk), 1))], 1).T).T[:, :3]
                        chain = pts[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]
                        bad = col["aa"][g] or col["ao"][g] or col["at"][g]
                        ac = "#d62728" if bad else ("#7a7a7a" if f["ik_ok"][e, g] else "#ff00ff")
                        ax3.plot(chain[:, 0], chain[:, 1], chain[:, 2], color=ac, lw=4, alpha=0.8, solid_capstyle="round")
                        ax2.plot(chain[:, 0], chain[:, 1], color=ac, lw=3, alpha=0.5)
                        hit = sorted(set(col["aa"][g]) | set(col["ao"][g]) | set(col["at"][g]))
                        if hit:
                            ax3.scatter(p_[g, hit, 0], p_[g, hit, 1], p_[g, hit, 2], s=90, c="#ff0000", alpha=0.85, depthshade=False)
                            ax2.scatter(p_[g, hit, 0], p_[g, hit, 1], s=60, c="#ff0000", alpha=0.7)
                        ax2.text(self.B[g, 0, 3], self.B[g, 1, 3], f"arm {g}", fontsize=8, color=GCOL[g], ha="center", fontweight="bold")
                # --- hands
                if f["parts"] is not None:
                    for g in (0, 1):
                        hc = "#d62728" if (hh or under[g]) else GCOL[g]
                        for i in range(f["parts"].shape[2]):
                            cr = box_corners(f["parts"][e, g, i], HALF[i], f["yaw"][e, g])
                            draw_box3d(ax3, cr, hc, lw=1.6)
                            ax2.add_patch(Polygon(foot(cr), closed=True, fill=False, ec=hc, lw=1.4))
                        if f["ik_ok"] is not None and not f["ik_ok"][e, g]:
                            pl = f["palm"][e, g]
                            ax3.scatter([pl[0]], [pl[1]], [pl[2]], s=160, marker="X", c="#ff00ff", depthshade=False)
                            ax3.text(pl[0], pl[1], pl[2] + 0.06, "IK fail", color="#ff00ff", fontsize=9, fontweight="bold")
                # --- events of this moment -> messages
                win = int(1.5 / self.dt)
                for tk in self.takes:
                    if tk.get("abort") and tk.get("abort_step") is not None and tk["abort_step"] <= st < tk["abort_step"] + win and (tk.get("_shown_at") in (None, st) or True):
                        pl = f["palm"][e, tk["g"]]
                        ax3.text(pl[0], pl[1], pl[2] + 0.10, f"ABORT {tk['abort']}", color="#d62728", fontsize=10, fontweight="bold")
                        ax2.text(pl[0], pl[1] + 0.04, f"ABORT {tk['abort']}", color="#d62728", fontsize=8, fontweight="bold", ha="center")
                for d in self.dist:
                    if d["step"] <= st < d["step"] + 2 * win:
                        k = d["obj"]; rec = bool(self.end_met[k]) if self.mov[k] else (f["static_moved_cm"][e][k] <= 3)
                        txt = f"{d['kind'].upper()} {short(self.ids[k])} (recovered by episode end: {'yes' if rec else 'no'})"
                        ax2.text(P[k, 0], P[k, 1] - 0.05, txt, color="#9900cc", fontsize=8, fontweight="bold", ha="center")
                        ax3.text(P[k, 0], P[k, 1], P[k, 2] + 0.15, d["kind"].upper(), color="#9900cc", fontsize=11, fontweight="bold")
                # --- views
                ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(-0.3, 0.6)
                ax3.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], 0.9)); ax3.view_init(elev=32, azim=-60 + 0.0 * fi)
                ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
                ax2.set_xlim(lo[0], hi[0]); ax2.set_ylim(lo[1], hi[1]); ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
                ax2.set_title("top view (dashed: hand route, dotted: zero-edit route, solid: carry path; rings: frozen goals, green = met)"
                              + ("\nfaint: the world model's plan for every object; bold: the chosen carry" if self.plans else ""), fontsize=8)
                # --- caption strip
                nm = int(f["met"][e][:self.n][self.mov].sum()); nmv = int(self.mov.sum())
                axc.text(0.0, 0.92, f"{self.label}   |   {self.env['name']}   |   t = {t:5.1f} s ({speed:.0f}x)   |   goals met {nm}/{nmv}", fontsize=12, fontweight="bold", va="top")
                for g in (0, 1):
                    ds = [d for d in self.dec if d["g"] == g and d["step"] <= st]
                    s_ = f"gripper {g} (arm {g}): "
                    if ds:
                        d = ds[-1]; s_ += TYPES.get(d["type"], d["type"])
                        if d["type"] == "take":
                            s_ += f" {short(self.ids[d['obj']])}, grasp yaw {math.degrees(d['gyaw']):.0f} deg, release u {d['rel_u']:.2f}"
                            tk = [x for x in self.takes if x["g"] == g and x["step"] == d["step"]]
                            if tk:
                                r_ = tk[0]
                                res = (f"ABORTED: {r_['abort']}" if r_.get("abort") and r_.get("abort_step", 1e9) <= st else
                                       ("carrying" if f["phase"][e, g] == 1 else "approaching" if f["phase"][e, g] == 4 else "released / settling" if f["phase"][e, g] == 2 else "done"))
                                s_ += f"  ->  {res}"
                        elif d["type"] == "move" and d.get("P") is not None:
                            s_ += f" to ({d['P'][0]:.2f}, {d['P'][1]:.2f}, {d['P'][2]:.2f})"
                    if f["retract"] is not None and f["retract"][e, g]:
                        s_ += "   [arm retracting the hand home after an abort]"
                    axc.text(0.0, 0.66 - 0.2 * g, s_, fontsize=10.5, color=GCOL[g], va="top")
                ev_ = []
                if col["aa"][0]:
                    ev_.append("ARM-ARM overlap")
                if col["ao"][0] or col["ao"][1]:
                    ev_.append("ARM-OBJECT overlap (" + ", ".join(short(self.ids[k]) for k in sorted(col["ao_obj"])) + ")")
                if col["at"][0] or col["at"][1]:
                    ev_.append("ARM-TABLE overlap")
                if hh:
                    ev_.append("HAND-HAND overlap")
                if any(under):
                    ev_.append("FINGERS BELOW THE TABLE (gripper " + ",".join(str(g) for g in (0, 1) if under[g]) + ")")
                if f["ik_ok"] is not None and not f["ik_ok"][e].all():
                    ev_.append("IK FAIL (arm " + ",".join(str(g) for g in (0, 1) if not f["ik_ok"][e, g]) + ")")
                if pushed:
                    ev_.append("PUSH WHILE HELD: " + ", ".join(f"{short(self.ids[k])} {v:.1f} cm" for k, v in pushed.items()))
                axc.text(0.0, 0.18, ("events: " + "  |  ".join(ev_)) if ev_ else "events: -", fontsize=10.5, color="#d62728" if ev_ else "#555", va="top")
                if self.ep is not None and st >= self.ep["s1"]:
                    axc.text(0.62, 0.92, f"EPISODE END: {self.ep['stop']}, carries {self.ep['carries']}, met {self.ep['met']:.2f}, all met {self.ep['all_met']}",
                             fontsize=11, color="#2a2", fontweight="bold", va="top")
                wr.grab_frame()
        plt.close(fig)
        return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("out"); ap.add_argument("--label", default=""); ap.add_argument("--speed", type=float, default=2.0)
    a = ap.parse_args()
    R = pickle.load(open(a.rec, "rb")); Path(a.out).mkdir(parents=True, exist_ok=True)
    lab = a.label or Path(a.rec).stem
    fps = int(round(a.speed / (R["dt"] * R["every"])))
    for e in range(len(R["env"])):
        c = Clip(R, e, lab)
        o = c.render(str(Path(a.out) / f"{lab}__{R['env'][e]['name']}.mp4"), speed=a.speed, fps=fps)
        print("wrote", o, len(c.F), "frames", flush=True)


if __name__ == "__main__":
    main()
