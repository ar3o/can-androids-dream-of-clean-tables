"""Goals and carry paths from the world-model teacher, for every env of a CleanupEnv.

At each reset the teacher sets the goal pose of every in-scope object (where it would put it). While an arm carries its
object, the teacher's imagined path for that object is re-planned on a new assignment, every `replan_every` control
steps, and when the object strays from the path. The teacher sees the objects only: it was trained and validated with
human hands, and robot palms in those slots change its choices.
"""

from __future__ import annotations

import time

import torch

from cleantables.paths import WM_RUNS
from cleantables.worldmodel.dataset import TABLE
from cleantables.worldmodel.teacher import WorldModelTeacher

from .geometry import yaw_of

PATH_STEPS = 32
SMOOTH_WINDOW = 5
FALLBACK_LIFT = 0.02  # m above the labelled container for objects the teacher has no token for


def teacher_checkpoints(run: str, seeds: tuple[int, ...]) -> list:
    return [WM_RUNS / run / f"final_seed{s}.pt" for s in seeds]


class TeacherGoals:
    def __init__(
        self,
        env,
        run: str,
        seeds: tuple[int, ...],
        samples: int,
        replan_every: int,
        offpath_dist: float,
        offpath_gap: int,
    ):
        self.env = env
        self.replan_every, self.offpath_dist, self.offpath_gap = replan_every, offpath_dist, offpath_gap
        dev, n, num_items = env.device, env.num_envs, env.num_items
        self.wm = WorldModelTeacher(
            env.scene_json["episode"], checkpoints=teacher_checkpoints(run, seeds), device=str(dev), samples=samples
        )
        ids = [str(x) for x in self.wm.Z["ids"]]
        missing = [i for i in ids if i not in env.names]
        assert not missing, f"world-model objects not in the scene: {missing}"
        names = env.names
        self.k_of = torch.tensor([names.index(i) for i in ids], device=dev)
        self.of_item = torch.tensor([ids.index(names[k]) if names[k] in ids else -1 for k in env.item_idx], device=dev)
        self.of_dest = torch.tensor(
            [ids.index(names[k]) if names[k] in ids else -1 for k in env.item_dst.tolist()], device=dev
        )
        self.height = torch.as_tensor(self.wm.Z["extents"][:, 2], dtype=torch.float32, device=dev)
        containers = [k for k, o in enumerate(env.scene_json["objects"]) if o["role"] == "container"]
        self.is_container = torch.tensor([names.index(i) in containers for i in ids] + [False] * 30, device=dev)

        self.path = torch.zeros(n, 2, PATH_STEPS + 1, 3, device=dev)  # per arm: imagined root positions, then the goal
        self.ok = torch.zeros(n, 2, dtype=torch.bool, device=dev)  # the path belongs to the arm's current object
        self.t_plan = torch.full((n, 2), -(10**6), dtype=torch.long, device=dev)
        self.assigned_at_plan = torch.full((n, 2), -1, dtype=torch.long, device=dev)
        self.p_next = torch.zeros(n, num_items, device=dev)  # P(object is moved next | task) at the last plan
        self.goal_root = torch.zeros(n, num_items, 3, device=dev)
        self.goal_dest = torch.full((n, num_items), -1, dtype=torch.long, device=dev)  # scene index, -1 = table
        self.goal_centre = torch.zeros(n, num_items, 3, device=dev)
        self.dest_stats = dict(n=0, container=0, right=0)
        self.cost = dict(calls=0, envs=0, seconds=0.0)
        self.reset_calls = 0
        print(
            f"teacher {env.scene_json['episode']}: {len(ids)} objects; items without a token: "
            f"{[env.item_names[i] for i in range(num_items) if int(self.of_item[i]) < 0]}; "
            f"ensemble {len(self.wm.models)} x {samples} samples",
            flush=True,
        )

    def _encode(self, ids: torch.Tensor, pc: torch.Tensor, qc: torch.Tensor):
        """World-model batch and per-model encodings for envs `ids`, plus P(next) averaged over the ensemble."""
        n = len(ids)
        b = self.wm.batch(pc[ids][:, self.k_of], yaw_of(qc[ids][:, self.k_of]), None)
        encodings = self.wm.encode(b)
        p_next = []
        for m, e in zip(self.wm.models, encodings, strict=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mover_logits, _ = m.pointers(e, torch.zeros(n, dtype=torch.long, device=pc.device))
            p_next.append(torch.softmax(mover_logits.float(), -1)[:, : self.wm.I])
        return b, encodings, torch.stack(p_next).mean(0)

    def _set_p_next(self, ids, p_next):
        oi = self.of_item.clamp(min=0)
        self.p_next[ids] = torch.where(self.of_item[None] >= 0, p_next[:, oi], torch.zeros_like(p_next[:, oi]))

    @torch.no_grad()
    def set_goals(self, ids: torch.Tensor) -> None:
        """Goal of every item in envs `ids`: the teacher's destination for it and the top mode of its end pose."""
        env, wm = self.env, self.wm
        pc, qc = env.object_state()
        n, num_items, dev = len(ids), env.num_items, env.device
        b, encodings, p_next = self._encode(ids, pc, qc)
        ends = torch.zeros(n, num_items, 3, device=dev)
        dest = torch.full((n, num_items), -1, dtype=torch.long, device=dev)
        for i in range(num_items):
            wi = int(self.of_item[i])
            if wi < 0:
                continue
            mover = torch.full((n,), wi, dtype=torch.long, device=dev)
            d = wm.destination(encodings, mover)
            ends[:, i] = wm.end_point(b, encodings, mover, d)
            dest[:, i] = torch.where(d < wm.I, self.k_of[d.clamp(max=wm.I - 1)], torch.full_like(d, -1))
        height = env.extents[env.item_obj, 2][None]
        root = ends.clone()
        root[..., 2] = (ends[..., 2] - height).clamp(min=0.0)
        for i in range(num_items):
            if int(self.of_item[i]) < 0:
                root[:, i] = pc[ids, env.item_dst[i]] + torch.tensor([0, 0, FALLBACK_LIFT], device=dev)
        self.goal_root[ids] = root
        self.goal_dest[ids] = dest
        centre = root.clone()
        centre[..., 2] = root[..., 2] + height / 2
        self.goal_centre[ids] = centre
        self._set_p_next(ids, p_next)
        self.reset_calls += 1
        if self.reset_calls == 1:
            e0 = int(ids[0])
            for i in range(num_items):
                kc = int(env.item_dst[i])
                print(
                    f"teacher goal env {e0} {env.item_names[i]}: "
                    f"start {pc[e0, env.item_obj[i]].cpu().numpy().round(3)} -> "
                    f"{root[0, i].cpu().numpy().round(3)} (labelled container {env.names[kc]} at "
                    f"{pc[e0, kc].cpu().numpy().round(3)}), P(next) {float(self.p_next[e0, i]):.2f}",
                    flush=True,
                )

    @torch.no_grad()
    def update_paths(self, pc: torch.Tensor, qc: torch.Tensor) -> None:
        """Re-plan the carry path of every arm that got a new object, is due, or has its object off the path."""
        env, wm = self.env, self.wm
        assigned, t, num_items = env.assigned, env.t, env.num_items
        obj = pc[env.arange[:, None], env.item_obj[assigned.clamp(max=num_items - 1)]]
        off = (self.path - obj[:, :, None]).norm(dim=-1).amin(-1) > self.offpath_dist
        age = t[:, None] - self.t_plan
        due = (assigned != self.assigned_at_plan) | (age >= self.replan_every)
        due |= off & self.ok & (age >= self.offpath_gap)  # off-path re-plans at most once per gap (they are costly)
        due &= assigned < num_items
        eidx = torch.nonzero(due.any(1))[:, 0]
        if len(eidx) == 0:
            return
        t0 = time.time()
        n, dev = len(eidx), env.device
        b, encodings, p_next = self._encode(eidx, pc, qc)
        self._set_p_next(eidx, p_next)
        arn = torch.arange(n, device=dev)
        lin = torch.linspace(0, 1, PATH_STEPS, device=dev)
        for ai in range(2):
            item = assigned[eidx, ai].clamp(max=num_items - 1)
            mover, labelled = self.of_item[item], self.of_dest[item]
            ok = (mover >= 0) & (assigned[eidx, ai] < num_items)
            mover = mover.clamp(min=0)
            dest = wm.destination(encodings, mover)
            to_container = (dest < wm.I) & self.is_container[dest.clamp(max=wm.I - 1)]
            right = to_container & (self.k_of[dest.clamp(max=wm.I - 1)] == env.item_dst[item])
            self.dest_stats["n"] += int(ok.sum())
            self.dest_stats["container"] += int((to_container & ok).sum())
            self.dest_stats["right"] += int((right & ok).sum())
            target = torch.where(to_container, dest, labelled)
            target = torch.where(target >= 0, target, torch.full_like(target, TABLE))
            goal = self.goal_root[eidx, item]
            samples = []
            for m, e in zip(wm.models, encodings, strict=True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    c, _, _ = m.heads_from(e, mover, target)
                    samples.append(m.sample_paths_anchored(e, c, K=wm.K).float()[..., :3])
            # median over all samples of the ensemble; the paths follow the object's top centre
            path = torch.cat(samples, 1).median(1).values + b["obj_geo"][arn, mover, :3][:, None]
            path = torch.nn.functional.avg_pool1d(
                path.transpose(1, 2), SMOOTH_WINDOW, 1, SMOOTH_WINDOW // 2, count_include_pad=False
            ).transpose(1, 2)
            path[..., 2] -= self.height[mover][:, None]
            path = path + lin[None, :, None] * (goal - path[:, -1])[:, None]  # end exactly at the goal
            self.path[eidx, ai] = torch.cat([path, goal[:, None]], 1)
            self.ok[eidx, ai] = ok
            self.t_plan[eidx, ai] = t[eidx]
            self.assigned_at_plan[eidx, ai] = assigned[eidx, ai]
        torch.cuda.synchronize()
        self.cost["calls"] += 1
        self.cost["envs"] += n
        self.cost["seconds"] += time.time() - t0
