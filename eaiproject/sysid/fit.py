"""Module 2: fit simulator parameters to a collected run with CMA-ES (PACE, Bjelonic et al.).

p = [armature, damping, frictionloss, bias] per joint (6 x 4) + one command delay = 25 parameters,
normalised to [-1, 1] inside their bounds (model.BOUNDS), CMA-ES started at the centre of the box.
Objective: time-averaged squared joint-position error over all training windows (rad^2).
The validation window (held-out multisine) is never used by the optimiser.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .model import ARM_JOINTS, N, PER_JOINT, ArmModel, SimParams
from .replay import Replayer, Window
from .report import load_run, plot_fit, plot_loss


def windows_from_run(meta: dict, log: dict, roles: tuple[str, ...], allow_partial: bool = False,
                     names: list[str] | None = None) -> list[Window]:
    out = []
    for s in meta["segments"]:
        if s["kind"] != "excite" or s["role"] not in roles or (names and s["name"] not in names):
            continue
        if not s["complete"] and not allow_partial:
            continue
        sl = slice(s["i0"], s["i1"])
        t_cmd = log["t_cmd"][sl]
        if len(t_cmd) < 2 or t_cmd[-1] - t_cmd[0] < 3.0:
            continue
        q_fb = log["fb_q"][sl]
        q0 = np.nanmean(q_fb[:5], axis=0)
        if not np.all(np.isfinite(q0)):
            q0 = log["q_cmd"][s["i0"]]
        out.append(Window(s["name"] + ("" if s["complete"] else "_partial"), s["role"], t_cmd, log["q_cmd"][sl],
                          log["qd_cmd"][sl], log["tau_ff"][sl], q0, log["fb_t"][sl], q_fb))
    return out


def _table(rows: list[tuple[str, np.ndarray]], header: str) -> str:
    n = max(28, *(len(name) for name, _ in rows))
    lines = [f"{header:{n}s} " + " ".join(f"{j:>8s}" for j in ARM_JOINTS) + f" {'mean':>8s}"]
    for name, v in rows:
        lines.append(f"{name:{n}s} " + " ".join(f"{x:8.3f}" for x in v) + f" {np.mean(v):8.3f}")
    return "\n".join(lines)


def params_table(ps: dict[str, SimParams]) -> str:
    lines = []
    for k, unit in (("armature", "kg m^2"), ("damping", "N m s/rad"), ("frictionloss", "N m"), ("bias", "deg")):
        rows = [(f"{k} [{unit}] {lab}", np.degrees(getattr(p, k)) if k == "bias" else getattr(p, k))
                for lab, p in ps.items()]
        lines.append(_table(rows, "parameter"))
    lines.append("delay [ms]: " + ", ".join(f"{lab} {1000 * p.delay:.2f}" for lab, p in ps.items()))
    return "\n\n".join(lines)


class Runs:
    """The windows of one or more runs, each replayed under its own run's torque cap (the gains must be
    the same: the fitted parameters describe the arm under one controller)."""

    def __init__(self, arm: ArmModel, groups: list[tuple[list[Window], np.ndarray]], kp, kd, nbatch: int, nthread: int):
        self.reps = [Replayer(arm, ws, kp, kd, cap, nbatch=nbatch, nthread=nthread) for ws, cap in groups if ws]
        self.windows = [(r, wi) for r in self.reps for wi in range(len(r.windows))]

    def loss(self, params: list[SimParams]) -> np.ndarray:
        """PACE objective over every window of every run: time-averaged squared position error, rad^2."""
        sse, cnt = 0.0, 0.0
        for r, wi in self.windows:
            e, c = r.errors(params, wi)
            sse, cnt = sse + e.sum(axis=1), cnt + c.sum()
        return sse / max(cnt, 1.0)

    def close(self) -> None:
        for r in self.reps[:1]:
            r.close()


def fit(run_dir, out_dir: str | Path | None = None, gens: int = 300, pop: int = 32,
        threads: int | None = None, sigma: float = 0.3, seed: int = 0, allow_partial: bool = False,
        patience: int = 40, eval_params: str | Path | None = None) -> Path:
    """`run_dir`: one run directory, or several to fit together (more data: other amplitudes, speeds,
    days). With several, each window is named <segment>_<run> and the output goes to the first run."""
    from cmaes import CMA

    run_dirs = [Path(r) for r in ([run_dir] if isinstance(run_dir, (str, Path)) else run_dir)]
    runs = [load_run(r) for r in run_dirs]
    meta = runs[0][0]
    out_dir = Path(out_dir) if out_dir else run_dirs[0] / f"fit_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kp, kd = (np.array(meta[k], float) for k in ("kp", "kd"))
    arm = ArmModel()
    train_g, val_g = [], []
    for r, (m, log) in zip(run_dirs, runs):
        if not (np.allclose(m["kp"], kp) and np.allclose(m["kd"], kd)):
            raise SystemExit(f"{r.name} was recorded under other PD gains than {run_dirs[0].name}: fit them separately")
        cap = np.array(m["torque_cap"], float)
        for group, role in ((train_g, "train"), (val_g, "val")):
            ws = windows_from_run(m, log, (role,), allow_partial)
            for w in ws if len(run_dirs) > 1 else []:
                w.name = f"{w.name}_{r.name}"
            group.append((ws, cap))
    train, val = [w for ws, _ in train_g for w in ws], [w for ws, _ in val_g for w in ws]
    if not train:
        raise SystemExit("no complete training segment in this run (try --allow-partial)")
    threads = threads or os.cpu_count() or 1
    print(f"run {', '.join(r.name for r in run_dirs)} ({meta['mode']}): train {[w.name for w in train]}, "
          f"val {[w.name for w in val]}")
    print(f"train data {sum(w.duration for w in train):.0f} s, physics dt {arm.dt * 1e3:.1f} ms, "
          f"population {pop}, {threads} threads")

    history: list[float] = []
    if eval_params:
        best = SimParams.load(eval_params)
        print(f"evaluating {eval_params} (no optimisation)")
    else:
        rep = Runs(arm, train_g, kp, kd, nbatch=pop, nthread=threads)
        lo, hi = SimParams.bounds()
        opt = CMA(mean=np.zeros(len(lo)), sigma=sigma, bounds=np.tile([-1.0, 1.0], (len(lo), 1)),
                  population_size=pop, seed=seed)
        best_x, best_f, since = None, np.inf, 0
        t_start = time.monotonic()
        for g in range(gens):
            xs = [opt.ask() for _ in range(opt.population_size)]
            fs = rep.loss([SimParams.from_normalized(x) for x in xs])
            opt.tell(list(zip(xs, fs.tolist())))
            i = int(np.argmin(fs))
            if fs[i] < best_f * (1 - 1e-3):
                since = 0
            else:
                since += 1
            if fs[i] < best_f:
                best_f, best_x = float(fs[i]), xs[i].copy()
            history.append(best_f)
            el = time.monotonic() - t_start
            if g == 0 or (g + 1) % 10 == 0:
                eta = el / (g + 1) * (gens - g - 1)
                print(f"  gen {g + 1:4d}/{gens}  best RMS {math.degrees(math.sqrt(best_f)):.3f} deg  "
                      f"sigma {opt._sigma:.3f}  {el:5.0f} s elapsed, <= {eta:5.0f} s left", flush=True)
            if since >= patience or opt.should_stop():
                print(f"  stopped at gen {g + 1}: no improvement > 0.1 % for {since} generations"
                      if since >= patience else f"  stopped at gen {g + 1}: CMA-ES converged")
                break
        rep.close()
        best = SimParams.from_normalized(best_x)
        x_edge = np.abs(best_x) > 0.98
        if x_edge.any():
            names = [f"{k}[{ARM_JOINTS[i]}]" for k in PER_JOINT for i in range(N)] + ["delay"]
            print("  note: at the search-box edge: " + ", ".join(n for n, e in zip(names, x_edge) if e))

    # ------------------------------------------------------------------ evaluation
    nominal = arm.nominal
    ev = Runs(arm, [(t + v, cap) for (t, cap), (v, _) in zip(train_g, val_g)], kp, kd, nbatch=2, nthread=2)
    rows, summary = [], {}
    for r, wi in ev.windows:
        w = r.windows[wi]
        r_nom, r_fit = (np.degrees(r.rmse(p, wi)) for p in (nominal, best))
        rows += [(f"{w.name} ({w.role}) nominal", r_nom), (f"{w.name} ({w.role}) fitted", r_fit)]
        summary[w.name] = {"role": w.role, "rmse_deg_nominal": r_nom.tolist(), "rmse_deg_fitted": r_fit.tolist()}
        q, _ = r.simulate([nominal, best], wi)
        plot_fit(out_dir, w.name, w.t_fb - w.t_cmd[0], w.q_fb, r.prep[wi]["t_state"] - w.t_cmd[0],
                 {"sim nominal": q[0], "sim fitted": q[1]})
    ev.close()
    plot_loss(out_dir, history) if history else None

    best.info = {"source_run": ", ".join(str(r) for r in run_dirs), "mode": meta["mode"],
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "kp": kp.tolist(), "kd": kd.tolist(), "torque_cap": [c.tolist() for _, c in train_g][0],
                 "train": [w.name for w in train], "val": [w.name for w in val],
                 "loss_rad2": history[-1] if history else None, "generations": len(history), "rmse": summary}
    best.save(out_dir / "sim_params.json")

    text = ["RMS joint-position error, deg (lower is better)", _table(rows, "window"), ""]
    ps = {"nominal": nominal, "fitted": best}
    if meta.get("sim_truth"):
        ps["truth"] = SimParams.from_json(meta["sim_truth"])
    text += ["Parameters", params_table(ps)]
    text = "\n".join(text)
    (out_dir / "fit_report.txt").write_text(text + "\n")
    (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(text)
    print(f"\nartifact: {out_dir / 'sim_params.json'}")
    return out_dir
