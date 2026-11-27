"""Data-quality report for a collected run and plotting helpers (matplotlib optional)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .model import N


def load_run(run_dir: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    with np.load(run_dir / "log.npz") as z:
        log = {k: z[k] for k in z.files}
    return meta, log


def quality(run_dir: str | Path) -> tuple[bool, str]:
    """Checks that decide whether the data is usable. Writes quality.txt / quality.json."""
    run_dir = Path(run_dir)
    meta, a = load_run(run_dir)
    lines, checks = [], []
    cap = np.array(meta["torque_cap"], float)
    rate = float(meta["rate_hz"])
    t = a["t_cmd"]
    if len(t) < 2:
        return False, "no data"
    dt = np.diff(t)
    med, p99, mx = np.median(dt), np.percentile(dt, 99), dt.max()
    lines.append(f"loop period: median {med * 1e3:.2f} ms ({1 / med:.0f} Hz, target {rate:.0f} Hz), "
                 f"p99 {p99 * 1e3:.2f} ms, max {mx * 1e3:.1f} ms")
    checks.append(("loop rate >= 90 % of target", 1 / med >= 0.9 * rate))
    checks.append(("p99 loop period < 3 x target", p99 < 3.0 / rate))

    exc = [s for s in meta["segments"] if s["kind"] == "excite"]
    planned = meta.get("planned_excitation", [s["name"] for s in exc])
    done = [s["name"] for s in exc if s["complete"]]
    checks.append((f"all planned excitations complete ({len(done)}/{len(planned)})", set(planned) <= set(done)))
    checks.append(("at least one train and one val segment",
                   any(s["role"] == "train" and s["complete"] for s in exc) and
                   any(s["role"] == "val" and s["complete"] for s in exc)))
    checks.append(("no abort", meta.get("abort") is None))

    lines.append("")
    lines.append(f"{'segment':15s} {'role':5s} {'done':5s} {'missing %':>9s} {'rms err deg (j1..j6)':>40s} "
                 f"{'peak tau/cap':>12s} {'sat %':>6s} {'max temp C':>10s}")
    worst_miss = worst_sat = 0.0
    for s in exc:
        sl = slice(s["i0"], s["i1"])
        q_cmd, q = a["q_cmd"][sl], a["fb_q"][sl]
        if len(q) == 0:
            continue
        miss = float(np.isnan(q).mean() * 100)
        rms = np.sqrt(np.nanmean((q_cmd - q) ** 2, axis=0))
        tau = np.abs(a["fb_tau"][sl])
        peak = float(np.nanmax(tau / cap))
        sat = float(np.nanmean(tau > 0.95 * cap) * 100)
        temp = float(np.nanmax(a["fb_temp"][sl]))
        worst_miss, worst_sat = max(worst_miss, miss), max(worst_sat, sat)
        lines.append(f"{s['name']:15s} {s['role']:5s} {str(s['complete']):5s} {miss:9.2f} "
                     f"{' '.join(f'{math.degrees(x):5.2f}' for x in rms):>40s} {peak:12.2f} {sat:6.2f} {temp:10.1f}")
    checks.append(("missing feedback < 1 %", worst_miss < 1.0))
    checks.append(("torque saturation < 0.5 % of samples", worst_sat < 0.5))

    ok = all(c[1] for c in checks)
    lines.append("")
    lines += [f"[{'OK' if c[1] else 'FAIL'}] {c[0]}" for c in checks]
    if meta.get("abort"):
        lines.append(f"abort: {meta['abort']}")
    lines.append("")
    lines.append("DATA OK: ready for scripts/sysid/fit.py" if ok else
                 "DATA NOT OK: see the FAIL lines (partial runs can still be fitted with --allow-partial)")
    text = "\n".join(lines)
    (run_dir / "quality.txt").write_text(text + "\n")
    (run_dir / "quality.json").write_text(json.dumps({"ok": ok, "checks": [[c[0], bool(c[1])] for c in checks]},
                                                     indent=2))
    return ok, text


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_run(run_dir: str | Path) -> list[Path]:
    """One figure per excitation segment: command vs measured, and the tracking error."""
    plt = _plt()
    if plt is None:
        return []
    run_dir = Path(run_dir)
    meta, a = load_run(run_dir)
    out = []
    for s in meta["segments"]:
        if s["kind"] != "excite" or s["i1"] - s["i0"] < 10:
            continue
        sl = slice(s["i0"], s["i1"])
        t = a["t_cmd"][sl] - a["t_cmd"][s["i0"]]
        fig, ax = plt.subplots(N, 2, figsize=(14, 13), sharex=True)
        for j in range(N):
            ax[j, 0].plot(t, np.degrees(a["q_cmd"][sl, j]), lw=0.8, label="command")
            ax[j, 0].plot(t, np.degrees(a["fb_q"][sl, j]), lw=0.8, label="measured")
            ax[j, 0].set_ylabel(f"joint{j + 1} [deg]")
            ax[j, 1].plot(t, np.degrees(a["q_cmd"][sl, j] - a["fb_q"][sl, j]), lw=0.6, color="C3")
            ax[j, 1].set_ylabel("error [deg]")
        ax[0, 0].legend(loc="upper right")
        ax[-1, 0].set_xlabel("time [s]")
        ax[-1, 1].set_xlabel("time [s]")
        fig.suptitle(f"{s['name']} ({s['role']})")
        fig.tight_layout()
        p = run_dir / f"segment_{s['name']}.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        out.append(p)
    return out


def plot_fit(out_dir: Path, name: str, t_meas: np.ndarray, q_meas: np.ndarray, t_sim: np.ndarray,
             sims: dict[str, np.ndarray]) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(N, 1, figsize=(12, 13), sharex=True)
    for j in range(N):
        ax[j].plot(t_meas[:, j], np.degrees(q_meas[:, j]), lw=1.2, color="k", label="real")
        for i, (lab, q) in enumerate(sims.items()):
            ax[j].plot(t_sim, np.degrees(q[:, j]), lw=0.8, color=f"C{i}", label=lab)
        ax[j].set_ylabel(f"joint{j + 1} [deg]")
    ax[0].legend(loc="upper right")
    ax[-1].set_xlabel("time [s]")
    fig.suptitle(f"{name}: real vs simulated joint positions")
    fig.tight_layout()
    p = out_dir / f"fit_{name}.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def plot_loss(out_dir: Path, history: list[float]) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.degrees(np.sqrt(history)))
    ax.set_xlabel("generation")
    ax.set_ylabel("best RMS position error [deg]")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fit_loss.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p
