#!/usr/bin/env python3
"""PDF report: existing CyberScape system (B0, no RL) vs heuristic (B1) vs PPO.

Covers the metrics of the "Experimental Design & Quantitative Comparison" slide from the
outputs of run_comparison_study.sh (eval_timing.py runs + CyberScape mission timing)
and the PPO training log. Needs matplotlib + reportlab:

  PYTHONPATH=<dir with reportlab> python make_comparison_report.py [--stamp cmp_YYYYmmdd_HHMMSS]

Writes results/comparison_report_<stamp>.pdf and the chart PNGs next to it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mission_timing import compare_paired  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONTROLLERS = ("b0", "rule", "ppo")
LABEL = {"b0": "B0 existing (LLM, no RL)", "rule": "B1 heuristic", "ppo": "PPO (RL)"}
SHORT = {"b0": "B0", "rule": "B1", "ppo": "PPO"}
# Reference categorical palette, slots 1-3 in fixed order (validated all-pairs for 3 series).
COLOR = {"b0": "#2a78d6", "rule": "#eb6834", "ppo": "#1baf7a"}
STATUS_ORDER = ("success", "collision", "timeout", "failed")
STATUS_COLOR = {"success": "#1baf7a", "collision": "#e34948", "timeout": "#eda100", "failed": "#8a8986"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONDITIONS = ("nominal", "generalization", "robustness")
CONDITION_TEXT = {
    "nominal": "seen-type layouts, V3 sensor model",
    "generalization": "new layouts + random start heading +-90 deg",
    "robustness": "15% delayed commands + doubled LiDAR noise/dropout",
}
BUDGET_S = 120.0


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load_rows(stamp: str) -> dict[str, list[dict]]:
    data = {}
    for condition in CONDITIONS:
        path = RESULTS / "timing" / f"{stamp}_{condition}" / "episodes.csv"
        if path.exists():
            data[condition] = list(csv.DictReader(path.open()))
    return data


def f(value, digits=1, suffix=""):
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def metrics(rows: list[dict], controller: str) -> dict:
    mine = [r for r in rows if r["controller"] == controller]
    n = len(mine)
    ok = [r for r in mine if r["success"] == "True"]
    times = [float(r["makespan_sim_s"]) for r in ok]
    # distance actually closed / distance driven (the 5 cm goal tolerance cannot push it above 1)
    eff = [(float(r["goal_distance_m"]) - float(r["final_distance_m"])) / float(r["path_length_m"])
           for r in ok if float(r["path_length_m"]) > 0]
    path = [float(r["path_length_m"]) for r in mine]
    rev_per_m = [float(r["turn_reversals"]) / max(float(r["path_length_m"]), 0.05) for r in mine]
    lat = [float(r["inference_p95_ms"]) for r in mine if r["inference_p95_ms"]]
    return {
        "n": n,
        "success": len(ok) / n if n else None,
        "collision": sum(r["status"] == "collision" for r in mine) / n if n else None,
        "timeout": sum(r["status"] == "timeout" for r in mine) / n if n else None,
        "failed": sum(r["status"] == "failed" for r in mine) / n if n else None,
        "time_mean": mean(times) if times else None,
        "time_median": median(times) if times else None,
        "penalized": mean(float(r["makespan_sim_s"]) if r["success"] == "True" else BUDGET_S for r in mine) if n else None,
        "return": mean(float(r["team_return"]) for r in mine) if n else None,
        "efficiency": mean(eff) if eff else None,
        "path": mean(path) if path else None,
        "reversals_per_m": mean(rev_per_m) if rev_per_m else None,
        "decisions": mean(float(r["decisions"]) for r in mine) if n else None,
        "latency_p95": median(lat) if lat else None,
        "rtf": median(float(r["rtf"]) for r in mine if r["rtf"]) if n else None,
    }


# ------------------------------------------------------------------ charts
def chart_outcomes(data, out: Path) -> Path:
    fig, axes = plt.subplots(1, len(data), figsize=(9.5, 3.0), sharey=True)
    axes = axes if len(data) > 1 else [axes]
    for ax, (condition, rows) in zip(axes, data.items()):
        x = range(len(CONTROLLERS))
        bottoms = [0.0] * len(CONTROLLERS)
        for status in STATUS_ORDER:
            vals = []
            for c in CONTROLLERS:
                mine = [r for r in rows if r["controller"] == c]
                vals.append(100 * sum(r["status"] == status for r in mine) / len(mine) if mine else 0)
            ax.bar(x, vals, bottom=bottoms, width=0.6, color=STATUS_COLOR[status], edgecolor="white",
                   linewidth=2, label=status)
            for i, v in enumerate(vals):
                if status == "success" and v >= 8:
                    ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold")
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(list(x), [SHORT[c] for c in CONTROLLERS])
        ax.set_title(f"{condition} (n={len(rows) // len(CONTROLLERS)})", fontsize=9, color=INK)
        style_axes(ax)
    axes[0].set_ylabel("% of episodes", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def chart_times(data, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    positions, labels = [], []
    pos = 0
    for condition, rows in data.items():
        for c in CONTROLLERS:
            times = [float(r["makespan_sim_s"]) for r in rows if r["controller"] == c and r["success"] == "True"]
            if times:
                jitter = [pos + (i % 7 - 3) * 0.05 for i in range(len(times))]
                ax.scatter(jitter, times, s=14, color=COLOR[c], alpha=0.85, linewidths=0)
                ax.plot([pos - 0.25, pos + 0.25], [median(times)] * 2, color=INK, linewidth=2)
            ax.text(pos, -9, f"n={len(times)}", ha="center", fontsize=7, color=INK2)
            positions.append(pos)
            labels.append(f"{SHORT[c]}\n{condition[:7]}")
            pos += 1
        pos += 0.6
    ax.set_xticks(positions, labels, fontsize=7)
    ax.set_ylabel("time to goal, successful episodes (sim s)", fontsize=8, color=INK2)
    ax.set_ylim(-14, BUDGET_S)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def chart_generalization(data, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    xs = [c for c in CONDITIONS if c in data]
    ends = {}
    for c in CONTROLLERS:
        ys = [100 * metrics(data[x], c)["success"] for x in xs]
        ax.plot(range(len(xs)), ys, color=COLOR[c], linewidth=2, marker="o", markersize=7)
        ends[c] = ys[-1]
    # direct labels at the line ends, nudged apart so equal values stay readable
    placed = []
    for c in sorted(ends, key=lambda k: ends[k]):
        y = ends[c]
        while any(abs(y - p) < 8 for p in placed):
            y += 8
        placed.append(y)
        ax.text(len(xs) - 1 + 0.1, y, LABEL[c], fontsize=8, color=INK, va="center")
    ax.set_xticks(range(len(xs)), xs, fontsize=8)
    ax.set_xlim(-0.2, len(xs) + 0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("success rate (%)", fontsize=8, color=INK2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def learning_curve(out: Path):
    logs = sorted(RESULTS.glob("train_isaac_point_nav_lidar_ppo_v3.log.*")) + [RESULTS / "train_isaac_point_nav_lidar_ppo_v3.log"]
    points = []
    for log in logs:
        if not log.exists():
            continue
        for line in log.open(errors="replace"):
            if "[TB03_PROGRESS]" in line:
                m = dict(re.findall(r"(\w+)=(\S+)", line))
                points.append((int(m["steps"]), m.get("outcome")))
    if not points:
        return None, {}
    points.sort()
    window = 500
    xs, ys = [], []
    for i in range(window, len(points) + 1, 100):
        chunk = points[i - window:i]
        xs.append(chunk[-1][0] / 1000)
        ys.append(100 * sum(o == "SUCCESS" for _, o in chunk) / window)
    blocks = [points[i:i + 1000] for i in range(0, len(points), 1000)]
    block_rates = [100 * sum(o == "SUCCESS" for _, o in b) / len(b) for b in blocks if len(b) >= 500]
    late = [y for x, y in zip(xs, ys) if x >= 500]
    stats = {
        "episodes": len(points), "steps": points[-1][0],
        "final_success": ys[-1] if ys else None,
        "late_mean": mean(late) if late else None, "late_std": pstdev(late) if len(late) > 1 else None,
        "block_rates": block_rates,
    }
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(xs, ys, color=COLOR["ppo"], linewidth=2)
    ax.axvline(1000, color=INK2, linewidth=0.8, linestyle="--")
    ax.text(1005, 5, "fine-tune (lr 1e-4)", fontsize=7, color=INK2)
    ax.set_xlabel("training steps (thousands)", fontsize=8, color=INK2)
    ax.set_ylabel("success, last 500 episodes (%)", fontsize=8, color=INK2)
    ax.set_ylim(0, 100)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out, stats


def mission_rows(stamp: str) -> list[dict]:
    path = RESULTS / "missions" / "missions.csv"
    if not path.exists():
        return []
    start = dt.datetime.strptime(stamp[4:], "%Y%m%d_%H%M%S")
    rows = []
    for r in csv.DictReader(path.open()):
        try:
            when = dt.datetime.strptime(r["mission_id"][:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if when >= start and r["mission_id"].count("_") >= 3:
            rows.append(r)
    return rows


# ------------------------------------------------------------------ statistics
def wilson(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion (as fractions)."""
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(rows, base: str, cand: str):
    """Paired success comparison on matched scenarios: discordant pairs + exact two-sided binomial p."""
    by = {}
    for r in rows:
        by.setdefault(r["scenario_id"], {})[r["controller"]] = r["success"] == "True"
    only_cand = sum(1 for v in by.values() if v.get(cand) and v.get(base) is False)
    only_base = sum(1 for v in by.values() if v.get(base) and v.get(cand) is False)
    n = only_cand + only_base
    if n == 0:
        return only_cand, only_base, 1.0
    k = min(only_cand, only_base)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return only_cand, only_base, min(1.0, 2 * p)


# ------------------------------------------------------------------ more charts
def chart_by_obstacles(rows, out: Path) -> Path:
    counts = sorted({int(r["obstacles"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    width = 0.26
    for k, c in enumerate(CONTROLLERS):
        ys, ns = [], []
        for n_obs in counts:
            mine = [r for r in rows if r["controller"] == c and int(r["obstacles"]) == n_obs]
            ns.append(len(mine))
            ys.append(100 * sum(r["success"] == "True" for r in mine) / len(mine) if mine else 0)
        xs = [i + (k - 1) * (width + 0.02) for i in range(len(counts))]
        ax.bar(xs, ys, width=width, color=COLOR[c], label=LABEL[c], edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(counts)),
                  [f"{n} obstacle{'s' if n != 1 else ''}\n(n={sum(1 for r in rows if r['controller'] == 'ppo' and int(r['obstacles']) == n)})"
                   for n in counts], fontsize=7.5)
    ax.set_ylabel("success (%)", fontsize=8, color=INK2)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def chart_strip(data, key: str, ylabel: str, out: Path, per_metre: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 2.7))
    pos, positions, labels = 0, [], []
    for condition, rows in data.items():
        for c in CONTROLLERS:
            vals = []
            for r in rows:
                if r["controller"] != c:
                    continue
                v = float(r[key])
                if per_metre:
                    v = v / max(float(r["path_length_m"]), 0.05)
                vals.append(v)
            if vals:
                ax.scatter([pos + (i % 7 - 3) * 0.05 for i in range(len(vals))], vals, s=12, color=COLOR[c],
                           alpha=0.85, linewidths=0)
                ax.plot([pos - 0.25, pos + 0.25], [median(vals)] * 2, color=INK, linewidth=2)
            positions.append(pos)
            labels.append(f"{SHORT[c]}\n{condition[:7]}")
            pos += 1
        pos += 0.6
    ax.set_xticks(positions, labels, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def mission_events(row: dict):
    path = RESULTS / "missions" / row["mission_id"] / "events.jsonl"
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.open()]
    starts, spans = {}, []
    for e in events:
        if e["event"] == "task_start":
            starts[e["task"]] = e["sim_s"]
        elif e["event"] in ("task_complete", "task_failed") and e["task"] in starts:
            spans.append((e["task"], starts[e["task"]], e["sim_s"], e["event"] == "task_complete"))
    return spans


def chart_mission_timeline(missions, out: Path):
    groups = [r for r in missions if mission_events(r)]
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(groups) + 1.0))
    labels = []
    for i, r in enumerate(groups):
        for task, t0, t1, ok in mission_events(r):
            moving = "move_to_point" in task or "move_forward" in task or "return_to_base" in task
            color = COLOR.get(r["controller"], "#888") if moving else "#bdbcb6"
            ax.barh(i, max(t1 - t0, 0.6), left=t0, height=0.6, color=color if ok else "#e34948",
                    edgecolor="white", linewidth=1)
        labels.append(f"{r['mission_id'].split('_')[2]} - {SHORT.get(r['controller'], r['controller'])}")
    ax.set_yticks(range(len(groups)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("mission time (sim s); coloured = driving commands, grey = scan / detect / rotate / report",
                  fontsize=7.5, color=INK2)
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ PDF
def build_pdf(stamp: str, out_pdf: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    data = load_rows(stamp)
    if not data:
        raise SystemExit(f"no results for {stamp}")
    figdir = out_pdf.parent / f"{out_pdf.stem}_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    nom = data.get("nominal", next(iter(data.values())))
    figs = {
        "outcomes": chart_outcomes(data, figdir / "outcomes.png"),
        "times": chart_times(data, figdir / "times.png"),
        "general": chart_generalization(data, figdir / "generalization.png"),
        "obstacles": chart_by_obstacles(nom, figdir / "by_obstacles.png"),
        "path": chart_strip(data, "path_length_m", "path length driven (m)", figdir / "path.png"),
        "reversals": chart_strip(data, "turn_reversals", "turn reversals per metre", figdir / "reversals.png", per_metre=True),
    }
    figs["learn"], learn = learning_curve(figdir / "learning_curve.png")
    missions = mission_rows(stamp)
    figs["timeline"] = chart_mission_timeline(missions, figdir / "mission_timeline.png")
    manifests = {c: json.loads((RESULTS / "timing" / f"{stamp}_{c}" / "manifest.json").read_text())
                 for c in data if (RESULTS / "timing" / f"{stamp}_{c}" / "manifest.json").exists()}

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.2, leading=12.6, alignment=TA_LEFT, spaceAfter=3)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10, textColor=colors.HexColor(INK2))
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=21, leading=25, textColor=colors.HexColor("#123a5c"),
                           alignment=TA_LEFT)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, textColor=colors.HexColor("#123a5c"), spaceBefore=4,
                        spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, textColor=colors.HexColor("#1b6b74"),
                        spaceBefore=9, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.6, leading=9.3, spaceAfter=0)
    mono = ParagraphStyle("mono", parent=cell, fontName="Courier", fontSize=7.2, leading=9)

    def table(rows, widths, header_rows=1, style=cell):
        t = Table([[c if not isinstance(c, str) else Paragraph(c, style) for c in r] for r in rows],
                  colWidths=widths, repeatRows=header_rows)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e8f1f2")),
            ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, colors.HexColor("#1b6b74")),
            ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, colors.HexColor(GRID)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ]))
        return t

    def bullet(text):
        return Paragraph("&bull;&nbsp;" + text, body)

    def figure(path, width_cm, height_cm, caption):
        return KeepTogether([Image(str(path), width=width_cm * cm, height=height_cm * cm), Paragraph(caption, small),
                             Spacer(1, 4)])

    pct = lambda x, d=0: f(100 * x, d, "%") if x is not None else "-"
    M = {cond: {c: metrics(rows, c) for c in CONTROLLERS} for cond, rows in data.items()}
    m_nom = M.get("nominal", next(iter(M.values())))
    n_nom = len(nom) // len(CONTROLLERS)

    def success_ci(rows, c):
        mine = [r for r in rows if r["controller"] == c]
        lo, hi = wilson(sum(r["success"] == "True" for r in mine), len(mine))
        return f"{pct(lo)}-{pct(hi)}" if lo is not None else "-"

    pair = {(b, c): compare_paired(nom, b, c, budget_s=BUDGET_S) for b, c in (("b0", "ppo"), ("rule", "ppo"), ("b0", "rule"))}
    signs = {k: sign_test(nom, *k) for k in pair}

    def verdict_time(p):
        lo, hi = p["delta_ci95_s"] or (None, None)
        if lo is None:
            return "inconclusive (too few scenarios solved by both)"
        if lo > 0:
            return "supported: the candidate is faster (whole 95% CI above 0)"
        if hi < 0:
            return "not supported: the candidate is slower (whole 95% CI below 0)"
        return "inconclusive: the 95% CI includes 0"

    def verdict_success(key):
        only_c, only_b, p = signs[key]
        better = "PPO" if key[1] == "ppo" else SHORT[key[1]]
        if p < 0.05:
            return (f"significant: {better} solved {only_c} scenarios the other did not, vs {only_b} the other way "
                    f"(exact sign test p = {p:.3f})") if only_c > only_b else (
                f"significant in the opposite direction: {only_b} vs {only_c} (p = {p:.3f})")
        return f"not significant: {only_c} vs {only_b} discordant scenarios (p = {p:.2f})"

    # ------------------------------------------------------------ cover + summary
    story = [
        Paragraph("Experimental Comparison Report", title),
        Paragraph("CyberScape mission execution without RL (original controller) vs. heuristic baseline vs. PPO", h2),
        Paragraph(f"TurtleBot3 Burger &middot; Isaac Sim 5.1 real-size scene &middot; generated {dt.date.today():%d %B %Y} "
                  f"&middot; run <font face='Courier'>{stamp}</font> &middot; AUC Graduation Project, Thesis II", small),
        Spacer(1, 10),
        Paragraph("Executive summary", h1),
        Paragraph("This report compares three ways of executing the movement commands of an approved CyberScape (LLM) "
                  "mission plan on a simulated TurtleBot3: <b>B0</b>, the execution controller the system used before "
                  "reinforcement learning; <b>B1</b>, a hand-coded heuristic; and <b>PPO</b>, the trained reinforcement-learning "
                  "navigation policy. All three drive the same robot in the same simulator on identical, seeded scenarios, "
                  f"and are scored with the metrics of the Experimental Design slide. {sum(len(r) for r in data.values())} "
                  f"navigation episodes and {len(missions)} full CyberScape missions were run.", body),
        Spacer(1, 4),
        table([
            ["Nominal condition", LABEL["b0"], LABEL["rule"], LABEL["ppo"]],
            ["Success rate (95% CI)"] + [f"{pct(m_nom[c]['success'])} ({success_ci(nom, c)})" for c in CONTROLLERS],
            ["Collision rate"] + [pct(m_nom[c]["collision"]) for c in CONTROLLERS],
            ["Timeout / gave up"] + [f"{pct(m_nom[c]['timeout'])} / {pct(m_nom[c]['failed'])}" for c in CONTROLLERS],
            ["Median time to goal (successes, sim s)"] + [f(m_nom[c]["time_median"]) for c in CONTROLLERS],
            ["Failure-penalized mean time (s)"] + [f(m_nom[c]["penalized"]) for c in CONTROLLERS],
            ["Mean team return"] + [f(m_nom[c]["return"], 2) for c in CONTROLLERS],
            ["Turn reversals per metre"] + [f(m_nom[c]["reversals_per_m"], 1) for c in CONTROLLERS],
        ] + ([["Success: generalization / robustness"] + [
            f"{pct(M['generalization'][c]['success']) if 'generalization' in M else '-'} / "
            f"{pct(M['robustness'][c]['success']) if 'robustness' in M else '-'}" for c in CONTROLLERS]]
             if len(M) > 1 else []),
            [5.6 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm]),
        Spacer(1, 6),
        Paragraph("Main findings", h2),
        bullet(f"<b>Reliability.</b> PPO vs B0 on success, paired by scenario: {verdict_success(('b0', 'ppo'))}. "
               f"PPO vs B1: {verdict_success(('rule', 'ppo'))}."),
        bullet(f"<b>Speed.</b> PPO vs B0 completion time on scenarios both solved: {verdict_time(pair[('b0', 'ppo')])} "
               f"(mean &Delta; {f(pair[('b0', 'ppo')]['mean_delta_s'])} s, n = {pair[('b0', 'ppo')]['scenarios_both_successful']}). "
               "B0 drives continuously at up to 0.15 m/s; PPO moves in trained stop-and-settle steps at 0.05 m/s, so "
               "when both succeed B0 tends to be faster by construction."),
        bullet(f"<b>Failure-aware time</b> (every attempt; failures count as the {BUDGET_S:.0f} s budget): "
               f"B0 {f(m_nom['b0']['penalized'])} s, B1 {f(m_nom['rule']['penalized'])} s, PPO {f(m_nom['ppo']['penalized'])} s."),
        bullet("<b>Safety.</b> Collision rates are reported next to every time figure; under the framework's decision rule "
               "a faster controller that collides more is not preferred."),
        bullet("<b>Scope.</b> Single robot. MAPPO (multi-robot) is not implemented yet; one PPO training seed. "
               "See section 9 for all limitations."),
        PageBreak(),
    ]

    # ------------------------------------------------------------ 1 introduction
    story += [
        Paragraph("1. Research question and hypotheses", h1),
        Paragraph("Thesis question: <i>How can LLM-driven mission planning be integrated with learning-based execution to "
                  "improve coordination, efficiency and adaptability in multi-robot systems?</i> This report tests the "
                  "single-robot foundation of that question: does a learned execution layer (PPO) execute an approved LLM "
                  "plan better than the original non-learning executor?", body),
        table([
            ["Hypothesis", "Statement (single-robot form)", "Tested by"],
            ["H1 efficiency", "PPO reduces matched-scenario completion time vs B0 while keeping an acceptable success rate.",
             "Paired &Delta; time with bootstrap CI; failure-penalized mean (sections 4-5)."],
            ["H1b reliability", "PPO reaches more goals than B0 on the same scenarios.",
             "Paired success, exact sign test on discordant scenarios (section 5)."],
            ["H3 adaptability", "PPO keeps its advantage on new layouts / initial headings and under disturbances.",
             "Generalization and robustness conditions (section 6)."],
            ["H2 coordination", "MAPPO reduces idle time and conflicts between robots.",
             "Not testable yet: needs MAPPO and several robots."],
        ], [2.8 * cm, 8.0 * cm, 6.2 * cm]),
        Paragraph("2. System under test", h1),
        Paragraph("CyberScape turns a natural-language mission into a reviewed plan: an LLM writes a high-level plan per robot, "
                  "a second LLM stage translates it into robot commands (e.g. <font face='Courier'>move_to_point((1.5, 0.75))</font>, "
                  "<font face='Courier'>scan_with_lidar()</font>, <font face='Courier'>detect_with_camera('survivor')</font>), "
                  "the plan is verified and parsed, and a human approves it in the HITL dashboard. The approved commands are "
                  "sent one at a time to the robot. <b>Only the execution of movement commands differs between the three "
                  "controllers</b>; the LLM plan, approval flow, robot, simulator and scenarios are identical.", body),
        Paragraph("2.1 B0 &mdash; existing system (LLM planning, no RL)", h2),
        Paragraph("The original <font face='Courier'>BurgerDriver.move_to_point</font> from "
                  "<font face='Courier'>Hardware/Turtlebot3/burger_comm_node.py</font>, ported to Isaac Sim without changing "
                  "its logic or constants (<font face='Courier'>baseline_b0.py</font>):", body),
        table([
            ["Step", "Behaviour", "Constants"],
            ["Heading", "Rotate in place while |heading error| &gt; 0.1 rad", "0.5 rad/s"],
            ["Drive", "v = min(0.15, 0.5&middot;distance), w = 1.5&middot;heading error, 10 Hz", "goal tolerance 0.05 m"],
            ["Blocked", "Front sector (&plusmn;18&deg;) minimum &lt; 0.35 m: stop", "OBSTACLE_STOP_DISTANCE 0.35 m"],
            ["Bypass", "Turn 90&deg; to a random side (50/50); if front &ge; 0.50 m drive 0.25 m; else turn 180&deg; and try "
                       "the other side; 3 attempts", "AVOIDANCE_STEP 0.25 m, CLEAR 0.50 m"],
            ["Give up", "More than 12 avoidance manoeuvres for one goal, or no bypass found", "MAX_AVOIDANCE_ATTEMPTS 12"],
        ], [2.2 * cm, 9.8 * cm, 5.0 * cm]),
        Paragraph("Porting choices (all to share the simulator with the other controllers): commands go through the same "
                  "wheel controller and physics; the front reading is the minimum valid LiDAR range in &plusmn;18&deg;, exactly "
                  "the robot node's <font face='Courier'>_sector_min_distance</font>; the random side uses a seeded generator "
                  "so each scenario is reproducible.", small),
        Paragraph("2.2 B1 &mdash; heuristic baseline", h2),
        Paragraph("Reactive rules on PPO's 6-number observation (goal distance, sin/cos goal bearing, front/left/right "
                  "clearance): if front &lt; 0.35 m turn toward the clearer side and then commit to 6 forward steps; else "
                  "if the goal is more than 0.20 rad off and that side has &ge; 0.25 m clearance, turn toward it; else drive "
                  "forward. Same discrete actions as PPO.", body),
        Paragraph("2.3 PPO &mdash; learned navigation policy", h2),
        table([
            ["Item", "Value"],
            ["Observation (6)", "goal distance / 2 m, sin and cos of goal bearing, front / left / right LiDAR clearance / 1 m "
                                "(front &plusmn;18&deg;, sides 18-120&deg;)"],
            ["Actions (4)", "FORWARD 0.05 m/s for 30 frames (0.5 s); TURN LEFT / RIGHT 0.8 rad/s for 15 frames; STAY; "
                            "then 8 settle frames (60 Hz physics)"],
            ["Reward", "5&middot;clip(progress, &plusmn;0.1 m) &minus; 0.01 per step; +10 goal (5 cm); &minus;10 collision; "
                       "proximity penalty up to 0.3 below 0.30 m"],
            ["Training", "Stable-Baselines3 PPO, MLP 64&times;64, lr 3e-4, &gamma; 0.99, GAE 0.95, entropy 0.01, 256 steps "
                         "per update; 1,000,000 steps + 250,000 fine-tune (lr 1e-4); evaluated checkpoint: 1,000,000"],
            ["Sensor model (V3)", "range noise &sigma; 1.5 cm, 10% ray dropout, 5% sector dropout, minimum range 0.15 m, "
                                  "obstacles 12-40 cm (real-robot lessons)"],
            ["Episode", "start pose fixed, goal 0.5-2 m from a bank of 7, 0-3 obstacles, 250 actions / 120 s max"],
        ], [3.2 * cm, 13.8 * cm]),
        PageBreak(),
        Paragraph("3. Experimental protocol", h1),
        Paragraph("3.1 Scenarios and conditions", h2),
        Paragraph("A scenario is fully determined by its seed: goal position, number, size and placement of obstacles "
                  "(placed with &ge; 0.40 m clearance to the start, &ge; 0.30 m to the goal and &ge; 0.30 m from each other). "
                  "Every controller runs every scenario of a condition; results are compared scenario by scenario.", body),
        table([["Condition", "Seeds", "Episodes per controller", "What changes vs nominal"]] + [
            [c, f"{manifests[c]['scenarios'][0]}-{manifests[c]['scenarios'][-1]}" if c in manifests else "-",
             str(len(data[c]) // len(CONTROLLERS)), CONDITION_TEXT[c] + (
                 "; LiDAR noise &sigma; 3 cm, 20% ray and 10% sector dropout; B0 also gets the range noise" if c == "robustness"
                 else "; the robot first turns in place to the random heading (not timed)" if c == "generalization" else "")]
            for c in CONDITIONS if c in data], [2.6 * cm, 2.4 * cm, 2.6 * cm, 9.4 * cm]),
        Paragraph("Perception: each controller uses its native pipeline on the same simulated LiDAR &mdash; PPO and B1 the V3 "
                  "sensor model PPO was trained with, B0 the raw front-sector minimum of the original node.", small),
        Paragraph("3.2 Clocks and timing (Mission Timing Evaluator)", h2),
        bullet("<b>Simulation clock</b>: Isaac's <font face='Courier'>SimulationContext.current_time</font>, read after "
               "physics steps. All times in this report are simulated seconds. Check: in every episode the clock equals "
               "physics frames &times; dt (1/60 s) to the millisecond."),
        bullet("<b>Wall clock</b>: <font face='Courier'>time.perf_counter_ns</font>; gives the real-time factor "
               "RTF = simulated / wall seconds (RTF &gt; 1 is faster than real time)."),
        bullet("<b>Decision clock</b>: wall time of each policy call (B1, PPO), reported as p50/p95/p99 after a warm-up call."),
        bullet("Episode start t0 = first controller decision after reset (and after the generalization pre-turn). "
               f"Budget: {BUDGET_S:.0f} s simulated for every controller. Collision = robot body within the obstacle's "
               "footprint plus robot radius (0.105 m)."),
        Paragraph("3.3 Metric definitions (Experimental Design slide)", h2),
        table([
            ["Slide metric", "Definition used here"],
            ["Mission success rate", "successful episodes / all episodes; 95% Wilson interval. Missions: all plan commands "
                                     "completed."],
            ["Completion time", "sim seconds from t0 to goal (5 cm), successful episodes only; failure-penalized mean counts "
                                "a failure as the 120 s budget; steps reported as decisions per episode."],
            ["Team return", "5&middot;(start distance &minus; end distance) &minus; 0.02&middot;T<sub>sim</sub> + 10&middot;"
                            "[success] &minus; 10&middot;[collision]: the PPO reward per simulated second, comparable across "
                            "decision rates (one robot, so the team is one agent)."],
            ["Collision / conflict rate", "episodes ending in obstacle contact / all episodes (inter-robot conflict needs "
                                          "several robots)."],
            ["Path / action efficiency", "distance closed / distance driven; path length; turn reversals per metre "
                                         "(left-right dithering); decisions per episode."],
            ["Generalization", "success and time on new layouts with a random initial heading."],
            ["Robustness", "success and time with delayed commands (agent delay) and degraded sensing (disturbance)."],
            ["Training stability", "learning curve and spread of success over training; variance across seeds "
                                   "(not available: one seed)."],
        ], [3.8 * cm, 13.2 * cm]),
        Paragraph("3.4 Statistics", h2),
        Paragraph("Time differences are paired by scenario and only use scenarios both controllers solved; &Delta; = baseline "
                  "&minus; candidate (positive = candidate faster); 95% percentile bootstrap over scenarios (5000 resamples). "
                  "Success is compared with an exact two-sided sign test on discordant scenarios (solved by exactly one of the "
                  "two). Failures and timeouts are always reported next to times.", body),
        PageBreak(),
    ]

    # ------------------------------------------------------------ 4 results nominal
    story += [
        Paragraph("4. Results", h1),
        figure(figs["outcomes"], 17, 5.4, "Figure 1. Outcome of every episode by condition and controller. failed = B0 gave "
                                          "up (no bypass found / more than 12 avoidance manoeuvres)."),
        figure(figs["times"], 17, 4.8, "Figure 2. Completion time of successful episodes (each dot one episode, bar = median). "
                                       "n under each column = number of successes."),
    ]
    head = ["Slide metric", "Measure", SHORT["b0"], SHORT["rule"], SHORT["ppo"]]
    for condition, rows in data.items():
        m = M[condition]
        val = lambda k, d=1: [f(m[c][k], d) for c in CONTROLLERS]
        story += [KeepTogether([
            Paragraph(f"4.{list(data).index(condition) + 1} All metrics &mdash; {condition} "
                      f"({len(rows) // len(CONTROLLERS)} scenarios)", h2),
            table([
                head,
                ["Mission success", "% episodes completed"] + [pct(m[c]["success"]) for c in CONTROLLERS],
                ["", "95% Wilson interval"] + [success_ci(rows, c) for c in CONTROLLERS],
                ["Collision / conflict", "% episodes with contact"] + [pct(m[c]["collision"]) for c in CONTROLLERS],
                ["", "% timeout / % gave up"] + [f"{pct(m[c]['timeout'])} / {pct(m[c]['failed'])}" for c in CONTROLLERS],
                ["Completion time", "mean (sim s), successes"] + val("time_mean"),
                ["", "median (sim s), successes"] + val("time_median"),
                ["", "failure-penalized mean (s)"] + val("penalized"),
                ["", "decisions (steps) per episode"] + val("decisions", 0),
                ["Team return", "mean per episode"] + val("return", 2),
                ["Path / action efficiency", "distance closed / driven"] + val("efficiency", 2),
                ["", "path length (m)"] + val("path", 2),
                ["", "turn reversals per metre"] + val("reversals_per_m", 1),
                ["Real-time", "decision latency p95 (ms)"] + [f(m[c]["latency_p95"], 3) for c in CONTROLLERS],
                ["", "RTF (sim s / wall s)"] + val("rtf", 2),
            ], [3.4 * cm, 5.2 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm]),
            Spacer(1, 4)])]
    story += [
        Paragraph("B0 is closed-form arithmetic with no learned model, so no decision latency is reported for it. Several "
                  "Isaac Sim processes ran in parallel, which lowers RTF for all controllers equally.", small),
        figure(figs["obstacles"], 13, 5.5, "Figure 3. Nominal success by number of obstacles in the scenario."),
        figure(figs["path"], 17, 4.6, "Figure 4. Path length driven per episode (all episodes)."),
        figure(figs["reversals"], 17, 4.6, "Figure 5. Turn-direction reversals per metre driven: how much a controller "
                                            "dithers left-right instead of committing to a direction."),
        PageBreak(),
    ]

    # ------------------------------------------------------------ 5 paired comparisons
    rows_pairs = [["Comparison", "Both solved", "Mean &Delta; (s)", "95% CI &Delta; (s)", "Mean impr.", "95% CI impr.",
                   "Penalized mean (s)", "Success: only cand / only base, p"]]
    for (base, cand), p in pair.items():
        ci = p["delta_ci95_s"] or (None, None)
        cip = p["paired_improvement_ci95_pct"] or (None, None)
        only_c, only_b, pv = signs[(base, cand)]
        rows_pairs.append([f"{SHORT[cand]} vs {SHORT[base]}", str(p["scenarios_both_successful"]), f(p["mean_delta_s"]),
                           f"{f(ci[0])} to {f(ci[1])}", f(p["mean_paired_improvement_pct"], 1, "%"),
                           f"{f(cip[0], 1)} to {f(cip[1], 1)}%",
                           f"{f(p['penalized_mean_s'][base])} &rarr; {f(p['penalized_mean_s'][cand])}",
                           f"{only_c} / {only_b}, p = {pv:.3f}"])
    story += [
        Paragraph("5. Paired comparisons (nominal)", h1),
        table(rows_pairs, [2.1 * cm, 1.5 * cm, 1.7 * cm, 2.3 * cm, 1.7 * cm, 2.4 * cm, 2.6 * cm, 2.7 * cm]),
        Paragraph("&Delta; = baseline time &minus; candidate time on the same scenario. Mean improvement = mean of per-scenario "
                  "100&middot;&Delta;/baseline. The last column counts scenarios solved by only one of the two controllers and "
                  "gives the exact sign-test p-value for 'equally likely to succeed'.", small),
        Paragraph("Hypothesis verdicts", h2),
        table([
            ["Hypothesis", "Verdict (nominal)"],
            ["H1 efficiency (PPO faster than B0)", verdict_time(pair[("b0", "ppo")])],
            ["H1b reliability (PPO succeeds more than B0)", verdict_success(("b0", "ppo"))],
            ["PPO vs heuristic B1, time", verdict_time(pair[("rule", "ppo")])],
            ["PPO vs heuristic B1, success", verdict_success(("rule", "ppo"))],
            ["B1 vs B0, success", verdict_success(("b0", "rule"))],
        ], [6.0 * cm, 11.0 * cm]),
        Paragraph("6. Generalization and robustness", h1),
        figure(figs["general"], 12, 5.6, "Figure 6. Success rate per condition."),
    ]
    if len(M) > 1:
        drop_rows = [["Controller", "Nominal", "Generalization", "&Delta; vs nominal", "Robustness", "&Delta; vs nominal"]]
        for c in CONTROLLERS:
            base_s = M["nominal"][c]["success"] if "nominal" in M else None
            g = M.get("generalization", {}).get(c, {}).get("success")
            r_ = M.get("robustness", {}).get(c, {}).get("success")
            drop_rows.append([LABEL[c], pct(base_s), pct(g), f((g - base_s) * 100, 0, " pts") if g is not None and base_s is not None else "-",
                              pct(r_), f((r_ - base_s) * 100, 0, " pts") if r_ is not None and base_s is not None else "-"])
        story += [table(drop_rows, [4.6 * cm, 2.2 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm, 2.6 * cm]),
                  Paragraph("Different seeds are used per condition (so that layouts are new), therefore these are "
                            "unpaired changes across conditions; within a condition all controllers are paired.", small)]
    story.append(PageBreak())

    # ------------------------------------------------------------ 7 failure analysis
    story += [Paragraph("7. Failure analysis", h1)]
    fail_rows = [["Condition", "Controller", "Collision", "Timeout", "Gave up", "Most common reason"]]
    for condition, rows in data.items():
        for c in CONTROLLERS:
            mine = [r for r in rows if r["controller"] == c and r["success"] != "True"]
            reasons = {}
            for r in mine:
                key = re.sub(r"\d+", "N", r["reason"] or r["status"])
                reasons[key] = reasons.get(key, 0) + 1
            common = max(reasons.items(), key=lambda kv: kv[1]) if reasons else ("-", 0)
            fail_rows.append([condition, SHORT[c], str(sum(r["status"] == "collision" for r in mine)),
                              str(sum(r["status"] == "timeout" for r in mine)), str(sum(r["status"] == "failed" for r in mine)),
                              f"{common[0]} ({common[1]})" if common[1] else "-"])
    story += [
        table(fail_rows, [2.8 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 6.8 * cm]),
        Paragraph("How each controller fails", h2),
        bullet("<b>B0</b> stops before obstacles (it rarely collides), but its bypass is a fixed 90&deg; turn and a 0.25 m "
               "step to a random side. With two or three boxes near the path it often turns back toward the goal, meets the "
               "same box, and repeats until it gives up (12 manoeuvres) or the time budget ends."),
        bullet("<b>B1</b> reacts only to the front sector and commits to 6 forward steps after every avoidance turn. It often "
               "drives away from the goal and runs out of time; it can also clip a box while committed."),
        bullet("<b>PPO</b> steers around obstacles using learned behaviour and reaches goals the rule-based controllers "
               "cannot, but its observation (3 LiDAR sectors, no memory) causes left-right dithering near obstacles "
               "(figure 5), and it collides when an obstacle edge falls between its sectors."),
    ]

    # ------------------------------------------------------------ 8 missions
    if missions:
        story += [
            Paragraph("8. Full CyberScape missions", h1),
            Paragraph("The same approved LLM plan is executed by each controller (headless). B1 and PPO run behind the RL "
                      "goal interface used in the demos (turn toward the goal first, stall recovery, spin guard, "
                      "person-safety zone); B0 runs exactly as the original node.", body),
            table([["Mission", "Ctrl", "Status", "Tasks", "Makespan (s)", "Mean task (s)", "Wall (s)", "Path (m)",
                    "Overrides / recoveries"]] + [
                [r["mission_id"].split("_")[2], SHORT.get(r["controller"], r["controller"]), r["status"],
                 f"{r['tasks_completed']}/{r['tasks_required']}", f(r["makespan_sim_s"]), f(r["mean_task_completion_sim_s"]),
                 f(r["wall_elapsed_s"], 0), f(r["path_length_m"], 1),
                 ", ".join(f"{k[6:].replace('_', ' ')} {v}" for k, v in r.items() if k.startswith("count_") and v) or "-"]
                for r in missions], [1.6 * cm, 1.0 * cm, 1.5 * cm, 1.1 * cm, 1.9 * cm, 1.9 * cm, 1.4 * cm, 1.3 * cm, 5.3 * cm]),
            Paragraph("Patrol: 6 commands (checkpoint A, scan, checkpoint B, scan, rotate 90&deg;, report). Disaster: 10 "
                      "commands (survivor 1: drive, scan, camera detection, report; survivor 2: the same; rotate; report). "
                      "Makespan = mission start to last command completed.", small),
        ]
        if figs["timeline"]:
            story.append(figure(figs["timeline"], 17, max(3.0, 0.75 * len(missions) + 1.6),
                                "Figure 7. Mission timelines: one bar per plan command; red = command failed."))
        story.append(PageBreak())

    # ------------------------------------------------------------ 9 training, discussion
    story += [Paragraph("9. Training stability (PPO)", h1)]
    if figs["learn"]:
        story += [
            figure(figs["learn"], 12, 5.6, "Figure 8. PPO training success over the last 500 episodes (stochastic actions)."),
            Paragraph(f"One training run: {learn['episodes']:,} episodes, {learn['steps']:,} steps (1M + 250k fine-tune). After "
                      f"500k steps, success over the last 500 episodes averaged {f(learn['late_mean'])}% with a standard "
                      f"deviation of {f(learn['late_std'])} points: training plateaued around that level. Success per block "
                      f"of 1000 episodes: {', '.join(f'{b:.0f}%' for b in learn['block_rates'])}.", body),
            Paragraph("<b>Variance across seeds is not measured</b>: only one seed was trained (about 4 days of simulation "
                      "per seed on this PC). The framework recommends at least 5 seeds before claiming stability.", body),
        ]
    story += [
        Paragraph("10. Discussion", h1),
        Paragraph("What the results mean for the thesis", h2),
        bullet("Replacing the executor changes <b>which goals get reached</b> more than how fast. The comparison must "
               "therefore be read as success first, then time on jointly solved scenarios, then the failure-penalized mean."),
        bullet("B0's continuous driving makes it fast on easy, open goals; its fixed bypass manoeuvre is what limits it in "
               "cluttered scenes. PPO is slower per metre because its trained action set is slow (0.05 m/s with stop-and-"
               "settle), not because it plans worse paths: a faster action set is the cheapest time improvement available."),
        bullet("The full-mission results include the goal-interface layer (alignment, recovery, guards) for B1 and PPO. "
               "This layer is part of the proposed system, but its interventions are counted and reported separately "
               "so that the policy's own contribution stays visible."),
        Paragraph("Threats to validity", h2),
        bullet("<b>Single robot, single training seed.</b> No multi-robot coordination is measured; PPO variability across "
               "seeds is unknown."),
        bullet("<b>Port of B0.</b> B0 runs in the simulator through the same wheel controller; on the real robot its timing "
               "also depends on ROS 2 latency and odometry, which are not modelled."),
        bullet("<b>Scenario distribution.</b> Scenarios come from PPO's training distribution (start pose and goal bank); "
               "this favours PPO in the nominal condition. The generalization condition partly addresses this."),
        bullet("<b>Sample size.</b> 30/20/20 scenarios per condition is a first budget; intervals in sections 4-5 show the "
               "remaining uncertainty."),
        bullet("<b>Simulated time.</b> All times are simulated seconds, not measured on the physical robot."),
        Paragraph("Recommendations", h2),
        bullet("Train V4: previous action and 12-16 LiDAR directions in the observation, a penalty on turn reversals, "
               "random start poses and goals, faster / arc actions. Then repeat this exact study (same seeds) to measure the gain."),
        bullet("Train at least 5 seeds per configuration and report the spread (training stability)."),
        bullet("Implement MAPPO for 2-3 robots and add B2 (independent PPO) and B3 (MAPPO) to the same harness; the timing "
               "evaluator already supports multiple robots and tasks."),
        bullet("Log the same events on the real TurtleBot3 (ROS 2 timestamps) to validate simulated times."),
        PageBreak(),
    ]

    # ------------------------------------------------------------ appendix
    story += [Paragraph("Appendix A. Per-scenario results", h1),
              Paragraph("Outcome and time per scenario (S = success with sim seconds, C = collision, T = timeout, F = gave up). "
                        "Obst. = number of obstacles; Dist. = start-to-goal distance (m).", small)]
    code = {"success": "S", "collision": "C", "timeout": "T", "failed": "F", "incomplete": "F"}
    for condition, rows in data.items():
        by = {}
        for r in rows:
            by.setdefault(r["scenario_id"], {})[r["controller"]] = r
        lines = [["Scenario", "Obst.", "Max width (m)", "Dist. (m)", "Heading offset"] + [SHORT[c] for c in CONTROLLERS]]
        for sid in sorted(by, key=int):
            any_row = next(iter(by[sid].values()))
            cells = []
            for c in CONTROLLERS:
                r = by[sid].get(c)
                if r is None:
                    cells.append("-")
                elif r["success"] == "True":
                    cells.append(f"S {f(r['makespan_sim_s'])} s")
                else:
                    cells.append(f"{code.get(r['status'], '?')} ({f(r['termination_elapsed_sim_s'], 0)} s)")
            lines.append([sid, any_row["obstacles"], f(any_row["max_obstacle_width_m"], 2), f(any_row["goal_distance_m"], 2),
                          f(any_row.get("initial_heading_offset_deg", 0) or 0, 0, "&deg;")] + cells)
        story += [Paragraph(f"A.{list(data).index(condition) + 1} {condition}", h2),
                  table(lines, [1.8 * cm, 1.2 * cm, 2.1 * cm, 1.7 * cm, 2.2 * cm, 2.7 * cm, 2.7 * cm, 2.6 * cm])]
    story += [
        PageBreak(),
        Paragraph("Appendix B. Reproducibility", h1),
        table([
            ["Item", "Value"],
            ["Code", "branch <font face='Courier'>sim</font>, folder <font face='Courier'>Hardware/Turtlebot3/rl_point_nav</font>"],
            ["Run everything", "<font face='Courier'>./run_comparison_study.sh</font> (episodes in parallel per condition, "
                               "then missions)"],
            ["This report", "<font face='Courier'>python make_comparison_report.py --stamp " + stamp + "</font>"],
            ["Raw data", f"<font face='Courier'>results/timing/{stamp}_&lt;condition&gt;/</font>: manifest.json (frozen "
                         "settings), episodes.csv (one row per episode), events/*.jsonl (every timed event), summary.json; "
                         "<font face='Courier'>results/missions/</font>: missions.csv + per-mission events.jsonl"],
            ["Simulator", "Isaac Sim 5.1, scene FInalSceneDemo_realsize.usd (robot at real scale, LiDAR at base_scan), "
                          f"physics dt {manifests.get('nominal', next(iter(manifests.values()), {})).get('physics_dt_s', 1 / 60):.4f} s"
                          if manifests else "Isaac Sim 5.1"],
            ["PPO model", Path(next(iter(manifests.values()))["model"]).name if manifests else "checkpoint_1000000_steps.zip"],
            ["Unit tests", "<font face='Courier'>python3 -m unittest test_mission_timing</font> (13 tests: parallel tasks, "
                           "unequal action durations, failed runs, asynchronous starts, paired statistics)"],
        ], [3.2 * cm, 13.8 * cm]),
    ]

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title="CyberScape: no-RL vs PPO comparison report",
                            author="AUC Thesis II team")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(INK2))
        canvas.drawString(2 * cm, 1.0 * cm, "AUC Graduation Project - Thesis II - CyberScape execution: no RL vs PPO")
        canvas.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=None)
    args = parser.parse_args()
    stamp = args.stamp or (RESULTS / "latest_comparison.txt").read_text().strip()
    out = RESULTS / f"comparison_report_{stamp}.pdf"
    build_pdf(stamp, out)
    print(out)


if __name__ == "__main__":
    main()
