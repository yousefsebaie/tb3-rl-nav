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
    for c in CONTROLLERS:
        ys = [100 * metrics(data[x], c)["success"] for x in xs]
        ax.plot(range(len(xs)), ys, color=COLOR[c], linewidth=2, marker="o", markersize=7)
        ax.text(len(xs) - 1 + 0.08, ys[-1], LABEL[c], fontsize=8, color=INK, va="center")
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


# ------------------------------------------------------------------ PDF
def build_pdf(stamp: str, out_pdf: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    data = load_rows(stamp)
    if not data:
        raise SystemExit(f"no results for {stamp}")
    figdir = out_pdf.parent / f"{out_pdf.stem}_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    f_outcomes = chart_outcomes(data, figdir / "outcomes.png")
    f_times = chart_times(data, figdir / "times.png")
    f_gen = chart_generalization(data, figdir / "generalization.png")
    f_learn, learn = learning_curve(figdir / "learning_curve.png")
    missions = mission_rows(stamp)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.2, leading=12.5, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10, textColor=colors.HexColor(INK2))
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, textColor=colors.HexColor("#123a5c"), spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, textColor=colors.HexColor("#1b6b74"),
                        spaceBefore=10, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.5)

    def table(rows, widths, header_rows=1):
        t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=widths, repeatRows=header_rows)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e8f1f2")),
            ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, colors.HexColor("#1b6b74")),
            ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, colors.HexColor(GRID)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        return t

    nom = data.get("nominal", next(iter(data.values())))
    m_nom = {c: metrics(nom, c) for c in CONTROLLERS}
    pair_b0 = compare_paired(nom, "b0", "ppo", budget_s=BUDGET_S)
    pair_b1 = compare_paired(nom, "rule", "ppo", budget_s=BUDGET_S)
    n_nom = len(nom) // len(CONTROLLERS)

    story = [
        Paragraph("Experimental Comparison: CyberScape without RL vs. with PPO", h1),
        Paragraph(f"TurtleBot3 Burger, Isaac Sim 5.1 real-size scene &middot; generated {dt.date.today():%d %B %Y} "
                  f"&middot; run <font face='Courier'>{stamp}</font>", small),
        Spacer(1, 8),
        Paragraph("Question", h2),
        Paragraph("Does replacing the original CyberScape execution controller with a learned PPO navigation policy "
                  "improve how an approved LLM mission plan is executed? Three controllers execute the same goals "
                  "on the same seeded scenarios, and are measured with the metrics of the Experimental Design slide.", body),
        Paragraph("Key results (nominal condition, %d paired scenarios)" % n_nom, h2),
    ]
    findings = [
        f"<b>Mission success</b>: B0 {f(100 * m_nom['b0']['success'], 0, '%')}, "
        f"B1 {f(100 * m_nom['rule']['success'], 0, '%')}, PPO {f(100 * m_nom['ppo']['success'], 0, '%')} of episodes reached the goal.",
        f"<b>Collisions</b>: B0 {f(100 * m_nom['b0']['collision'], 0, '%')}, B1 {f(100 * m_nom['rule']['collision'], 0, '%')}, "
        f"PPO {f(100 * m_nom['ppo']['collision'], 0, '%')}.",
        f"<b>Failure-aware completion time</b> (failures count as the {BUDGET_S:.0f} s budget): "
        f"B0 {f(m_nom['b0']['penalized'])} s, B1 {f(m_nom['rule']['penalized'])} s, PPO {f(m_nom['ppo']['penalized'])} s.",
        f"<b>Paired time, PPO vs B0</b>: {pair_b0['scenarios_both_successful']} scenarios solved by both; "
        f"mean improvement {f(pair_b0['mean_paired_improvement_pct'], 1, '%')} "
        f"(95% CI {f((pair_b0['paired_improvement_ci95_pct'] or (None, None))[0], 1)} to "
        f"{f((pair_b0['paired_improvement_ci95_pct'] or (None, None))[1], 1, '%')}; positive = PPO faster).",
        f"<b>Team return</b> (PPO reward in continuous time): B0 {f(m_nom['b0']['return'], 2)}, "
        f"B1 {f(m_nom['rule']['return'], 2)}, PPO {f(m_nom['ppo']['return'], 2)}.",
    ]
    story += [Paragraph("&bull; " + x, body) for x in findings]
    story += [
        Spacer(1, 6),
        Paragraph("Read success and collision together with time: a controller that is fast but collides or times out more "
                  "often is not better (framework decision rule). Time comparisons use only scenarios where both "
                  "controllers succeeded; the failure-penalized mean includes every attempt.", small),
        Paragraph("Speed is partly a design difference, not only intelligence: PPO acts in stop-and-settle steps "
                  "(forward 0.05 m/s for 0.5 s, turns 0.8 rad/s for 0.25 s, then a 0.13 s settle) because it was trained "
                  "that way, while B0 drives continuously at up to 0.15 m/s. On clear paths B0 is therefore faster by "
                  "construction; PPO's advantage, where it has one, is reaching goals B0 cannot (success rate).", small),
        Paragraph("Setup", h2),
        table([
            ["Controller", "What it is"],
            [LABEL["b0"], "The execution controller the LLM plans ran on before RL: <font face='Courier'>burger_comm_node.move_to_point</font> "
                          "ported unchanged (rotate to goal, drive v=min(0.15, 0.5d), w=1.5&middot;error; front &lt; 0.35 m: turn 90&deg; "
                          "to a random side and sidestep 0.25 m; 12 attempts max). Raw front-sector LiDAR minimum."],
            [LABEL["rule"], "Hand-coded reactive rules on the same 6-D observation as PPO (turn away from a close front obstacle, "
                            "commit forward, else steer to goal). V3 sensor model."],
            [LABEL["ppo"], "Trained V3 PPO policy (1M steps, MLP 64x64, 4 discrete actions), deterministic. "
                           "V3 sensor model it was trained and deployed with."],
        ], [3.6 * cm, 13.4 * cm]),
        Spacer(1, 6),
        table([["Condition", "Scenarios", "What changes"]] +
              [[c, str(len(data[c]) // len(CONTROLLERS)), CONDITION_TEXT[c]] for c in CONDITIONS if c in data],
              [3.6 * cm, 2.2 * cm, 11.2 * cm]),
        Spacer(1, 4),
        Paragraph("Each scenario = one goal 0.5-2 m away with 0-3 boxes (12-40 cm), identical for every controller (seeded). "
                  "Budget 120 s simulated per goal. Time is the Isaac simulation clock read after physics steps; "
                  "it reconciles exactly with physics frames &times; dt.", small),
        PageBreak(),
        Paragraph("1. Mission success, collision / conflict rate", h2),
        Image(str(f_outcomes), width=17 * cm, height=5.4 * cm),
        Paragraph("Share of episodes by outcome. failed = B0 gave up (no bypass found / too many avoidance attempts).", small),
        Paragraph("2. Completion time", h2),
        Image(str(f_times), width=17 * cm, height=5.4 * cm),
        Paragraph("Each dot is one successful episode; the bar is the median. Only successful episodes have a completion "
                  "time; see the outcome chart for how many there are.", small),
    ]

    head = ["Metric (slide)", "Measure"] + [SHORT[c] for c in CONTROLLERS]
    for condition, rows in data.items():
        m = {c: metrics(rows, c) for c in CONTROLLERS}
        pct = lambda k: [f(100 * m[c][k], 0, "%") if m[c][k] is not None else "-" for c in CONTROLLERS]
        val = lambda k, d=1, s="": [f(m[c][k], d, s) for c in CONTROLLERS]
        story += [
            Paragraph(f"3. All metrics &mdash; {condition}", h2) if condition == "nominal" else
            Paragraph(f"All metrics &mdash; {condition}", h2),
            table([
                head,
                ["Mission success rate", "% episodes completed"] + pct("success"),
                ["Collision / conflict", "% episodes with contact"] + pct("collision"),
                ["", "% timeouts / % gave up"] + [f"{f(100 * m[c]['timeout'], 0, '%')} / {f(100 * m[c]['failed'], 0, '%')}" for c in CONTROLLERS],
                ["Completion time", "mean / median sim s (successes)"] + [f"{f(m[c]['time_mean'])} / {f(m[c]['time_median'])}" for c in CONTROLLERS],
                ["", "failure-penalized mean (s)"] + val("penalized"),
                ["Team return", "mean, PPO reward in continuous time"] + val("return", 2),
                ["Path / action efficiency", "distance closed / driven (successes)"] + val("efficiency", 2),
                ["", "path length (m), all episodes"] + val("path", 2),
                ["", "turn reversals per metre"] + val("reversals_per_m", 1),
                ["", "decisions per episode"] + val("decisions", 0),
                ["Real-time feasibility", "decision latency p95 (ms) / RTF"] + [
                    f"{f(m[c]['latency_p95'], 3)} / {f(m[c]['rtf'], 2)}" for c in CONTROLLERS],
            ], [3.6 * cm, 5.2 * cm, 2.7 * cm, 2.7 * cm, 2.8 * cm]),
        ]
    story += [
        Paragraph("B0 has no learned policy: its per-decision cost is closed-form arithmetic, so no latency is reported. "
                  "Team return = 5&middot;(start distance &minus; end distance) &minus; 0.02&middot;T<sub>sim</sub> + 10&middot;[success] "
                  "&minus; 10&middot;[collision]: the PPO training reward per second of simulated time, so controllers with "
                  "different decision rates are comparable.", small),
        PageBreak(),
        Paragraph("4. Paired comparisons (nominal)", h2),
    ]
    rows_pairs = [["Comparison", "Both solved", "Mean &Delta; (s)", "95% CI &Delta; (s)", "Mean improvement", "95% CI (%)",
                   "Penalized mean (s)"]]
    for base, cand in (("b0", "ppo"), ("rule", "ppo"), ("b0", "rule")):
        p = compare_paired(nom, base, cand, budget_s=BUDGET_S)
        ci = p["delta_ci95_s"] or (None, None)
        cip = p["paired_improvement_ci95_pct"] or (None, None)
        rows_pairs.append([f"{SHORT[cand]} vs {SHORT[base]}", str(p["scenarios_both_successful"]), f(p["mean_delta_s"]),
                           f"{f(ci[0])} to {f(ci[1])}", f(p["mean_paired_improvement_pct"], 1, "%"),
                           f"{f(cip[0], 1)} to {f(cip[1], 1)}",
                           f"{f(p['penalized_mean_s'][base])} &rarr; {f(p['penalized_mean_s'][cand])}"])
    story += [
        table(rows_pairs, [2.6 * cm, 1.8 * cm, 2.0 * cm, 2.6 * cm, 2.4 * cm, 2.6 * cm, 3.0 * cm]),
        Paragraph("&Delta; = baseline time &minus; candidate time on the same scenario (positive = candidate faster). "
                  "Bootstrap over scenarios, 5000 resamples. With few scenarios solved by both, the interval is wide "
                  "and the time result is not conclusive; the success and penalized columns carry the comparison.", small),
        Paragraph("5. Generalization and robustness", h2),
        Image(str(f_gen), width=12 * cm, height=5.6 * cm),
        Paragraph("Success rate per condition. Generalization: obstacle layouts never used for these runs before and a "
                  "start heading the policy never trained with. Robustness: agent delay (15% of commands arrive one "
                  "decision late) and degraded sensing.", small),
    ]
    if missions:
        story += [
            Paragraph("6. Full CyberScape missions (approved LLM plan, same plan per controller)", h2),
            table([["Mission", "Controller", "Status", "Tasks", "Makespan (sim s)", "Mean task (s)", "Path (m)", "Overrides"]] + [
                [r["mission_id"].split("_")[2], SHORT.get(r["controller"], r["controller"]), r["status"],
                 f"{r['tasks_completed']}/{r['tasks_required']}", f(r["makespan_sim_s"]), f(r["mean_task_completion_sim_s"]),
                 f(r["path_length_m"], 1),
                 ", ".join(f"{k[6:]}={v}" for k, v in r.items() if k.startswith("count_") and v) or "-"]
                for r in missions], [1.8 * cm, 1.6 * cm, 1.6 * cm, 1.2 * cm, 2.2 * cm, 2.0 * cm, 1.4 * cm, 5.2 * cm]),
            Paragraph("Patrol: 6 commands (two checkpoints with obstacle avoidance, scans, turn, report). Disaster: 10 commands "
                      "(two survivors: drive, scan, camera detection, report). B1 and PPO run behind the RL goal interface "
                      "(turn to goal, stall recovery, spin guard, person-safety zone); B0 runs exactly as the original node.", small),
        ]
    story += [PageBreak(), Paragraph("7. Training stability (PPO)", h2)]
    if f_learn:
        blocks = ", ".join(f"{b:.0f}%" for b in learn["block_rates"])
        story += [
            Image(str(f_learn), width=12 * cm, height=5.6 * cm),
            Paragraph(f"One training run: {learn['episodes']} episodes, {learn['steps']:,} steps (1M + 250k fine-tune). "
                      f"Success over the last 500 episodes after 500k steps: mean {f(learn['late_mean'])}%, standard deviation "
                      f"{f(learn['late_std'])} points. Success per block of 1000 episodes: {blocks}. Training success is lower "
                      f"than evaluation because training samples actions stochastically.", body),
            Paragraph("Variance across training seeds is <b>not yet measured</b>: only one seed was trained. The framework "
                      "asks for &ge; 5 seeds before claiming stability.", body),
        ]
    story += [
        Paragraph("8. Limitations and what is not compared yet", h2),
        Paragraph("&bull; <b>MAPPO</b> (the last column of the slide) does not exist yet; this report covers the single-robot "
                  "layer: existing system vs heuristic vs PPO. The same harness and timing evaluator take MAPPO once trained.", body),
        Paragraph("&bull; B0 was ported from the robot node to the simulator; the robot node itself was never timed in Isaac "
                  "(the old Isaac listener teleported the robot, so it has no physical timing).", body),
        Paragraph("&bull; One PPO training seed; scenario counts (30/20/20) are a first budget, not a power analysis.", body),
        Paragraph("&bull; Simulation seconds are not physical robot seconds; real-robot timing needs ROS 2 event logs.", body),
        Paragraph("&bull; LLM planning overhead is measured separately (wall clock, results/missions/planning.csv) and is the "
                  "same for every controller, because all of them execute the same approved plan.", body),
        Spacer(1, 8),
        Paragraph("Reproduce: <font face='Courier'>./run_comparison_study.sh</font> then "
                  "<font face='Courier'>python make_comparison_report.py</font>. Raw per-episode CSVs, JSONL event logs "
                  f"and manifests: <font face='Courier'>results/timing/{stamp}_*/</font>.", small),
    ]

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title="CyberScape: no-RL vs PPO comparison", author="Thesis II team")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(INK2))
        canvas.drawString(2 * cm, 1.0 * cm, "AUC Graduation Project - Thesis II - controller comparison")
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
