#!/usr/bin/env python3
# Copyright 2026 Individual Contributor: Michael Lavery
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract per-step convergence curves from exclamation-task training logs.

verl's console logger emits one line per step as ``step:N - key:value - key:value ...``
(``verl/utils/logger/aggregate_logger.py``). This turns a set of those logs into a
side-by-side table so the loss-interpolation sweep (Stage F of ``sky_sdpo_test.yaml``)
can be read as "which mix climbs fastest", not just "did it crash".

Why a parser rather than a grep: the question is a *rate*, so the metric has to stay
paired with its step index and stay ordered. A bare grep of matching substrings loses
both -- it cannot tell a run that hit 0.9 at step 3 from one that hit 0.9 at step 19.

Usage:
    python examples/injecagent/parse_exclaim_curves.py ~/exclaim_sdpo_blend_a*.log \
        [--csv-dir ~/exclaim_curves] [--metric critic/rewards/mean]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# `step:12 - critic/rewards/mean:0.5 - sdpo/reprompt_sample_fraction:0.9 - ...`
_STEP_RE = re.compile(r"\bstep:(\d+)\b")
_PAIR_RE = re.compile(r"([A-Za-z_][\w/]*):(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")

# Reported for every run. The first is the headline convergence signal; the rest explain it.
# Matched by suffix: actor-worker metrics are re-emitted with an `actor/` prefix
# (`rename_dict` in verl/utils/py_functional.py), so `distillation/loss` reaches the console as
# `actor/distillation/loss`, while trainer-side ones like `sdpo/reprompt_sample_fraction` do not.
# Suffix matching catches both without hard-coding which side of the fence each metric sits on.
_TRACKED = (
    "critic/rewards/mean",
    "sdpo/reprompt_sample_fraction",
    "distillation/loss",
    "pg_loss",
)


def _tracked_keys(row: dict[str, float]) -> list[str]:
    """Keys of ``row`` matching a tracked metric, in ``_TRACKED`` order."""
    matched = []
    for tracked in _TRACKED:
        matched.extend(k for k in row if k == tracked or k.endswith("/" + tracked))
    return matched


def resolve_metric(steps: dict[int, dict[str, float]], metric: str) -> str:
    """Resolve ``metric`` against the keys actually logged, tolerating an ``actor/`` prefix."""
    keys = {k for row in steps.values() for k in row}
    if metric in keys:
        return metric
    candidates = sorted(k for k in keys if k.endswith("/" + metric))
    return candidates[0] if candidates else metric


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    """Return ``{step: {metric: value}}`` for one training log."""
    steps: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        step_match = _STEP_RE.search(line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        # A step's metrics may be split across repeated prints; merge rather than overwrite.
        row = steps.setdefault(step, {})
        for key, value in _PAIR_RE.findall(line):
            if key == "step":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                continue
    return steps


def steps_to_threshold(curve: list[tuple[int, float]], threshold: float) -> int | None:
    """First step whose value reaches ``threshold``, or None if never reached."""
    for step, value in curve:
        if value >= threshold:
            return step
    return None


def summarize(name: str, steps: dict[int, dict[str, float]], metric: str) -> dict[str, object]:
    """Reduce one run's curve to the numbers that answer 'how fast did it climb?'."""
    curve = [(s, steps[s][metric]) for s in sorted(steps) if metric in steps[s]]
    if not curve:
        return {"run": name, "n_steps": len(steps), "note": f"no '{metric}' logged"}
    values = [v for _, v in curve]
    first, final = values[0], values[-1]
    best_step, best = max(curve, key=lambda sv: sv[1])
    # Mean slope per step: the headline "goes up faster" number, robust to a noisy final step.
    slope = (final - first) / (curve[-1][0] - curve[0][0]) if len(curve) > 1 else 0.0
    return {
        "run": name,
        "n_steps": len(curve),
        "first": first,
        "final": final,
        "delta": final - first,
        "slope_per_step": slope,
        "best": best,
        "best_step": best_step,
        "steps_to_0.5": steps_to_threshold(curve, 0.5),
        "steps_to_0.9": steps_to_threshold(curve, 0.9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="training logs to parse")
    parser.add_argument(
        "--metric",
        default="critic/rewards/mean",
        help="metric whose convergence is summarized (default: critic/rewards/mean)",
    )
    parser.add_argument("--csv-dir", type=Path, default=None, help="also write per-run per-step CSVs here")
    args = parser.parse_args()

    summaries = []
    for log in args.logs:
        if not log.exists():
            print(f"missing: {log}")
            continue
        steps = parse_log(log)
        name = log.stem
        metric = resolve_metric(steps, args.metric)
        summaries.append(summarize(name, steps, metric))

        if args.csv_dir:
            args.csv_dir.mkdir(parents=True, exist_ok=True)
            columns = sorted({k for row in steps.values() for k in _tracked_keys(row)})
            out = args.csv_dir / f"{name}.csv"
            with out.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", *columns])
                for step in sorted(steps):
                    writer.writerow([step, *(steps[step].get(c, "") for c in columns)])
            print(f"wrote {out}")

        # Per-step trace, so a stalled or collapsing run is visible and not just summarized away.
        print(f"\n### {name}  (metric: {metric})")
        for step in sorted(steps):
            shown = [f"{k}={steps[step][k]:.4f}" for k in _tracked_keys(steps[step])]
            if shown:
                print(f"  step {step:>3}: " + "  ".join(shown))

    if not summaries:
        print("\nno runs parsed")
        return 1

    print(f"\n=== convergence summary ({args.metric}) ===")
    header = ["run", "n_steps", "first", "final", "delta", "slope_per_step", "best", "best_step", "->0.5", "->0.9"]
    print("  ".join(h.ljust(28 if h == "run" else 9) for h in header))
    for row in summaries:
        if "note" in row:
            print(f"{row['run'].ljust(28)}  {str(row['n_steps']).ljust(9)}  {row['note']}")
            continue
        cells = [
            row["run"].ljust(28),
            str(row["n_steps"]).ljust(9),
            f"{row['first']:.4f}".ljust(9),
            f"{row['final']:.4f}".ljust(9),
            f"{row['delta']:+.4f}".ljust(9),
            f"{row['slope_per_step']:+.5f}".ljust(9),
            f"{row['best']:.4f}".ljust(9),
            str(row["best_step"]).ljust(9),
            str(row["steps_to_0.5"]).ljust(9),
            str(row["steps_to_0.9"]).ljust(9),
        ]
        print("  ".join(cells))

    ranked = [r for r in summaries if "slope_per_step" in r]
    if ranked:
        ranked.sort(key=lambda r: r["slope_per_step"], reverse=True)
        print("\nfastest climb first: " + ", ".join(f"{r['run']} ({r['slope_per_step']:+.5f}/step)" for r in ranked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
