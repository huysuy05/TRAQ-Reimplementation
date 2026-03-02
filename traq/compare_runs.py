import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


DISPLAY_LABELS = {
    "v2": "AR-CP RAG",
    "adaptive_v2": "AR-CP RAG ES",
    "vanilla": "Vanilla",
}


def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_gold_aliases_norm(ex: Dict[str, Any]) -> List[str]:
    ans = ex.get("Answer") or ex.get("answer") or {}
    golds: List[str] = []
    if isinstance(ans, dict):
        val = ans.get("Value") if "Value" in ans else ans.get("value")
        if val:
            golds.append(str(val))
        aliases = ans.get("Aliases") if "Aliases" in ans else (ans.get("aliases") or [])
        for alias in aliases or []:
            if alias:
                golds.append(str(alias))
    out: List[str] = []
    seen = set()
    for gold in golds:
        ng = norm_text(gold)
        if ng and ng not in seen:
            out.append(ng)
            seen.add(ng)
    return out


def any_gold_match(pred: str, golds_norm: List[str]) -> bool:
    p = norm_text(pred)
    if not p:
        return False
    return any((g in p) or (p in g) for g in golds_norm)


def exact_match_norm(pred: str, golds_norm: List[str]) -> bool:
    p = norm_text(pred)
    if not p:
        return False
    return p in golds_norm


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_coverage_curve_csv(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            exp = row.get("expected_coverage")
            emp = row.get("empirical_coverage")
            if exp in (None, "") or emp in (None, ""):
                continue
            rows.append(
                {
                    "expected_coverage": float(exp),
                    "empirical_coverage": float(emp),
                }
            )
    return rows


def parse_run_spec(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --run segment '{part}'. Expected key=value.")
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    if "method" not in out or "jsonl" not in out:
        raise ValueError("--run must include at least method=... and jsonl=...")
    if "label" not in out:
        out["label"] = out["method"]
    return out


def compute_metrics_from_rows(rows: List[Dict[str, Any]], eval_start: Optional[int]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    n = len(rows)
    metrics["num_rows"] = n

    c_ret_vals = [float(ex["C_ret_size"]) for ex in rows if ex.get("C_ret_size") is not None]
    if c_ret_vals:
        metrics["avg_c_ret"] = float(np.mean(c_ret_vals))

    p_star_vals = [bool(ex["p_star_in_C_ret"]) for ex in rows if ex.get("p_star_in_C_ret") is not None]
    if p_star_vals:
        metrics["retriever_coverage_rate"] = float(np.mean(p_star_vals))

    pred_sizes: List[int] = []
    said_cant_flags: List[bool] = []
    labels: List[int] = []
    hit_gold_vals: List[bool] = []
    for ex in rows:
        c_agg = ex.get("C_agg") or []
        if isinstance(c_agg, list):
            pred_sizes.append(len(c_agg))
            said_cant_flags.append(any("can't answer" in str(a).lower() for a in c_agg))
        label = ex.get("rej_label_ans")
        if label is not None:
            labels.append(int(label))
        if ex.get("hit_gold") is not None:
            hit_gold_vals.append(bool(ex["hit_gold"]))

    metrics["prediction_set_sizes"] = pred_sizes
    if pred_sizes:
        metrics["avg_c_agg"] = float(np.mean(pred_sizes))
        metrics["avg_semantic_clusters"] = float(np.mean(pred_sizes))

    if hit_gold_vals:
        metrics["coverage_proxy"] = float(np.mean(hit_gold_vals))

    if labels and len(labels) == len(said_cant_flags):
        n_labels = len(labels)
        p_unans = sum(1 for y in labels if y == 0) / n_labels
        p_reject = sum(1 for r in said_cant_flags if r) / n_labels
        p_joint = sum(1 for y, r in zip(labels, said_cant_flags) if y == 0 and r) / n_labels
        r_refuse = (p_joint / p_unans) if p_unans > 0 else 0.0
        p_refuse = (p_joint / p_reject) if p_reject > 0 else 0.0
        denom = max(2.0 - p_refuse * r_refuse, 1e-12)
        f1_paper = (p_refuse * r_refuse) / denom
        metrics["p_unans"] = float(p_unans)
        metrics["p_reject"] = float(p_reject)
        metrics["p_joint"] = float(p_joint)
        metrics["r_refuse"] = float(r_refuse)
        metrics["p_refuse"] = float(p_refuse)
        metrics["f1_refuse"] = float(f1_paper)

    ecr_exact_vals = [ex.get("ecr_exact") for ex in rows if ex.get("ecr_exact") is not None]
    ecr_substr_vals = [ex.get("ecr_substr") for ex in rows if ex.get("ecr_substr") is not None]
    if ecr_exact_vals:
        metrics["exact_ecr"] = float(np.mean([bool(v) for v in ecr_exact_vals]))
    if ecr_substr_vals:
        metrics["substring_ecr"] = float(np.mean([bool(v) for v in ecr_substr_vals]))

    if "exact_ecr" not in metrics or "substring_ecr" not in metrics:
        eval_from = int(eval_start) if eval_start is not None else 0
        exact_hits = 0
        substr_hits = 0
        total = 0
        for idx, ex in enumerate(rows):
            if idx < eval_from:
                continue
            c_agg = ex.get("C_agg") or []
            if not isinstance(c_agg, list):
                continue
            golds = get_gold_aliases_norm(ex)
            if not golds:
                continue
            label = ex.get("rej_label_ans")
            if label is not None and int(label) == 0:
                hit_exact = any("can't answer" in str(a).lower() for a in c_agg)
                hit_substr = hit_exact
            else:
                hit_exact = any(exact_match_norm(str(a), golds) for a in c_agg)
                hit_substr = any(any_gold_match(str(a), golds) for a in c_agg)
            exact_hits += int(hit_exact)
            substr_hits += int(hit_substr)
            total += 1
        if total > 0:
            metrics["exact_ecr"] = float(exact_hits / total)
            metrics["substring_ecr"] = float(substr_hits / total)

    if "step1_avg_batches" not in metrics:
        step1_vals = [float(ex["ne_batches_wo"]) for ex in rows if ex.get("ne_batches_wo") is not None]
        if step1_vals:
            metrics["step1_avg_batches"] = float(np.mean(step1_vals))
    if "step2_avg_batches" not in metrics:
        step2_vals = [float(ex["ne_batches_with"]) for ex in rows if ex.get("ne_batches_with") is not None]
        if step2_vals:
            metrics["step2_avg_batches"] = float(np.mean(step2_vals))

    return metrics


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_metrics_csv(path: str, run_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    headers = [
        "row_type",
        "method",
        "label",
        "run_index",
        "method_run_count",
        "target_coverage_run",
        "jsonl_path",
        "curve_path",
        "num_rows",
        "avg_c_ret",
        "retriever_coverage_rate",
        "p_joint",
        "p_unans",
        "p_reject",
        "r_refuse",
        "p_refuse",
        "f1_refuse",
        "avg_c_agg",
        "avg_semantic_clusters",
        "coverage_proxy",
        "exact_ecr",
        "substring_ecr",
        "step1_avg_batches",
        "step2_avg_batches",
        "step1_early_stop_rate",
        "step2_early_stop_rate",
        "saved_calls_step1",
    ]
    std_headers = [
        "avg_c_ret_std",
        "retriever_coverage_rate_std",
        "p_joint_std",
        "p_unans_std",
        "p_reject_std",
        "r_refuse_std",
        "p_refuse_std",
        "f1_refuse_std",
        "avg_c_agg_std",
        "avg_semantic_clusters_std",
        "coverage_proxy_std",
        "exact_ecr_std",
        "substring_ecr_std",
        "step1_avg_batches_std",
        "step2_avg_batches_std",
        "step1_early_stop_rate_std",
        "step2_early_stop_rate_std",
        "saved_calls_step1_std",
    ]
    headers.extend(std_headers)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in run_rows + summary_rows:
            out = {key: row.get(key, "") for key in headers}
            writer.writerow(out)


def save_prediction_set_boxplot(path: str, method_to_sizes: Dict[str, List[int]], method_to_avg_clusters: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    if not method_to_sizes:
        return
    ensure_parent(path)
    methods = list(method_to_sizes.keys())
    labels = [
        f"{DISPLAY_LABELS.get(method, method)}\nmean={method_to_avg_clusters.get(method, 0.0):.2f}"
        for method in methods
    ]
    data = [method_to_sizes[method] for method in methods]
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots(figsize=(max(6, 2 + 1.8 * len(methods)), 6))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Prediction Set Size |C_agg|")
    ax.set_title("Prediction Set Size by Method\n(labels show mean semantic clusters)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def save_coverage_comparison_plot(
    path: str,
    run_rows: List[Dict[str, Any]],
    coverage_metric: str,
    target_coverage: float,
) -> None:
    import matplotlib.pyplot as plt

    if not run_rows:
        return
    ensure_parent(path)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[row["method"]].append(row)

    target_values = sorted(
        {
            float(row["target_coverage_run"])
            for row in run_rows
            if row.get("target_coverage_run") not in ("", None)
        }
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(grouped), 1)))
    fig, ax = plt.subplots(figsize=(max(9, 3 + 1.8 * max(len(target_values), 1)), 5.5))

    for color, (method, rows) in zip(colors, sorted(grouped.items())):
        display_method = DISPLAY_LABELS.get(method, method)
        curve_points: Dict[float, List[float]] = defaultdict(list)
        for row in rows:
            curve_path = row.get("curve_path")
            if curve_path in ("", None):
                continue
            if not os.path.exists(str(curve_path)):
                continue
            for point in load_coverage_curve_csv(str(curve_path)):
                x = float(point["expected_coverage"])
                curve_points[x].append(float(point["empirical_coverage"]))

        if curve_points:
            xs = sorted(curve_points.keys())
            ys = [float(np.mean(curve_points[x])) for x in xs]
            errs = [float(np.std(curve_points[x])) if len(curve_points[x]) > 1 else 0.0 for x in xs]
            target_values = sorted(set(target_values).union(xs))
        else:
            if not target_values:
                target_values = [float(target_coverage)]
            by_target: Dict[float, List[float]] = defaultdict(list)
            for row in rows:
                y = row.get(coverage_metric)
                if y in ("", None):
                    continue
                t = row.get("target_coverage_run")
                if t in ("", None):
                    continue
                tf = float(t)
                by_target[tf].append(float(y))

            if by_target:
                xs = sorted(by_target.keys())
                ys = [float(np.mean(by_target[x])) for x in xs]
                errs = [float(np.std(by_target[x])) if len(by_target[x]) > 1 else 0.0 for x in xs]
            else:
                # Vanilla/non-conformal baseline: plot as a flat line across all target coverages.
                vals = [float(row.get(coverage_metric, 0.0) or 0.0) for row in rows if row.get(coverage_metric) not in ("", None)]
                if not vals:
                    continue
                xs = list(target_values)
                ys = [float(np.mean(vals)) for _ in xs]
                errs = [float(np.std(vals)) if len(vals) > 1 else 0.0 for _ in xs]

        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=display_method)
        if any(err > 0 for err in errs):
            ax.errorbar(xs, ys, yerr=errs, color=color, capsize=4, linewidth=1, linestyle="none")

    if not target_values:
        target_values = [float(target_coverage)]
    ax.plot(target_values, target_values, color="black", linestyle="--", linewidth=1.5, label="Reference")
    tick_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{int(round(x * 100))}" for x in tick_values])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.49, 0.91)
    ax.set_xlabel("Target Coverage (%)")
    ax.set_ylabel("Empirical")
    # ax.set_title(f"Coverage Comparison ({coverage_metric})")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def summarize_by_method(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[row["method"]].append(row)

    metric_keys = [
        "avg_c_ret",
        "retriever_coverage_rate",
        "p_joint",
        "p_unans",
        "p_reject",
        "r_refuse",
        "p_refuse",
        "f1_refuse",
        "avg_c_agg",
        "avg_semantic_clusters",
        "coverage_proxy",
        "exact_ecr",
        "substring_ecr",
        "step1_avg_batches",
        "step2_avg_batches",
        "step1_early_stop_rate",
        "step2_early_stop_rate",
        "saved_calls_step1",
    ]

    summaries: List[Dict[str, Any]] = []
    for method, rows in grouped.items():
        summary: Dict[str, Any] = {
            "row_type": "summary",
            "method": method,
            "label": f"{method}_summary",
            "run_index": "",
            "method_run_count": len(rows),
            "jsonl_path": "",
            "curve_path": "",
        }
        for key in metric_keys:
            vals = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
            if not vals:
                continue
            summary[key] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        summary["num_rows"] = int(np.mean([int(row["num_rows"]) for row in rows if row.get("num_rows") not in ("", None)]))
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser("Aggregate TRAQ runs into one CSV and comparison plots")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: method=<name>,label=<run_label>,jsonl=<path>,target=<coverage>,curve=<coverage_csv>",
    )
    parser.add_argument("--metrics_csv_path", type=str, default="out/run_metrics_comparison.csv")
    parser.add_argument("--boxplot_path", type=str, default="out/prediction_set_boxplot.png")
    parser.add_argument("--coverage_plot_path", type=str, default="out/method_coverage_comparison.png")
    parser.add_argument(
        "--coverage_metric",
        type=str,
        default="substring_ecr",
        choices=["retriever_coverage_rate", "coverage_proxy", "exact_ecr", "substring_ecr"],
    )
    parser.add_argument("--target_coverage", type=float, default=0.9)
    args = parser.parse_args()

    specs = [parse_run_spec(spec) for spec in args.run]
    method_counts: Dict[str, int] = defaultdict(int)
    for spec in specs:
        method_counts[spec["method"]] += 1

    run_rows: List[Dict[str, Any]] = []
    method_to_sizes: Dict[str, List[int]] = defaultdict(list)
    method_to_avg_clusters_accum: Dict[str, List[float]] = defaultdict(list)
    method_run_index: Dict[str, int] = defaultdict(int)

    for spec in specs:
        method = spec["method"]
        label = spec["label"]
        method_run_index[method] += 1

        rows = load_jsonl(spec["jsonl"])
        row_metrics = compute_metrics_from_rows(rows, eval_start=None)
        merged = dict(row_metrics)

        merged["row_type"] = "run"
        merged["method"] = method
        merged["label"] = label
        merged["run_index"] = method_run_index[method]
        merged["method_run_count"] = method_counts[method]
        target_run = spec.get("target")
        if target_run is not None and target_run != "":
            merged["target_coverage_run"] = float(target_run)
        elif method.lower().startswith("vanilla"):
            merged["target_coverage_run"] = ""
        else:
            merged["target_coverage_run"] = float(args.target_coverage)
        merged["jsonl_path"] = spec["jsonl"]
        merged["curve_path"] = spec.get("curve", "")
        run_rows.append(merged)

        pred_sizes = row_metrics.get("prediction_set_sizes", [])
        if pred_sizes:
            method_to_sizes[method].extend(int(v) for v in pred_sizes)
        if row_metrics.get("avg_semantic_clusters") is not None:
            method_to_avg_clusters_accum[method].append(float(row_metrics["avg_semantic_clusters"]))

    summary_rows = summarize_by_method(run_rows)
    write_metrics_csv(args.metrics_csv_path, run_rows, summary_rows)

    method_to_avg_clusters = {
        method: float(np.mean(vals)) for method, vals in method_to_avg_clusters_accum.items() if vals
    }
    save_prediction_set_boxplot(args.boxplot_path, method_to_sizes, method_to_avg_clusters)
    save_coverage_comparison_plot(
        args.coverage_plot_path,
        run_rows=run_rows,
        coverage_metric=args.coverage_metric,
        target_coverage=float(args.target_coverage),
    )

    print(f"Saved metrics CSV: {args.metrics_csv_path}")
    print(f"Saved prediction set boxplot: {args.boxplot_path}")
    print(f"Saved coverage comparison plot: {args.coverage_plot_path}")


if __name__ == "__main__":
    main()
