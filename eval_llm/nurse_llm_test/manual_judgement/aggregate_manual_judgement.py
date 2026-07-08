#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path


COLUMNS = [
    "result_file",
    "scenario",
    "prompt_variant",
    "model",
    "overall_score",
    "faithfulness",
    "expected_alignment",
    "nursing_correctness",
    "safety_escalation",
    "structure_completeness",
    "actionability",
    "no_hallucination",
    "json_quality",
    "unsafe",
    "brief_rationale",
]

NUMERIC_COLUMNS = [
    "overall_score",
    "faithfulness",
    "expected_alignment",
    "nursing_correctness",
    "safety_escalation",
    "structure_completeness",
    "actionability",
    "no_hallucination",
    "json_quality",
]


def parse_table(path: Path):
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if "result_file" in line and "overall_score" in line:
            continue
        if set(line.replace("|", "").strip()) <= {"-", ":"}:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != len(COLUMNS):
            continue
        row = dict(zip(COLUMNS, parts))
        if not row["result_file"].endswith("_results.json"):
            continue
        row["_source_file"] = str(path)
        for col in NUMERIC_COLUMNS:
            try:
                row[col] = float(row[col])
            except ValueError:
                row[col] = None
        row["unsafe"] = row["unsafe"].strip().lower()
        rows.append(row)
    return rows


def fmt(value, digits=2):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def avg(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(values) if values else None


def scenario_from_result_file(result_file: str):
    parts = Path(result_file).parts
    try:
        idx = parts.index("nurse_llm_test")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ""


def prompt_variant_from_result_file(result_file: str):
    parts = Path(result_file).parts
    if "professional_guided" in parts:
        return "professional_guided"
    if "simple_guided" in parts:
        return "simple_guided"
    return "default"


def build_markdown(grouped_rows):
    lines = []
    lines.append("# Manual Judgement Average Scores")
    lines.append("")
    lines.append("每个结果文件由 3 个 xhigh 评审智能体独立打分，表中为三者平均值。")
    lines.append("")

    lines.append("## Overall Ranking")
    lines.append("")
    lines.append(
        "| rank | result_file | scenario | prompt_variant | model | avg_overall | unsafe_votes | avg_safety | avg_nursing | avg_structure | agent_scores |"
    )
    lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---|")

    summary_rows = []
    for result_file, rows in grouped_rows.items():
        avg_overall = avg([row["overall_score"] for row in rows])
        summary_rows.append((avg_overall if avg_overall is not None else -1, result_file, rows))

    for rank, (_, result_file, rows) in enumerate(sorted(summary_rows, reverse=True), start=1):
        first = rows[0]
        scenario = first.get("scenario") or scenario_from_result_file(result_file)
        prompt_variant = first.get("prompt_variant") or prompt_variant_from_result_file(result_file)
        model = first.get("model", "")
        unsafe_votes = sum(1 for row in rows if row.get("unsafe") == "yes")
        agent_scores = ", ".join(fmt(row["overall_score"], 1) for row in rows)
        lines.append(
            "| {rank} | {result_file} | {scenario} | {prompt_variant} | {model} | {avg_overall} | {unsafe_votes} | {avg_safety} | {avg_nursing} | {avg_structure} | {agent_scores} |".format(
                rank=rank,
                result_file=result_file,
                scenario=scenario,
                prompt_variant=prompt_variant,
                model=model,
                avg_overall=fmt(avg(row["overall_score"] for row in rows), 2),
                unsafe_votes=unsafe_votes,
                avg_safety=fmt(avg(row["safety_escalation"] for row in rows), 2),
                avg_nursing=fmt(avg(row["nursing_correctness"] for row in rows), 2),
                avg_structure=fmt(avg(row["structure_completeness"] for row in rows), 2),
                agent_scores=agent_scores,
            )
        )

    lines.append("")
    lines.append("## Scenario Model Averages")
    lines.append("")
    lines.append("| scenario | prompt_variant | model | avg_overall | files | unsafe_votes |")
    lines.append("|---|---|---|---:|---:|---:|")

    scenario_model = defaultdict(list)
    for result_file, rows in grouped_rows.items():
        first = rows[0]
        key = (
            first.get("scenario") or scenario_from_result_file(result_file),
            first.get("prompt_variant") or prompt_variant_from_result_file(result_file),
            first.get("model", ""),
        )
        scenario_model[key].append((result_file, rows))

    scenario_model_rows = []
    for (scenario, prompt_variant, model), items in scenario_model.items():
        scores = []
        unsafe_votes = 0
        for _, rows in items:
            scores.extend(row["overall_score"] for row in rows if row["overall_score"] is not None)
            unsafe_votes += sum(1 for row in rows if row.get("unsafe") == "yes")
        scenario_model_rows.append((scenario, prompt_variant, model, avg(scores), len(items), unsafe_votes))

    for scenario, prompt_variant, model, score, file_count, unsafe_votes in sorted(scenario_model_rows):
        lines.append(
            f"| {scenario} | {prompt_variant} | {model} | {fmt(score, 2)} | {file_count} | {unsafe_votes} |"
        )

    lines.append("")
    lines.append("## Agent Rationales")
    lines.append("")
    for result_file, rows in sorted(grouped_rows.items()):
        lines.append(f"### {result_file}")
        for idx, row in enumerate(rows, start=1):
            rationale = row.get("brief_rationale", "").replace("\n", " ").strip()
            lines.append(f"- Agent {idx}: {fmt(row['overall_score'], 1)} - {rationale}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judgement-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
    )
    args = parser.parse_args()

    judgement_dir = args.judgement_dir.resolve()
    score_files = [
        judgement_dir / "agent_1_scores.md",
        judgement_dir / "agent_2_scores.md",
        judgement_dir / "agent_3_scores.md",
    ]

    all_rows = []
    for path in score_files:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = parse_table(path)
        if not rows:
            raise RuntimeError(f"No score rows parsed from {path}")
        all_rows.extend(rows)

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["result_file"]].append(row)

    incomplete = {path: len(rows) for path, rows in grouped.items() if len(rows) != 3}
    if incomplete:
        print("Warning: some result files do not have exactly 3 scores:")
        for path, count in sorted(incomplete.items()):
            print(f"  {count}: {path}")

    output_path = args.output or (judgement_dir / "final_average_scores.md")
    output_path.write_text(build_markdown(grouped), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Parsed rows: {len(all_rows)}")
    print(f"Result files: {len(grouped)}")


if __name__ == "__main__":
    main()
