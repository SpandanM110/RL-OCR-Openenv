"""Graders for OCR Table RL Environment.

All graders return a float in (0.0, 1.0) — strictly between 0 and 1.
"""
from __future__ import annotations
import re

# Clamp final task scores to open interval (0, 1)
_EPS = 0.001


def _clamp(score: float) -> float:
    """Clamp score to strictly within (0, 1)."""
    return max(_EPS, min(1.0 - _EPS, score))


# ---------------------------------------------------------------------------
# Character Error Rate
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        prev = dp[0]
        dp[0] = i + 1
        for j, cb in enumerate(b):
            temp = dp[j + 1]
            dp[j + 1] = prev if ca == cb else 1 + min(prev, dp[j], dp[j + 1])
            prev = temp
    return dp[len(b)]


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate: edit_distance / len(reference), clamped to [0,1]."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = _levenshtein(hypothesis.strip(), reference.strip())
    return min(1.0, dist / max(len(reference.strip()), 1))


# ---------------------------------------------------------------------------
# Markdown table parsing + scoring
# ---------------------------------------------------------------------------

def _parse_md_table(md: str) -> list[list[str]]:
    """Return list of rows (each a list of cell strings), skip separator rows."""
    rows = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip separator rows like |---|---|
        if re.match(r"^\|[\s\-:]+\|", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def markdown_score(md_output: str, md_ground_truth: str) -> float:
    """Cell-level exact match: correct_cells / total_gt_cells."""
    if not md_output:
        return 0.0
    gt_rows = _parse_md_table(md_ground_truth)
    out_rows = _parse_md_table(md_output)
    if not gt_rows:
        return 1.0
    total = sum(len(r) for r in gt_rows)
    correct = 0
    for i, gt_row in enumerate(gt_rows):
        if i >= len(out_rows):
            break
        out_row = out_rows[i]
        for j, gt_cell in enumerate(gt_row):
            if j < len(out_row) and out_row[j].strip() == gt_cell.strip():
                correct += 1
    return correct / total


# ---------------------------------------------------------------------------
# KPI field scoring
# ---------------------------------------------------------------------------

def _numeric_close(a: str, b: str, tol: float = 0.01) -> bool:
    """True if both parse as numbers and are within tol * |b|."""
    try:
        fa = float(re.sub(r"[,$%]", "", a))
        fb = float(re.sub(r"[,$%]", "", b))
        if fb == 0:
            return fa == 0
        return abs(fa - fb) / abs(fb) <= tol
    except ValueError:
        return False


def kpi_score(kpis: dict, ground_truth: dict) -> float:
    """Per-field match: numeric within 1%, string exact. Returns correct/total."""
    if not ground_truth:
        return 1.0
    if not kpis:
        return 0.0
    correct = 0
    for key, gt_val in ground_truth.items():
        agent_val = str(kpis.get(key, "")).strip()
        gt_str = str(gt_val).strip()
        if agent_val == gt_str or _numeric_close(agent_val, gt_str):
            correct += 1
    return correct / len(ground_truth)


def kpi_hallucination_penalty(kpis: dict, ground_truth: dict) -> float:
    """Return count of keys in kpis not present in ground_truth."""
    if not kpis:
        return 0
    return sum(1 for k in kpis if k not in ground_truth)


# ---------------------------------------------------------------------------
# Calibration (Brier score, inverted to reward)
# ---------------------------------------------------------------------------

def calibration_score(confidences: list, gt_cells: dict) -> float:
    """
    Brier score component: mean((conf - correct)^2), inverted to [0,1].
    gt_cells: dict of "(row,col)" -> bool (True if correct).
    """
    if not confidences:
        return 0.5  # neutral if no confidences given
    brier = 0.0
    for c in confidences:
        key = f"({c.get('row')},{c.get('col')})"
        is_correct = float(gt_cells.get(key, False))
        conf = float(c.get("confidence", 0.5))
        brier += (conf - is_correct) ** 2
    mean_brier = brier / len(confidences)
    return max(0.0, 1.0 - mean_brier)  # perfect calibration → 1.0


# ---------------------------------------------------------------------------
# Final task scorers
# ---------------------------------------------------------------------------

def score_task1(markdown: str, kpis: dict, gt_md: str, gt_kpis: dict) -> float:
    md = markdown_score(markdown or "", gt_md)
    kp = kpi_score(kpis or {}, gt_kpis)
    return _clamp(round(0.5 * md + 0.5 * kp, 4))


def score_task2(
    markdown: str,
    kpis: dict,
    confidences: list,
    gt_md: str,
    gt_kpis: dict,
    gt_cells: dict,
) -> float:
    md = markdown_score(markdown or "", gt_md)
    kp = kpi_score(kpis or {}, gt_kpis)
    cal = calibration_score(confidences or [], gt_cells)
    return _clamp(round(0.4 * md + 0.4 * kp + 0.2 * cal, 4))


def score_task3(
    markdown: str,
    kpis: dict,
    gt_md: str,
    gt_kpis: dict,
    steps_used: int,
    optimal_steps: int = 8,
) -> float:
    md = markdown_score(markdown or "", gt_md)
    kp = kpi_score(kpis or {}, gt_kpis)
    efficiency = max(0.0, 1.0 - 0.02 * max(0, steps_used - optimal_steps))
    return _clamp(round(0.35 * md + 0.35 * kp + 0.15 * efficiency + 0.15 * md, 4))
