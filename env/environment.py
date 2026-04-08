"""Core OCR Table RL Environment."""
from __future__ import annotations
from typing import Optional

from .models import OCRAction, OCRObservation, OCRState
from .graders import (
    cer, markdown_score, kpi_score, kpi_hallucination_penalty,
    calibration_score, score_task1, score_task2, score_task3,
)
from .tasks import TASK_REGISTRY


class OCREnvironment:
    """
    OpenEnv-compatible environment for OCR Table Extraction RL.

    Three tasks:
      - clean_table      (easy)
      - noisy_financial  (medium)
      - degraded_report  (hard)

    Supports:
      step(action)  -> (OCRObservation, reward, done, info)
      reset(task)   -> OCRObservation
      state()       -> OCRState
    """

    def __init__(self):
        self._task_name: str = "clean_table"
        self._task_data: dict = {}
        self._step: int = 0
        self._max_steps: int = 5
        self._done: bool = False

        # Submitted outputs
        self._markdown: Optional[str] = None
        self._kpis: Optional[dict] = None
        self._confidences: list = []

        # State tracking
        self._best_cer: float = 1.0
        self._kpi_fields_correct: int = 0
        self._active_table: int = 1
        self._last_action_type: Optional[str] = None
        self._repeat_count: int = 0
        self._cropped_hint: Optional[str] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task: str = "clean_table") -> OCRObservation:
        if task not in TASK_REGISTRY:
            task = "clean_table"
        self._task_name = task
        self._task_data = TASK_REGISTRY[task]()
        self._step = 0
        self._max_steps = self._task_data["max_steps"]
        self._done = False
        self._markdown = None
        self._kpis = None
        self._confidences = []
        self._best_cer = 1.0
        self._kpi_fields_correct = 0
        self._active_table = 1
        self._last_action_type = None
        self._repeat_count = 0
        self._cropped_hint = None

        return OCRObservation(
            image_b64=self._task_data.get("image_b64"),
            text_hint=self._task_data.get("text_hint", ""),
            reward=0.0,
            done=False,
            cer=None,
            kpi_score=None,
            error=None,
            metadata={
                "task": self._task_name,
                "max_steps": self._max_steps,
                "instructions": self._instructions(),
            },
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: OCRAction) -> tuple[OCRObservation, float, bool, dict]:
        if self._done:
            obs = OCRObservation(
                text_hint=self._task_data.get("text_hint", ""),
                reward=0.0,
                done=True,
                error="Episode already done. Call reset().",
            )
            return obs, 0.0, True, {"error": "Episode already done."}

        self._step += 1
        reward = 0.0
        error = None

        # Loop detection
        if action.action_type == self._last_action_type:
            self._repeat_count += 1
        else:
            self._repeat_count = 0
        self._last_action_type = action.action_type

        if self._repeat_count >= 2:
            reward -= 0.10

        # Process action
        if action.action_type == "extract_table_md":
            reward += self._handle_extract_md(action)

        elif action.action_type == "extract_kpis":
            reward += self._handle_extract_kpis(action)

        elif action.action_type == "crop_region":
            reward, error = self._handle_crop_region(action)

        elif action.action_type == "retry_region":
            reward += self._handle_retry_region()

        elif action.action_type == "correct_cell":
            reward += self._handle_correct_cell(action)

        elif action.action_type == "switch_table":
            self._active_table = 2 if self._active_table == 1 else 1
            reward = 0.0

        elif action.action_type == "finalize":
            reward, self._done = self._handle_finalize()

        # Max steps exceeded
        if self._step >= self._max_steps and not self._done:
            self._done = True
            # Final score on timeout
            final = self._compute_final_score()
            reward = max(reward, final)

        gt_kpis = self._gt_kpis()
        current_kpi_score = kpi_score(self._kpis or {}, gt_kpis) if gt_kpis else None
        current_cer = cer(self._markdown or "", self._gt_md()) if self._markdown else None

        obs = OCRObservation(
            image_b64=self._cropped_hint,
            text_hint=self._cropped_hint or self._task_data.get("text_hint", ""),
            reward=round(reward, 4),
            done=self._done,
            cer=round(current_cer, 4) if current_cer is not None else None,
            kpi_score=round(current_kpi_score, 4) if current_kpi_score is not None else None,
            error=error,
            metadata={
                "step": self._step,
                "max_steps": self._max_steps,
                "active_table": self._active_table,
            },
        )

        self._cropped_hint = None  # reset after one step
        return obs, round(reward, 4), self._done, {"error": error}

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> OCRState:
        gt_kpis = self._gt_kpis()
        correct = 0
        if self._kpis and gt_kpis:
            correct = sum(1 for k, v in gt_kpis.items()
                          if str(self._kpis.get(k, "")).strip() == str(v).strip())
        return OCRState(
            task_name=self._task_name,
            step=self._step,
            max_steps=self._max_steps,
            best_cer=round(self._best_cer, 4),
            kpi_fields_correct=correct,
            kpi_fields_total=len(gt_kpis) if gt_kpis else 0,
            markdown_submitted=self._markdown is not None,
            kpis_submitted=self._kpis is not None,
            active_table=self._active_table,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gt_md(self) -> str:
        td = self._task_data
        if self._task_name == "degraded_report":
            if self._active_table == 1:
                return td["gt_md"].split("\n---\n")[0]
            else:
                parts = td["gt_md"].split("\n---\n")
                return parts[1] if len(parts) > 1 else td["gt_md"]
        return td.get("gt_md", "")

    def _gt_kpis(self) -> dict:
        td = self._task_data
        if self._task_name == "degraded_report":
            return td.get("gt_kpis", {})
        return td.get("gt_kpis", {})

    def _handle_extract_md(self, action: OCRAction) -> float:
        if not action.markdown:
            return -0.05
        prev_cer = self._best_cer
        gt = self._gt_md()
        new_cer = cer(action.markdown, gt)
        self._markdown = action.markdown
        if self._confidences is not None and action.confidences:
            self._confidences = [c if isinstance(c, dict) else c.model_dump()
                                  for c in action.confidences]
        if new_cer < self._best_cer:
            self._best_cer = new_cer
            return round(0.1 * (prev_cer - new_cer), 4)
        return 0.0

    def _handle_extract_kpis(self, action: OCRAction) -> float:
        if not action.kpis:
            return -0.05
        gt = self._gt_kpis()
        reward = 0.0
        prev_correct = self._kpi_fields_correct
        new_correct = sum(1 for k, v in gt.items()
                          if str(action.kpis.get(k, "")).strip() == str(v).strip())
        gained = max(0, new_correct - prev_correct)
        reward += 0.10 * gained
        hallucinations = kpi_hallucination_penalty(action.kpis, gt)
        reward -= 0.05 * hallucinations
        self._kpis = action.kpis
        self._kpi_fields_correct = new_correct
        return round(reward, 4)

    def _handle_crop_region(self, action: OCRAction) -> tuple[float, Optional[str]]:
        # Simulate zoom: return a sub-hint from text_hint
        hint = self._task_data.get("text_hint", "")
        region = action.region or {}
        r1 = int(region.get("r1", 0))
        r2 = int(region.get("r2", 999))
        lines = hint.splitlines()
        sub = "\n".join(lines[r1:r2]) if lines else hint
        self._cropped_hint = sub  # will be returned as text_hint next step
        return 0.0, None

    def _handle_retry_region(self) -> float:
        # If the cropped region would improve CER, small bonus
        if self._markdown:
            prev = self._best_cer
            gt = self._gt_md()
            # Use clean GT lines as a proxy (real system would re-OCR)
            new_cer = cer(self._markdown, gt)
            if new_cer < prev:
                self._best_cer = new_cer
                return 0.05
        return 0.0

    def _handle_correct_cell(self, action: OCRAction) -> float:
        if action.cell_row is None or action.cell_col is None or not action.cell_value:
            return -0.02
        if not self._markdown:
            return -0.02
        # Apply cell correction to markdown
        lines = self._markdown.splitlines()
        data_lines = [l for l in lines if l.strip().startswith("|") and "---" not in l]
        ri = action.cell_row
        ci = action.cell_col
        if ri < len(data_lines):
            cells = data_lines[ri].strip("|").split("|")
            if ci < len(cells):
                cells[ci] = f" {action.cell_value} "
                data_lines[ri] = "|" + "|".join(cells) + "|"
                # Reconstruct markdown
                full = []
                di = 0
                for l in lines:
                    if l.strip().startswith("|") and "---" not in l:
                        full.append(data_lines[di] if di < len(data_lines) else l)
                        di += 1
                    else:
                        full.append(l)
                self._markdown = "\n".join(full)
                new_cer = cer(self._markdown, self._gt_md())
                if new_cer < self._best_cer:
                    self._best_cer = new_cer
                    return 0.05
        return 0.0

    def _handle_finalize(self) -> tuple[float, bool]:
        if not self._markdown and not self._kpis:
            return -0.20, True
        score = self._compute_final_score()
        return round(score, 4), True

    def _compute_final_score(self) -> float:
        td = self._task_data
        gt_md = td.get("gt_md", "")
        gt_kpis = self._gt_kpis()
        task_id = td.get("task_id", 1)

        if task_id == 1:
            return score_task1(self._markdown or "", self._kpis or {}, gt_md, gt_kpis)
        elif task_id == 2:
            gt_cells = td.get("gt_cells", {})
            return score_task2(
                self._markdown or "", self._kpis or {}, self._confidences,
                gt_md, gt_kpis, gt_cells,
            )
        else:
            return score_task3(
                self._markdown or "", self._kpis or {}, gt_md, gt_kpis,
                steps_used=self._step,
            )

    def _instructions(self) -> str:
        base = (
            "You are an OCR agent. Extract the table(s) from the document.\n"
            "Actions: extract_table_md, extract_kpis, crop_region, retry_region, "
            "correct_cell, switch_table (task 3 only), finalize.\n"
            "Output BOTH a Markdown table AND a JSON KPI dict before calling finalize.\n"
        )
        if self._task_name == "degraded_report":
            base += "This document has TWO tables. Use switch_table to toggle between them.\n"
        return base
