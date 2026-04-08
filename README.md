---
title: OCR Table RL Environment
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
tags:
  - openenv
  - rl-environment
  - ocr
  - table-extraction
pinned: false
---

# OCR Table RL Environment

[![openenv](https://img.shields.io/badge/openenv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

An OpenEnv-compatible Reinforcement Learning environment for **structured document table extraction**. Agents learn to extract complex tables from document images into structured **Markdown** and **JSON KPIs** — a genuinely hard, real-world problem where current OCR solutions fall short.

---

## Motivation

Current OCR pipelines break on:
- Multi-level headers and merged cells
- Noisy / low-contrast scans
- Rotated or skewed documents
- Semantic labeling of extracted values (KPI naming)

This environment creates an RL feedback loop: agents iterate over a document using crop, retry, and correct actions, receiving incremental rewards as their extraction quality improves — mimicking how a human analyst would refine an OCR pass.

**Novel contributions:**
- **Dual output grading** — Markdown table fidelity + JSON KPI semantic labeling, both graded
- **Multi-pass refinement** — `crop_region` → `retry_region` → `correct_cell` loop
- **KPI hallucination penalty** — agent penalized for inventing field names not in ground truth
- **Calibrated confidence** — optional per-cell confidence scoring via Brier score component
- **Noise curriculum** — clean → medium noise → heavy degradation + rotation

---

## Action Space

| `action_type` | Fields | Description |
|---|---|---|
| `extract_table_md` | `markdown: str` | Submit Markdown table extraction |
| `extract_kpis` | `kpis: dict` | Submit JSON dict of labeled KPIs |
| `crop_region` | `region: {r1, r2}` | Zoom into row range of the document (returns sub-hint) |
| `retry_region` | — | Re-extract after crop |
| `correct_cell` | `cell_row, cell_col, cell_value` | Fix a specific cell in current Markdown |
| `switch_table` | — | Toggle between Table 1 / Table 2 (Task 3 only) |
| `finalize` | — | Commit current outputs and end episode |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `image_b64` | `str \| null` | Base64-encoded PNG of the document (on reset, after crop) |
| `text_hint` | `str` | Noisy simulated OCR text (always available) |
| `reward` | `float` | Step reward |
| `done` | `bool` | Episode complete |
| `cer` | `float \| null` | Current Character Error Rate vs ground truth |
| `kpi_score` | `float \| null` | Current KPI field accuracy |
| `error` | `str \| null` | Error message if last action failed |
| `metadata` | `dict` | Step, max_steps, active_table |

---

## Tasks

### Task 1 — Clean Printed Table (Easy)
- **Document**: Single-table sales report (5 cols × 7 data rows), clean font, white background
- **Agent goal**: Markdown table + `{ "total_q1", "total_q4", "best_product" }`
- **Optimal steps**: 2 | **Max steps**: 5
- **Grader**: 50% Markdown cell accuracy + 50% KPI field match

### Task 2 — Noisy Financial Statement (Medium)
- **Document**: Multi-header annual financial table with Gaussian noise + 1.5° rotation
- **Agent goal**: Markdown table + `{ "revenue_fy2024", "ebitda_margin", "yoy_growth_pct", "net_income_fy2024" }`
- **Optimal steps**: 4 | **Max steps**: 10
- **Grader**: 40% Markdown + 40% KPI + 20% calibration (optional confidence scoring)

### Task 3 — Degraded Multi-Table Report (Hard)
- **Document**: Two tables (Operational KPIs + Financial Summary) on one page, heavy noise, 3.5° rotation
- **Agent goal**: Markdown for both tables (separated by `---`) + nested JSON `{ "table1": {...}, "table2": {...} }`
- **Optimal steps**: 8 | **Max steps**: 15
- **Grader**: 35% per-table Markdown + 35% KPI + 15% cross-table consistency + 15% efficiency

---

## Reward Function

| Signal | Value |
|---|---|
| `extract_table_md` improves CER | `+0.1 × ΔCER` |
| `extract_kpis` — new correct field | `+0.10` per field |
| `extract_kpis` — hallucinated key | `−0.05` per key |
| `crop_region` + `retry_region` improving CER | `+0.05` |
| `correct_cell` fixing an error | `+0.05` |
| Same action 3× in a row (loop) | `−0.10` |
| `finalize` with empty submission | `−0.20` |
| `finalize` with full submission | Final grader score (0.0–1.0) |
| Steps over efficiency budget | `−0.02` per excess step |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all 3 tasks with metadata |
| `POST` | `/reset` | `{"task": "clean_table\|noisy_financial\|degraded_report"}` |
| `POST` | `/step` | OCRAction JSON → `{observation, reward, done, info}` |
| `GET` | `/state` | Current environment state |
| `GET` | `/docs` | Interactive API docs (Swagger UI) |

---

## Setup & Usage

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t ocr-table-rl .
docker run -p 8000:8000 -e HF_TOKEN=your_token ocr-table-rl
```

### Run Inference Baseline

```bash
# Start env server first, then:
export HF_TOKEN=your_token
export MODEL_NAME=gpt-4.1-mini        # or any OpenAI-compatible model
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

### Quick Test

```bash
# Reset to easy task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "clean_table"}'

# Submit a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "extract_table_md", "markdown": "| Product | Q1 |\n|---|---|\n| Widget A | 1200 |"}'

# Finalize
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "finalize"}'
```

---

## Inference Output Format

```
[START] task=clean_table env=ocr-table-rl model=gpt-4.1-mini
[STEP] step=1 action={"action_type":"extract_table_md",...} reward=0.05 done=false error=null
[STEP] step=2 action={"action_type":"extract_kpis",...} reward=0.30 done=false error=null
[STEP] step=3 action={"action_type":"finalize"} reward=0.82 done=true error=null
[END] success=true steps=3 rewards=0.05,0.30,0.82
```

---

## Baseline Performance (gpt-4.1-mini)

| Task | Avg Score | Avg Steps |
|---|---|---|
| clean_table | ~0.75 | 2–3 |
| noisy_financial | ~0.50 | 4–6 |
| degraded_report | ~0.30 | 8–12 |

*Scores improve significantly with vision-capable models (GPT-4o, Claude 3.5 Sonnet) when using `image_b64`.*

---

## Environment Variables

| Variable | Default | Required |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4.1-mini` | No |
| `HF_TOKEN` | — | **Yes** |
| `ENV_URL` | `http://localhost:8000` | No |

---

## License

MIT
