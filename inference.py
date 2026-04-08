"""
Baseline inference script for OCR Table RL Environment.

Usage:
    HF_TOKEN=<your_token> python inference.py

Environment variables:
    API_BASE_URL   - LLM API endpoint (default: https://api-inference.huggingface.co/v1)
    MODEL_NAME     - Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       - API key (required for LLM calls)
    ENV_BASE_URL   - Environment server URL. If not set, runs environment in-process.
"""
from __future__ import annotations
import os
import json
import sys
import time
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["clean_table", "noisy_financial", "degraded_report"]
MAX_STEPS = 15
BENCHMARK_NAME = "ocr-table-rl"

# ---------------------------------------------------------------------------
# Environment access — in-process or remote
# ---------------------------------------------------------------------------

_local_env = None


def _get_local_env():
    """Lazy-init a local in-process environment."""
    global _local_env
    if _local_env is None:
        from env.environment import OCREnvironment
        _local_env = OCREnvironment()
    return _local_env


def env_reset(task: str) -> dict:
    if ENV_BASE_URL:
        resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    else:
        env = _get_local_env()
        obs = env.reset(task=task)
        return obs.model_dump()


def env_step(action: dict) -> dict:
    if ENV_BASE_URL:
        resp = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
        resp.raise_for_status()
        return resp.json()
    else:
        from env.models import OCRAction
        env = _get_local_env()
        act = OCRAction(**action)
        obs, reward, done, info = env.step(act)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert OCR agent that extracts structured tables from documents.
You receive a text_hint (noisy OCR output) and sometimes an image (base64 PNG).

Your goal:
1. Extract the table as a proper Markdown table
2. Extract key KPIs as a JSON dict with semantic labels
3. Call finalize when ready

Available action_types:
- extract_table_md: submit markdown table (field: "markdown")
- extract_kpis: submit KPI JSON dict (field: "kpis")
- crop_region: zoom into region (field: "region": {"r1": int, "r2": int})
- retry_region: re-extract after crop
- correct_cell: fix a cell (fields: "cell_row", "cell_col", "cell_value")
- switch_table: toggle between table1/table2 (task degraded_report only)
- finalize: commit outputs and end episode

Always respond with a single JSON object matching one action.
Example: {"action_type": "extract_table_md", "markdown": "| A | B |\\n|---|---|\\n| 1 | 2 |"}
"""


def build_user_message(obs: dict, step_num: int, task: str) -> str:
    text_hint = obs.get("text_hint", "")
    cer_val = obs.get("cer")
    kpi_val = obs.get("kpi_score")
    error = obs.get("error")

    msg = f"Step {step_num} | Task: {task}\n"
    msg += f"Text hint (OCR output):\n{text_hint}\n\n"
    if cer_val is not None:
        msg += f"Current CER: {cer_val:.3f} (lower is better)\n"
    if kpi_val is not None:
        msg += f"Current KPI score: {kpi_val:.3f}\n"
    if error:
        msg += f"Last error: {error}\n"
    msg += "\nRespond with one action JSON."
    return msg


def call_agent(obs: dict, history: list, step_num: int, task: str) -> dict:
    """Call LLM via OpenAI client and return a parsed action dict."""
    if not HF_TOKEN:
        # No LLM available — use a simple heuristic fallback
        return _heuristic_action(obs, step_num, task)

    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

        history.append({"role": "user", "content": build_user_message(obs, step_num, task)})
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": content})

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        action = json.loads(content)
        return action
    except Exception as e:
        print(f"LLM call failed: {e}", file=sys.stderr)
        return {"action_type": "finalize"}


def _heuristic_action(obs: dict, step_num: int, task: str) -> dict:
    """Simple heuristic agent when no LLM is available."""
    text_hint = obs.get("text_hint", "")

    if step_num == 1:
        # First step: try to extract markdown from the text hint
        # Parse the hint as a rough markdown table
        lines = text_hint.strip().splitlines()
        md_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("("):
                # Convert to table row
                cells = [c.strip() for c in stripped.split("  ") if c.strip()]
                if cells:
                    md_lines.append("| " + " | ".join(cells) + " |")

        if len(md_lines) >= 2:
            # Insert separator after header
            ncols = md_lines[0].count("|") - 1
            sep = "| " + " | ".join(["---"] * max(ncols, 1)) + " |"
            md = md_lines[0] + "\n" + sep + "\n" + "\n".join(md_lines[1:])
        else:
            md = text_hint

        return {"action_type": "extract_table_md", "markdown": md}

    elif step_num == 2:
        # Second step: extract KPIs from the hint
        kpis = {}
        lines = text_hint.strip().splitlines()
        for line in lines:
            parts = line.strip().split("  ")
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                key = parts[0].lower().replace(" ", "_").replace("/", "_")
                key = "".join(c for c in key if c.isalnum() or c == "_").strip("_")
                # Find first value that looks numeric
                for v in parts[1:]:
                    v_clean = v.replace(",", "").replace("$", "").replace("%", "")
                    if any(c.isdigit() for c in v_clean):
                        kpis[key] = v.strip()
                        break
        if kpis:
            return {"action_type": "extract_kpis", "kpis": kpis}
        return {"action_type": "extract_kpis", "kpis": {"total": "0"}}

    else:
        return {"action_type": "finalize"}


# ---------------------------------------------------------------------------
# Main loop — strict [START] [STEP] [END] format
# ---------------------------------------------------------------------------

def run_task(task: str) -> tuple[bool, int, list[float]]:
    """Run one task episode. Returns (success, steps, rewards)."""
    print(f"[START] task={task} env={BENCHMARK_NAME} model={MODEL_NAME}")

    obs = env_reset(task)

    rewards: list[float] = []
    history: list[dict] = []
    step_num = 0
    done = False

    while not done and step_num < MAX_STEPS:
        step_num += 1
        action = call_agent(obs, history, step_num, task)

        result = env_step(action)
        obs = result["observation"]
        reward = float(result["reward"])
        done = bool(result["done"])
        last_error = result.get("info", {}).get("error")
        error_str = last_error if last_error else "null"

        action_str = json.dumps(action, separators=(",", ":"))
        if len(action_str) > 120:
            action_str = action_str[:117] + "..."

        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_str}"
        )
        rewards.append(reward)

    success = max(rewards) >= 0.7 if rewards else False
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={reward_str}")

    return success, step_num, rewards


def main():
    if ENV_BASE_URL:
        # Wait for remote environment to be ready
        print(f"Connecting to environment at {ENV_BASE_URL} ...", file=sys.stderr)
        start = time.time()
        while time.time() - start < 60:
            try:
                resp = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
    else:
        print("Running environment in-process (no ENV_BASE_URL set)", file=sys.stderr)

    all_success = True
    for task in TASKS:
        try:
            success, steps, rewards = run_task(task)
            if not success:
                all_success = False
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.00")
            print(f"ERROR running task {task}: {e}", file=sys.stderr)
            all_success = False
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
