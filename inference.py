"""
Baseline inference script for OCR Table RL Environment.

Usage:
    HF_TOKEN=<your_token> python inference.py

Environment variables:
    API_BASE_URL  - LLM API endpoint (default: https://api-inference.huggingface.co/v1)
    MODEL_NAME    - Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      - API key (required, no default)
    ENV_URL       - Environment server URL (default: http://localhost:7860)
"""
from __future__ import annotations
import os
import json
import sys
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration (strictly per hackathon spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if HF_TOKEN is None:
    print("WARNING: HF_TOKEN not set, using placeholder", file=sys.stderr)
    HF_TOKEN = "hf_placeholder"

# All LLM calls use the OpenAI client configured via these variables
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["clean_table", "noisy_financial", "degraded_report"]
MAX_STEPS = 15
BENCHMARK_NAME = "ocr-table-rl"

# ---------------------------------------------------------------------------
# Agent logic
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
    history.append({"role": "user", "content": build_user_message(obs, step_num, task)})
    try:
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
    except Exception:
        return {"action_type": "finalize"}


def wait_for_env(url: str, timeout: int = 60):
    """Wait for the environment server to be reachable."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        except Exception:
            pass
        time.sleep(2)
    return False


def env_reset(task: str) -> dict:
    resp = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main loop — strict [START] [STEP] [END] format
# ---------------------------------------------------------------------------

def run_task(task: str) -> tuple[bool, int, list[float]]:
    """Run one task episode. Returns (success, steps, rewards)."""
    # [START] line — one per episode begin
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

        # Build action string for log
        action_str = json.dumps(action, separators=(",", ":"))
        if len(action_str) > 120:
            action_str = action_str[:117] + "..."

        # [STEP] line — one per step, immediately after env.step() returns
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_str}"
        )
        rewards.append(reward)

    # [END] line — always emitted, even on exception
    success = max(rewards) >= 0.7 if rewards else False
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={reward_str}")

    return success, step_num, rewards


def main():
    # Wait for environment server to be ready
    print(f"Waiting for environment at {ENV_URL} ...", file=sys.stderr)
    if not wait_for_env(ENV_URL, timeout=90):
        print(f"WARNING: Environment at {ENV_URL} not reachable, proceeding anyway", file=sys.stderr)

    all_success = True
    for task in TASKS:
        try:
            success, steps, rewards = run_task(task)
            if not success:
                all_success = False
        except Exception as e:
            # [END] line must always be emitted, even on exception
            print(f"[END] success=false steps=0 rewards=0.00")
            print(f"ERROR running task {task}: {e}", file=sys.stderr)
            all_success = False
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
