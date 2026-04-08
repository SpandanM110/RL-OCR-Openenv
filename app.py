"""
Gradio UI for OCR Table RL Environment.
Mounted onto the FastAPI app so both the OpenEnv API and the demo
share port 7860 on HuggingFace Spaces.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from env.environment import OCREnvironment
from env.tasks import TASK_METADATA

# One shared env instance for the demo
_env = OCREnvironment()

TASK_IDS = [t["id"] for t in TASK_METADATA]
TASK_LABELS = {
    "clean_table": "📄 Clean Printed Table (Easy)",
    "noisy_financial": "📊 Noisy Financial Statement (Medium)",
    "degraded_report": "🌫️ Degraded Multi-Table Report (Hard)",
}

ACTION_TYPES = [
    "extract_table_md",
    "extract_kpis",
    "crop_region",
    "retry_region",
    "correct_cell",
    "switch_table",
    "finalize",
]


def do_reset(task_id: str):
    obs = _env.reset(task=task_id)
    image = None
    if obs.image_b64:
        import base64, io
        from PIL import Image
        image = Image.open(io.BytesIO(base64.b64decode(obs.image_b64)))
    hint = obs.text_hint or "(no text hint)"
    instructions = obs.metadata.get("instructions", "")
    history = f"**Task reset:** `{task_id}`\n\n{instructions}"
    return image, hint, history, "", "", 0.0


def do_step(action_type: str, markdown: str, kpis_str: str, history: str):
    from env.models import OCRAction

    kpis = None
    if kpis_str.strip():
        try:
            kpis = json.loads(kpis_str)
        except Exception:
            kpis = None

    action = OCRAction(
        action_type=action_type,
        markdown=markdown.strip() if markdown.strip() else None,
        kpis=kpis,
    )

    obs, reward, done, info = _env.step(action)
    state = _env.state()

    error_msg = f" ⚠️ `{obs.error}`" if obs.error else ""
    cer_msg = f" | CER: `{obs.cer:.3f}`" if obs.cer is not None else ""
    kpi_msg = f" | KPI score: `{obs.kpi_score:.3f}`" if obs.kpi_score is not None else ""
    done_msg = " ✅ **Done!**" if done else ""

    step_line = (
        f"\n\n**Step {state.step}** | `{action_type}` | "
        f"Reward: `{reward:.3f}`{cer_msg}{kpi_msg}{error_msg}{done_msg}"
    )
    new_history = history + step_line

    hint = obs.text_hint or "(no updated hint)"
    return hint, new_history, reward


with gr.Blocks(
    title="OCR Table RL Environment",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 1100px; margin: auto; }",
) as demo:
    gr.Markdown(
        """
# 📄 OCR Table RL Environment
**OpenEnv-compatible RL benchmark** for structured document table extraction.

Agents extract complex tables from document images into **Markdown** + **JSON KPIs**.
Three tasks of increasing difficulty: clean → noisy → degraded multi-table.

> API endpoints: `/reset` · `/step` · `/state` · `/health` · `/tasks` · `/docs`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Controls")
            task_dd = gr.Dropdown(
                choices=list(TASK_LABELS.values()),
                value=list(TASK_LABELS.values())[0],
                label="Select Task",
            )
            reset_btn = gr.Button("🔄 Reset Episode", variant="primary")

            gr.Markdown("### ⚡ Take a Step")
            action_dd = gr.Dropdown(
                choices=ACTION_TYPES,
                value="extract_table_md",
                label="Action Type",
            )
            md_input = gr.Textbox(
                label="Markdown (for extract_table_md / correct_cell)",
                placeholder="| Col1 | Col2 |\n|---|---|\n| val | val |",
                lines=5,
            )
            kpis_input = gr.Textbox(
                label="KPIs JSON (for extract_kpis)",
                placeholder='{"total_q1": "10220", "best_product": "Widget E"}',
                lines=3,
            )
            step_btn = gr.Button("▶️ Step", variant="secondary")
            reward_out = gr.Number(label="Last Step Reward", value=0.0, precision=4)

        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Document Image")
            img_out = gr.Image(label="Document (on reset / after crop)", type="pil", height=350)
            gr.Markdown("### 📝 OCR Text Hint")
            hint_out = gr.Textbox(label="Noisy OCR Text", lines=8, interactive=False)
            gr.Markdown("### 📜 Episode History")
            history_out = gr.Markdown("*Reset an episode to start.*")

    # --- Task label → id mapping ---
    label_to_id = {v: k for k, v in TASK_LABELS.items()}

    def _reset(task_label):
        task_id = label_to_id.get(task_label, "clean_table")
        return do_reset(task_id)

    reset_btn.click(
        fn=_reset,
        inputs=[task_dd],
        outputs=[img_out, hint_out, history_out, md_input, kpis_input, reward_out],
    )

    step_btn.click(
        fn=do_step,
        inputs=[action_dd, md_input, kpis_input, history_out],
        outputs=[hint_out, history_out, reward_out],
    )

    with gr.Accordion("📋 Task Descriptions", open=False):
        for t in TASK_METADATA:
            gr.Markdown(
                f"**{t['title']}** (`{t['id']}`) — Difficulty: `{t['difficulty']}`\n\n"
                f"{t['description']}\n\n"
                f"Optimal steps: `{t['optimal_steps']}` | Max steps: `{t['max_steps']}`"
            )

    with gr.Accordion("🔌 API Usage", open=False):
        gr.Markdown(
            """
```bash
# Reset
curl -X POST https://SpandanM110-RL-OCR-Openenv.hf.space/reset \\
  -H 'Content-Type: application/json' \\
  -d '{"task": "clean_table"}'

# Step
curl -X POST https://SpandanM110-RL-OCR-Openenv.hf.space/step \\
  -H 'Content-Type: application/json' \\
  -d '{"action_type": "extract_table_md", "markdown": "| A | B |\\n|---|---|\\n| 1 | 2 |"}'

# Health
curl https://SpandanM110-RL-OCR-Openenv.hf.space/health
```
            """
        )
