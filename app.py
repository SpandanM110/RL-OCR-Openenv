"""
OCR Table RL Environment — HuggingFace Space (Gradio SDK).
Gradio UI for interactive demo + OpenEnv API endpoints.
"""
import json
import sys
import os
import base64
import io

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from pydantic import BaseModel

from env.environment import OCREnvironment
from env.models import OCRAction, OCRObservation, OCRState
from env.tasks import TASK_METADATA

# ===========================================================================
# Environment instances
# ===========================================================================
_ui_env = OCREnvironment()
_api_env = OCREnvironment()

TASK_LABELS = {
    "clean_table": "Clean Printed Table (Easy)",
    "noisy_financial": "Noisy Financial Statement (Medium)",
    "degraded_report": "Degraded Multi-Table Report (Hard)",
}
ACTION_TYPES = [
    "extract_table_md", "extract_kpis", "crop_region",
    "retry_region", "correct_cell", "switch_table", "finalize",
]
label_to_id = {v: k for k, v in TASK_LABELS.items()}


# ===========================================================================
# Gradio callbacks
# ===========================================================================

def do_reset(task_label: str):
    task_id = label_to_id.get(task_label, "clean_table")
    obs = _ui_env.reset(task=task_id)
    image = None
    if obs.image_b64:
        from PIL import Image as PILImage
        image = PILImage.open(io.BytesIO(base64.b64decode(obs.image_b64)))
    hint = obs.text_hint or "(no text hint)"
    instructions = obs.metadata.get("instructions", "")
    history = f"**Task reset:** `{task_id}`\n\n{instructions}"
    return image, hint, history, "", "", 0.0


def do_step(action_type, markdown, kpis_str, history):
    kpis = None
    if kpis_str and kpis_str.strip():
        try:
            kpis = json.loads(kpis_str)
        except Exception:
            pass
    action = OCRAction(
        action_type=action_type,
        markdown=markdown.strip() if markdown and markdown.strip() else None,
        kpis=kpis,
    )
    obs, reward, done, info = _ui_env.step(action)
    state = _ui_env.state()
    parts = [f"\n\n**Step {state.step}** | `{action_type}` | Reward: `{reward:.3f}`"]
    if obs.cer is not None:
        parts.append(f"CER: `{obs.cer:.3f}`")
    if obs.kpi_score is not None:
        parts.append(f"KPI: `{obs.kpi_score:.3f}`")
    if obs.error:
        parts.append(f"Error: `{obs.error}`")
    if done:
        parts.append("**DONE**")
    return obs.text_hint or "(no hint)", history + " | ".join(parts), reward


# ===========================================================================
# Build Gradio app
# ===========================================================================

with gr.Blocks(
    title="OCR Table RL Environment",
) as demo:
    gr.Markdown(
        "# OCR Table RL Environment\n"
        "**OpenEnv-compatible RL benchmark** for structured document table extraction.\n\n"
        "Extract complex tables from document images into **Markdown** + **JSON KPIs**.\n"
        "Three tasks: clean (easy) / noisy financial (medium) / degraded multi-table (hard)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            task_dd = gr.Dropdown(choices=list(TASK_LABELS.values()),
                                  value=list(TASK_LABELS.values())[0], label="Select Task")
            reset_btn = gr.Button("Reset Episode", variant="primary")
            gr.Markdown("### Take a Step")
            action_dd = gr.Dropdown(choices=ACTION_TYPES, value="extract_table_md", label="Action Type")
            md_input = gr.Textbox(label="Markdown", placeholder="| Col1 | Col2 |\n|---|---|\n| v | v |", lines=5)
            kpis_input = gr.Textbox(label="KPIs JSON", placeholder='{"key": "value"}', lines=3)
            step_btn = gr.Button("Step", variant="secondary")
            reward_out = gr.Number(label="Last Reward", value=0.0, precision=4)
        with gr.Column(scale=2):
            gr.Markdown("### Document Image")
            img_out = gr.Image(label="Document", type="pil", height=350)
            gr.Markdown("### OCR Text Hint")
            hint_out = gr.Textbox(label="Noisy OCR Text", lines=8, interactive=False)
            gr.Markdown("### Episode Log")
            history_out = gr.Markdown("*Reset an episode to start.*")

    reset_btn.click(do_reset, [task_dd],
                    [img_out, hint_out, history_out, md_input, kpis_input, reward_out])
    step_btn.click(do_step, [action_dd, md_input, kpis_input, history_out],
                   [hint_out, history_out, reward_out])

    with gr.Accordion("Task Descriptions", open=False):
        for t in TASK_METADATA:
            gr.Markdown(f"**{t['title']}** (`{t['id']}`) -- `{t['difficulty']}`\n\n"
                        f"{t['description']}\n\n"
                        f"Optimal: `{t['optimal_steps']}` steps | Max: `{t['max_steps']}` steps")

    with gr.Accordion("OpenEnv API Endpoints", open=False):
        gr.Markdown(
            "```\n"
            "GET  /health   -> {status, env, version}\n"
            "GET  /tasks    -> {tasks: [...]}\n"
            "GET  /state    -> OCRState\n"
            "POST /reset    -> OCRObservation   body: {task: 'clean_table'}\n"
            "POST /step     -> StepResponse     body: OCRAction JSON\n"
            "```"
        )


# ===========================================================================
# Mount OpenEnv API on Gradio's underlying FastAPI
# ===========================================================================

class ResetRequest(BaseModel):
    task: str = "clean_table"


# Gradio Blocks has an internal FastAPI app we can add routes to
_app = demo.app


@_app.get("/health")
async def health():
    return {"status": "healthy", "env": "ocr-table-rl", "version": "1.0.0"}


@_app.get("/tasks")
async def tasks():
    return {"tasks": TASK_METADATA}


@_app.post("/reset")
async def reset(req: ResetRequest):
    valid = {"clean_table", "noisy_financial", "degraded_report"}
    t = req.task if req.task in valid else "clean_table"
    return _api_env.reset(task=t).model_dump()


@_app.post("/step")
async def step(action: OCRAction):
    obs, reward, done, info = _api_env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@_app.get("/state")
async def state():
    return _api_env.state().model_dump()


# ===========================================================================
# Launch (HF Spaces will auto-detect `demo` and call demo.launch())
# ===========================================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
