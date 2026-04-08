"""
OCR Table RL Environment — HuggingFace Space (Gradio SDK).
Gradio UI for interactive demo + OpenEnv API endpoints.
Supports uploading custom PDFs/images for table extraction.
"""
import json
import sys
import os
import base64
import io

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from pydantic import BaseModel
from PIL import Image as PILImage

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
# Upload helpers
# ===========================================================================

def _file_to_pil(file_path: str) -> PILImage.Image:
    """Convert uploaded file (image or PDF) to PIL Image."""
    if file_path.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=200)
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except ImportError:
            # Fallback: try pdf2image
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(file_path, first_page=1, last_page=1, dpi=200)
                return images[0]
            except ImportError:
                return None
    else:
        return PILImage.open(file_path).convert("RGB")


def _pil_to_b64(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def handle_upload(file):
    """Process uploaded PDF/image and return preview + extracted text hint."""
    if file is None:
        return None, "(no file uploaded)", ""

    file_path = file if isinstance(file, str) else file.name
    img = _file_to_pil(file_path)
    if img is None:
        return None, "Error: Could not process this file. Install PyMuPDF for PDF support.", ""

    # Generate a noisy text hint (simulate what an OCR would produce)
    b64 = _pil_to_b64(img)
    hint = (
        "Uploaded document detected. The image has been loaded.\n"
        "Use the RL environment tasks for graded evaluation, "
        "or paste extracted text/markdown below to test grading."
    )
    return img, hint, b64


# ===========================================================================
# Gradio callbacks
# ===========================================================================

def do_reset(task_label: str):
    task_id = label_to_id.get(task_label, "clean_table")
    obs = _ui_env.reset(task=task_id)
    image = None
    if obs.image_b64:
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

    with gr.Tabs():
        # ===============================================================
        # Tab 1: RL Environment (graded tasks)
        # ===============================================================
        with gr.Tab("RL Environment"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Controls")
                    task_dd = gr.Dropdown(
                        choices=list(TASK_LABELS.values()),
                        value=list(TASK_LABELS.values())[0],
                        label="Select Task",
                    )
                    reset_btn = gr.Button("Reset Episode", variant="primary")

                    gr.Markdown("### Take a Step")
                    action_dd = gr.Dropdown(
                        choices=ACTION_TYPES,
                        value="extract_table_md",
                        label="Action Type",
                    )
                    md_input = gr.Textbox(
                        label="Markdown (for extract_table_md)",
                        placeholder="| Col1 | Col2 |\n|---|---|\n| v | v |",
                        lines=5,
                    )
                    kpis_input = gr.Textbox(
                        label="KPIs JSON (for extract_kpis)",
                        placeholder='{"total_q1": "10220", "best_product": "Widget E"}',
                        lines=3,
                    )
                    step_btn = gr.Button("Step", variant="secondary")
                    reward_out = gr.Number(label="Last Reward", value=0.0, precision=4)

                with gr.Column(scale=2):
                    gr.Markdown("### Document Image")
                    img_out = gr.Image(label="Document", type="pil", height=350)
                    gr.Markdown("### OCR Text Hint")
                    hint_out = gr.Textbox(label="Noisy OCR Text", lines=8, interactive=False)
                    gr.Markdown("### Episode Log")
                    history_out = gr.Markdown("*Reset an episode to start.*")

            reset_btn.click(
                do_reset, [task_dd],
                [img_out, hint_out, history_out, md_input, kpis_input, reward_out],
            )
            step_btn.click(
                do_step, [action_dd, md_input, kpis_input, history_out],
                [hint_out, history_out, reward_out],
            )

        # ===============================================================
        # Tab 2: Upload Your Document
        # ===============================================================
        with gr.Tab("Upload Document"):
            gr.Markdown(
                "### Upload a PDF or Image\n"
                "Upload your own document containing tables. "
                "The image will be displayed and you can use the RL environment "
                "to practice extraction, or manually test markdown/KPI grading."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.File(
                        label="Upload PDF or Image",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("Process Document", variant="primary")
                    upload_b64_state = gr.State("")

                    gr.Markdown("### Test Grading")
                    gr.Markdown(
                        "Paste your extracted markdown table and KPIs below "
                        "to see how the grader scores them against ground truth."
                    )
                    upload_md = gr.Textbox(
                        label="Your Markdown Table",
                        placeholder="| Header1 | Header2 |\n|---|---|\n| val1 | val2 |",
                        lines=8,
                    )
                    upload_kpis = gr.Textbox(
                        label="Your KPIs (JSON)",
                        placeholder='{"revenue": "$1,000", "growth": "15%"}',
                        lines=4,
                    )
                    grade_btn = gr.Button("Grade Extraction", variant="secondary")
                    grade_result = gr.Markdown("")

                with gr.Column(scale=2):
                    gr.Markdown("### Document Preview")
                    upload_img_out = gr.Image(
                        label="Uploaded Document", type="pil", height=450
                    )
                    upload_hint = gr.Textbox(
                        label="Document Info", lines=4, interactive=False
                    )

            upload_btn.click(
                handle_upload, [upload_input],
                [upload_img_out, upload_hint, upload_b64_state],
            )

            def grade_extraction(md_text, kpis_text):
                from env.graders import markdown_score, kpi_score, cer
                results = []

                if md_text and md_text.strip():
                    # Score against task 1 ground truth as a reference
                    from env.tasks import TASK1_DATA, _md_from_data
                    gt_md = _md_from_data(TASK1_DATA)
                    md_sc = markdown_score(md_text, gt_md)
                    char_err = cer(md_text, gt_md)
                    results.append(f"**Markdown Score** (vs clean_table GT): `{md_sc:.3f}`")
                    results.append(f"**Character Error Rate**: `{char_err:.3f}`")
                else:
                    results.append("No markdown provided.")

                if kpis_text and kpis_text.strip():
                    try:
                        kpis = json.loads(kpis_text)
                        from env.tasks import TASK1_DATA
                        gt_kpis = TASK1_DATA["kpis"]
                        kp_sc = kpi_score(kpis, gt_kpis)
                        results.append(f"**KPI Score** (vs clean_table GT): `{kp_sc:.3f}`")
                        results.append(f"GT KPI keys: `{list(gt_kpis.keys())}`")
                    except json.JSONDecodeError:
                        results.append("Invalid JSON for KPIs.")
                else:
                    results.append("No KPIs provided.")

                return "\n\n".join(results)

            grade_btn.click(
                grade_extraction, [upload_md, upload_kpis], [grade_result]
            )

        # ===============================================================
        # Tab 3: Info
        # ===============================================================
        with gr.Tab("Info"):
            with gr.Accordion("Task Descriptions", open=True):
                for t in TASK_METADATA:
                    gr.Markdown(
                        f"**{t['title']}** (`{t['id']}`) -- Difficulty: `{t['difficulty']}`\n\n"
                        f"{t['description']}\n\n"
                        f"Optimal: `{t['optimal_steps']}` steps | Max: `{t['max_steps']}` steps"
                    )

            with gr.Accordion("OpenEnv API Endpoints", open=True):
                gr.Markdown(
                    "The OpenEnv API runs alongside this UI:\n\n"
                    "```\n"
                    "GET  /health   -> {status, env, version}\n"
                    "GET  /tasks    -> {tasks: [...]}\n"
                    "GET  /state    -> OCRState\n"
                    "POST /reset    -> OCRObservation   body: {task: 'clean_table'}\n"
                    "POST /step     -> StepResponse     body: OCRAction JSON\n"
                    "```\n\n"
                    "**Supported tasks:** `clean_table` | `noisy_financial` | `degraded_report`\n\n"
                    "**Environment Variables:**\n"
                    "- `API_BASE_URL` — LLM endpoint (default: HF Inference API)\n"
                    "- `MODEL_NAME` — Model ID (default: Qwen/Qwen2.5-72B-Instruct)\n"
                    "- `HF_TOKEN` — API key (required for inference.py)"
                )

            with gr.Accordion("Reward Function", open=False):
                gr.Markdown(
                    "| Signal | Value |\n"
                    "|---|---|\n"
                    "| `extract_table_md` improves CER | +0.1 x delta |\n"
                    "| `extract_kpis` new correct field | +0.10 per field |\n"
                    "| `extract_kpis` hallucinated key | -0.05 per key |\n"
                    "| `crop_region` + `retry_region` improves CER | +0.05 |\n"
                    "| `correct_cell` fixes error | +0.05 |\n"
                    "| Same action 3x in a row | -0.10 |\n"
                    "| `finalize` empty submission | -0.20 |\n"
                    "| `finalize` with full submission | Final grader score 0.0-1.0 |\n"
                    "| Steps over budget | -0.02 per excess step |"
                )


# ===========================================================================
# Mount OpenEnv API on Gradio's underlying FastAPI
# ===========================================================================

class ResetRequest(BaseModel):
    task: str = "clean_table"


_app = demo.app


@_app.get("/health")
async def health():
    return {"status": "healthy", "env": "ocr-table-rl", "version": "1.0.0"}


@_app.get("/tasks")
async def tasks_endpoint():
    return {"tasks": TASK_METADATA}


@_app.post("/reset")
async def reset_endpoint(req: ResetRequest):
    valid = {"clean_table", "noisy_financial", "degraded_report"}
    t = req.task if req.task in valid else "clean_table"
    return _api_env.reset(task=t).model_dump()


@_app.post("/step")
async def step_endpoint(action: OCRAction):
    obs, reward, done, info = _api_env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@_app.get("/state")
async def state_endpoint():
    return _api_env.state().model_dump()


# ===========================================================================
# Launch
# ===========================================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
