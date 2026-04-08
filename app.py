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
    """Convert uploaded file (image or PDF page 1) to PIL Image."""
    if file_path.lower().endswith(".pdf"):
        import fitz
        doc = fitz.open(file_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    return PILImage.open(file_path).convert("RGB")


def extract_document(file_path: str) -> dict:
    """
    Extract tables, text, and KPIs from PDF or image using PyMuPDF.
    Handles large multi-page PDFs efficiently — no ML models, no GPU.
    The RL loop handles refinement and accuracy improvement.
    """
    import fitz

    is_pdf = file_path.lower().endswith(".pdf")

    if not is_pdf:
        # For images, just return the raw text via OCR-like extraction
        return {
            "full_text": "(Image uploaded — use RL environment tasks for graded extraction)",
            "tables": [],
            "kpis": {},
            "num_tables": 0,
            "num_pages": 1,
            "pages_with_tables": [],
        }

    doc = fitz.open(file_path)
    total_pages = len(doc)
    full_text = ""
    all_tables = []
    all_kpis = {}
    pages_with_tables = []

    for page_num in range(total_pages):
        page = doc[page_num]

        # Extract raw text
        page_text = page.get_text("text")
        full_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"

        # Extract tables
        try:
            tab_finder = page.find_tables()
            for tab_idx, tab in enumerate(tab_finder.tables):
                rows = tab.extract()
                if not rows or len(rows) < 2:
                    continue

                # Clean and build markdown
                headers = [str(c or "").strip() for c in rows[0]]
                # Skip tables where all headers are empty
                if not any(headers):
                    continue

                md = f"**Page {page_num + 1}, Table {tab_idx + 1}**\n\n"
                md += "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

                for row in rows[1:]:
                    cells = [str(c or "").strip() for c in row]
                    md += "| " + " | ".join(cells) + " |\n"

                all_tables.append(md.strip())
                pages_with_tables.append(page_num + 1)

                # Auto-extract KPIs: first column = label, rest = values
                for row in rows[1:]:
                    cells = [str(c or "").strip() for c in row]
                    if len(cells) >= 2 and cells[0]:
                        # Clean key name
                        key = cells[0].lower().strip()
                        key = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                        key = key.replace("(", "").replace(")", "").replace(",", "")
                        key = "".join(c for c in key if c.isalnum() or c == "_")
                        key = key.strip("_")
                        if not key or len(key) < 2:
                            continue
                        # Find first numeric value
                        for v in cells[1:]:
                            v_clean = v.replace(",", "").replace("$", "").replace("%", "")
                            if any(c.isdigit() for c in v_clean) and v.strip():
                                all_kpis[key] = v.strip()
                                break
        except Exception:
            pass

    doc.close()

    return {
        "full_text": full_text,
        "tables": all_tables,
        "kpis": all_kpis,
        "num_tables": len(all_tables),
        "num_pages": total_pages,
        "pages_with_tables": sorted(set(pages_with_tables)),
    }


def handle_upload(file):
    """Process uploaded PDF/image with PyMuPDF. Fast, reliable, handles any size."""
    if file is None:
        return None, "(no file uploaded)", "", "", ""

    file_path = file if isinstance(file, str) else file.name

    # Get image preview (page 1)
    try:
        img = _file_to_pil(file_path)
    except Exception:
        img = None

    # Extract tables + text + KPIs
    try:
        result = extract_document(file_path)
    except Exception as e:
        return img, f"Extraction failed: {e}", "", "", ""

    # Build summary
    table_pages = result.get("pages_with_tables", [])
    hint = (
        f"PyMuPDF extracted {result['num_tables']} table(s) "
        f"from {result['num_pages']} page(s).\n"
        f"Tables found on pages: {table_pages if table_pages else 'none'}\n"
        f"KPIs auto-detected: {len(result['kpis'])}"
    )

    # Combine tables
    tables_combined = "\n\n---\n\n".join(result["tables"]) if result["tables"] else "(no tables found)"

    # KPIs as JSON
    kpis_json = json.dumps(result["kpis"], indent=2) if result["kpis"] else "{}"

    return img, hint, tables_combined, kpis_json, result["full_text"]


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
                "Upload any document containing tables — even large multi-page PDFs. "
                "**PyMuPDF** extracts tables, text, and KPIs instantly. "
                "The RL environment then handles iterative refinement.\n"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.File(
                        label="Upload PDF or Image",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("Extract Tables", variant="primary")

                    gr.Markdown("### Document Preview")
                    upload_img_out = gr.Image(
                        label="Uploaded Document", type="pil", height=350
                    )
                    upload_hint = gr.Textbox(
                        label="Extraction Info", lines=3, interactive=False
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Extracted Tables (Markdown)")
                    upload_md = gr.Textbox(
                        label="Extracted Markdown Tables",
                        lines=12,
                        interactive=True,
                    )
                    gr.Markdown("### Auto-Detected KPIs (JSON)")
                    upload_kpis = gr.Textbox(
                        label="Extracted KPIs",
                        lines=6,
                        interactive=True,
                    )
                    with gr.Accordion("Full Document Text", open=False):
                        full_md_out = gr.Textbox(
                            label="Full Extracted Text",
                            lines=15,
                            interactive=False,
                        )

            with gr.Row():
                gr.Markdown("### Test Grading")
            with gr.Row():
                grade_btn = gr.Button("Grade Against Task 1 Ground Truth", variant="secondary")
                grade_result = gr.Markdown("")

            upload_btn.click(
                handle_upload, [upload_input],
                [upload_img_out, upload_hint, upload_md, upload_kpis, full_md_out],
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
# Standalone FastAPI app with OpenEnv API + Gradio mounted on top
# ===========================================================================

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


class ResetRequest(BaseModel):
    task: str = "clean_table"


app = FastAPI(title="OCR Table RL Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "env": "ocr-table-rl", "version": "1.0.0"}


@app.get("/tasks")
async def tasks_endpoint():
    return {"tasks": TASK_METADATA}


@app.post("/reset")
async def reset_endpoint(request: Request):
    valid = {"clean_table", "noisy_financial", "degraded_report"}
    t = "clean_table"
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("task") in valid:
            t = body["task"]
    except Exception:
        pass
    return _api_env.reset(task=t).model_dump()


@app.post("/step")
async def step_endpoint(action: OCRAction):
    obs, reward, done, info = _api_env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
async def state_endpoint():
    return _api_env.state().model_dump()


# Mount Gradio UI at root — AFTER defining API routes so they take priority
app = gr.mount_gradio_app(app, demo, path="/")


# ===========================================================================
# Launch
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
