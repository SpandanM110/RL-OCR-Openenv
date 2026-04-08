"""FastAPI server for OCR Table RL Environment (with Gradio UI mounted at /ui)."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from env.environment import OCREnvironment
from env.models import OCRAction, OCRObservation, OCRState
from env.tasks import TASK_METADATA

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OCR Table RL Environment",
    description=(
        "OpenEnv-compatible RL environment for structured table extraction. "
        "Agents extract complex tables into Markdown + JSON KPIs from synthetic document images."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (one session at a time — sufficient for eval)
env = OCREnvironment()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "clean_table"


class StepResponse(BaseModel):
    observation: OCRObservation
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "ocr-table-rl", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    return {"tasks": TASK_METADATA}


@app.post("/reset", response_model=OCRObservation)
async def reset(request: ResetRequest):
    """Reset the environment for a given task and return the initial observation."""
    valid_tasks = {"clean_table", "noisy_financial", "degraded_report"}
    task = request.task if request.task in valid_tasks else "clean_table"
    obs = env.reset(task=task)
    return obs


@app.post("/step", response_model=StepResponse)
async def step(action: OCRAction):
    """Execute one environment step with the given action."""
    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=OCRState)
async def get_state():
    """Return the current environment state."""
    return env.state()


# ---------------------------------------------------------------------------
# Mount Gradio UI at /ui
# ---------------------------------------------------------------------------

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gradio_app",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"),
    )
    gradio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gradio_module)
    demo = gradio_module.demo
    app = gr.mount_gradio_app(app, demo, path="/ui")
except Exception as e:
    print(f"Warning: Could not mount Gradio UI: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
