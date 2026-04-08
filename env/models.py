from pydantic import BaseModel, Field
from typing import Literal, Optional


class CellConfidence(BaseModel):
    row: int
    col: int
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class OCRAction(BaseModel):
    action_type: Literal[
        "extract_table_md",
        "extract_kpis",
        "crop_region",
        "retry_region",
        "correct_cell",
        "switch_table",
        "finalize",
    ]
    markdown: Optional[str] = None
    kpis: Optional[dict] = None
    region: Optional[dict] = None
    cell_row: Optional[int] = None
    cell_col: Optional[int] = None
    cell_value: Optional[str] = None
    confidences: Optional[list] = None


class OCRObservation(BaseModel):
    image_b64: Optional[str] = None
    text_hint: str = ""
    reward: float = 0.0
    done: bool = False
    cer: Optional[float] = None
    kpi_score: Optional[float] = None
    error: Optional[str] = None
    metadata: dict = {}


class OCRState(BaseModel):
    task_name: str
    step: int
    max_steps: int
    best_cer: float = 1.0
    kpi_fields_correct: int = 0
    kpi_fields_total: int = 0
    markdown_submitted: bool = False
    kpis_submitted: bool = False
    active_table: int = 1
