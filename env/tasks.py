"""Task definitions with PIL-generated synthetic document images and ground truth."""
from __future__ import annotations
import base64
import io
import math
import random

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _try_font(size: int = 14):
    if not PIL_AVAILABLE:
        return None
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _add_noise(img, intensity: float = 30.0):
    """Add Gaussian-like noise to a PIL image."""
    import random as rnd
    pixels = img.load()
    w, h = img.size
    for x in range(w):
        for y in range(h):
            r, g, b = pixels[x, y]
            noise = int(rnd.gauss(0, intensity))
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )
    return img


def _rotate_image(img, degrees: float):
    return img.rotate(degrees, expand=True, fillcolor=(255, 255, 255))


# ---------------------------------------------------------------------------
# Task 1 — Clean Printed Table
# ---------------------------------------------------------------------------

TASK1_DATA = {
    "headers": ["Product", "Q1 Sales", "Q2 Sales", "Q3 Sales", "Q4 Sales"],
    "rows": [
        ["Widget A", "1200", "1350", "1100", "1500"],
        ["Widget B", "800", "950", "870", "920"],
        ["Widget C", "2100", "2300", "2050", "2400"],
        ["Widget D", "450", "480", "510", "490"],
        ["Widget E", "3200", "3100", "3300", "3500"],
        ["Widget F", "670", "720", "690", "750"],
        ["Widget G", "1800", "1950", "1870", "2000"],
    ],
    "kpis": {
        "total_q1": "10220",
        "total_q4": "13060",
        "best_product": "Widget E",
    },
}


def _md_from_data(data: dict) -> str:
    headers = data["headers"]
    rows = data["rows"]
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_task1() -> dict:
    gt_md = _md_from_data(TASK1_DATA)
    gt_kpis = TASK1_DATA["kpis"]

    if not PIL_AVAILABLE:
        return {
            "image_b64": None,
            "text_hint": gt_md,
            "gt_md": gt_md,
            "gt_kpis": gt_kpis,
            "gt_cells": {},
            "max_steps": 5,
            "task_id": 1,
        }

    headers = TASK1_DATA["headers"]
    rows = TASK1_DATA["rows"]
    col_widths = [110, 80, 80, 80, 80]
    row_h = 28
    pad = 20
    W = sum(col_widths) + pad * 2
    H = row_h * (len(rows) + 2) + pad * 2

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _try_font(13)
    hfont = _try_font(13)

    # Title
    draw.text((pad, pad), "Sales Report — Annual Summary", fill=(30, 30, 30), font=hfont)

    y = pad + row_h
    x0 = pad
    # Header row
    for i, (h, cw) in enumerate(zip(headers, col_widths)):
        x = x0 + sum(col_widths[:i])
        draw.rectangle([x, y, x + cw, y + row_h], fill=(50, 100, 180))
        draw.text((x + 4, y + 6), h, fill=(255, 255, 255), font=font)
    y += row_h

    # Data rows
    for ri, row in enumerate(rows):
        fill = (240, 245, 255) if ri % 2 == 0 else (255, 255, 255)
        for i, (cell, cw) in enumerate(zip(row, col_widths)):
            x = x0 + sum(col_widths[:i])
            draw.rectangle([x, y, x + cw, y + row_h], fill=fill, outline=(200, 200, 200))
            draw.text((x + 4, y + 7), cell, fill=(20, 20, 20), font=font)
        y += row_h

    # KPI summary
    y += 5
    draw.text((pad, y), f"Total Q1: {gt_kpis['total_q1']}  |  Total Q4: {gt_kpis['total_q4']}  |  Best: {gt_kpis['best_product']}", fill=(80, 80, 80), font=font)

    return {
        "image_b64": _b64_png(img),
        "text_hint": gt_md,  # clean hint for task 1
        "gt_md": gt_md,
        "gt_kpis": gt_kpis,
        "gt_cells": {},
        "max_steps": 5,
        "task_id": 1,
    }


# ---------------------------------------------------------------------------
# Task 2 — Noisy Financial Statement
# ---------------------------------------------------------------------------

TASK2_DATA = {
    "headers": ["Metric", "FY2022", "FY2023", "FY2024", "YoY%"],
    "rows": [
        ["Revenue", "$4,200K", "$5,100K", "$6,300K", "+23.5%"],
        ["Gross Profit", "$1,890K", "$2,346K", "$3,024K", "+28.9%"],
        ["EBITDA", "$840K", "$1,020K", "$1,386K", "+35.9%"],
        ["Net Income", "$504K", "$663K", "$945K", "+42.5%"],
        ["EPS", "$1.26", "$1.66", "$2.36", "+42.2%"],
    ],
    "kpis": {
        "revenue_fy2024": "$6,300K",
        "ebitda_margin": "22.0%",
        "yoy_growth_pct": "+23.5%",
        "net_income_fy2024": "$945K",
    },
    "gt_cells": {
        "(0,0)": True, "(0,1)": True, "(0,2)": True,
        "(1,0)": True, "(1,3)": True,
        "(2,0)": True, "(2,3)": True,
    },
}


def generate_task2() -> dict:
    gt_md = _md_from_data(TASK2_DATA)
    gt_kpis = TASK2_DATA["kpis"]
    gt_cells = TASK2_DATA["gt_cells"]

    # Noisy text hint — simulate OCR errors
    noisy_hint = gt_md.replace("$", "S").replace("%", "°/o").replace("K", "lK").replace("+", "÷")

    if not PIL_AVAILABLE:
        return {
            "image_b64": None,
            "text_hint": noisy_hint,
            "gt_md": gt_md,
            "gt_kpis": gt_kpis,
            "gt_cells": gt_cells,
            "max_steps": 10,
            "task_id": 2,
        }

    headers = TASK2_DATA["headers"]
    rows = TASK2_DATA["rows"]
    col_widths = [120, 80, 80, 80, 70]
    row_h = 30
    pad = 20
    W = sum(col_widths) + pad * 2
    H = row_h * (len(rows) + 3) + pad * 2

    img = Image.new("RGB", (W, H), (252, 252, 248))
    draw = ImageDraw.Draw(img)
    font = _try_font(13)

    draw.text((pad, pad), "Annual Financial Statement — CONFIDENTIAL", fill=(20, 20, 80), font=font)

    y = pad + row_h
    x0 = pad

    # Two-level header simulation
    draw.text((x0, y), "Financials (USD)", fill=(100, 100, 100), font=font)
    y += row_h // 2

    for i, (h, cw) in enumerate(zip(headers, col_widths)):
        x = x0 + sum(col_widths[:i])
        draw.rectangle([x, y, x + cw, y + row_h], fill=(30, 60, 120))
        draw.text((x + 4, y + 7), h, fill=(255, 255, 255), font=font)
    y += row_h

    for ri, row in enumerate(rows):
        fill = (245, 248, 255) if ri % 2 == 0 else (255, 255, 255)
        for i, (cell, cw) in enumerate(zip(row, col_widths)):
            x = x0 + sum(col_widths[:i])
            draw.rectangle([x, y, x + cw, y + row_h], fill=fill, outline=(180, 180, 200))
            color = (0, 120, 0) if "+" in cell else (180, 20, 20) if "-" in cell else (20, 20, 20)
            draw.text((x + 4, y + 8), cell, fill=color, font=font)
        y += row_h

    # Add noise + slight rotation
    img = _add_noise(img, intensity=18)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    img = _rotate_image(img, -1.5)

    return {
        "image_b64": _b64_png(img),
        "text_hint": noisy_hint,
        "gt_md": gt_md,
        "gt_kpis": gt_kpis,
        "gt_cells": gt_cells,
        "max_steps": 10,
        "task_id": 2,
    }


# ---------------------------------------------------------------------------
# Task 3 — Degraded Multi-Table Report
# ---------------------------------------------------------------------------

TASK3_DATA = {
    "table1": {
        "headers": ["KPI", "Target", "Actual", "Variance"],
        "rows": [
            ["Customer Satisfaction", "90%", "87%", "-3%"],
            ["Ticket Resolution Rate", "95%", "93%", "-2%"],
            ["Avg Handle Time", "4.5min", "5.1min", "+0.6min"],
            ["First Call Resolution", "80%", "78%", "-2%"],
        ],
        "kpis": {
            "csat_actual": "87%",
            "ticket_resolution": "93%",
            "avg_handle_time": "5.1min",
        },
    },
    "table2": {
        "headers": ["Category", "Budget", "Spent", "Remaining"],
        "rows": [
            ["Operations", "$500K", "$472K", "$28K"],
            ["Marketing", "$200K", "$198K", "$2K"],
            ["R&D", "$300K", "$275K", "$25K"],
            ["Support", "$150K", "$163K", "-$13K"],
        ],
        "kpis": {
            "operations_budget": "$500K",
            "support_overspend": "-$13K",
            "total_remaining": "$42K",
        },
    },
}


def generate_task3() -> dict:
    t1 = TASK3_DATA["table1"]
    t2 = TASK3_DATA["table2"]
    gt_md1 = _md_from_data(t1)
    gt_md2 = _md_from_data(t2)
    gt_md = gt_md1 + "\n---\n" + gt_md2

    gt_kpis = {
        "table1": t1["kpis"],
        "table2": t2["kpis"],
    }

    # Very noisy hint
    noisy = (gt_md
             .replace("$", "S")
             .replace("%", "°/o")
             .replace("K", "lK")
             .replace("-", "—")
             .replace("+", "÷"))

    if not PIL_AVAILABLE:
        return {
            "image_b64": None,
            "text_hint": noisy,
            "gt_md": gt_md,
            "gt_kpis": gt_kpis,
            "gt_cells": {},
            "max_steps": 15,
            "task_id": 3,
        }

    col_widths1 = [180, 80, 80, 80]
    col_widths2 = [120, 80, 80, 80]
    row_h = 28
    pad = 20

    W = max(sum(col_widths1), sum(col_widths2)) + pad * 2
    H = row_h * (len(t1["rows"]) + len(t2["rows"]) + 7) + pad * 2

    img = Image.new("RGB", (W, H), (248, 245, 240))
    draw = ImageDraw.Draw(img)
    font = _try_font(12)

    y = pad

    def draw_table(data, col_widths, title_str):
        nonlocal y
        draw.text((pad, y), title_str, fill=(50, 30, 10), font=font)
        y += row_h
        x0 = pad
        for i, (h, cw) in enumerate(zip(data["headers"], col_widths)):
            x = x0 + sum(col_widths[:i])
            draw.rectangle([x, y, x + cw, y + row_h], fill=(80, 50, 20))
            draw.text((x + 3, y + 7), h, fill=(255, 240, 200), font=font)
        y += row_h
        for ri, row in enumerate(data["rows"]):
            fill = (250, 245, 235) if ri % 2 == 0 else (240, 235, 225)
            for i, (cell, cw) in enumerate(zip(row, col_widths)):
                x = x0 + sum(col_widths[:i])
                draw.rectangle([x, y, x + cw, y + row_h], fill=fill, outline=(170, 150, 130))
                draw.text((x + 3, y + 8), cell, fill=(30, 20, 10), font=font)
            y += row_h
        y += 10

    draw_table(t1, col_widths1, "Table 1: Operational KPIs")
    draw.line([(pad, y), (W - pad, y)], fill=(100, 80, 60), width=2)
    y += 10
    draw_table(t2, col_widths2, "Table 2: Financial Summary")

    # Heavy degradation
    img = _add_noise(img, intensity=35)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img = _rotate_image(img, 3.5)

    return {
        "image_b64": _b64_png(img),
        "text_hint": noisy,
        "gt_md": gt_md,
        "gt_kpis": gt_kpis,
        "gt_cells": {},
        "max_steps": 15,
        "task_id": 3,
    }


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "clean_table": generate_task1,
    "noisy_financial": generate_task2,
    "degraded_report": generate_task3,
}

TASK_METADATA = [
    {
        "id": "clean_table",
        "title": "Clean Printed Table",
        "difficulty": "easy",
        "description": "Extract a clean single-table sales report into Markdown + JSON KPIs.",
        "optimal_steps": 2,
        "max_steps": 5,
    },
    {
        "id": "noisy_financial",
        "title": "Noisy Financial Statement",
        "difficulty": "medium",
        "description": "Extract a noisy multi-header financial table with merged context. Output Markdown + labeled KPIs. Optional per-cell confidence scoring.",
        "optimal_steps": 4,
        "max_steps": 10,
    },
    {
        "id": "degraded_report",
        "title": "Degraded Multi-Table Report",
        "difficulty": "hard",
        "description": "Extract two tables from a heavily degraded rotated document. Output separate Markdowns + cross-table KPI JSON.",
        "optimal_steps": 8,
        "max_steps": 15,
    },
]
