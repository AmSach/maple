"""
MAPLE Local Agent Desktop Application (Native PySide6 / Qt6)
=============================================================
Genuine native Windows desktop application for your 4-tier local AI stack:
  - DeepSeek-Coder-V2-Lite MAPLE (54 Experts, Sub-5GB): Hardcore Systems & C++20
  - Gemma 4 26B MAPLE (Sub-5GB): General Python, Web, Refactoring
  - Ternary Bonsai 2 27B (PTQ1_0): Deep Reasoning, Architecture & Decisions
  - Qwen 2.5 Coder 1.5B: Ultra-Fast Triage, Git Commits & Shell One-Liners

Features:
  - Automatic Model Chooser (routes every prompt to the right model)
  - Live Workspace File Tree (left pane)
  - Live-rendered Markdown + Pygments chat & tool stream (center pane)
  - Live Code / Diff Highlighter + Rendered Preview + Terminal Log (right pane)
  - Live RTX 4050 6GB VRAM meter + automatic VRAM eviction when swapping models

Author: Aman Sachan (amansachan92905@gmail.com)
"""

import sys
import os
import re
import csv
import json
import time
import base64
import shutil
import difflib
import tempfile
import subprocess
from typing import Optional, Dict, Any

import fitz
import openpyxl
from pptx import Presentation
import requests
import markdown
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QDir, QRegularExpression, QFileSystemWatcher
from PySide6.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QAction, QTextCursor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeView, QFileSystemModel, QTextBrowser, QPlainTextEdit,
    QPushButton, QButtonGroup, QLabel, QProgressBar, QTabWidget, QLineEdit,
    QFileDialog, QFrame, QGraphicsDropShadowEffect, QSizePolicy
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_agent_harness import (
    MODEL_ROSTER,
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    DEFAULT_OLLAMA_API,
    DEFAULT_BONSAI_API,
    Tools,
    auto_choose_model,
    micro_model_route,
    load_agent_rules,
    ensure_vram_for_model,
)

class DocumentRenderer:
    """Renders PDF, Excel/CSV, PowerPoint (PPTX), LaTeX (TEX), Images, Markdown, and HTML to rich Qt HTML."""

    @staticmethod
    def _esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def render_pdf(path: str, max_pages: int = 25) -> str:
        doc = fitz.open(path)
        total = len(doc)
        parts = [f"<h3>PDF Document: {DocumentRenderer._esc(os.path.basename(path))} ({total} pages)</h3>"]
        for i in range(min(total, max_pages)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            parts.append(
                f'<div style="margin:12px 0; padding:8px; background:#090d12; border:1px solid #232f42; border-radius:8px;">'
                f'<div style="color:#8b9bb4; font-size:11px; margin-bottom:6px;">Page {i + 1} / {total}</div>'
                f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" />'
                f'</div>'
            )
        if total > max_pages:
            parts.append(f"<p><i>Showing first {max_pages} of {total} pages.</i></p>")
        doc.close()
        return "".join(parts)

    @staticmethod
    def render_excel(path: str, max_rows: int = 200, max_cols: int = 25) -> str:
        lower = path.lower()
        parts = [f"<h3>Spreadsheet: {DocumentRenderer._esc(os.path.basename(path))}</h3>"]
        if lower.endswith(".csv"):
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.reader(f))
            parts.append(DocumentRenderer._rows_to_table("CSV", rows, max_rows, max_cols))
            return "".join(parts)

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        for name in wb.sheetnames:
            ws = wb[name]
            rows = []
            for idx, row in enumerate(ws.iter_rows(values_only=True)):
                if idx >= max_rows:
                    break
                rows.append(["" if c is None else str(c) for c in row[:max_cols]])
            parts.append(DocumentRenderer._rows_to_table(name, rows, max_rows, max_cols))
        wb.close()
        return "".join(parts)

    @staticmethod
    def _rows_to_table(sheet_name: str, rows: list, max_rows: int, max_cols: int) -> str:
        html = [f'<h4 style="color:#10b981; margin-top:14px;">Sheet: {DocumentRenderer._esc(sheet_name)}</h4>']
        html.append('<table border="1" cellspacing="0" cellpadding="5" style="border-collapse:collapse; border-color:#232f42; font-size:12px; width:100%;">')
        for r_idx, row in enumerate(rows[:max_rows]):
            bg = "#161e2c" if r_idx == 0 else ("#111722" if r_idx % 2 == 1 else "#0d121a")
            tag = "th" if r_idx == 0 else "td"
            html.append(f'<tr style="background:{bg};">')
            for cell in row[:max_cols]:
                html.append(f'<{tag} style="border:1px solid #232f42; padding:4px 8px; text-align:left;">{DocumentRenderer._esc(cell)}</{tag}>')
            html.append("</tr>")
        html.append("</table>")
        return "".join(html)

    @staticmethod
    def render_pptx(path: str) -> str:
        prs = Presentation(path)
        total = len(prs.slides)
        parts = [f"<h3>Presentation: {DocumentRenderer._esc(os.path.basename(path))} ({total} slides)</h3>"]
        for idx, slide in enumerate(prs.slides, start=1):
            title = slide.shapes.title.text.strip() if (slide.shapes.title and slide.shapes.title.has_text_frame) else f"Slide {idx}"
            card = [
                f'<div style="margin:14px 0; padding:14px; background:#0d131d; border:1px solid #2d3d54; border-left:4px solid #38bdf8; border-radius:8px;">',
                f'<div style="color:#8b9bb4; font-size:11px; margin-bottom:4px;">SLIDE {idx} / {total}</div>',
                f'<h3 style="color:#ffffff; margin:0 0 8px 0;">{DocumentRenderer._esc(title)}</h3>',
            ]
            for shape in slide.shapes:
                if shape == slide.shapes.title:
                    continue
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        txt = p.text.strip()
                        if txt:
                            indent = "&nbsp;&nbsp;" * (p.level or 0)
                            card.append(f'<div style="margin:3px 0; color:#e6edf3;">{indent}- {DocumentRenderer._esc(txt)}</div>')
                if shape.has_table:
                    t_rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                    card.append(DocumentRenderer._rows_to_table(f"Slide {idx} Table", t_rows, 50, 15))
                if shape.shape_type == 13:  # Picture
                    try:
                        img_bytes = shape.image.blob
                        ctype = shape.image.content_type or "image/png"
                        b64 = base64.b64encode(img_bytes).decode("ascii")
                        card.append(f'<div style="margin:8px 0;"><img src="data:{ctype};base64,{b64}" style="max-width:95%;" /></div>')
                    except Exception:
                        pass
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    card.append(f'<div style="margin-top:8px; padding:6px; background:#090d12; color:#8b9bb4; font-size:11.5px; border-radius:4px;"><b>Notes:</b> {DocumentRenderer._esc(notes)}</div>')
            card.append("</div>")
            parts.append("".join(card))
        return "".join(parts)

    @staticmethod
    def render_tex(path: str, tex_source: str) -> str:
        compiler = shutil.which("pdflatex") or shutil.which("xelatex")
        if compiler:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tex_copy = os.path.join(tmpdir, "doc.tex")
                    with open(tex_copy, "w", encoding="utf-8") as f:
                        f.write(tex_source)
                    proc = subprocess.run(
                        [compiler, "-interaction=nonstopmode", "-output-directory", tmpdir, tex_copy],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
                    )
                    pdf_out = os.path.join(tmpdir, "doc.pdf")
                    if proc.returncode == 0 and os.path.exists(pdf_out):
                        return DocumentRenderer.render_pdf(pdf_out)
            except Exception:
                pass

        # Clean HTML fallback preview for LaTeX when compiler is not installed
        html = [f"<h3>LaTeX Document Preview: {DocumentRenderer._esc(os.path.basename(path))}</h3>"]
        title_m = re.search(r"\\title\{([^}]*)\}", tex_source)
        if title_m:
            html.append(f'<h2 style="color:#ffffff;">{DocumentRenderer._esc(title_m.group(1))}</h2>')
        body_m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex_source, re.DOTALL)
        body = body_m.group(1) if body_m else tex_source
        body = DocumentRenderer._esc(body)
        body = re.sub(r"\\section\*?\{([^}]*)\}", r'<h3 style="color:#38bdf8; margin-top:12px;">\1</h3>', body)
        body = re.sub(r"\\subsection\*?\{([^}]*)\}", r'<h4 style="color:#10b981; margin-top:8px;">\1</h4>', body)
        body = re.sub(r"\\textbf\{([^}]*)\}", r"<b>\1</b>", body)
        body = re.sub(r"\\textit\{([^}]*)\}", r"<i>\1</i>", body)
        body = re.sub(r"\$\$(.*?)\$\$", r'<div style="background:#090d12; padding:8px; margin:6px 0; border-left:3px solid #a78bfa; font-family:Consolas,monospace;">\1</div>', body, flags=re.DOTALL)
        body = body.replace("\n\n", "<br><br>").replace("\n", "<br>")
        html.append(f'<div style="padding:10px; background:#0d131d; border:1px solid #232f42; border-radius:6px;">{body}</div>')
        return "".join(html)

    @staticmethod
    def render_image(path: str) -> str:
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f'<h3>Image: {DocumentRenderer._esc(os.path.basename(path))}</h3><img src="data:image/{ext};base64,{b64}" style="max-width:100%;" />'

    @staticmethod
    def is_rich_doc(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".pdf", ".xlsx", ".xlsm", ".csv", ".pptx", ".tex", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".md", ".html", ".htm"}

    @staticmethod
    def render_file(path: str, text_content: Optional[str] = None) -> str:
        lower = path.lower()
        try:
            if lower.endswith(".pdf"):
                return DocumentRenderer.render_pdf(path)
            if lower.endswith((".xlsx", ".xlsm", ".csv")):
                return DocumentRenderer.render_excel(path)
            if lower.endswith(".pptx"):
                return DocumentRenderer.render_pptx(path)
            if lower.endswith(".tex"):
                if text_content is None:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        text_content = f.read()
                return DocumentRenderer.render_tex(path, text_content)
            if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                return DocumentRenderer.render_image(path)
            if text_content is None:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text_content = f.read()
            if lower.endswith(".md"):
                return markdown.markdown(text_content, extensions=["fenced_code", "tables"])
            if lower.endswith((".html", ".htm")):
                return text_content
            return f"<h3>{DocumentRenderer._esc(os.path.basename(path))}</h3><pre>{DocumentRenderer._esc(text_content)}</pre>"
        except Exception as e:
            return f"<p style='color:#ef4444;'>Failed to render {DocumentRenderer._esc(path)}: {DocumentRenderer._esc(e)}</p>"


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #090c12;
    color: #e6edf3;
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
}
QFrame#TopBar {
    background-color: #0e141f;
    border: 1px solid #1e293b;
    border-radius: 14px;
    margin: 8px 10px 4px 10px;
}
QFrame#PaneCard {
    background-color: #0e141f;
    border: 1px solid #1e293b;
    border-radius: 14px;
}
QLineEdit, QPlainTextEdit, QTextBrowser, QTreeView {
    background-color: #0b1018;
    color: #e6edf3;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 8px 10px;
    selection-background-color: #1e3a5f;
    selection-color: #ffffff;
}
QLineEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #38bdf8;
}
QTreeView {
    padding: 6px;
    outline: none;
}
QTreeView::item {
    padding: 4px 6px;
    border-radius: 6px;
}
QTreeView::item:hover {
    background-color: #162032;
}
QTreeView::item:selected {
    background-color: #1a3350;
    color: #38bdf8;
}
QPushButton {
    background-color: #141c2b;
    color: #cbd5e1;
    border: 1px solid #233044;
    border-radius: 10px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover {
    border-color: #38bdf8;
    color: #ffffff;
}
QPushButton#PillBtn {
    background-color: #111827;
    color: #94a3b8;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#PillBtn:hover {
    color: #e2e8f0;
    border-color: #334155;
}
QPushButton#PillBtn:checked {
    background-color: #0ea5e9;
    color: #04101c;
    border: 1px solid #38bdf8;
}
QPushButton#ModePill {
    background-color: #111827;
    color: #94a3b8;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#ModePill:checked {
    background-color: #10b981;
    color: #04150e;
    border: 1px solid #34d399;
}
QPushButton#RunBtn {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #38bdf8, stop:1 #0284c7);
    color: #04101c;
    border: none;
    border-radius: 12px;
    padding: 8px 18px;
    font-weight: 700;
}
QPushButton#RunBtn:hover {
    background-color: #7dd3fc;
}
QPushButton#RunBtn:disabled {
    background-color: #1e293b;
    color: #64748b;
}
QPushButton#StopBtn {
    background-color: #2a1218;
    color: #fca5a5;
    border: 1px solid #7f1d1d;
    border-radius: 12px;
    padding: 7px 16px;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    border-radius: 14px;
    background-color: #0b1018;
    top: -1px;
}
QTabBar::tab {
    background-color: #0e141f;
    color: #94a3b8;
    border: 1px solid #1e293b;
    padding: 7px 16px;
    margin-right: 4px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #141d2d;
    color: #10b981;
    border-bottom: 2px solid #10b981;
}
QProgressBar {
    background-color: #090d12;
    border: 1px solid #1e293b;
    border-radius: 10px;
    text-align: center;
    color: #e6edf3;
    font-size: 11px;
    font-weight: 600;
    height: 22px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #38bdf8);
    border-radius: 9px;
}
QSplitter::handle {
    background-color: transparent;
    width: 6px;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 4px 0;
}
QScrollBar::handle:vertical {
    background: #233044;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #38bdf8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0 4px;
}
QScrollBar::handle:horizontal {
    background: #233044;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    width: 0px;
}
"""


class CodeAndDiffHighlighter(QSyntaxHighlighter):
    """Native Qt syntax highlighter supporting code keywords, strings, comments, and unified diffs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []

        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#38bdf8"))
        kw_fmt.setFontWeight(QFont.Bold)
        keywords = [
            "def", "class", "import", "from", "return", "if", "elif", "else",
            "for", "while", "try", "except", "with", "as", "async", "await",
            "template", "typename", "concept", "requires", "constexpr", "struct",
            "auto", "void", "int", "bool", "const", "std", "include", "namespace",
            "function", "const", "let", "var", "export", "True", "False", "None"
        ]
        for kw in keywords:
            self.rules.append((QRegularExpression(rf"\b{kw}\b"), kw_fmt))

        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("#a5d6ff"))
        self.rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), str_fmt))
        self.rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), str_fmt))

        cmt_fmt = QTextCharFormat()
        cmt_fmt.setForeground(QColor("#6a7b91"))
        self.rules.append((QRegularExpression(r"#[^\n]*"), cmt_fmt))
        self.rules.append((QRegularExpression(r"//[^\n]*"), cmt_fmt))

        self.diff_add_fmt = QTextCharFormat()
        self.diff_add_fmt.setForeground(QColor("#6ee7b7"))
        self.diff_add_fmt.setBackground(QColor(16, 185, 129, 32))

        self.diff_del_fmt = QTextCharFormat()
        self.diff_del_fmt.setForeground(QColor("#fca5a5"))
        self.diff_del_fmt.setBackground(QColor(239, 68, 68, 32))

        self.diff_hunk_fmt = QTextCharFormat()
        self.diff_hunk_fmt.setForeground(QColor("#38bdf8"))

    def highlightBlock(self, text: str):
        if text.startswith("+") and not text.startswith("+++"):
            self.setFormat(0, len(text), self.diff_add_fmt)
            return
        if text.startswith("-") and not text.startswith("---"):
            self.setFormat(0, len(text), self.diff_del_fmt)
            return
        if text.startswith("@@"):
            self.setFormat(0, len(text), self.diff_hunk_fmt)
            return

        for pattern, fmt in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class AgentWorker(QThread):
    """Background worker thread running model inference and local tool execution."""

    model_selected = Signal(str, str, str, str)
    step_started = Signal(int, int)
    token_received = Signal(str)
    file_written = Signal(str, str)
    file_edited = Signal(str, str, str)
    file_read = Signal(str, str, int)
    cmd_executed = Signal(str, str, int, str)
    telemetry = Signal(int, float, float)
    error_occurred = Signal(str)
    finished_run = Signal()

    def __init__(self, prompt: str, model_key: str, mode: str, workdir: str):
        super().__init__()
        self.prompt = prompt
        self.model_key = model_key
        self.mode = mode
        self.workdir = workdir
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.workdir or os.getcwd(), path))

    def _stream_llm(self, chosen: str, prompt: str, system: Optional[str] = None) -> str:
        cfg = MODEL_ROSTER.get(chosen, MODEL_ROSTER["gemma"])
        ensure_vram_for_model(chosen)

        endpoint = f"{DEFAULT_OLLAMA_API}/api/generate"
        if cfg["backend"] == "bonsai_or_router":
            try:
                r = requests.get(f"{DEFAULT_OLLAMA_API}/api/tags", timeout=1.0)
                has_bonsai = any("bonsai" in m.get("name", "") for m in r.json().get("models", []))
                if not has_bonsai:
                    endpoint = f"{DEFAULT_BONSAI_API}/completion"
            except Exception:
                endpoint = f"{DEFAULT_BONSAI_API}/completion"

        start_t = time.time()
        collected = []

        if endpoint.endswith("/completion"):
            prompt_fmt = (
                f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                if system
                else f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_payload = {
                "prompt": prompt_fmt,
                "stream": True,
                "n_predict": cfg["max_tokens"],
                "temperature": cfg["temperature"],
                "stop": ["<|im_end|>", "<|im_start|>"],
            }
            resp = requests.post(endpoint, json=llama_payload, stream=True, timeout=180)
            for line in resp.iter_lines():
                if self._stop_flag:
                    break
                if not line:
                    continue
                ls = line.decode("utf-8", errors="replace")
                if ls.startswith("data: "):
                    ds = ls[6:].strip()
                    if ds == "[DONE]":
                        break
                    try:
                        tok = json.loads(ds).get("content", "")
                        if tok:
                            collected.append(tok)
                            self.token_received.emit(tok)
                    except Exception:
                        pass
        else:
            payload = {
                "model": cfg["id"],
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": cfg["temperature"],
                    "num_predict": cfg["max_tokens"],
                    "num_ctx": 4096,
                    "stop": ["\n[Step ", "\n\n[Step "],
                },
            }
            if system:
                payload["system"] = system
            resp = requests.post(endpoint, json=payload, stream=True, timeout=180)
            for line in resp.iter_lines():
                if self._stop_flag:
                    break
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8", errors="replace"))
                    tok = chunk.get("response", "")
                    if tok:
                        collected.append(tok)
                        self.token_received.emit(tok)
                    if chunk.get("done", False):
                        break
                except Exception:
                    pass

        elapsed = round(time.time() - start_t, 2)
        full_text = "".join(collected)
        tokens_est = max(1, len(full_text) // 4)
        tps = round(tokens_est / elapsed, 1) if elapsed > 0 else 0.0
        self.telemetry.emit(tokens_est, elapsed, tps)
        return full_text

    def run(self):
        try:
            if self.model_key == "auto":
                chosen, route_reason = micro_model_route(self.prompt)
            else:
                chosen, route_reason = self.model_key, "Manual selection"
            if chosen not in MODEL_ROSTER:
                chosen = "gemma"
            cfg = MODEL_ROSTER[chosen]
            self.model_selected.emit(chosen, cfg["name"], cfg["role"], route_reason)

            rules_text = load_agent_rules()
            if self.mode == "chat":
                self._stream_llm(chosen, self.prompt, system=rules_text)
                self.finished_run.emit()
                return

            system_prompt = (
                f"You are an autonomous engineering agent operating in working directory: {self.workdir}\n"
                f"Role: {cfg['role']}\n"
                f"{rules_text}\n"
                f"{TOOL_SCHEMAS}\n"
                f"Execution Protocol:\n"
                f"1. To execute work, write TOOL: <name> followed by ARGS: <json>.\n"
                f"2. After tools execute, inspect their outputs and decide the next step.\n"
                f"3. Only when all tool results have been observed and the task is accomplished, output 'FINAL ANSWER:' followed by your concise report.\n"
                f"4. Ban placeholders. Write complete, working solutions."
            )
            files_touched = []
            recent_steps = []
            max_steps = 8

            for step in range(1, max_steps + 1):
                if self._stop_flag:
                    break
                self.step_started.emit(step, max_steps)

                mem_lines = [f"GOAL: {self.prompt}"]
                if files_touched:
                    mem_lines.append("FILES_TOUCHED: " + ", ".join(files_touched[-10:]))
                if recent_steps:
                    mem_lines.append("RECENT STEPS (compact):")
                    mem_lines.extend(recent_steps[-2:])
                step_prompt = "\n".join(mem_lines) + f"\n\n[Step {step}/{max_steps}] What is your next action?"

                response_text = self._stream_llm(chosen, step_prompt, system=system_prompt)
                if self._stop_flag:
                    break

                tool_matches = list(re.finditer(r"TOOL:\s*(\w+)\s*\nARGS:\s*({.*?})", response_text, re.DOTALL))
                executed_any = False
                step_obs = []

                for m in tool_matches:
                    tool_name = m.group(1).strip()
                    try:
                        args = json.loads(m.group(2).strip())
                    except Exception:
                        args = {}

                    if tool_name not in TOOL_REGISTRY:
                        step_obs.append(f"- ERROR: Unknown tool '{tool_name}'")
                        continue

                    executed_any = True
                    if tool_name == "write_file":
                        fpath = self._resolve_path(args.get("path", "output.txt"))
                        content = args.get("content", "")
                        os.makedirs(os.path.dirname(fpath), exist_ok=True)
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(content)
                        result = {"success": True, "path": fpath, "bytes_written": len(content)}
                        if fpath not in files_touched:
                            files_touched.append(fpath)
                        self.file_written.emit(fpath, content)

                    elif tool_name == "edit_file":
                        fpath = self._resolve_path(args.get("path", ""))
                        target = args.get("target", "")
                        replacement = args.get("replacement", "")
                        if not os.path.exists(fpath):
                            result = {"error": f"File not found: {fpath}"}
                        else:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                old_data = f.read()
                            if target not in old_data:
                                result = {"error": "Target string not found in file."}
                            else:
                                new_data = old_data.replace(target, replacement)
                                with open(fpath, "w", encoding="utf-8") as f:
                                    f.write(new_data)
                                diff_str = "\n".join(difflib.unified_diff(
                                    old_data.splitlines(),
                                    new_data.splitlines(),
                                    fromfile=f"a/{os.path.basename(fpath)}",
                                    tofile=f"b/{os.path.basename(fpath)}",
                                    lineterm=""
                                ))
                                result = {"success": True, "path": fpath}
                                if fpath not in files_touched:
                                    files_touched.append(fpath)
                                self.file_edited.emit(fpath, diff_str, new_data)

                    elif tool_name == "read_file":
                        fpath = self._resolve_path(args.get("path", ""))
                        if not os.path.exists(fpath):
                            result = {"error": f"File not found: {fpath}"}
                        else:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                lines = f.readlines()
                            start_l = max(1, int(args.get("start_line", 1)))
                            end_l = min(len(lines), int(args.get("end_line", min(len(lines), start_l + 149))))
                            raw_txt = "".join(lines[start_l - 1:end_l])
                            result = {"path": fpath, "start_line": start_l, "end_line": end_l, "total_lines": len(lines), "content": raw_txt[:4000]}
                            if fpath not in files_touched:
                                files_touched.append(fpath)
                            self.file_read.emit(fpath, "".join(lines), len(lines))

                    elif tool_name == "run_command":
                        cmd = args.get("cmd", "")
                        result = Tools.run_command(f'cd /d "{self.workdir}" && {cmd}' if os.name == "nt" else cmd, silent=True)
                        out_txt = (result.get("stdout") or "") + (f"\n[STDERR]\n{result.get('stderr')}" if result.get("stderr") else "")
                        self.cmd_executed.emit("SHELL", cmd, result.get("exit_code", -1), out_txt)

                    elif tool_name == "python_eval":
                        code = args.get("code", "")
                        result = Tools.python_eval(code)
                        out_txt = (result.get("stdout") or "") + (f"\n[STDERR]\n{result.get('stderr')}" if result.get("stderr") else "")
                        self.cmd_executed.emit("PYTHON", code.splitlines()[0][:80] if code else "", result.get("exit_code", -1), out_txt)

                    else:
                        try:
                            result = TOOL_REGISTRY[tool_name](**args)
                        except Exception as ex:
                            result = {"error": str(ex)}
                        if tool_name in ("web_search", "fetch_url", "search_files"):
                            self.cmd_executed.emit(tool_name.upper(), json.dumps(args)[:90], 0, json.dumps(result, indent=2)[:1500])

                    step_obs.append(f"- {tool_name}({json.dumps(args)[:160]}) -> {json.dumps(result)[:1200]}")

                if step_obs:
                    recent_steps.append(f"Step {step}:\n" + "\n".join(step_obs))

                if not executed_any and ("FINAL ANSWER:" in response_text or "DONE" in response_text):
                    break

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished_run.emit()


class MapleDesktopWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAPLE Local Agent Desktop")
        self.resize(1440, 860)

        self.workdir = r"E:\OllamaModels" if os.path.exists(r"E:\OllamaModels") else os.getcwd()
        self.worker: Optional[AgentWorker] = None
        self.chat_markdown_blocks = []
        self.current_stream_buf = ""
        self._last_chat_refresh_t = 0.0
        self.active_file_path = ""
        self.active_file_content = ""
        self.selected_model_key = "auto"
        self.selected_mode = "task"
        self._anim_lines = []
        self._anim_idx = 0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(12)
        self._anim_timer.timeout.connect(self._tick_live_edit_anim)

        self.fs_watcher = QFileSystemWatcher(self)
        self.fs_watcher.fileChanged.connect(self._on_watched_file_changed)

        self._build_ui()
        self._start_vram_timer()

    def _apply_shadow(self, widget: QWidget):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        widget.setGraphicsEffect(shadow)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 8)
        root_layout.setSpacing(6)

        # Top Bar (Rounded Card)
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        self._apply_shadow(top_bar)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(10)

        title_lbl = QLabel("MAPLE")
        title_lbl.setStyleSheet("font-weight: 800; font-size: 16px; color: #38bdf8; letter-spacing: 1px;")
        top_layout.addWidget(title_lbl)

        # Segmented Model Pills
        self.model_group = QButtonGroup(self)
        self.model_group.setExclusive(True)
        model_pills = [
            ("Auto 0.5B", "auto"),
            ("DeepSeek", "deepseek"),
            ("Gemma 4", "gemma"),
            ("Bonsai 27B", "bonsai"),
            ("Fast 1.5B", "fast"),
        ]
        for label, key in model_pills:
            btn = QPushButton(label)
            btn.setObjectName("PillBtn")
            btn.setCheckable(True)
            if key == "auto":
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, k=key: setattr(self, "selected_model_key", k))
            self.model_group.addButton(btn)
            top_layout.addWidget(btn)

        # Segmented Mode Pills
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for label, mode_val in [("Agent Task", "task"), ("Direct Chat", "chat")]:
            btn = QPushButton(label)
            btn.setObjectName("ModePill")
            btn.setCheckable(True)
            if mode_val == "task":
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, m=mode_val: setattr(self, "selected_mode", m))
            self.mode_group.addButton(btn)
            top_layout.addWidget(btn)

        self.workdir_edit = QLineEdit(self.workdir)
        self.workdir_edit.setMinimumWidth(180)
        self.workdir_edit.setPlaceholderText("Working directory...")
        top_layout.addWidget(self.workdir_edit, 1)

        browse_btn = QPushButton("Folder")
        browse_btn.clicked.connect(self._choose_workdir)
        top_layout.addWidget(browse_btn)

        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 6141)
        self.vram_bar.setFormat("VRAM: %v / %m MiB")
        self.vram_bar.setFixedWidth(165)
        top_layout.addWidget(self.vram_bar)

        root_layout.addWidget(top_bar)

        # 3-Pane Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setContentsMargins(10, 2, 10, 2)

        # Left Pane Card: Workspace File Tree
        left_card = QFrame()
        left_card.setObjectName("PaneCard")
        left_card.setMinimumWidth(220)
        self._apply_shadow(left_card)
        left_lay = QVBoxLayout(left_card)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)
        tree_hdr = QLabel("WORKSPACE")
        tree_hdr.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700; letter-spacing: 1px; padding-left: 4px;")
        left_lay.addWidget(tree_hdr)

        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath(self.workdir)
        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(self.workdir))
        for col in (1, 2, 3):
            self.tree.hideColumn(col)
        self.tree.setHeaderHidden(True)
        self.tree.clicked.connect(self._on_tree_clicked)
        left_lay.addWidget(self.tree, 1)
        splitter.addWidget(left_card)

        # Center Pane Card: Chat & Prompt
        center_widget = QFrame()
        center_widget.setObjectName("PaneCard")
        center_widget.setMinimumWidth(400)
        self._apply_shadow(center_widget)
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(8)

        self.active_model_badge = QLabel("Router (qwen2.5:0.5b CPU) | Ready")
        self.active_model_badge.setStyleSheet(
            "background-color: #111827; color: #38bdf8; border: 1px solid #1e293b; "
            "border-radius: 10px; padding: 5px 10px; font-family: 'Cascadia Code', Consolas, monospace; font-size: 11.5px;"
        )
        center_layout.addWidget(self.active_model_badge)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setHtml(self._wrap_html("<p><b>MAPLE Local Agent Desktop Ready.</b><br>Micro-router (0.5B CPU), persistent rules, web search, and live document rendering active.</p>"))
        center_layout.addWidget(self.chat_view, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setPlaceholderText("Enter prompt or autonomous task... (Ctrl+Enter to run)")
        self.prompt_input.setFixedHeight(80)
        input_row.addWidget(self.prompt_input, 1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("RunBtn")
        self.run_btn.clicked.connect(self._start_run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        btn_col.addWidget(self.run_btn)
        btn_col.addWidget(self.stop_btn)
        input_row.addLayout(btn_col)
        center_layout.addLayout(input_row)

        splitter.addWidget(center_widget)

        # Right Pane Card: Live Code / Diff + Rendered View + Terminal Log
        right_card = QFrame()
        right_card.setObjectName("PaneCard")
        right_card.setMinimumWidth(380)
        self._apply_shadow(right_card)
        right_lay = QVBoxLayout(right_card)
        right_lay.setContentsMargins(8, 8, 8, 8)
        self.right_tabs = QTabWidget()

        self.code_editor = QPlainTextEdit()
        self.code_editor.setReadOnly(True)
        self.code_editor.setFont(QFont("Cascadia Code", 10))
        self.highlighter = CodeAndDiffHighlighter(self.code_editor.document())
        self.right_tabs.addTab(self.code_editor, "Live File / Diff")

        self.rendered_view = QTextBrowser()
        self.rendered_view.setOpenExternalLinks(True)
        self.right_tabs.addTab(self.rendered_view, "Rendered View")

        self.terminal_log = QPlainTextEdit()
        self.terminal_log.setReadOnly(True)
        self.terminal_log.setFont(QFont("Cascadia Code", 10))
        self.right_tabs.addTab(self.terminal_log, "Terminal / Tools")

        right_lay.addWidget(self.right_tabs, 1)
        splitter.addWidget(right_card)
        
        # Proportional splitter distribution across all 3 panes
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 4)
        splitter.setSizes([260, 600, 580])

        root_layout.addWidget(splitter, 1)

        # Ctrl+Enter shortcut
        run_shortcut = QAction(self)
        run_shortcut.setShortcut("Ctrl+Return")
        run_shortcut.triggered.connect(self._start_run)
        self.addAction(run_shortcut)

    def _wrap_html(self, inner_html: str) -> str:
        return f"""
        <html><head><style>
        body {{ background-color: #111722; color: #e6edf3; font-family: 'Segoe UI', sans-serif; font-size: 13.5px; line-height: 1.5; }}
        pre {{ background-color: #090d12; border: 1px solid #232f42; padding: 10px; border-radius: 6px; font-family: 'Cascadia Code', Consolas, monospace; font-size: 12px; white-space: pre-wrap; }}
        code {{ background-color: #161e2c; padding: 1px 4px; border-radius: 4px; font-family: 'Cascadia Code', Consolas, monospace; }}
        h1, h2, h3 {{ color: #38bdf8; }}
        .user-box {{ background-color: #141d2b; border-left: 3px solid #38bdf8; padding: 8px 12px; margin: 8px 0; border-radius: 4px; }}
        .tool-box {{ background-color: #0d131d; border-left: 3px solid #10b981; padding: 6px 10px; margin: 6px 0; font-family: 'Cascadia Code', Consolas, monospace; font-size: 12px; }}
        </style></head><body>{inner_html}</body></html>
        """

    def _refresh_chat_html(self):
        blocks = list(self.chat_markdown_blocks)
        if self.current_stream_buf:
            rendered = markdown.markdown(self.current_stream_buf, extensions=["fenced_code", "tables"])
            blocks.append(rendered)
        self.chat_view.setHtml(self._wrap_html("".join(blocks)))
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _choose_workdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Working Directory", self.workdir)
        if d:
            self.workdir = d
            self.workdir_edit.setText(d)
            self.fs_model.setRootPath(d)
            self.tree.setRootIndex(self.fs_model.index(d))

    def _on_tree_clicked(self, index):
        path = self.fs_model.filePath(index)
        if os.path.isfile(path):
            lower = path.lower()
            if lower.endswith((".pdf", ".xlsx", ".xlsm", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                self._show_file_in_right_pane(path, f"[Binary document: {os.path.basename(path)}]")
            else:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    self._show_file_in_right_pane(path, content)
                except Exception:
                    pass

    def _watch_active_file(self, path: str):
        existing = self.fs_watcher.files()
        if existing:
            self.fs_watcher.removePaths(existing)
        if path and os.path.isfile(path):
            self.fs_watcher.addPath(path)

    def _on_watched_file_changed(self, path: str):
        if path != self.active_file_path or not os.path.isfile(path):
            return
        lower = path.lower()
        if lower.endswith((".pdf", ".xlsx", ".xlsm", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            self._show_file_in_right_pane(path, f"[Binary document: {os.path.basename(path)}]")
        else:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if content != self.active_file_content:
                    self._show_file_in_right_pane(path, content, animate=False)
            except Exception:
                pass
        if os.path.isfile(path) and path not in self.fs_watcher.files():
            self.fs_watcher.addPath(path)

    def _start_live_edit_anim(self, text: str):
        self._anim_timer.stop()
        lines = text.splitlines()
        if len(lines) <= 2 or len(lines) > 400:
            self.code_editor.setPlainText(text)
            return
        self._anim_lines = lines
        self._anim_idx = 0
        self.code_editor.clear()
        self._anim_timer.start()

    def _tick_live_edit_anim(self):
        if self._anim_idx >= len(self._anim_lines):
            self._anim_timer.stop()
            return
        batch_end = min(len(self._anim_lines), self._anim_idx + 4)
        chunk = "\n".join(self._anim_lines[self._anim_idx:batch_end])
        self.code_editor.appendPlainText(chunk)
        self._anim_idx = batch_end
        sb = self.code_editor.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_file_in_right_pane(self, path: str, content: str, animate: bool = False):
        self.active_file_path = path
        self.active_file_content = content
        self._watch_active_file(path)
        if animate:
            self._start_live_edit_anim(content)
        else:
            self._anim_timer.stop()
            self.code_editor.setPlainText(content)
        self.right_tabs.setTabText(0, f"Live File: {os.path.basename(path)}")
        lower = path.lower()
        rendered_html = DocumentRenderer.render_file(path, None if lower.endswith((".pdf", ".xlsx", ".xlsm", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".bmp")) else content)
        if lower.endswith((".html", ".htm")):
            self.rendered_view.setHtml(rendered_html)
        else:
            self.rendered_view.setHtml(self._wrap_html(rendered_html))
        if DocumentRenderer.is_rich_doc(path):
            self.right_tabs.setCurrentIndex(1)

    def _start_vram_timer(self):
        self.vram_timer = QTimer(self)
        self.vram_timer.timeout.connect(self._update_vram)
        self.vram_timer.start(3000)
        self._update_vram()

    def _update_vram(self):
        try:
            v = Tools.vram_status()
            used = int(v.get("used_mib", 0))
            total = int(v.get("total_mib", 6141))
            self.vram_bar.setMaximum(total)
            self.vram_bar.setValue(used)
        except Exception:
            pass

    def _start_run(self):
        if self.worker and self.worker.isRunning():
            return
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            return
        self.workdir = self.workdir_edit.text().strip() or self.workdir
        self.prompt_input.clear()

        safe_prompt = prompt.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        self.chat_markdown_blocks.append(f'<div class="user-box"><b>YOU:</b><br>{safe_prompt}</div>')
        self.current_stream_buf = ""
        self._refresh_chat_html()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = AgentWorker(
            prompt=prompt,
            model_key=self.selected_model_key,
            mode=self.selected_mode,
            workdir=self.workdir,
        )
        self.worker.model_selected.connect(self._on_model_selected)
        self.worker.step_started.connect(self._on_step_started)
        self.worker.token_received.connect(self._on_token)
        self.worker.file_written.connect(self._on_file_written)
        self.worker.file_edited.connect(self._on_file_edited)
        self.worker.file_read.connect(self._on_file_read)
        self.worker.cmd_executed.connect(self._on_cmd_executed)
        self.worker.telemetry.connect(self._on_telemetry)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.start()

    def _stop_run(self):
        if self.worker:
            self.worker.stop()
            self.active_model_badge.setText("Stopping...")

    def _on_model_selected(self, key: str, name: str, role: str, reason: str):
        self.active_model_badge.setText(f"Active: {key.upper()} ({name}) | Router: {reason}")

    def _on_step_started(self, step: int, max_steps: int):
        if self.current_stream_buf:
            self.chat_markdown_blocks.append(markdown.markdown(self.current_stream_buf, extensions=["fenced_code", "tables"]))
            self.current_stream_buf = ""
        self.chat_markdown_blocks.append(f"<p><b>[Step {step}/{max_steps}]</b></p>")
        self._refresh_chat_html()

    def _on_token(self, tok: str):
        self.current_stream_buf += tok
        now = time.time()
        # Stream live every ~60ms without blocking or recursive event pumping
        if now - self._last_chat_refresh_t > 0.06 or "\n" in tok or tok.endswith((".", ":", "`", "}", "]")):
            self._last_chat_refresh_t = now
            self._refresh_chat_html()

    def _on_file_written(self, path: str, content: str):
        self._show_file_in_right_pane(path, content, animate=True)
        self.right_tabs.setCurrentIndex(1 if DocumentRenderer.is_rich_doc(path) else 0)
        self.chat_markdown_blocks.append(f'<div class="tool-box">WRITE FILE: {path} ({len(content)} chars)</div>')
        self._refresh_chat_html()

    def _on_file_edited(self, path: str, diff_str: str, new_content: str):
        self.active_file_path = path
        self.active_file_content = new_content
        self._watch_active_file(path)
        self._start_live_edit_anim(diff_str or new_content)
        self.right_tabs.setTabText(0, f"Diff: {os.path.basename(path)}")
        rendered_html = DocumentRenderer.render_file(path, new_content)
        self.rendered_view.setHtml(rendered_html if path.lower().endswith((".html", ".htm")) else self._wrap_html(rendered_html))
        self.right_tabs.setCurrentIndex(1 if DocumentRenderer.is_rich_doc(path) else 0)
        self.chat_markdown_blocks.append(f'<div class="tool-box">EDIT DIFF: {path}</div>')
        self._refresh_chat_html()

    def _on_file_read(self, path: str, content: str, total_lines: int):
        self._show_file_in_right_pane(path, content)
        self.chat_markdown_blocks.append(f'<div class="tool-box">READ FILE: {path} ({total_lines} lines)</div>')
        self._refresh_chat_html()

    def _on_cmd_executed(self, title: str, cmd: str, exit_code: int, out_txt: str):
        self.terminal_log.appendPlainText(f"[{title} | EXIT {exit_code}] $ {cmd}\n{out_txt}\n{'-'*60}")
        self.chat_markdown_blocks.append(f'<div class="tool-box">{title}: {cmd} | EXIT {exit_code}</div>')
        self._refresh_chat_html()
        # Auto-render any generated PDF/XLSX/PPTX/TEX/PNG referenced in command or output
        for match in re.findall(r'([A-Za-z0-9_./\\:-]+\.(?:pdf|xlsx|xlsm|csv|pptx|tex|png|jpg|jpeg))', cmd + "\n" + out_txt, re.IGNORECASE):
            cand = match if os.path.isabs(match) else os.path.abspath(os.path.join(self.workdir, match))
            if os.path.isfile(cand):
                self._show_file_in_right_pane(cand, f"[Generated document: {os.path.basename(cand)}]")
                break

    def _on_telemetry(self, tokens: int, elapsed: float, tps: float):
        curr = self.active_model_badge.text().split(" | Telemetry:")[0]
        self.active_model_badge.setText(f"{curr} | Telemetry: ~{tokens} tok in {elapsed}s ({tps} TPS)")

    def _on_error(self, msg: str):
        self.chat_markdown_blocks.append(f'<p style="color:#ef4444;">[Error: {msg}]</p>')
        self._refresh_chat_html()

    def _on_finished(self):
        if self.current_stream_buf:
            self.chat_markdown_blocks.append(markdown.markdown(self.current_stream_buf, extensions=["fenced_code", "tables"]))
            self.current_stream_buf = ""
        self._refresh_chat_html()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._update_vram()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MAPLE Local Agent Desktop")
    parser.add_argument("--workdir", default=None, help="Initial working directory")
    parser.add_argument("--task", default=None, help="Initial prompt/task to prefill")
    parser.add_argument("--model", default="auto", choices=["auto", "deepseek", "gemma", "bonsai", "fast"], help="Initial model")
    parser.add_argument("--autorun", action="store_true", help="Automatically start the prefilled task")
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0]] + qt_args)
    app.setStyleSheet(DARK_STYLESHEET)
    win = MapleDesktopWindow()
    if args.workdir and os.path.isdir(args.workdir):
        win.workdir = os.path.abspath(args.workdir)
        win.workdir_edit.setText(win.workdir)
        win.fs_model.setRootPath(win.workdir)
        win.tree.setRootIndex(win.fs_model.index(win.workdir))
    if args.model in ("auto", "deepseek", "gemma", "bonsai", "fast"):
        win.selected_model_key = args.model
    if args.task:
        win.prompt_input.setPlainText(args.task)
    win.show()
    if args.autorun and args.task:
        QTimer.singleShot(400, win._start_run)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
