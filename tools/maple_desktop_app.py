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
import json
import time
import difflib
from typing import Optional, Dict, Any

import requests
import markdown
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QDir, QRegularExpression
from PySide6.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QAction, QTextCursor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeView, QFileSystemModel, QTextBrowser, QPlainTextEdit,
    QPushButton, QComboBox, QLabel, QProgressBar, QTabWidget, QLineEdit,
    QFileDialog, QFrame
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
    ensure_vram_for_model,
)

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0b0f14;
    color: #e6edf3;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QFrame#TopBar {
    background-color: #111722;
    border-bottom: 1px solid #232f42;
}
QComboBox, QLineEdit, QPlainTextEdit, QTextBrowser, QTreeView {
    background-color: #111722;
    color: #e6edf3;
    border: 1px solid #232f42;
    border-radius: 6px;
    padding: 4px 8px;
}
QTreeView::item:selected {
    background-color: #1f334d;
    color: #38bdf8;
}
QPushButton {
    background-color: #161e2c;
    color: #e6edf3;
    border: 1px solid #232f42;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover {
    border-color: #38bdf8;
}
QPushButton#RunBtn {
    background-color: #38bdf8;
    color: #071019;
    border: none;
}
QPushButton#RunBtn:disabled {
    background-color: #1e3a4c;
    color: #6b7c93;
}
QPushButton#StopBtn {
    background-color: #3b1820;
    color: #fca5a5;
    border: 1px solid #7f1d1d;
}
QTabWidget::pane {
    border: 1px solid #232f42;
    border-radius: 6px;
    background-color: #0d121a;
}
QTabBar::tab {
    background-color: #111722;
    color: #8b9bb4;
    border: 1px solid #232f42;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background-color: #161e2c;
    color: #10b981;
    border-bottom-color: #10b981;
}
QProgressBar {
    background-color: #090d12;
    border: 1px solid #232f42;
    border-radius: 5px;
    text-align: center;
    color: #e6edf3;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #10b981;
    border-radius: 4px;
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

    model_selected = Signal(str, str, str)
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
                "options": {"temperature": cfg["temperature"], "num_predict": cfg["max_tokens"]},
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
            chosen = auto_choose_model(self.prompt) if self.model_key == "auto" else self.model_key
            if chosen not in MODEL_ROSTER:
                chosen = "gemma"
            cfg = MODEL_ROSTER[chosen]
            self.model_selected.emit(chosen, cfg["name"], cfg["role"])

            if self.mode == "chat":
                self._stream_llm(chosen, self.prompt)
                self.finished_run.emit()
                return

            system_prompt = (
                f"You are an autonomous engineering agent operating in working directory: {self.workdir}\n"
                f"Role: {cfg['role']}\n"
                f"{TOOL_SCHEMAS}\n"
                f"Rules:\n"
                f"1. To execute work, write TOOL: <name> followed by ARGS: <json>.\n"
                f"2. After tools execute, inspect their outputs and decide the next step.\n"
                f"3. Only when all tool results have been observed and the task is accomplished, output 'FINAL ANSWER:' followed by your concise report.\n"
                f"4. Ban placeholders. Write complete, working solutions."
            )
            history = f"TASK GOAL: {self.prompt}\n"
            max_steps = 6

            for step in range(1, max_steps + 1):
                if self._stop_flag:
                    break
                self.step_started.emit(step, max_steps)
                response_text = self._stream_llm(chosen, history + "\nWhat is your next action?", system=system_prompt)
                if self._stop_flag:
                    break

                tool_matches = list(re.finditer(r"TOOL:\s*(\w+)\s*\nARGS:\s*({.*?})", response_text, re.DOTALL))
                executed_any = False

                for m in tool_matches:
                    tool_name = m.group(1).strip()
                    try:
                        args = json.loads(m.group(2).strip())
                    except Exception:
                        args = {}

                    if tool_name not in TOOL_REGISTRY:
                        history += f"\nERROR: Unknown tool '{tool_name}'\n"
                        continue

                    executed_any = True
                    if tool_name == "write_file":
                        fpath = self._resolve_path(args.get("path", "output.txt"))
                        content = args.get("content", "")
                        os.makedirs(os.path.dirname(fpath), exist_ok=True)
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(content)
                        result = {"success": True, "path": fpath, "bytes_written": len(content)}
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
                                self.file_edited.emit(fpath, diff_str, new_data)

                    elif tool_name == "read_file":
                        fpath = self._resolve_path(args.get("path", ""))
                        if not os.path.exists(fpath):
                            result = {"error": f"File not found: {fpath}"}
                        else:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                lines = f.readlines()
                            raw_txt = "".join(lines)
                            result = {"path": fpath, "total_lines": len(lines), "content": raw_txt[:8000]}
                            self.file_read.emit(fpath, raw_txt, len(lines))

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
                        result = TOOL_REGISTRY[tool_name](**args)

                    history += f"\nACTION TAKEN: {tool_name} with {args}\nRESULT: {json.dumps(result)}\n"

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
        self.active_file_path = ""
        self.active_file_content = ""

        self._build_ui()
        self._start_vram_timer()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top Bar
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.setSpacing(10)

        title_lbl = QLabel("MAPLE Desktop Harness")
        title_lbl.setStyleSheet("font-weight: 700; font-size: 15px; color: #38bdf8;")
        top_layout.addWidget(title_lbl)

        top_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Auto Chooser (Recommended)", "auto")
        self.model_combo.addItem("DeepSeek 54-Exp (Systems/C++)", "deepseek")
        self.model_combo.addItem("Gemma 4 26B (General/Web)", "gemma")
        self.model_combo.addItem("Bonsai 27B (Reasoning/Decisions)", "bonsai")
        self.model_combo.addItem("Fast 1.5B (Triage/Commits)", "fast")
        top_layout.addWidget(self.model_combo)

        top_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Agent Task (Tools)", "task")
        self.mode_combo.addItem("Direct Chat", "chat")
        top_layout.addWidget(self.mode_combo)

        top_layout.addWidget(QLabel("Workdir:"))
        self.workdir_edit = QLineEdit(self.workdir)
        self.workdir_edit.setMinimumWidth(200)
        top_layout.addWidget(self.workdir_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._choose_workdir)
        top_layout.addWidget(browse_btn)

        top_layout.addStretch(1)

        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 6141)
        self.vram_bar.setFormat("VRAM: %v / %m MiB")
        self.vram_bar.setFixedWidth(180)
        top_layout.addWidget(self.vram_bar)

        root_layout.addWidget(top_bar)

        # 3-Pane Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left Pane: Workspace File Tree
        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath(self.workdir)
        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(self.workdir))
        for col in (1, 2, 3):
            self.tree.hideColumn(col)
        self.tree.setHeaderHidden(True)
        self.tree.clicked.connect(self._on_tree_clicked)
        splitter.addWidget(self.tree)

        # Center Pane: Chat & Prompt
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        self.active_model_badge = QLabel("Ready | Auto Model Chooser active")
        self.active_model_badge.setStyleSheet("color: #8b9bb4; font-family: 'Cascadia Code', Consolas, monospace; font-size: 11.5px;")
        center_layout.addWidget(self.active_model_badge)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setHtml(self._wrap_html("<p><b>MAPLE Local Agent Desktop Ready.</b><br>Ask a coding/architecture question or give an autonomous file task.</p>"))
        center_layout.addWidget(self.chat_view, 1)

        input_row = QHBoxLayout()
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setPlaceholderText("Enter prompt or autonomous task... (Ctrl+Enter to run)")
        self.prompt_input.setFixedHeight(78)
        input_row.addWidget(self.prompt_input, 1)

        btn_col = QVBoxLayout()
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

        # Right Pane: Live Code / Diff + Rendered View + Terminal Log
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

        splitter.addWidget(self.right_tabs)
        splitter.setSizes([230, 620, 590])

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
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                self._show_file_in_right_pane(path, content)
            except Exception:
                pass

    def _show_file_in_right_pane(self, path: str, content: str):
        self.active_file_path = path
        self.active_file_content = content
        self.code_editor.setPlainText(content)
        self.right_tabs.setTabText(0, f"Live File: {os.path.basename(path)}")
        lower = path.lower()
        if lower.endswith(".md"):
            html = markdown.markdown(content, extensions=["fenced_code", "tables"])
            self.rendered_view.setHtml(self._wrap_html(html))
        elif lower.endswith(".html") or lower.endswith(".htm"):
            self.rendered_view.setHtml(content)
        else:
            self.rendered_view.setHtml(self._wrap_html(f"<h3>{os.path.basename(path)}</h3><pre>{content.replace('<', '&lt;')}</pre>"))

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
            model_key=self.model_combo.currentData(),
            mode=self.mode_combo.currentData(),
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

    def _on_model_selected(self, key: str, name: str, role: str):
        self.active_model_badge.setText(f"Active: {key.upper()} ({name}) | {role}")

    def _on_step_started(self, step: int, max_steps: int):
        if self.current_stream_buf:
            self.chat_markdown_blocks.append(markdown.markdown(self.current_stream_buf, extensions=["fenced_code", "tables"]))
            self.current_stream_buf = ""
        self.chat_markdown_blocks.append(f"<p><b>[Step {step}/{max_steps}]</b></p>")
        self._refresh_chat_html()

    def _on_token(self, tok: str):
        self.current_stream_buf += tok
        self._refresh_chat_html()

    def _on_file_written(self, path: str, content: str):
        self._show_file_in_right_pane(path, content)
        self.right_tabs.setCurrentIndex(0)
        self.chat_markdown_blocks.append(f'<div class="tool-box">WRITE FILE: {path} ({len(content)} chars)</div>')
        self._refresh_chat_html()

    def _on_file_edited(self, path: str, diff_str: str, new_content: str):
        self.active_file_path = path
        self.active_file_content = new_content
        self.code_editor.setPlainText(diff_str or new_content)
        self.right_tabs.setTabText(0, f"Diff: {os.path.basename(path)}")
        self.right_tabs.setCurrentIndex(0)
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
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    win = MapleDesktopWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
