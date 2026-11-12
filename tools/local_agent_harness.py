"""
MAPLE Local Agent Tool Harness & Model Chooser (v2.0)
=====================================================
A unified developer tool harness and intelligent model router designed for real-world engineering work.
Manages a 4-tier model roster tailored for specialized tasks with automated VRAM safety on 6GB GPUs.

Model Roster:
  1. deepseek (DeepSeek-Coder-V2-Lite MoE Sub-5GB, 54 Experts):
     Hardcore Systems & Algorithmic Engineering (C++20, Rust, lock-free concurrency, low-level data structures).
  2. gemma (Gemma 4 26B Sub-5GB in-situ):
     Fast General Coding, Web/Frontend, Scripts, Code Review & Refactoring.
  3. bonsai (Ternary-Bonsai-2 27B PTQ1_0):
     Chief Architect, Strategic Reasoner & Decision Maker (Architectural trade-offs, decision-relevant pragmatism, game theory).
  4. fast (Qwen 2.5 Coder 1.5B / Sub-1GB):
     Ultra-Fast Triage, Git Commits, Regex Patterns, Instant Shell One-Liners (< 300ms latency).

Built-in Engineering Tools:
  - run_command: Run shell commands (PowerShell/bash) with timeout and stdout/stderr capture.
  - read_file: Read files with optional line-range limits.
  - write_file: Atomically write or overwrite files.
  - edit_file: Exact search-and-replace string replacement.
  - list_dir: List directory contents with file metadata.
  - python_eval: Run isolated Python snippets and capture return values.
  - git_status: Inspect git status, modified files, and branch info.
  - vram_status: Query live GPU VRAM usage and active processes.

Usage:
  # One-shot prompt with auto-routing
  python local_agent_harness.py "Write a C++20 lock-free single-producer single-consumer ring buffer"

  # Explicit model selection
  python local_agent_harness.py --model bonsai "Evaluate the strategic trade-offs of microservices vs monolith"

  # Autonomous agent task with tools
  python local_agent_harness.py --task "Read README.md, find any mentions of 48-experts, and update them to 54-experts"

  # Interactive REPL mode
  python local_agent_harness.py --chat

Author: Aman Sachan (amansachan92905@gmail.com)
"""

import sys
import os
import re
import json
import time
import shutil
import difflib
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple

import requests

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Service Endpoints
DEFAULT_OLLAMA_API = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434")
DEFAULT_BONSAI_API = os.environ.get("BONSAI_API", "http://127.0.0.1:8081")

MODEL_ROSTER = {
    "deepseek": {
        "id": "deepseek-coder-maple:latest",
        "name": "DeepSeek-Coder-V2-Lite MAPLE (Sub-5GB MoE)",
        "role": "Hardcore Systems, Concurrency & Algorithmic Specialist",
        "specialties": ["C++20", "Rust", "Lock-Free Structures", "Memory Order Semantics", "Algorithms", "Complex Math"],
        "backend": "ollama",
        "temperature": 0.3,
        "max_tokens": 4096
    },
    "gemma": {
        "id": "gemma4:latest",
        "name": "Gemma 4 26B MAPLE (Sub-5GB In-Situ)",
        "role": "General Software, Frontend & Code Review Specialist",
        "specialties": ["Python Scripts", "Web/API Development", "Frontend (HTML/CSS/JS)", "Refactoring", "Documentation"],
        "backend": "ollama",
        "temperature": 0.5,
        "max_tokens": 4096
    },
    "bonsai": {
        "id": "bonsai:latest",
        "name": "Ternary Bonsai 2 27B (PTQ1_0 100% VRAM)",
        "role": "Chief Architect, Strategic Reasoner & Decision Maker",
        "specialties": ["Architectural Decisions", "Decision-Relevant Pragmatism", "Game Theory", "Strategic Moats", "System Design"],
        "backend": "bonsai_or_router",
        "temperature": 0.6,
        "max_tokens": 4096
    },
    "fast": {
        "id": "qwen2.5-coder:1.5b",
        "name": "Qwen 2.5 Coder 1.5B (Ultra-Fast Triage)",
        "role": "Sub-Second Triage, Git Commits & Terminal Copilot",
        "specialties": ["Git Commits", "Regex", "Shell One-Liners", "Quick Syntax Checks", "Fast Edits (<300ms)"],
        "backend": "ollama",
        "temperature": 0.2,
        "max_tokens": 2048
    }
}

# Heuristic Intent Classifier (Model Chooser)
def auto_choose_model(prompt: str) -> str:
    p_lower = prompt.lower()

    # 1. Strategic Reasoning / Decision Making / Architecture -> Bonsai
    reasoning_keywords = [
        "decision", "trade-off", "tradeoff", "strategy", "architect", "architecture",
        "game theory", "why should we", "evaluate", "ostrom", "governance", "commons",
        "pragmatism", "rubric", "cfo", "m&a", "monetization", "du-pont", "dupont",
        "swot", "pestle", "porter", "pros and cons", "compare approaches"
    ]
    if any(k in p_lower for k in reasoning_keywords):
        return "bonsai"

    # 2. Hardcore Systems / Concurrency / Low-level -> DeepSeek
    hardcore_keywords = [
        "c++", "cpp", "c++20", "lock-free", "concurrency", "mutex", "spin-lock", "spinlock",
        "memory_order", "atomic", "ring buffer", "hypergraph", "dijkstra", "pointer",
        "segmentation", "assembly", "simd", "avx", "cuda", "kernel", "zero-copy",
        "cache-line", "false sharing", "template metaprogramming", "concepts"
    ]
    if any(k in p_lower for k in hardcore_keywords):
        return "deepseek"

    # 3. Super Small / Instant Dev Triage -> Fast
    fast_keywords = [
        "git commit", "commit message", "regex", "regular expression", "powershell one-liner",
        "bash one-liner", "quick one-liner", "shell command", "grep", "sed", "awk",
        "explain this line", "fix typo", "quick syntax"
    ]
    if any(k in p_lower for k in fast_keywords):
        return "fast"

    # 4. General Software / Scripts / Frontend -> Gemma
    return "gemma"

def load_agent_rules() -> str:
    """Loads persistent synced rules from maple_agent_rules.md."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "maple_agent_rules.md"),
        r"E:\OllamaModels\maple_agent_rules.md",
        r"C:\Users\amans\.gemini\antigravity\scratch\repos\maple\maple_agent_rules.md",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    return f.read().strip()
            except Exception:
                pass
    return ""

def micro_model_route(prompt: str) -> Tuple[str, str]:
    """Uses a dedicated 0.5B CPU-resident micro-model (qwen2.5:0.5b, 0 MB VRAM) to classify intent."""
    router_prompt = (
        "Pick ONE model for this task:\n"
        "deepseek = C++, Rust, CUDA, lock-free concurrency, systems, algorithms\n"
        "gemma = Python, web/HTML/CSS/JS, APIs, scripts, PDF/Excel/PPTX/LaTeX files\n"
        "bonsai = Architecture trade-offs, strategy, decision-making, game theory, reasoning\n"
        "fast = Git commits, regex, shell one-liners, quick syntax fixes\n\n"
        f"Task: {prompt[:450]}\n\n"
        'Reply ONLY JSON: {"model": "deepseek|gemma|bonsai|fast", "reason": "5 words max"}'
    )
    try:
        r = requests.post(
            f"{DEFAULT_OLLAMA_API}/api/generate",
            json={
                "model": "qwen2.5:0.5b",
                "prompt": router_prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "20m",
                "options": {"num_gpu": 0, "temperature": 0.0, "num_predict": 96},
            },
            timeout=15.0,
        )
        if r.status_code == 200:
            raw = r.json().get("response", "").strip()
            try:
                data = json.loads(raw)
                chosen = str(data.get("model", "")).strip().lower()
                reason = str(data.get("reason", "0.5B micro-router")).strip()
                if chosen in MODEL_ROSTER:
                    return chosen, reason
            except Exception:
                m = re.search(r'"model"\s*:\s*"(deepseek|gemma|bonsai|fast)"', raw, re.IGNORECASE)
                if m:
                    return m.group(1).lower(), "0.5B micro-router"
    except Exception:
        pass
    fallback = auto_choose_model(prompt)
    return fallback, "Heuristic fallback (micro-router offline)"

# VRAM Arbitrator for 6GB Laptop GPU
def ensure_vram_for_model(model_key: str):
    target_backend = MODEL_ROSTER.get(model_key, {}).get("backend", "ollama")

    if target_backend == "bonsai_or_router":
        # Unload Ollama models to free 4.8GB
        try:
            r = requests.get(f"{DEFAULT_OLLAMA_API}/api/ps", timeout=1.5)
            if r.status_code == 200:
                for m in r.json().get("models", []):
                    name = m.get("name") or m.get("model")
                    if name:
                        requests.post(f"{DEFAULT_OLLAMA_API}/api/generate", json={"model": name, "keep_alive": 0}, timeout=2.0)
        except Exception:
            pass

    elif target_backend == "ollama":
        # If the gateway router is active on DEFAULT_OLLAMA_API, it manages VRAM eviction automatically.
        router_active = False
        try:
            tags_r = requests.get(f"{DEFAULT_OLLAMA_API}/api/tags", timeout=1.0)
            if tags_r.status_code == 200:
                router_active = any("bonsai" in m.get("name", "") for m in tags_r.json().get("models", []))
        except Exception:
            pass

        if not router_active:
            # Check if standalone llama-server is holding VRAM on 8081
            try:
                r = requests.get(f"{DEFAULT_BONSAI_API}/health", timeout=0.8)
                if r.status_code == 200:
                    # Terminate only standalone Bonsai llama-server, not Ollama's internal runner
                    subprocess.run(
                        ["powershell", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { $_.ExecutablePath -notlike '*Ollama*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    time.sleep(1)
            except Exception:
                pass

ENABLE_RICH_UI = True

def _guess_lexer(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mapping = {
        "py": "python", "cpp": "cpp", "hpp": "cpp", "h": "cpp", "c": "c",
        "js": "javascript", "ts": "typescript", "tsx": "tsx", "jsx": "jsx",
        "json": "json", "md": "markdown", "bat": "bat", "ps1": "powershell",
        "sh": "bash", "yaml": "yaml", "yml": "yaml", "html": "html", "css": "css",
        "rs": "rust", "go": "go", "sql": "sql", "toml": "toml"
    }
    return mapping.get(ext, "text")

class VisualRenderer:
    """Renders live file writes, unified diffs, terminal runs, and markdown in the console."""

    @staticmethod
    def show_write_file(path: str, content: str):
        if not (RICH_AVAILABLE and ENABLE_RICH_UI):
            return
        lexer = _guess_lexer(path)
        lines = content.splitlines()
        max_preview = 60
        preview_lines = lines[:max_preview]
        truncated = len(lines) > max_preview

        # Animate progressive line reveal for live "writing to file" effect
        step = max(1, len(preview_lines) // 12)
        with Live(console=console, refresh_per_second=15, transient=False) as live:
            for idx in range(step, len(preview_lines) + step, step):
                chunk = "\n".join(preview_lines[:min(idx, len(preview_lines))])
                if truncated and idx >= len(preview_lines):
                    chunk += f"\n# ... ({len(lines) - max_preview} more lines written)"
                syn = Syntax(chunk, lexer, theme="monokai", line_numbers=True, word_wrap=True)
                live.update(Panel(syn, title=f"[bold green]WRITE FILE[/bold green] {path}", border_style="green"))
                if idx < len(preview_lines):
                    time.sleep(0.03)

    @staticmethod
    def show_edit_diff(path: str, old_text: str, new_text: str):
        if not (RICH_AVAILABLE and ENABLE_RICH_UI):
            return
        diff_lines = list(difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{os.path.basename(path)}",
            tofile=f"b/{os.path.basename(path)}",
            lineterm=""
        ))
        diff_str = "\n".join(diff_lines[:80])
        if len(diff_lines) > 80:
            diff_str += f"\n... ({len(diff_lines) - 80} more diff lines)"
        if not diff_str:
            diff_str = "(no changes)"
        syn = Syntax(diff_str, "diff", theme="monokai", word_wrap=True)
        console.print(Panel(syn, title=f"[bold yellow]EDIT DIFF[/bold yellow] {path}", border_style="yellow"))

    @staticmethod
    def show_read_file(path: str, content: str, total_lines: int):
        if not (RICH_AVAILABLE and ENABLE_RICH_UI):
            return
        lexer = _guess_lexer(path)
        lines = content.splitlines()
        preview = "\n".join(lines[:35])
        if len(lines) > 35:
            preview += f"\n... ({len(lines) - 35} more lines, total {total_lines})"
        syn = Syntax(preview, lexer, theme="monokai", word_wrap=True)
        console.print(Panel(syn, title=f"[bold cyan]READ FILE[/bold cyan] {path} ({total_lines} lines)", border_style="cyan"))

    @staticmethod
    def show_command_result(title: str, cmd_str: str, res: Dict[str, Any]):
        if not (RICH_AVAILABLE and ENABLE_RICH_UI):
            return
        code = res.get("exit_code", -1)
        badge = f"[bold green]EXIT {code}[/bold green]" if code == 0 else f"[bold red]EXIT {code}[/bold red]"
        out = (res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        body = f"$ {cmd_str}\n"
        if out:
            body += f"\n{out[:1200]}"
        if err:
            body += f"\n[STDERR]\n{err[:600]}"
        border = "blue" if code == 0 else "red"
        console.print(Panel(body, title=f"[bold]{title}[/bold] | {badge}", border_style=border))

# Built-in Engineering Tools Implementation
class Tools:
    @staticmethod
    def run_command(cmd: str, timeout: int = 45, silent: bool = False) -> Dict[str, Any]:
        """Runs a shell command and captures stdout, stderr, and returncode."""
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            res = {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr
            }
            if not silent:
                VisualRenderer.show_command_result("SHELL COMMAND", cmd, res)
            return res
        except subprocess.TimeoutExpired:
            res = {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
            if not silent:
                VisualRenderer.show_command_result("SHELL COMMAND", cmd, res)
            return res
        except Exception as e:
            res = {"exit_code": -1, "stdout": "", "stderr": str(e)}
            if not silent:
                VisualRenderer.show_command_result("SHELL COMMAND", cmd, res)
            return res

    @staticmethod
    def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Dict[str, Any]:
        """Reads content of a file with optional line numbers."""
        if not os.path.exists(path):
            return {"error": f"File '{path}' does not exist."}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total_lines = len(lines)
            s = max(0, (start_line - 1)) if start_line else 0
            e = min(total_lines, end_line) if end_line else total_lines
            selected = lines[s:e]
            raw_content = "".join(selected)
            content = "".join([f"{s + idx + 1:4d} | {line}" for idx, line in enumerate(selected)])
            VisualRenderer.show_read_file(path, raw_content, total_lines)
            return {"path": path, "total_lines": total_lines, "content": content}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def write_file(path: str, content: str) -> Dict[str, Any]:
        """Writes content atomically to a file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            VisualRenderer.show_write_file(path, content)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "path": path, "bytes_written": len(content)}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def edit_file(path: str, target: str, replacement: str) -> Dict[str, Any]:
        """Replaces exact target string with replacement in a file."""
        if not os.path.exists(path):
            return {"error": f"File '{path}' does not exist."}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            if target not in data:
                return {"error": "Target string not found in file."}
            count = data.count(target)
            new_data = data.replace(target, replacement)
            VisualRenderer.show_edit_diff(path, data, new_data)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_data)
            return {"success": True, "path": path, "replacements": count}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def list_dir(path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        """Lists files and directories with sizes."""
        if not os.path.exists(path):
            return {"error": f"Path '{path}' does not exist."}
        entries = []
        try:
            if recursive:
                for root, dirs, files in os.walk(path):
                    for f in files:
                        fp = os.path.join(root, f)
                        entries.append({"name": fp, "size": os.path.getsize(fp)})
            else:
                for it in os.scandir(path):
                    entries.append({
                        "name": it.name,
                        "is_dir": it.is_dir(),
                        "size": it.stat().st_size if not it.is_dir() else 0
                    })
            return {"path": path, "entries": entries[:100]}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def python_eval(code: str) -> Dict[str, Any]:
        """Runs Python snippet in an isolated subprocess."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            res = {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr
            }
            VisualRenderer.show_command_result("PYTHON EVAL", code[:120].replace("\n", "; "), res)
            return res
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

    @staticmethod
    def git_status() -> Dict[str, Any]:
        """Returns git branch and modified files."""
        res = Tools.run_command("git status --short", silent=True)
        branch_res = Tools.run_command("git branch --show-current", silent=True)
        return {
            "branch": branch_res.get("stdout", "").strip(),
            "status": res.get("stdout", "").strip()
        }

    @staticmethod
    def vram_status() -> Dict[str, Any]:
        """Returns live GPU VRAM usage via nvidia-smi."""
        res = Tools.run_command("nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits", silent=True)
        if res["exit_code"] == 0:
            parts = [p.strip() for p in res["stdout"].strip().split(",")]
            if len(parts) >= 4:
                return {
                    "total_mib": parts[0],
                    "used_mib": parts[1],
                    "free_mib": parts[2],
                    "gpu_util_pct": parts[3]
                }
        return {"raw": res["stdout"]}

    @staticmethod
    def search_files(pattern: str, path: str = ".", max_hits: int = 30) -> Dict[str, Any]:
        """Searches text files for a pattern so the agent locates lines before reading."""
        hits = []
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fn in files:
                if fn.lower().endswith((".gguf", ".bin", ".exe", ".dll", ".png", ".jpg", ".pdf", ".xlsx", ".pptx")):
                    continue
                fp = os.path.join(root, fn)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for idx, line in enumerate(f, start=1):
                            if rx.search(line):
                                hits.append(f"{fp}:{idx}: {line.strip()[:140]}")
                                if len(hits) >= max_hits:
                                    return {"matches": hits, "truncated": True}
                except Exception:
                    continue
        return {"matches": hits, "truncated": False}

    @staticmethod
    def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
        """Searches official JSON APIs (DuckDuckGo Instant Answer, StackOverflow, Wikipedia) with zero CAPTCHAs."""
        results = []
        headers = {"User-Agent": "MAPLE-Local-Agent/2.0"}
        # 1. DuckDuckGo Instant Answer JSON
        try:
            r = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                headers=headers,
                timeout=8,
            )
            if r.status_code == 200:
                d = r.json()
                if d.get("AbstractText"):
                    results.append({"title": d.get("Heading", query), "url": d.get("AbstractURL", ""), "snippet": d["AbstractText"]})
                for topic in d.get("RelatedTopics", [])[:3]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({"title": topic["Text"][:60], "url": topic.get("FirstURL", ""), "snippet": topic["Text"]})
        except Exception:
            pass
        # 2. StackOverflow API (great for coding/library queries)
        try:
            r = requests.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={"order": "desc", "sort": "relevance", "q": query, "site": "stackoverflow", "pagesize": max_results},
                headers=headers,
                timeout=8,
            )
            if r.status_code == 200:
                for item in r.json().get("items", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": f"Tags: {', '.join(item.get('tags', []))} | Answered: {item.get('is_answered')}"
                    })
        except Exception:
            pass
        # 3. Wikipedia API (great for general/conceptual queries)
        if len(results) < max_results:
            try:
                r = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3},
                    headers=headers,
                    timeout=8,
                )
                if r.status_code == 200:
                    for item in r.json().get("query", {}).get("search", []):
                        snip = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                        title = item.get("title", "")
                        results.append({"title": title, "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", "snippet": snip})
            except Exception:
                pass
        return {"query": query, "results": results[:max_results]}

    @staticmethod
    def fetch_url(url: str, max_chars: int = 3500) -> Dict[str, Any]:
        """Fetches a webpage and extracts clean readable text within finite context limits."""
        try:
            import html as html_lib
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=12,
            )
            txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.DOTALL | re.IGNORECASE)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = html_lib.unescape(re.sub(r"\s+", " ", txt)).strip()
            return {"url": url, "status": r.status_code, "text": txt[:max_chars]}
        except Exception as e:
            return {"url": url, "error": str(e)}

TOOL_REGISTRY = {
    "run_command": Tools.run_command,
    "read_file": Tools.read_file,
    "write_file": Tools.write_file,
    "edit_file": Tools.edit_file,
    "list_dir": Tools.list_dir,
    "search_files": Tools.search_files,
    "web_search": Tools.web_search,
    "fetch_url": Tools.fetch_url,
    "python_eval": Tools.python_eval,
    "git_status": Tools.git_status,
    "vram_status": Tools.vram_status
}

TOOL_SCHEMAS = """
Available Local & Web Tools:
1. search_files(pattern="<regex/text>", path=".") - Locate symbols/lines before reading!
2. read_file(path="<path>", start_line=<int>, end_line=<int>) - Read only needed line ranges.
3. write_file(path="<path>", content="<content>")
4. edit_file(path="<path>", target="<exact string>", replacement="<replacement string>")
5. list_dir(path="<dir>", recursive=<bool>)
6. run_command(cmd="<shell command>")
7. python_eval(code="<python code>")
8. web_search(query="<search terms>") - Search the web for current info/docs.
9. fetch_url(url="<https://...>") - Read text from a webpage.
10. git_status()
11. vram_status()

To call a tool, format your output strictly as:
TOOL: <tool_name>
ARGS: {"arg1": "value"}
"""

# Inference Dispatcher
def query_model(model_key: str, prompt: str, system: Optional[str] = None, stream: bool = True) -> str:
    cfg = MODEL_ROSTER.get(model_key, MODEL_ROSTER["gemma"])
    ensure_vram_for_model(model_key)

    model_id = cfg["id"]
    endpoint = f"{DEFAULT_OLLAMA_API}/api/generate"

    # If Bonsai direct backend is active and requested without router
    if cfg["backend"] == "bonsai_or_router":
        # Check if router is available on 11434, else fallback to 8081
        try:
            r = requests.get(f"{DEFAULT_OLLAMA_API}/api/tags", timeout=1.0)
            has_bonsai = any("bonsai" in m.get("name", "") for m in r.json().get("models", []))
            if not has_bonsai:
                # Use direct backend on 8081
                endpoint = f"{DEFAULT_BONSAI_API}/completion"
        except Exception:
            endpoint = f"{DEFAULT_BONSAI_API}/completion"

    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": cfg["temperature"],
            "num_predict": cfg["max_tokens"]
        }
    }
    if system:
        payload["system"] = system

    start_t = time.time()
    collected = []
    use_rich = RICH_AVAILABLE and ENABLE_RICH_UI

    def _emit_token(tok: str, live_obj=None):
        collected.append(tok)
        if use_rich and live_obj is not None:
            txt = "".join(collected)
            live_obj.update(Panel(Markdown(txt), title=f"[bold magenta]{cfg['name']}[/bold magenta]", border_style="magenta"))
        else:
            sys.stdout.write(tok)
            sys.stdout.flush()

    try:
        live_ctx = Live(Panel("Thinking...", title=f"[bold magenta]{cfg['name']}[/bold magenta]", border_style="magenta"), console=console, refresh_per_second=12) if (use_rich and stream) else None
        if live_ctx:
            live_ctx.__enter__()
        try:
            if endpoint.endswith("/completion"):
                prompt_formatted = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" if system else f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                llama_payload = {
                    "prompt": prompt_formatted,
                    "stream": stream,
                    "n_predict": cfg["max_tokens"],
                    "temperature": cfg["temperature"],
                    "stop": ["<|im_end|>", "<|im_start|>"]
                }
                resp = requests.post(endpoint, json=llama_payload, stream=stream, timeout=120)
                if stream:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_s = line_str[6:].strip()
                            if data_s == "[DONE]":
                                break
                            try:
                                token = json.loads(data_s).get("content", "")
                                if token:
                                    _emit_token(token, live_ctx)
                            except Exception:
                                pass
                else:
                    out = resp.json().get("content", "")
                    collected.append(out)
                    if use_rich:
                        console.print(Panel(Markdown(out), title=f"[bold magenta]{cfg['name']}[/bold magenta]", border_style="magenta"))
                    else:
                        sys.stdout.write(out)

            else:
                resp = requests.post(endpoint, json=payload, stream=stream, timeout=120)
                if stream:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            token = chunk.get("response", "")
                            if token:
                                _emit_token(token, live_ctx)
                            if chunk.get("done", False):
                                break
                        except Exception:
                            pass
                else:
                    out = resp.json().get("response", "")
                    collected.append(out)
                    if use_rich:
                        console.print(Panel(Markdown(out), title=f"[bold magenta]{cfg['name']}[/bold magenta]", border_style="magenta"))
                    else:
                        sys.stdout.write(out)
        finally:
            if live_ctx:
                live_ctx.__exit__(None, None, None)

    except requests.exceptions.ConnectionError:
        print(f"\n[!] Connection failed to {endpoint}. Ensure Ollama or the Bonsai router is running.")
        return ""
    except Exception as e:
        print(f"\n[!] Error during inference: {e}")
        return ""

    elapsed = time.time() - start_t
    full_text = "".join(collected)
    tokens_est = len(full_text) // 4
    tps = tokens_est / elapsed if elapsed > 0 else 0
    print(f"\n[Telemetry: ~{tokens_est} tokens in {elapsed:.2f}s | ~{tps:.1f} TPS | Model: {cfg['name']}]")
    return full_text

# Autonomous ReAct Agent Loop
def run_autonomous_agent(task: str, model_key: str = "auto", max_steps: int = 8):
    if model_key == "auto":
        actual_model, route_reason = micro_model_route(task)
    else:
        actual_model, route_reason = model_key, "Manual user selection"
    cfg = MODEL_ROSTER.get(actual_model, MODEL_ROSTER["gemma"])

    print("=" * 70)
    print("  MAPLE Autonomous Agent Harness")
    print(f"  Task: {task}")
    print(f"  Assigned Model: {cfg['name']} ({cfg['role']})")
    print(f"  Router Reason:  {route_reason}")
    print("=" * 70)

    rules_text = load_agent_rules()
    system_prompt = (
        f"You are an autonomous engineering agent operating with local & web tools.\n"
        f"Role: {cfg['role']}\n"
        f"{rules_text}\n"
        f"{TOOL_SCHEMAS}\n"
        f"Execution Protocol:\n"
        f"1. To execute work, output TOOL: <name> followed by ARGS: <json>.\n"
        f"2. After tools execute, inspect their outputs and decide the next step.\n"
        f"3. Only when all tool results have been observed and the task is accomplished, output 'FINAL ANSWER:' followed by your concise report.\n"
        f"4. Ban placeholders. Write complete, working solutions."
    )

    files_touched = []
    recent_steps = []

    for step in range(1, max_steps + 1):
        print(f"\n>>> [Step {step}/{max_steps}] Thinking with {actual_model}...")
        mem_block = [f"GOAL: {task}"]
        if files_touched:
            mem_block.append("FILES_TOUCHED: " + ", ".join(files_touched[-10:]))
        if recent_steps:
            mem_block.append("RECENT STEPS (compact):")
            mem_block.extend(recent_steps[-2:])
        prompt = "\n".join(mem_block) + f"\n\n[Step {step}/{max_steps}] What is your next action?"
        response = query_model(actual_model, prompt=prompt, system=system_prompt, stream=True)

        # Check for tool invocations (support one or multiple tool calls per turn)
        tool_matches = list(re.finditer(r"TOOL:\s*(\w+)\s*\nARGS:\s*({.*?})", response, re.DOTALL))
        executed_any_tool = False
        step_observations = []
        for tool_match in tool_matches:
            tool_name = tool_match.group(1).strip()
            args_raw = tool_match.group(2).strip()
            try:
                args = json.loads(args_raw)
            except Exception as e:
                args = {}
                print(f"[!] Failed to parse tool args: {e}")

            if tool_name in TOOL_REGISTRY:
                executed_any_tool = True
                print(f"\n[*] Executing tool: {tool_name}({args})")
                tool_func = TOOL_REGISTRY[tool_name]
                try:
                    result = tool_func(**args)
                except Exception as e:
                    result = {"error": str(e)}
                res_json = json.dumps(result)
                print(f"[< Result]: {res_json[:200]}...")
                if tool_name in ("write_file", "edit_file", "read_file") and isinstance(args.get("path"), str):
                    p = args["path"]
                    if p not in files_touched:
                        files_touched.append(p)
                step_observations.append(f"- {tool_name}({json.dumps(args)[:180]}) -> {res_json[:1200]}")
            else:
                print(f"[!] Unknown tool: {tool_name}")
                step_observations.append(f"- ERROR: Unknown tool '{tool_name}'")

        if step_observations:
            recent_steps.append(f"Step {step}:\n" + "\n".join(step_observations))

        if not executed_any_tool and ("FINAL ANSWER:" in response or "DONE" in response):
            print("\n[+] Task goal completed!")
            break

# Interactive Terminal REPL
def run_interactive_repl(initial_model: str = "auto"):
    current_model = initial_model
    print("=" * 70)
    print("  MAPLE Local Agent Harness - Interactive Session")
    print("  Commands:")
    print("    /model <deepseek|gemma|bonsai|fast|auto>  : Switch active model")
    print("    /roster                                  : Display all models & roles")
    print("    /vram                                    : Display live GPU VRAM usage")
    print("    /tools                                   : List available system tools")
    print("    /task <goal>                             : Launch autonomous tool agent")
    print("    /clear                                   : Clear conversation screen")
    print("    /exit                                    : Quit")
    print("=" * 70)

    while True:
        try:
            display_model = current_model if current_model != "auto" else "auto-routing"
            user_input = input(f"\n[{display_model}] >> ").strip()
            if not user_input:
                continue

            if user_input.startswith("/exit") or user_input.startswith("/quit"):
                print("[*] Exiting harness. Goodbye!")
                break

            if user_input.startswith("/model "):
                choice = user_input.split()[1].strip().lower()
                if choice in MODEL_ROSTER or choice == "auto":
                    current_model = choice
                    print(f"[*] Active model switched to: {current_model}")
                else:
                    print(f"[!] Unknown model '{choice}'. Valid: {list(MODEL_ROSTER.keys())} or 'auto'")
                continue

            if user_input == "/roster":
                print("\nModel Roster:")
                for k, v in MODEL_ROSTER.items():
                    print(f"  - {k:10s} : {v['name']} -> {v['role']}")
                continue

            if user_input == "/vram":
                v_info = Tools.vram_status()
                print(f"\nGPU VRAM: Total: {v_info.get('total_mib')} MiB | Used: {v_info.get('used_mib')} MiB | Free: {v_info.get('free_mib')} MiB | GPU Util: {v_info.get('gpu_util_pct')}%")
                continue

            if user_input == "/tools":
                print(TOOL_SCHEMAS)
                continue

            if user_input.startswith("/task "):
                task_goal = user_input[6:].strip()
                run_autonomous_agent(task_goal, model_key=current_model)
                continue

            if user_input == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue

            # Standard prompt query
            chosen = auto_choose_model(user_input) if current_model == "auto" else current_model
            print(f"\n[Routing to: {chosen.upper()} ({MODEL_ROSTER[chosen]['role']})]\n")
            query_model(chosen, prompt=user_input, stream=True)

        except KeyboardInterrupt:
            print("\n[*] Interrupted.")
        except EOFError:
            break

# CLI Argument Parser
def main():
    parser = argparse.ArgumentParser(description="MAPLE Local Agent Tool Harness & Model Chooser")
    parser.add_argument("prompt", nargs="?", help="Direct prompt to execute")
    parser.add_argument("--model", "-m", choices=["deepseek", "gemma", "bonsai", "fast", "auto"], default="auto", help="Select model (default: auto)")
    parser.add_argument("--task", "-t", help="Run multi-step autonomous agent with tools")
    parser.add_argument("--chat", "-c", action="store_true", help="Launch interactive REPL session")
    parser.add_argument("--vram", action="store_true", help="Print GPU VRAM status and exit")
    parser.add_argument("--roster", action="store_true", help="Print available models and exit")
    parser.add_argument("--no-ui", action="store_true", help="Disable Rich live panels and print plain text")

    args = parser.parse_args()
    if args.no_ui:
        global ENABLE_RICH_UI
        ENABLE_RICH_UI = False

    if args.vram:
        print(Tools.vram_status())
        return

    if args.roster:
        for k, v in MODEL_ROSTER.items():
            print(f"[{k}] {v['name']}\n  Role: {v['role']}\n  Specialties: {', '.join(v['specialties'])}\n")
        return

    if args.task:
        run_autonomous_agent(args.task, model_key=args.model)
        return

    if args.chat or not args.prompt:
        run_interactive_repl(initial_model=args.model)
        return

    # One-shot prompt
    chosen = auto_choose_model(args.prompt) if args.model == "auto" else args.model
    print(f"\n[Model Chooser]: Selected '{chosen.upper()}' ({MODEL_ROSTER[chosen]['name']})")
    query_model(chosen, prompt=args.prompt, stream=True)

if __name__ == "__main__":
    main()
