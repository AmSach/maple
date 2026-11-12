# MAPLE Local Agent Tool Harness

**Author**: Aman Sachan ([amansachan92905@gmail.com](mailto:amansachan92905@gmail.com))
**Version**: 2.0
**Target Hardware**: NVIDIA RTX 4050 6GB Laptop GPU (Windows 11, CUDA 12.4)

---

## Overview

The MAPLE Local Agent Tool Harness is a unified developer CLI and autonomous agent framework that routes engineering prompts to the right local LLM for the task, manages VRAM arbitration on 6GB consumer GPUs, and exposes a structured tool execution loop for multi-step agentic workflows.

Two components work together:

| File | Role |
|:---|:---|
| `tools/local_agent_harness.py` | CLI, model chooser, autonomous agent loop, tool execution |
| `tools/ollama_bonsai_router.py` | FastAPI gateway unifying Ollama and Bonsai backends under one API surface |

---

## 4-Tier Model Roster

The harness maintains four specialist tiers, each targeted at a distinct engineering domain. All four models fit entirely within 6GB VRAM when arbitrated correctly (they are never loaded simultaneously).

| Tier Key | Model | Size | Backend | Temperature | Max Tokens |
|:---|:---|:---:|:---:|:---:|:---:|
| `deepseek` | DeepSeek-Coder-V2-Lite MAPLE (54 of 64 Experts, Sub-5GB MoE) | 4.50 GB | Ollama | 0.3 | 4096 |
| `gemma` | Gemma 4 26B-A4B MAPLE (In-Situ Sub-5GB) | 4.75 GB | Ollama | 0.5 | 4096 |
| `bonsai` | Ternary Bonsai 2 27B (PTQ1_0, 100% VRAM) | 5.69 GB | llama-server on :8081 | 0.6 | 4096 |
| `fast` | Qwen 2.5 Coder 1.5B (Sub-1GB) | ~0.9 GB | Ollama | 0.2 | 2048 |

### Tier Specializations

**deepseek - Hardcore Systems and Algorithmic Engineering**

Specialties: C++20, Rust, lock-free data structures, `memory_order` semantics, atomic operations, ring buffers, SIMD/AVX, CUDA kernels, hypergraph algorithms, Dijkstra solvers.

Activation keywords: `c++`, `cpp`, `lock-free`, `concurrency`, `mutex`, `spinlock`, `memory_order`, `atomic`, `ring buffer`, `hypergraph`, `dijkstra`, `assembly`, `simd`, `avx`, `cuda`, `kernel`, `zero-copy`, `cache-line`, `false sharing`, `template metaprogramming`, `concepts`.

**gemma - General Software, Frontend, and Code Review**

Specialties: Python scripts, web and API development, HTML/CSS/JavaScript, refactoring, documentation, general code review.

This is the default fallback tier when no other keyword matches.

**bonsai - Chief Architect and Strategic Reasoner**

Specialties: Architectural trade-offs, decision-relevant pragmatism, game theory, strategic moats, system design, comparative evaluation.

Activation keywords: `decision`, `trade-off`, `tradeoff`, `strategy`, `architect`, `architecture`, `game theory`, `evaluate`, `ostrom`, `governance`, `commons`, `pragmatism`, `swot`, `pestle`, `porter`, `pros and cons`, `compare approaches`.

**fast - Ultra-Fast Triage and Terminal Copilot**

Specialties: Git commit messages, regex patterns, PowerShell one-liners, bash one-liners, quick syntax checks, fast file edits. Target latency: under 300ms.

Activation keywords: `git commit`, `commit message`, `regex`, `regular expression`, `powershell one-liner`, `bash one-liner`, `shell command`, `grep`, `sed`, `awk`, `explain this line`, `fix typo`, `quick syntax`.

---

## Automatic Model Routing

When `--model auto` (the default), the harness classifies the prompt using a priority-ordered keyword match:

```
1. Strategic / Architectural keywords  ->  bonsai
2. Hardcore systems / concurrency keywords  ->  deepseek
3. Triage / terminal keywords  ->  fast
4. Everything else  ->  gemma (default fallback)
```

Pass `--model <key>` to bypass the classifier entirely.

---

## CLI Flags

```
python local_agent_harness.py [OPTIONS] [PROMPT]

Positional:
  PROMPT                One-shot prompt text (optional if --task or --chat is set)

Options:
  --model, -m  <tier>   Select model tier: deepseek | gemma | bonsai | fast | auto
                         Default: auto (heuristic intent classifier)

  --task, -t  <goal>    Run the autonomous multi-step agent loop with tool execution.
                         The agent iterates up to 20 steps, calling tools and reasoning
                         until it emits "FINAL ANSWER:" or "DONE".

  --chat, -c            Launch the interactive REPL session.

  --vram                Print live GPU VRAM usage (via nvidia-smi) and exit.

  --roster              Print the full model roster with roles and specialties and exit.
```

### Examples

```powershell
# One-shot prompt, auto-routed
python local_agent_harness.py "Write a C++20 lock-free SPSC ring buffer"

# Force bonsai for strategic analysis
python local_agent_harness.py --model bonsai "Evaluate microservices vs monolith for a 50-engineer team"

# Autonomous agent: multi-step task with tool execution
python local_agent_harness.py --task "Read README.md, find any mentions of 48-experts, update them to 54-experts, and verify the change"

# Interactive REPL starting on deepseek
python local_agent_harness.py --chat --model deepseek

# Check VRAM before loading a model
python local_agent_harness.py --vram

# Print roster
python local_agent_harness.py --roster
```

### Interactive REPL Commands

Inside `--chat` mode:

| Command | Action |
|:---|:---|
| `/model <key>` | Switch active model tier (deepseek, gemma, bonsai, fast, auto) |
| `/roster` | Print all model tiers and roles |
| `/vram` | Print live GPU VRAM (Total / Used / Free / Utilization) |
| `/tools` | List all available tool schemas |
| `/task <goal>` | Launch the autonomous agent loop on a goal |
| `/clear` | Clear the terminal screen |
| `/exit` | Quit the harness |

---

## Built-in Engineering Tool Schemas

The autonomous agent loop (`--task`) exposes these tools to the LLM. The LLM emits a `TOOL:` / `ARGS:` block which the harness parses and executes.

### Tool Call Format

```
TOOL: <tool_name>
ARGS: {"arg1": "value", "arg2": "value"}
```

### Tool Reference

**1. run_command**

Run a shell command and capture stdout, stderr, and exit code.

```json
{
  "tool": "run_command",
  "args": {
    "cmd": "<shell command string>",
    "timeout": 45
  },
  "returns": {
    "exit_code": "<int>",
    "stdout": "<string>",
    "stderr": "<string>"
  }
}
```

**2. read_file**

Read a file with optional line-range filtering.

```json
{
  "tool": "read_file",
  "args": {
    "path": "<absolute or relative path>",
    "start_line": "<int, optional>",
    "end_line": "<int, optional>"
  },
  "returns": {
    "path": "<string>",
    "total_lines": "<int>",
    "content": "<string with line numbers>"
  }
}
```

**3. write_file**

Atomically write or overwrite a file (creates parent directories).

```json
{
  "tool": "write_file",
  "args": {
    "path": "<path>",
    "content": "<full file content>"
  },
  "returns": {
    "success": true,
    "path": "<string>",
    "bytes_written": "<int>"
  }
}
```

**4. edit_file**

Exact search-and-replace on a file. Replaces every occurrence of `target` with `replacement`.

```json
{
  "tool": "edit_file",
  "args": {
    "path": "<path>",
    "target": "<exact string to find>",
    "replacement": "<replacement string>"
  },
  "returns": {
    "success": true,
    "path": "<string>",
    "replacements": "<int>"
  }
}
```

**5. list_dir**

List directory entries with sizes. Set `recursive` to true for a full tree.

```json
{
  "tool": "list_dir",
  "args": {
    "path": "<directory path>",
    "recursive": false
  },
  "returns": {
    "path": "<string>",
    "entries": [{"name": "<string>", "is_dir": true, "size": 0}]
  }
}
```

**6. python_eval**

Execute a Python snippet in an isolated subprocess and capture output.

```json
{
  "tool": "python_eval",
  "args": {
    "code": "<python source string>"
  },
  "returns": {
    "exit_code": "<int>",
    "stdout": "<string>",
    "stderr": "<string>"
  }
}
```

**7. git_status**

Inspect the current git branch and working-tree status (short format).

```json
{
  "tool": "git_status",
  "args": {},
  "returns": {
    "branch": "<string>",
    "status": "<string>"
  }
}
```

**8. vram_status**

Query live GPU VRAM usage via `nvidia-smi`.

```json
{
  "tool": "vram_status",
  "args": {},
  "returns": {
    "total_mib": "<string>",
    "used_mib": "<string>",
    "free_mib": "<string>",
    "gpu_util_pct": "<string>"
  }
}
```

---

## Ollama-Bonsai Gateway Router

**File**: `tools/ollama_bonsai_router.py`
**Launch script**: `E:\OllamaModels\start_ollama_bonsai_router.bat`

The router is a FastAPI service that presents a single Ollama-compatible API surface on `127.0.0.1:11434`. It transparently routes requests to the correct backend depending on whether the requested model is Bonsai or an Ollama model.

### Architecture

```
Client (Ollama CLI, WebUI, harness)
        |
        v
  Router  :11434  (ollama_bonsai_router.py)
        |
  +-----+-----+
  |           |
  v           v
Ollama     llama-server
:11435     :8081
(deepseek,  (bonsai)
 gemma,
 fast)
```

### Exposed Endpoints

| Method | Path | Description |
|:---|:---|:---|
| GET | `/api/tags` | Lists all available models (Ollama models merged with injected Bonsai entry) |
| POST | `/api/show` | Returns model metadata; Bonsai served natively, others proxied to Ollama |
| POST | `/api/generate` | Ollama-format text generation with streaming ndjson |
| POST | `/api/chat` | Ollama-format chat completion with ChatML formatting for Bonsai |
| GET | `/api/version` | Returns router version string |
| GET | `/v1/models` | OpenAI-compatible model listing |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |

### Configuration

Environment variables (all optional, defaults shown):

| Variable | Default | Description |
|:---|:---|:---|
| `OLLAMA_UPSTREAM_URL` | `http://127.0.0.1:11435` | Upstream Ollama daemon URL |
| `ROUTER_HOST` | `127.0.0.1` | Interface the router binds to |
| `ROUTER_PORT` | `11434` | Port the router listens on |

Hardcoded paths (edit source to change):

| Constant | Value | Description |
|:---|:---|:---|
| `BONSAI_BACKEND_PORT` | `8081` | Port for the Bonsai llama-server backend |
| `BONSAI_BIN_PATH` | `E:\OllamaModels\Bonsai-demo\bin\cuda\llama-server.exe` | llama-server binary |
| `BONSAI_MODEL_PATH` | `E:\OllamaModels\Bonsai-2-27B-PTQ1_0-CRACK.gguf` (fallback: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`) | Bonsai GGUF model file |
| `LLAMA_LOG_PATH` | `E:\OllamaModels\llama_server.log` | Bonsai server log |

### Bonsai Model Name Aliases

The router recognizes any of the following strings as a Bonsai request:

```
bonsai
bonsai:latest
bonsai-27b
bonsai-27b:latest
ternary-bonsai
ternary-bonsai:latest
```

Any name starting with `bonsai` is also matched.

### Bonsai Launch Parameters

When Bonsai is first requested, the router starts `llama-server` with these flags:

| Flag | Value | Purpose |
|:---|:---|:---|
| `-ngl` | `99` | Offload all 99 layers to GPU (100% VRAM) |
| `-c` | `4096` | Context window in tokens |
| `-np` | `1` | Single parallel slot |
| `-fa on` | on | Flash Attention enabled |
| `-ctk` | `q4_0` | KV cache key quantization |
| `-ctv` | `q4_0` | KV cache value quantization |
| `-b` | `1024` | Physical batch size |
| `-ub` | `256` | Micro-batch size |
| `-t` | `4` | CPU threads for prompt pre-fill |
| `--poll` | `1` | Polling mode (reduces latency on Windows) |
| `--prio` | `2` | Process priority level |
| `--jinja` | - | Enable Jinja template rendering |
| `--reasoning-format` | `deepseek` | DeepSeek reasoning chain format |
| `--reasoning-budget` | `250` | Maximum reasoning tokens per turn |

---

## VRAM Arbitration

The 6GB RTX 4050 Laptop GPU (usable: approximately 5,350 MB after Windows DWM overhead) cannot hold two large models simultaneously. The harness and router enforce dynamic eviction before each model switch.

### Arbitration Logic

**Switching to Bonsai (bonsai_or_router backend):**

1. Query `GET /api/ps` on the upstream Ollama daemon to find loaded Ollama models.
2. For each loaded model, POST `{"model": "<name>", "keep_alive": 0}` to `/api/generate` to force immediate VRAM eviction.
3. Boot the Bonsai `llama-server` process on port 8081 with all GPU layers enabled.
4. Poll `/health` for up to 30 seconds until Bonsai reports ready.

**Switching to any Ollama model (deepseek, gemma, fast):**

1. Check if `llama-server` is responding on `http://127.0.0.1:8081/health`.
2. If Bonsai is running, terminate the `llama-server` process and sweep any orphan `llama-server.exe` processes via `psutil`.
3. Wait 1 second for VRAM to drain.
4. Ollama then loads the requested model into the freed VRAM.

### VRAM Footprint by Tier

| Tier | Model File | Approx. VRAM |
|:---|:---|:---:|
| deepseek | DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf | ~4,850 MiB |
| gemma | gemma_4_26b_sub5gb_insitu.gguf | ~4,870 MiB |
| bonsai | Bonsai-2-27B-PTQ1_0-CRACK.gguf | ~5,500 MiB |
| fast | qwen2.5-coder:1.5b (quantized) | ~900 MiB |

---

## Setup and Installation

### Prerequisites

- Python 3.10 or later
- Ollama installed and running (native port 11435 when router is active)
- NVIDIA GPU with CUDA 12.4 and 6GB VRAM
- `llama-server.exe` from the Bonsai-demo distribution at `E:\OllamaModels\Bonsai-demo\bin\cuda\`

### Python Dependencies

```
pip install requests fastapi uvicorn httpx psutil
```

### Ollama Models

Pull the Ollama-managed models:

```powershell
# Reconfigure Ollama to run on port 11435 when the router is active
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama pull deepseek-coder-maple:latest
ollama pull gemma4:latest
ollama pull qwen2.5-coder:1.5b
```

The Bonsai GGUF files are managed directly on disk at `E:\OllamaModels\`.

### Starting the Router

```powershell
cd E:\OllamaModels
.\start_ollama_bonsai_router.bat
```

The router starts on `127.0.0.1:11434`. All Ollama-compatible tools and clients can now point to this address and transparently access all four model tiers.

### Running the Harness

```powershell
cd C:\Users\amans\.gemini\antigravity\scratch\repos\maple\tools
python local_agent_harness.py --chat
```

---

## Service Endpoints Summary

| Service | URL | Description |
|:---|:---|:---|
| Router (unified) | `http://127.0.0.1:11434` | Primary API surface for all clients |
| Ollama upstream | `http://127.0.0.1:11435` | Native Ollama daemon (not for direct client use when router is active) |
| Bonsai backend | `http://127.0.0.1:8081` | llama-server for Bonsai (started on demand by router) |

---

## Contact

Aman Sachan - [amansachan92905@gmail.com](mailto:amansachan92905@gmail.com)
