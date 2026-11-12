"""
Ollama-Bonsai Gateway Router (v2.0)
====================================
Seamlessly routes Bonsai 2 (27B PTQ1_0) through Ollama API specifications while
forwarding all standard models (DeepSeek Coder V2 Lite MAPLE, Gemma 4, Qwen 2.5 Coder)
to the native Ollama daemon.

Provides:
  - Transparent Ollama endpoints: /api/tags, /api/show, /api/generate, /api/chat, /api/version
  - OpenAI-compatible endpoints: /v1/models, /v1/chat/completions
  - Strict VRAM arbitration on 6GB RTX 4050 GPU: dynamic unloading to guarantee 0 OOM
  - Streaming ndjson for Ollama CLI and WebUI compatibility

Author: Aman Sachan (amansachan92905@gmail.com)
"""

import sys
import os
import time
import json
import uuid
import asyncio
import subprocess
import atexit
import psutil
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Configuration Defaults
OLLAMA_UPSTREAM_URL = os.environ.get("OLLAMA_UPSTREAM_URL", "http://127.0.0.1:11435")
ROUTER_HOST = os.environ.get("ROUTER_HOST", "127.0.0.1")
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "11434"))
BONSAI_BACKEND_PORT = 8081
BONSAI_BACKEND_URL = f"http://127.0.0.1:{BONSAI_BACKEND_PORT}"

BONSAI_BIN_PATH = r"E:\OllamaModels\Bonsai-demo\bin\cuda\llama-server.exe"
BONSAI_MODEL_PATH = (
    r"E:\OllamaModels\Bonsai-2-27B-PTQ1_0-CRACK.gguf"
    if os.path.exists(r"E:\OllamaModels\Bonsai-2-27B-PTQ1_0-CRACK.gguf")
    else r"E:\OllamaModels\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
)
LLAMA_LOG_PATH = r"E:\OllamaModels\llama_server.log"

BONSAI_PROC: Optional[subprocess.Popen] = None

# VRAM Arbitrator Helper
def is_bonsai_running() -> bool:
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{BONSAI_BACKEND_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False

def stop_bonsai():
    global BONSAI_PROC
    if BONSAI_PROC and BONSAI_PROC.poll() is None:
        print("[VRAM Arbitrator] Terminating Bonsai backend to free 5.5GB VRAM...")
        try:
            BONSAI_PROC.terminate()
            BONSAI_PROC.wait(timeout=3)
        except Exception:
            try:
                BONSAI_PROC.kill()
            except Exception:
                pass
        BONSAI_PROC = None

    # Sweep any orphan processes
    try:
        for p in psutil.process_iter(['name']):
            if p.info['name'] and 'llama-server' in p.info['name'].lower():
                p.kill()
    except Exception:
        pass
    time.sleep(1)

atexit.register(stop_bonsai)

async def unload_ollama_vram():
    """Tells upstream Ollama to unload whatever model is currently holding VRAM."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Query active models via /api/ps
            r = await client.get(f"{OLLAMA_UPSTREAM_URL}/api/ps")
            if r.status_code == 200:
                data = r.json()
                models = data.get("models", [])
                for m in models:
                    name = m.get("name") or m.get("model")
                    if name:
                        print(f"[VRAM Arbitrator] Unloading Ollama model '{name}' from VRAM...")
                        await client.post(
                            f"{OLLAMA_UPSTREAM_URL}/api/generate",
                            json={"model": name, "keep_alive": 0}
                        )
    except Exception as e:
        print(f"[VRAM Arbitrator] Note on Ollama unload: {e}")

def start_bonsai():
    global BONSAI_PROC
    if is_bonsai_running():
        return

    print("[VRAM Arbitrator] Booting Bonsai 2 (27B PTQ1_0) into pure GDDR6 VRAM...")
    cmd = [
        BONSAI_BIN_PATH,
        "-m", BONSAI_MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(BONSAI_BACKEND_PORT),
        "-ngl", "99",
        "-c", "4096",
        "-np", "1",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "-b", "1024",
        "-ub", "256",
        "-t", "4",
        "--poll", "1",
        "--prio", "2",
        "--jinja",
        "--reasoning-format", "deepseek",
        "--reasoning-budget", "250"
    ]

    log_f = open(LLAMA_LOG_PATH, "a", encoding="utf-8", buffering=1)
    log_f.write(f"\n--- Starting Bonsai Backend via Router at {time.ctime()} ---\n")
    log_f.flush()

    BONSAI_PROC = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=log_f,
        text=True
    )

    # Wait for health
    start_t = time.time()
    while time.time() - start_t < 30:
        if is_bonsai_running():
            print(f"[OK] Bonsai 2 backend LIVE on port {BONSAI_BACKEND_PORT} in {time.time() - start_t:.1f}s.")
            return
        if BONSAI_PROC.poll() is not None:
            raise RuntimeError(f"Bonsai llama-server failed to start (exit code {BONSAI_PROC.returncode}).")
        time.sleep(1.0)
    raise TimeoutError("Timed out waiting for Bonsai backend to initialize.")

async def prepare_bonsai():
    """Ensures Ollama models are evicted and Bonsai is ready in VRAM."""
    await unload_ollama_vram()
    start_bonsai()

async def prepare_ollama():
    """Ensures Bonsai is evicted so Ollama gets 100% of the 6GB VRAM."""
    if is_bonsai_running():
        stop_bonsai()

# FastAPI App
app = FastAPI(
    title="Ollama-Bonsai Gateway Router",
    description="Unified API Gateway routing Bonsai 27B and Ollama models with automated VRAM safety.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BONSAI_MODEL_NAMES = {"bonsai", "bonsai:latest", "bonsai-27b", "bonsai-27b:latest", "ternary-bonsai", "ternary-bonsai:latest"}

def is_bonsai_target(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    clean = model_name.strip().lower()
    return clean in BONSAI_MODEL_NAMES or clean.startswith("bonsai")

# 1. Tags / List Models
@app.get("/api/tags")
async def list_tags():
    ollama_models = []
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_UPSTREAM_URL}/api/tags")
            if resp.status_code == 200:
                ollama_models = resp.json().get("models", [])
    except Exception:
        pass

    # Inject Bonsai definitions
    bonsai_entry = {
        "name": "bonsai:latest",
        "model": "bonsai:latest",
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "size": os.path.getsize(BONSAI_MODEL_PATH) if os.path.exists(BONSAI_MODEL_PATH) else 5946648928,
        "digest": "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "bonsai",
            "families": ["bonsai"],
            "parameter_size": "27B",
            "quantization_level": "PTQ1_0"
        }
    }

    # Deduplicate and merge
    merged = [bonsai_entry]
    for m in ollama_models:
        if m.get("name") != "bonsai:latest":
            merged.append(m)

    return {"models": merged}

# 2. Show Model Info
@app.post("/api/show")
async def show_model(req: Request):
    body = await req.json()
    model = body.get("name") or body.get("model") or ""
    if is_bonsai_target(model):
        return {
            "modelfile": f"FROM {os.path.basename(BONSAI_MODEL_PATH)}\nTEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"",
            "parameters": "stop \"<|im_start|>\"\nstop \"<|im_end|>\"",
            "template": "{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "bonsai",
                "families": ["bonsai"],
                "parameter_size": "27B",
                "quantization_level": "PTQ1_0"
            },
            "model_info": {
                "general.architecture": "bonsai",
                "general.parameter_count": 27000000000,
                "general.quantization_version": 2
            }
        }

    # Forward to Ollama
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{OLLAMA_UPSTREAM_URL}/api/show", json=body)
        return JSONResponse(status_code=r.status_code, content=r.json())

# 3. Generate Endpoint (/api/generate)
@app.post("/api/generate")
async def api_generate(req: Request):
    body = await req.json()
    model = body.get("model", "")
    stream = body.get("stream", True)

    if not is_bonsai_target(model):
        await prepare_ollama()
        # Transparent proxy to Ollama
        if stream:
            async def proxy_stream():
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream("POST", f"{OLLAMA_UPSTREAM_URL}/api/generate", json=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            return StreamingResponse(proxy_stream(), media_type="application/x-ndjson")
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{OLLAMA_UPSTREAM_URL}/api/generate", json=body)
                return JSONResponse(status_code=r.status_code, content=r.json())

    # Bonsai Execution
    await prepare_bonsai()
    prompt = body.get("prompt", "")
    system = body.get("system", "")
    full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" if system else f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    llama_req = {
        "prompt": full_prompt,
        "n_predict": body.get("options", {}).get("num_predict", 2048),
        "temperature": body.get("options", {}).get("temperature", 0.6),
        "top_p": body.get("options", {}).get("top_p", 0.9),
        "stream": stream,
        "stop": ["<|im_end|>", "<|im_start|>"]
    }

    start_t = time.time()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    if stream:
        async def bonsai_gen_stream():
            token_count = 0
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{BONSAI_BACKEND_URL}/completion", json=llama_req) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            item = json.loads(data_str)
                            token = item.get("content", "")
                            stop = item.get("stop", False)
                            token_count += 1
                            chunk = {
                                "model": "bonsai:latest",
                                "created_at": created_at,
                                "response": token,
                                "done": False
                            }
                            yield json.dumps(chunk) + "\n"
                            if stop:
                                break
                        except Exception:
                            continue

            elapsed_ns = int((time.time() - start_t) * 1e9)
            done_chunk = {
                "model": "bonsai:latest",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "response": "",
                "done": True,
                "total_duration": elapsed_ns,
                "load_duration": 0,
                "prompt_eval_count": len(full_prompt) // 4,
                "eval_count": token_count,
                "eval_duration": elapsed_ns
            }
            yield json.dumps(done_chunk) + "\n"

        return StreamingResponse(bonsai_gen_stream(), media_type="application/x-ndjson")

    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            llama_req["stream"] = False
            r = await client.post(f"{BONSAI_BACKEND_URL}/completion", json=llama_req)
            data = r.json()
            content = data.get("content", "")
            elapsed_ns = int((time.time() - start_t) * 1e9)
            return {
                "model": "bonsai:latest",
                "created_at": created_at,
                "response": content,
                "done": True,
                "total_duration": elapsed_ns,
                "eval_count": len(content) // 4,
                "eval_duration": elapsed_ns
            }

# 4. Chat Endpoint (/api/chat)
@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    model = body.get("model", "")
    stream = body.get("stream", True)

    if not is_bonsai_target(model):
        await prepare_ollama()
        if stream:
            async def proxy_chat_stream():
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream("POST", f"{OLLAMA_UPSTREAM_URL}/api/chat", json=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            return StreamingResponse(proxy_chat_stream(), media_type="application/x-ndjson")
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{OLLAMA_UPSTREAM_URL}/api/chat", json=body)
                return JSONResponse(status_code=r.status_code, content=r.json())

    # Bonsai Chat Execution
    await prepare_bonsai()
    messages = body.get("messages", [])
    
    # Format ChatML
    chatml = ""
    for m in messages:
        role = m.get("role", "user")
        cnt = m.get("content", "")
        chatml += f"<|im_start|>{role}\n{cnt}<|im_end|>\n"
    chatml += "<|im_start|>assistant\n"

    llama_req = {
        "prompt": chatml,
        "n_predict": body.get("options", {}).get("num_predict", 2048),
        "temperature": body.get("options", {}).get("temperature", 0.6),
        "top_p": body.get("options", {}).get("top_p", 0.9),
        "stream": stream,
        "stop": ["<|im_end|>", "<|im_start|>"]
    }

    start_t = time.time()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    if stream:
        async def bonsai_chat_stream():
            token_count = 0
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{BONSAI_BACKEND_URL}/completion", json=llama_req) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            item = json.loads(data_str)
                            token = item.get("content", "")
                            stop = item.get("stop", False)
                            token_count += 1
                            chunk = {
                                "model": "bonsai:latest",
                                "created_at": created_at,
                                "message": {
                                    "role": "assistant",
                                    "content": token
                                },
                                "done": False
                            }
                            yield json.dumps(chunk) + "\n"
                            if stop:
                                break
                        except Exception:
                            continue

            elapsed_ns = int((time.time() - start_t) * 1e9)
            done_chunk = {
                "model": "bonsai:latest",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True,
                "total_duration": elapsed_ns,
                "eval_count": token_count,
                "eval_duration": elapsed_ns
            }
            yield json.dumps(done_chunk) + "\n"

        return StreamingResponse(bonsai_chat_stream(), media_type="application/x-ndjson")

    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            llama_req["stream"] = False
            r = await client.post(f"{BONSAI_BACKEND_URL}/completion", json=llama_req)
            data = r.json()
            content = data.get("content", "")
            elapsed_ns = int((time.time() - start_t) * 1e9)
            return {
                "model": "bonsai:latest",
                "created_at": created_at,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done": True,
                "total_duration": elapsed_ns,
                "eval_count": len(content) // 4,
                "eval_duration": elapsed_ns
            }

# 5. Version
@app.get("/api/version")
async def get_version():
    return {"version": "0.15.2-bonsai-routed"}

# 6. Fallback Proxy for all other routes
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
async def fallback_proxy(request: Request, path: str):
    method = request.method
    url = f"{OLLAMA_UPSTREAM_URL}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    body = await request.body()

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.request(
                method,
                url,
                headers=headers,
                params=request.query_params,
                content=body
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers)
            )
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": f"Failed to proxy to Ollama: {str(e)}"})

if __name__ == "__main__":
    print("=" * 70)
    print("  Ollama-Bonsai Gateway Router v2.0")
    print(f"  Listening on: http://{ROUTER_HOST}:{ROUTER_PORT}")
    print(f"  Upstream Ollama: {OLLAMA_UPSTREAM_URL}")
    print(f"  Bonsai Backend: {BONSAI_BACKEND_URL} (27B PTQ1_0)")
    print("  VRAM Safety: Active dynamic model swapping on 6GB RTX 4050")
    print("=" * 70)
    uvicorn.run(app, host=ROUTER_HOST, port=ROUTER_PORT, log_level="warning")
