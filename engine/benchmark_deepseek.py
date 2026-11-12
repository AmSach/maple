"""
DeepSeek-Coder-V2-Lite MAPLE Benchmark Suite
Runs 3 complex engineering prompts using llama-completion.exe on CUDA backend.
Records exact Prompt TPS, Generation TPS, duration, and captures full generated code.
"""

import os
import sys
import time
import subprocess
import json

MODEL_PATH = r"D:\models\deepseek_coder_v2_maple\DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf"
COMPLETION_EXE = r"D:\tools\llama-cpp\llama-completion.exe"
OUTPUT_JSON = r"E:\OllamaModels\deepseek_maple_benchmark_results.json"

PROMPTS = [
    {
        "id": "spsc_ring_buffer",
        "title": "Lock-Free SPSC Ring Buffer (C++20)",
        "prompt": "User: Implement an ultra-high-performance Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer in C++20. Requirements: cacheline alignment (alignas(64)) to eliminate false sharing, std::atomic with memory_order_acquire / memory_order_release, power-of-two capacity with bitwise masking, zero dynamic memory allocations on push/pop, and full move semantics.\n\nAssistant:"
    },
    {
        "id": "directed_hypergraph_dp",
        "title": "Directed Hypergraph Shortest Path DP Solver (Python)",
        "prompt": "User: Implement a production-grade Directed Hypergraph Shortest Path solver in Python using Dynamic Programming and Dijkstra-style min-heap traversal. A B-hyperedge connects a set of tail vertices (tail(e)) to a single head vertex (head(e)) with a weight w(e), where the cost of traversing the hyperedge is w(e) + max(cost(v) for v in tail(e)). Include full type hints, docstrings, cycle detection, and an example graph test.\n\nAssistant:"
    },
    {
        "id": "distributed_rate_limiter",
        "title": "Distributed Sliding-Window Token Bucket Rate Limiter (Redis Lua + Python)",
        "prompt": "User: Implement a high-throughput distributed sliding-window token bucket rate limiter in Python backed by an atomic Redis Lua script. The Lua script must atomically calculate fractional token replenishment based on millisecond timestamp deltas, check capacity, consume tokens if available, and return [allowed: 0/1, remaining_tokens, retry_after_ms] without race conditions. Include Redis connection pooling and fallback handling.\n\nAssistant:"
    }
]

def run_single_prompt(p_info):
    print("=" * 80)
    print(f"  RUNNING BENCHMARK: {p_info['title']}")
    print("=" * 80)
    
    cmd = [
        COMPLETION_EXE,
        "-m", MODEL_PATH,
        "-ngl", "99",
        "-c", "4096",
        "-n", "768",
        "--temp", "0.2",
        "--top-p", "0.95",
        "-no-cnv",
        "-e",
        "-p", p_info["prompt"]
    ]
    
    env = os.environ.copy()
    env["PATH"] = r"C:\Users\amans\AppData\Local\Programs\Ollama\lib\ollama;D:\tools\llama-cpp;" + env.get("PATH", "")
    
    t0 = time.time()
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    total_time = time.time() - t0
    output = res.stdout
    
    # Parse generation and prompt eval TPS from output
    # eval time =    3344.32 ms /   255 runs   (   13.11 ms per token,    76.25 tokens per second)
    # prompt eval time =     386.39 ms /    16 tokens (   24.15 ms per token,    41.41 tokens per second)
    prompt_tps = 0.0
    gen_tps = 0.0
    
    for line in output.splitlines():
        if "prompt eval time =" in line and "tokens per second" in line:
            try:
                prompt_tps = float(line.split("tokens per second")[0].split(",")[-1].strip())
            except Exception:
                pass
        if "eval time =" in line and "tokens per second" in line and "prompt eval" not in line:
            try:
                gen_tps = float(line.split("tokens per second")[0].split(",")[-1].strip())
            except Exception:
                pass
                
    # Extract code response
    response_text = output
    if p_info["prompt"] in output:
        response_text = output.split(p_info["prompt"])[-1]
    if "common_perf_print:" in response_text:
        response_text = response_text.split("common_perf_print:")[0]
        
    result = {
        "id": p_info["id"],
        "title": p_info["title"],
        "duration_sec": round(total_time, 2),
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "code_output": response_text.strip()
    }
    print(f"[+] Completed {p_info['title']} in {total_time:.2f}s")
    print(f"    Prompt Speed: {prompt_tps:.2f} TPS | Generation Speed: {gen_tps:.2f} TPS")
    return result

def main():
    all_results = []
    for p in PROMPTS:
        r = run_single_prompt(p)
        all_results.append(r)
        
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] All benchmark results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
