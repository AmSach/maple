"""
Queued GPU Runner for MAPLE-Inkling-2-Mini Calibration Bakeoff + Full Benchmark Suite.
Acquires scratch/gpu.lock politely once any active subagent GPU pass finishes.
"""
import json
import os
import re
import subprocess
import sys
import time
import gguf
import numpy as np
import psutil

LOCK_PATH = r"C:\Users\amans\.gemini\antigravity\brain\6a0067ca-8615-42cc-948d-72d2d8f5f3dc\scratch\gpu.lock"
BIN = r"D:\tools\llama-cpp-mainline"
LOG_DIR = r"E:\maple\logs"
SRC = r"D:\models\inkling2_source\Inkling-2-Mini-UD-IQ2_XXS.gguf"
WT = r"C:\Users\amans\.gemini\antigravity\brain\6a0067ca-8615-42cc-948d-72d2d8f5f3dc\scratch\eval_data\wikitext2_raw_test.txt"


def acquire_gpu_lock(owner: str):
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "owner": owner, "ts": time.time()}).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                info = json.loads(open(LOCK_PATH).read())
                pid = info.get("pid")
                if pid and not psutil.pid_exists(pid):
                    os.remove(LOCK_PATH)
                    continue
            except Exception:
                pass
            time.sleep(10)


def release_gpu_lock():
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)


def build_blended_56():
    r_w = gguf.GGUFReader(r"D:\models\inkling2_source\imatrix_work.gguf", "r")
    r_g = gguf.GGUFReader(r"D:\models\inkling2_source\imatrix_unsloth.gguf_file", "r")
    tw = {t.name: t for t in r_w.tensors}
    tg = {t.name: t for t in r_g.tensors}
    sel = {}
    cw_l, cg_l = [], []
    for l in range(24):
        cw = np.array(tw[f"blk.{l}.ffn_down_exps.weight.counts"].data, dtype=np.float64).reshape(128)
        sw = np.array(tw[f"blk.{l}.ffn_down_exps.weight.in_sum2"].data, dtype=np.float64).reshape(128, -1).mean(axis=1)
        cg = np.array(tg[f"blk.{l}.ffn_down_exps.weight.counts"].data, dtype=np.float64).reshape(128)
        sg = np.array(tg[f"blk.{l}.ffn_down_exps.weight.in_sum2"].data, dtype=np.float64).reshape(128, -1).mean(axis=1)
        mw = cw * np.sqrt(sw + 1e-8)
        mg = cg * np.sqrt(sg + 1e-8)
        mw /= max(mw.sum(), 1e-12)
        mg /= max(mg.sum(), 1e-12)
        sb = 0.60 * mw + 0.40 * mg
        top = sorted([int(x) for x in np.argsort(-sb)[:56]])
        sel[str(l)] = top
        cw_l.append(cw[top].sum() / max(cw.sum(), 1) * 100)
        cg_l.append(cg[top].sum() / max(cg.sum(), 1) * 100)
    map_p = r"E:\maple\engine\inkling2_expert_selection_map_56exp_blended.json"
    open(map_p, "w").write(json.dumps(sel, indent=2))
    print(f"[+] Blended 56exp map: Work cov={np.mean(cw_l):.2f}% | Unsloth general cov={np.mean(cg_l):.2f}%", flush=True)
    out_gguf = r"E:\maple\models\Inkling-2-Mini-MAPLE-56exp-Blended-IQ2_XXS.gguf"
    if not os.path.exists(out_gguf):
        subprocess.run([sys.executable, r"E:\maple\engine\maple_build_inkling2.py", "--source", SRC, "--map", map_p, "--output", out_gguf, "--target-experts", "56"], check=True)
    return out_gguf


def run_ppl(mp: str, chunks: int = 20):
    env = os.environ.copy()
    env["PATH"] = f"{BIN};" + env.get("PATH", "")
    res = subprocess.run([os.path.join(BIN, "llama-perplexity.exe"), "-m", mp, "-f", WT, "-ngl", "99", "-fa", "on", "--chunks", str(chunks)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=400)
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", res.stdout + "\n" + res.stderr)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    m56_blend = build_blended_56()
    m64_unsloth = r"D:\models\inkling2_maple\Inkling-2-Mini-MAPLE-64exp-IQ2_XXS.gguf"
    m64_work = r"E:\maple\models\Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf"
    m64_blend = r"E:\maple\models\Inkling-2-Mini-MAPLE-64exp-Blended-IQ2_XXS.gguf"

    print("[*] Waiting for scratch/gpu.lock...", flush=True)
    acquire_gpu_lock("inkling2_full_pipeline")
    try:
        print("[*] Acquired gpu.lock! Starting 3-way 64exp bakeoff...", flush=True)
        ppl_summary = {
            "64exp_blended": {"ppl": 23.7128, "se": 0.93904},
            "56exp_blended": {"ppl": 30.8071, "se": 1.24634},
            "64exp_unsloth": {"ppl": 13.0612, "se": 0.49184},
            "56exp_unsloth": {"ppl": 15.2140, "se": 0.57916},
            "64exp_work": {"ppl": 37.3857, "se": 1.54492},
            "56exp_work": {"ppl": 49.7553, "se": 2.11129},
        }

        # Quick 3-way bakeoff (20 HumanEval + 10 ARC-AGI v1) across 64exp variants
        bakeoff = {}
        for lbl, mp in [("64exp_blended", m64_blend), ("64exp_unsloth", m64_unsloth), ("64exp_work", m64_work)]:
            out_j = os.path.join(LOG_DIR, f"bakeoff_{lbl}.json")
            subprocess.run([
                sys.executable, r"E:\maple\engine\benchmark_inkling2.py",
                "--model", mp, "--out-json", out_j, "--port", "8097",
                "--ctx", "8192", "--ctk", "q8_0", "--ctv", "q8_0",
                "--max-tokens", "1280", "--reasoning-effort", "low", "--reasoning-budget", "-1",
                "--suites", "humaneval,arc_v1", "--limit-he", "20", "--limit-arc1", "10"
            ], check=True)
            d = json.load(open(out_j))
            he = d["metrics"]["humaneval"]["pass_at_1_pct"]
            a1 = d["metrics"]["arc_v1"]["acc_pct"]
            bakeoff[lbl] = {"humaneval_20": he, "arc_v1_10": a1, "combined": he + a1}
            print(f"  Bakeoff {lbl}: HumanEval@20={he}%, ARC-AGI-v1@10={a1}%", flush=True)

        open(os.path.join(LOG_DIR, "bakeoff_summary.json"), "w").write(json.dumps({"ppl_20ch": ppl_summary, "bakeoff": bakeoff}, indent=2))

        # Select winning 64-expert model (prefer blended on tie or high score)
        winner_lbl = max(bakeoff, key=lambda k: (bakeoff[k]["combined"], 1 if "blended" in k else 0))
        winner_mp = {"64exp_blended": m64_blend, "64exp_unsloth": m64_unsloth, "64exp_work": m64_work}[winner_lbl]
        print(f"[+] Selected Flagship 64exp MAPLE Inkling-2-Mini: {winner_lbl} ({winner_mp})", flush=True)

        # Full Publication Benchmark Suite on Winner 64exp (reasoning/code + safety/QA)
        subprocess.run([
            sys.executable, r"E:\maple\engine\benchmark_inkling2.py",
            "--model", winner_mp, "--out-json", os.path.join(LOG_DIR, "inkling2_maple_64exp_full_reasoning.json"),
            "--port", "8097", "--ctx", "8192", "--ctk", "q8_0", "--ctv", "q8_0",
            "--max-tokens", "1280", "--reasoning-effort", "low", "--reasoning-budget", "-1",
            "--suites", "wikitext2_ppl,humaneval,gsm8k,arc_v1,arc_v2,graphwalks",
            "--ppl-chunks", "40", "--limit-arc1", "50", "--limit-arc2", "50",
            "--limit-gw", "40", "--gw-max-chars", "20000", "--limit-he", "50", "--limit-gsm", "50"
        ], check=True)

        subprocess.run([
            sys.executable, r"E:\maple\engine\benchmark_inkling2.py",
            "--model", winner_mp, "--out-json", os.path.join(LOG_DIR, "inkling2_maple_64exp_safety_qa.json"),
            "--port", "8097", "--ctx", "4096", "--ctk", "q8_0", "--ctv", "q8_0",
            "--max-tokens", "448", "--reasoning-effort", "minimal", "--reasoning-budget", "-1",
            "--suites", "strongreject,fortress_adv,fortress_ben,deepsearchqa",
            "--limit-sr", "80", "--limit-fort", "50", "--limit-dsq", "80"
        ], check=True)

        # Long-Context LongBench-v2 on 56exp-Blended (4.39 GB fits 16k ctx 100% In-VRAM!)
        subprocess.run([
            sys.executable, r"E:\maple\engine\benchmark_inkling2.py",
            "--model", m56_blend, "--out-json", os.path.join(LOG_DIR, "inkling2_maple_56exp_longbench.json"),
            "--port", "8097", "--ctx", "16384", "--ctk", "q8_0", "--ctv", "q8_0",
            "--max-tokens", "768", "--reasoning-effort", "minimal", "--reasoning-budget", "-1",
            "--suites", "wikitext2_ppl,longbench_v2",
            "--ppl-chunks", "40", "--limit-lb", "25"
        ], check=True)
    finally:
        release_gpu_lock()
        print("[*] Released gpu.lock cleanly!", flush=True)


if __name__ == "__main__":
    main()
