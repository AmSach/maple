"""
Resume MAPLE-Inkling-2-Mini evaluation pipeline cleanly behind scratch/gpu.lock.
Consolidates bakeoff summary, preserves completed HumanEval@50 & WikiText-2 PPL (40ch),
runs remaining reasoning/safety/long-context benchmarks with reasoning_effort='minimal' + rescue,
and writes final consolidated JSON logs for HF/GitHub/Site publishing.
"""
import json
import os
import subprocess
import sys
import time
import psutil

LOCK_PATH = r"C:\Users\amans\.gemini\antigravity\brain\6a0067ca-8615-42cc-948d-72d2d8f5f3dc\scratch\gpu.lock"
LOG_DIR = r"E:\maple\logs"
M64_WORK = r"E:\maple\models\Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf"
M56_BLEND = r"E:\maple\models\Inkling-2-Mini-MAPLE-56exp-Blended-IQ2_XXS.gguf"


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


def build_bakeoff_summary():
    bakeoff = {}
    for lbl in ("64exp_blended", "64exp_unsloth", "64exp_work"):
        p = os.path.join(LOG_DIR, f"bakeoff_{lbl}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            he = d["metrics"]["humaneval"]["pass_at_1_pct"]
            a1 = d["metrics"]["arc_v1"]["acc_pct"]
            bakeoff[lbl] = {"humaneval_20": he, "arc_v1_10": a1, "combined": he + a1}
    ppl_summary = {
        "64exp_blended": {"ppl": 23.7128, "se": 0.93904},
        "56exp_blended": {"ppl": 30.8071, "se": 1.24634},
        "64exp_unsloth": {"ppl": 13.0612, "se": 0.49184},
        "56exp_unsloth": {"ppl": 15.2140, "se": 0.57916},
        "64exp_work": {"ppl": 37.3857, "se": 1.54492},
        "56exp_work": {"ppl": 49.7553, "se": 2.11129},
    }
    open(os.path.join(LOG_DIR, "bakeoff_summary.json"), "w").write(json.dumps({"ppl_20ch": ppl_summary, "bakeoff": bakeoff}, indent=2))


def main():
    build_bakeoff_summary()
    print("[*] Waiting for scratch/gpu.lock...", flush=True)
    acquire_gpu_lock("inkling2_remaining_evals")
    try:
        print("[*] Acquired gpu.lock! Running remaining MAPLE-Inkling-2-Mini suites...", flush=True)
        rem_json = os.path.join(LOG_DIR, "inkling2_maple_64exp_remaining.json")
        subprocess.run([
            sys.executable, r"E:\maple\engine\benchmark_inkling2.py",
            "--model", M64_WORK, "--out-json", rem_json,
            "--port", "8097", "--ctx", "8192", "--ctk", "q8_0", "--ctv", "q8_0",
            "--max-tokens", "1280", "--reasoning-effort", "minimal", "--reasoning-budget", "-1",
            "--suites", "gsm8k,arc_v1,arc_v2,graphwalks",
            "--limit-gsm", "50", "--limit-arc1", "40", "--limit-arc2", "40",
            "--limit-gw", "40", "--gw-max-chars", "20000",
        ], check=True)

        # Merge remaining metrics into inkling2_maple_64exp_full_reasoning.json cleanly
        full_p = os.path.join(LOG_DIR, "inkling2_maple_64exp_full_reasoning.json")
        full_d = json.load(open(full_p))
        rem_d = json.load(open(rem_json))
        full_d["metrics"].update(rem_d["metrics"])
        open(full_p, "w", encoding="utf-8").write(json.dumps(full_d, indent=2))
        print("[+] Merged remaining reasoning suites into inkling2_maple_64exp_full_reasoning.json", flush=True)

        # Safety + Closed-Book DeepSearchQA on 64exp-Work
        subprocess.run([
            sys.executable, r"E:\maple\engine\benchmark_inkling2.py",
            "--model", M64_WORK, "--out-json", os.path.join(LOG_DIR, "inkling2_maple_64exp_safety_qa.json"),
            "--port", "8097", "--ctx", "4096", "--ctk", "q8_0", "--ctv", "q8_0",
            "--max-tokens", "448", "--reasoning-effort", "minimal", "--reasoning-budget", "-1",
            "--suites", "strongreject,fortress_adv,fortress_ben,deepsearchqa",
            "--limit-sr", "80", "--limit-fort", "50", "--limit-dsq", "80",
        ], check=True)

        # LongBench-v2 <=64k + WikiText-2 (40ch) on 56exp-Blended (4.39 GB at -c 16384)
        subprocess.run([
            sys.executable, r"E:\maple\engine\benchmark_inkling2.py",
            "--model", M56_BLEND, "--out-json", os.path.join(LOG_DIR, "inkling2_maple_56exp_longbench.json"),
            "--port", "8097", "--ctx", "16384", "--ctk", "q8_0", "--ctv", "q8_0",
            "--max-tokens", "768", "--reasoning-effort", "minimal", "--reasoning-budget", "-1",
            "--suites", "wikitext2_ppl,longbench_v2",
            "--ppl-chunks", "40", "--limit-lb", "25",
        ], check=True)
        print("[+] ALL Inkling-2-Mini MAPLE benchmarks COMPLETE!", flush=True)
    finally:
        release_gpu_lock()
        print("[*] Released gpu.lock cleanly!", flush=True)


if __name__ == "__main__":
    main()
