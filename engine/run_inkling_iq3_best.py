"""
Automated Best-Quality Sub-5GB MAPLE-Inkling-2-Mini (UD-IQ3_XXS) Pipeline.

Waits cleanly for Inkling-2-Mini-UD-IQ3_XXS.gguf (11.66 GB, KLD=0.0269) to finish
downloading, acquires scratch/gpu.lock once any active GPU job yields, then:
  1. Collects native UD-IQ3_XXS 3-pass imatrix on calibration_corpus_inkling_work.txt.
  2. Benchmarks unpruned UD-IQ3_XXS (-ncmoe 18) llama-bench r=5 speed baseline.
  3. Builds k'=44 (4.89 GB) candidates across {unsloth, blend25, blend40}, verifies
     100% GPU residency at -c 8192 q8_0 KV cache (falling back to k'=42 if WDDM spills).
  4. Runs WikiText-2 PPL (20ch) screening + Top-2 Bakeoff (HumanEval@25, GSM8K@25, ARC@15).
  5. Slices winning 16k-ctx companion k'=40 (4.56 GB) and deletes the 11.66 GB source GGUF
     plus all losing candidate GGUFs immediately to keep disk clean.
  6. Executes full flagship suite (llama-bench r=10, WikiText-2 40ch, HumanEval@50,
     GSM8K@50, ARC-v1@30, ARC-v2@20, GraphWalks@25, StrongREJECT@60, FORTRESS@40x2,
     DeepSearchQA@60) + LongBench-v2<=64k@25 on k'=40 at -c 16384 q8_0.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import psutil

BIN = r"D:\tools\llama-cpp-mainline"
LOCK_PATH = r"C:\Users\amans\.gemini\antigravity\brain\6a0067ca-8615-42cc-948d-72d2d8f5f3dc\scratch\gpu.lock"
SRC_IQ3 = r"D:\models\inkling2_source\Inkling-2-Mini-UD-IQ3_XXS.gguf"
EXPECTED_IQ3_BYTES = 11663310784
IM_UNSLOTH = r"D:\models\inkling2_source\imatrix_unsloth.gguf_file"
IM_IQ3_WORK = r"D:\models\inkling2_source\imatrix_iq3_work.gguf"
WORK_CORPUS = r"E:\maple\engine\calibration_corpus_inkling_work.txt"
WT_TXT = r"C:\Users\amans\.gemini\antigravity\brain\6a0067ca-8615-42cc-948d-72d2d8f5f3dc\scratch\eval_data\wikitext2_raw_test.txt"
ENG_DIR = r"E:\maple\engine"
MOD_DIR = r"E:\maple\models"
LOG_DIR = r"E:\maple\logs"


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


def llama_env():
    env = os.environ.copy()
    env["PATH"] = f"{BIN};" + env.get("PATH", "")
    return env


def collect_iq3_imatrix():
    if os.path.exists(IM_IQ3_WORK) and os.path.getsize(IM_IQ3_WORK) > 40_000_000:
        print(f"[=] Reusing cached {IM_IQ3_WORK}", flush=True)
        return
    cmd = [
        os.path.join(BIN, "llama-imatrix.exe"),
        "-m", SRC_IQ3,
        "-f", WORK_CORPUS,
        "-o", IM_IQ3_WORK,
        "--output-format", "gguf",
        "-ngl", "99",
        "-ncmoe", "18",
        "-fa", "on",
        "--parse-special",
        "-c", "512",
        "-t", "6",
        "--no-ppl",
        "--chunks", "6",
    ]
    print(f"[*] Running native UD-IQ3_XXS 3-pass imatrix (-ncmoe 18)...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=llama_env(), timeout=900)
    open(os.path.join(LOG_DIR, "inkling2_iq3_imatrix.log"), "w", encoding="utf-8").write(res.stdout + "\n" + res.stderr)
    if res.returncode != 0:
        print((res.stdout + "\n" + res.stderr)[-1200:])
        raise RuntimeError(f"llama-imatrix failed rc={res.returncode}")
    print(f"[+] Saved {IM_IQ3_WORK} in {time.time()-t0:.1f}s ({os.path.getsize(IM_IQ3_WORK):,} B)", flush=True)


def build_blended_map(target_k: int, w_work: float, tag: str) -> str:
    import gguf
    rw = gguf.GGUFReader(IM_IQ3_WORK, "r")
    rg = gguf.GGUFReader(IM_UNSLOTH, "r")
    tw = {t.name: t for t in rw.tensors}
    tg = {t.name: t for t in rg.tensors}
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
        sb = w_work * mw + (1.0 - w_work) * mg
        top = sorted([int(x) for x in np.argsort(-sb)[:target_k]])
        sel[str(l)] = top
        cw_l.append(cw[top].sum() / max(cw.sum(), 1.0) * 100.0)
        cg_l.append(cg[top].sum() / max(cg.sum(), 1.0) * 100.0)
    map_path = os.path.join(ENG_DIR, f"inkling2_expert_selection_map_{target_k}exp_iq3_{tag}.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=2)
    print(f"[+] Map {target_k}exp_{tag}: work_cov={np.mean(cw_l):.2f}% | unsloth_cov={np.mean(cg_l):.2f}%", flush=True)
    return map_path


def slice_iq3(map_path: str, out_gguf: str, target_k: int):
    if os.path.exists(out_gguf) and os.path.getsize(out_gguf) > 3_500_000_000:
        print(f"[=] Reusing {out_gguf}", flush=True)
        return
    subprocess.run([
        sys.executable, os.path.join(ENG_DIR, "maple_build_inkling2.py"),
        "--source", SRC_IQ3,
        "--map", map_path,
        "--output", out_gguf,
        "--source-experts", "128",
        "--target-experts", str(target_k),
    ], check=True)


def run_bench(model_path: str, out_json: str, ncmoe: int = 0, reps: int = 10):
    cmd = [os.path.join(BIN, "llama-bench.exe"), "-m", model_path, "-ngl", "99", "-r", str(reps), "-fa", "1", "-o", "json"]
    if ncmoe > 0:
        cmd += ["-ncmoe", str(ncmoe)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=llama_env(), timeout=600)
    open(out_json + ".stderr.log", "w", encoding="utf-8").write(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"llama-bench failed rc={res.returncode}: {res.stderr[-800:]}")
    rows = json.loads(res.stdout)
    open(out_json, "w", encoding="utf-8").write(json.dumps(rows, indent=2))
    return rows


def check_vram_fit(model_path: str, ctx: int = 8192) -> tuple[bool, str]:
    """Verify 100% GPU layer offload at target context using llama-completion --verbose -fit off."""
    log_p = os.path.join(LOG_DIR, f"vram_probe_{os.path.basename(model_path)}_{ctx}.log")
    cmd = [
        os.path.join(BIN, "llama-completion.exe"),
        "-m", model_path, "-c", str(ctx), "-ngl", "99", "-fa", "on",
        "-ctk", "q8_0", "-ctv", "q8_0", "-ub", "256", "-fit", "off",
        "-n", "1", "-p", "Hi", "--no-conversation", "--verbose",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=llama_env(), timeout=180)
    txt = res.stdout + "\n" + res.stderr
    open(log_p, "w", encoding="utf-8").write(txt)
    m_off = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", txt)
    m_cpu = re.search(r"CPU_Mapped\s+model\s+buffer\s+size\s*=\s*([\d.]+)\s*MiB", txt)
    m_cuda = re.search(r"CUDA0\s+model\s+buffer\s+size\s*=\s*([\d.]+)\s*MiB", txt)
    m_kv = re.findall(r"CUDA0\s+KV\s+buffer\s+size\s*=\s*([\d.]+)\s*MiB", txt)
    m_comp = re.search(r"sched_reserve:\s*CUDA0\s+compute\s+buffer\s+size\s*=\s*([\d.]+)\s*MiB", txt)
    full_off = bool(m_off and m_off.group(1) == m_off.group(2))
    cpu_mib = float(m_cpu.group(1)) if m_cpu else 9999.0
    cuda_mib = float(m_cuda.group(1)) if m_cuda else 0.0
    kv_mib = sum(float(x) for x in m_kv)
    comp_mib = float(m_comp.group(1)) if m_comp else 0.0
    # CPU_Mapped should only contain embeddings/output (~235-350 MiB), zero MoE layer offload (<1000 MiB)
    fits = (res.returncode == 0 and full_off and cpu_mib < 1000.0 and cuda_mib > 2500.0)
    buf_str = (
        f"rc={res.returncode} offload={m_off.group(0) if m_off else '?'} "
        f"cuda={cuda_mib:.1f}MiB kv={kv_mib:.1f}MiB comp={comp_mib:.1f}MiB cpu_mapped={cpu_mib:.1f}MiB"
    )
    print(f"  [VRAM CHECK c={ctx}] {os.path.basename(model_path)}: {buf_str}", flush=True)
    return fits, buf_str


def run_ppl(model_path: str, chunks: int = 20) -> tuple[float, float]:
    cmd = [
        os.path.join(BIN, "llama-perplexity.exe"),
        "-m", model_path, "-f", WT_TXT, "-c", "512", "-ngl", "99", "-fa", "on", "--chunks", str(chunks),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=llama_env(), timeout=600)
    out = res.stdout + "\n" + res.stderr
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", out)
    if not m:
        raise RuntimeError(f"PPL failed for {model_path}: {out[-600:]}")
    return float(m.group(1)), float(m.group(2))


def run_eval(model_path: str, out_json: str, ctx: int, max_tokens: int, effort: str, budget: int, suites: str, extra_flags: list[str]) -> dict:
    cmd = [
        sys.executable, os.path.join(ENG_DIR, "benchmark_inkling2.py"),
        "--model", model_path, "--out-json", out_json,
        "--port", "8097", "--ctx", str(ctx), "--ctk", "q8_0", "--ctv", "q8_0",
        "--max-tokens", str(max_tokens), "--reasoning-effort", effort, "--reasoning-budget", str(budget),
        "--suites", suites,
    ] + extra_flags
    subprocess.run(cmd, check=True)
    return json.load(open(out_json, encoding="utf-8"))


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MOD_DIR, exist_ok=True)

    print(f"[*] Waiting for UD-IQ3_XXS download completion ({EXPECTED_IQ3_BYTES:,} B)...", flush=True)
    while not (os.path.exists(SRC_IQ3) and os.path.getsize(SRC_IQ3) == EXPECTED_IQ3_BYTES):
        time.sleep(15)
    print(f"[+] Verified base file present: {SRC_IQ3} ({os.path.getsize(SRC_IQ3):,} B)", flush=True)

    print("[*] Waiting politely on scratch/gpu.lock...", flush=True)
    acquire_gpu_lock("inkling2_iq3_best_quality")
    try:
        print("[*] Acquired gpu.lock! Running UD-IQ3_XXS Best-Quality pipeline...", flush=True)

        # 1. Native IQ3 3-pass work imatrix + Unpruned IQ3 speed baseline
        collect_iq3_imatrix()
        bunp_path = os.path.join(LOG_DIR, "bench_unpruned_iq3_ncmoe18.json")
        if not os.path.exists(bunp_path):
            rows_unp = run_bench(SRC_IQ3, bunp_path, ncmoe=18, reps=5)
            pp = next(r for r in rows_unp if r["n_prompt"] == 512 and r["n_gen"] == 0)["avg_ts"]
            tg = next(r for r in rows_unp if r["n_prompt"] == 0 and r["n_gen"] == 128)["avg_ts"]
            print(f"[+] Unpruned IQ3 (-ncmoe 18): pp512={pp:.2f} t/s | tg128={tg:.2f} t/s", flush=True)

        # 2. Check max expert count k' in {44, 42} fitting 100% in VRAM at -c 8192 q8_0
        probe_k = 44
        m_probe = os.path.join(MOD_DIR, "_probe_44exp_iq3_b25.gguf")
        map_probe = build_blended_map(probe_k, 0.25, "probe")
        slice_iq3(map_probe, m_probe, probe_k)
        fits_44, vram_info_44 = check_vram_fit(m_probe, ctx=8192)
        os.remove(m_probe)
        if not fits_44:
            print("[!] k'=44 spilled host memory on -c 8192, stepping down to k'=42 for 100% VRAM...", flush=True)
            probe_k = 42
        else:
            print(f"[+] Confirmed k'={probe_k} fits 100% in VRAM @ -c 8192 q8_0!", flush=True)

        # Free superseded intermediate 64exp Blended IQ2 copy (4.85 GB) before IQ3 candidate builds
        old_b64 = os.path.join(MOD_DIR, "Inkling-2-Mini-MAPLE-64exp-Blended-IQ2_XXS.gguf")
        if os.path.exists(old_b64):
            os.remove(old_b64)
            print(f"[del] Removed intermediate {old_b64} (4.85 GB freed for IQ3 builds)", flush=True)

        # 3. Build & screen 3 calibration candidates at k'=probe_k (keep max 2 on disk to conserve space)
        candidates = []
        ppl_results = {}
        for tag, ww in [("uns", 0.0), ("b25", 0.25), ("b40", 0.40)]:
            mp = os.path.join(MOD_DIR, f"Inkling-2-Mini-MAPLE-{probe_k}exp-IQ3_XXS-{tag}.gguf")
            mpath = build_blended_map(probe_k, ww, tag)
            slice_iq3(mpath, mp, probe_k)
            ppl, se = run_ppl(mp, chunks=20)
            ppl_results[tag] = {"model": mp, "map": mpath, "work_weight": ww, "ppl_20ch": ppl, "ppl_se": se}
            print(f"  [PPL-20ch] {probe_k}exp-IQ3-{tag}: {ppl:.4f} +/- {se:.4f}", flush=True)
            candidates.append((tag, mp, ppl))

        # Select Top-2 by PPL (preserving at least one blended work variant for bakeoff comparison)
        candidates.sort(key=lambda x: x[2])
        top2_tags = [c[0] for c in candidates[:2]]
        if "b25" not in top2_tags and "b40" not in top2_tags:
            top2_tags[1] = "b25"
        for tag, mp, _ in candidates:
            if tag not in top2_tags and os.path.exists(mp):
                os.remove(mp)
                print(f"[del] Pruned #3 PPL candidate {os.path.basename(mp)}", flush=True)

        # 4. Head-to-head Bakeoff on top-2 IQ3 builds: HumanEval@25 + GSM8K@25 + ARC-v1@15
        bake_summary = {"k_exp": probe_k, "vram_44": vram_info_44, "ppl_20ch": ppl_results, "bakeoff": {}}
        best_tag, best_score = None, -1e9
        for tag in top2_tags:
            mp = ppl_results[tag]["model"]
            out_b = os.path.join(LOG_DIR, f"bakeoff_iq3_{probe_k}exp_{tag}.json")
            res_b = run_eval(
                mp, out_b, ctx=8192, max_tokens=1024, effort="low", budget=512,
                suites="humaneval,gsm8k,arc_v1",
                extra_flags=["--limit-he", "25", "--limit-gsm", "25", "--limit-arc1", "15"],
            )
            he = res_b["metrics"]["humaneval"]["pass_at_1_pct"]
            gsm = res_b["metrics"]["gsm8k"]["acc_pct"]
            arc = res_b["metrics"]["arc_v1"]["acc_pct"]
            ppl = ppl_results[tag]["ppl_20ch"]
            score = he + gsm + 0.5 * arc - 0.25 * max(ppl - 15.0, 0.0)
            bake_summary["bakeoff"][tag] = {"humaneval_25": he, "gsm8k_25": gsm, "arc_v1_15": arc, "ppl_20ch": ppl, "score": round(score, 3)}
            print(f"  [Bakeoff] {probe_k}exp-IQ3-{tag}: HE={he}%, GSM={gsm}%, ARC1={arc}%, PPL={ppl:.2f} -> score={score:.2f}", flush=True)
            if score > best_score:
                best_score = score
                best_tag = tag

        bake_summary["winner_tag"] = best_tag
        bake_summary["winner_work_weight"] = ppl_results[best_tag]["work_weight"]
        open(os.path.join(LOG_DIR, "iq3_bakeoff_summary.json"), "w", encoding="utf-8").write(json.dumps(bake_summary, indent=2))

        # 5. Finalize Flagship + 16k Companion filenames, then DELETE 11.66 GB source & losers!
        win_src = ppl_results[best_tag]["model"]
        win_map = ppl_results[best_tag]["map"]
        flagship_gguf = os.path.join(MOD_DIR, f"Inkling-2-Mini-MAPLE-{probe_k}exp-IQ3_XXS.gguf")
        if os.path.exists(flagship_gguf):
            os.remove(flagship_gguf)
        os.rename(win_src, flagship_gguf)

        win_map_final = os.path.join(ENG_DIR, f"inkling2_expert_selection_map_{probe_k}exp_iq3_winner.json")
        open(win_map_final, "w", encoding="utf-8").write(open(win_map, encoding="utf-8").read())

        for tag, _, _ in candidates:
            mp = ppl_results[tag]["model"]
            if tag != best_tag and os.path.exists(mp):
                os.remove(mp)
                print(f"[del] Removed losing IQ3 candidate {mp}", flush=True)

        # Slice 16k-ctx companion (k'=40, 4.56 GB) using the winning calibration weight BEFORE deleting source
        map_40 = build_blended_map(40, ppl_results[best_tag]["work_weight"], "winner")
        comp_40_gguf = os.path.join(MOD_DIR, "Inkling-2-Mini-MAPLE-40exp-IQ3_XXS.gguf")
        slice_iq3(map_40, comp_40_gguf, 40)

        # Delete 11.66 GB source GGUF immediately
        if os.path.exists(SRC_IQ3):
            os.remove(SRC_IQ3)
            print(f"[del] Deleted {SRC_IQ3} (11.66 GB freed)", flush=True)

        # 6. Full Benchmark Suite on Champion IQ3 Flagship + k'=40 16k Companion
        print(f"\n[*] Running FULL Flagship Benchmarks on {os.path.basename(flagship_gguf)} ...", flush=True)
        run_bench(flagship_gguf, os.path.join(LOG_DIR, f"bench_maple_{probe_k}exp_iq3.json"), ncmoe=0, reps=10)
        run_eval(
            flagship_gguf, os.path.join(LOG_DIR, "inkling2_maple_iq3_flagship.json"),
            ctx=8192, max_tokens=1024, effort="low", budget=512,
            suites="wikitext2_ppl,humaneval,gsm8k,arc_v1,arc_v2,graphwalks",
            extra_flags=["--ppl-chunks", "40", "--limit-he", "50", "--limit-gsm", "50", "--limit-arc1", "30", "--limit-arc2", "20", "--limit-gw", "25", "--gw-max-chars", "20000"],
        )
        run_eval(
            flagship_gguf, os.path.join(LOG_DIR, "inkling2_maple_iq3_safety_qa.json"),
            ctx=4096, max_tokens=448, effort="minimal", budget=256,
            suites="strongreject,fortress_adv,fortress_ben,deepsearchqa",
            extra_flags=["--limit-sr", "60", "--limit-fort", "40", "--limit-dsq", "60"],
        )

        # Companion k'=40 at -c 16384 for LongBench-v2<=64k
        fits_40, _ = check_vram_fit(comp_40_gguf, ctx=16384)
        comp_ctx = 16384 if fits_40 else 12288
        run_bench(comp_40_gguf, os.path.join(LOG_DIR, "bench_maple_40exp_iq3.json"), ncmoe=0, reps=10)
        run_eval(
            comp_40_gguf, os.path.join(LOG_DIR, "inkling2_maple_iq3_40exp_longbench.json"),
            ctx=comp_ctx, max_tokens=768, effort="low", budget=384,
            suites="wikitext2_ppl,longbench_v2",
            extra_flags=["--ppl-chunks", "40", "--limit-lb", "25"],
        )

        print(f"[+] COMPLETE! Champion={flagship_gguf} (winner={best_tag}) | 16k Companion={comp_40_gguf}", flush=True)
    finally:
        release_gpu_lock()
        print("[*] Released gpu.lock cleanly!", flush=True)


if __name__ == "__main__":
    main()
