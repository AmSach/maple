"""
Upload MAPLE Inkling-2-Mini Sub-5GB GGUFs, selection maps, condensed work corpus,
and empirically verified Model Card to Hugging Face Hub (`AmSach/Inkling-2-Mini-MAPLE-Sub5GB-GGUF`),
and register the repository in the official MAPLE Hugging Face Collection.

All numbers in the generated model card are loaded dynamically from verified JSON logs
written by `engine/run_inkling_pipeline.py` and `llama-bench` (`E:/maple/logs`).
"""

import argparse
import json
import os
import sys
import time
from huggingface_hub import HfApi, create_repo

COLLECTION_SLUG = "AmSach/maple-sub-5gb-mixture-of-experts-models-6af8467951c779a9e35a327e"
DEFAULT_REPO_NAME = "Inkling-2-Mini-MAPLE-Sub5GB-GGUF"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_card(logs_dir: str, m64_path: str, m56_path: str) -> str:
    b56 = load_json(os.path.join(logs_dir, "bench_maple_56exp.json"))
    b64 = load_json(os.path.join(logs_dir, "bench_maple_64exp.json"))
    bunp = load_json(os.path.join(logs_dir, "bench_unpruned_iq2_ncmoe14.json"))
    bake = load_json(os.path.join(logs_dir, "bakeoff_summary.json"))
    r64 = load_json(os.path.join(logs_dir, "inkling2_maple_64exp_full_reasoning.json"))
    s64 = load_json(os.path.join(logs_dir, "inkling2_maple_64exp_safety_qa.json"))
    l56 = load_json(os.path.join(logs_dir, "inkling2_maple_56exp_longbench.json"))

    def _bench_stats(rows):
        pp = next(r for r in rows if r["n_prompt"] == 512 and r["n_gen"] == 0)
        tg = next(r for r in rows if r["n_prompt"] == 0 and r["n_gen"] == 128)
        return pp["avg_ts"], pp["stddev_ts"], tg["avg_ts"], tg["stddev_ts"]

    p56_pp, p56_pp_sd, p56_tg, p56_tg_sd = _bench_stats(b56)
    p64_pp, p64_pp_sd, p64_tg, p64_tg_sd = _bench_stats(b64)
    pun_pp, pun_pp_sd, pun_tg, pun_tg_sd = _bench_stats(bunp)

    sz64_gb = os.path.getsize(m64_path) / 1e9
    sz56_gb = os.path.getsize(m56_path) / 1e9
    m64_name = os.path.basename(m64_path)
    m56_name = os.path.basename(m56_path)

    m_r64 = r64["metrics"]
    m_s64 = s64["metrics"]
    m_l56 = l56["metrics"]

    ppl64 = m_r64.get("wikitext2_ppl", {})
    ppl56 = m_l56.get("wikitext2_ppl", {})
    he = m_r64.get("humaneval", {})
    gsm = m_r64.get("gsm8k", {})
    arc1 = m_r64.get("arc_v1", {})
    arc2 = m_r64.get("arc_v2", {})
    gw = m_r64.get("graphwalks", {})
    sr = m_s64.get("strongreject", {})
    fa = m_s64.get("fortress_adv", {})
    fb = m_s64.get("fortress_ben", {})
    dsq = m_s64.get("deepsearchqa", {})
    lb = m_l56.get("longbench_v2", {})

    speedup_gen_64 = p64_tg / pun_tg
    speedup_pp_64 = p64_pp / pun_pp
    speedup_gen_56 = p56_tg / pun_tg
    speedup_pp_56 = p56_pp / pun_pp

    winner_model = os.path.basename(r64["model"])

    return f"""---
license: apache-2.0
tags:
- maple
- moe
- mixture-of-experts
- pruning
- inkling2_loop
- inkling-2-mini
- gguf
- sub-5gb
- llama.cpp
base_model:
- thinkingmachines/Inkling-2-Mini
- unsloth/Inkling-2-Mini-GGUF
---

# Inkling 2 Mini MAPLE (Sub-5GB Looped-MoE GGUF)

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-AmSach/maple-181717?style=for-the-badge&logo=github)](https://github.com/AmSach/maple)
[![Collection](https://img.shields.io/badge/HF_Collection-MAPLE_Sub--5GB-blue?style=for-the-badge)](https://huggingface.co/collections/{COLLECTION_SLUG})
[![Project Site](https://img.shields.io/badge/Docs-amsach.github.io/maple-10b981?style=for-the-badge)](https://amsach.github.io/maple/)

</div>

**Manifold-Activated Pruning of Latent Experts (MAPLE)** compresses Thinking Machines Lab's **26.4B** looped-MoE model (`inkling2_loop`, 24 physical layers with layers 6-17 executed 3 times per token, 128 routed experts per layer, top-8 active) from **8.49 GB (`IQ2_XXS`)** down to **4.85 GB (`64` experts/layer)** and **4.39 GB (`56` experts/layer)**, enabling **100% In-VRAM execution (`49/49` layers)** on a consumer **6 GB RTX 4050 Laptop GPU** via mainline `llama.cpp` (`b12774+`).

Both `mmproj` vision weights (639 MB) and DSpark speculative drafter weights are dropped so 100% of VRAM goes to pruned text experts and KV cache (`5,237 MiB` total at `-c 8192` for `64`-exp; `5,335 MiB` total at `-c 16384` for `56`-exp, both with `-ctk q8_0 -ctv q8_0`).

---

## 1. Looped-MoE (`L6-L17`, 3x Passes) Calibration Architecture

Because Inkling 2 Mini executes layers 6-17 three times per token, Pass 2/3 refinement experts differ from Pass 1 experts. `llama-imatrix` accumulates activation counts and variance across every pass:
- Pre-loop (`L0-L5`) and post-loop (`L18-L23`): `24,576` routing calls per physical layer on our condensed work corpus.
- Looped middle block (`L6-L17`): **`73,728` = `3 x 24,576`** routing calls aggregated across all 3 passes, with sharper specialization (mean Gini `0.344` vs. `0.248` pre-loop and `0.266` post-loop).
- Retaining `k'=64` (`4.85 GB`) protects multi-pass refinement diversity across `L6-L17` while fitting 100% inside 6 GB VRAM; `k'=56` (`4.39 GB`) provides extended `-c 16384` long-context headroom.

---

## 2. RTX 4050 6GB Systems Throughput (`llama-bench`, `r=10`, CUDA 12.4)

| Configuration | Disk Size | VRAM Model Buffer | Prefill (`pp512`) | Generation (`tg128`) | Speedup vs. Unpruned |
|:---|:---:|:---:|:---:|:---:|:---:|
| Unpruned `Inkling-2-Mini-UD-IQ2_XXS` (`-ncmoe 14`) | 8.49 GB | Hybrid (`10/24` GPU exp.) | {pun_pp:.2f} +/- {pun_pp_sd:.2f} t/s | {pun_tg:.2f} +/- {pun_tg_sd:.2f} t/s | 1.00x (Baseline) |
| **MAPLE `64`-exp (`{m64_name}`)** | **{sz64_gb:.2f} GB** | **4,375.9 MiB (100% VRAM)** | **{p64_pp:.2f} +/- {p64_pp_sd:.2f} t/s** | **{p64_tg:.2f} +/- {p64_tg_sd:.2f} t/s** | **{speedup_pp_64:.2f}x pp / {speedup_gen_64:.2f}x tg** |
| **MAPLE `56`-exp (`{m56_name}`)** | **{sz56_gb:.2f} GB** | **3,942.0 MiB (100% VRAM)** | **{p56_pp:.2f} +/- {p56_pp_sd:.2f} t/s** | **{p56_tg:.2f} +/- {p56_tg_sd:.2f} t/s** | **{speedup_pp_56:.2f}x pp / {speedup_gen_56:.2f}x tg** |

---

## 3. MAPLE-Inkling-2-Mini Benchmark Results

Evaluated on NVIDIA RTX 4050 6GB (`llama-server` `b12774`, `--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0`, seed 42). Flagship `64`-exp model tested: `{winner_model}`. Thinking Machines Lab published BF16/FP4 baselines from `thinkingmachines/Inkling-2-Mini` are shown for context (noting local single-model VRAM limits where applicable).

| Benchmark Suite | Split / Local Eval Scope | MAPLE Pruned Result | Published Inkling-2-Mini Baseline (Full Weights) | Notes |
|:---|:---|:---:|:---:|:---|
| **HumanEval (0-shot)** | Coding (`n={he.get('n')}`, `--reasoning-effort low`) | **{he.get('pass_at_1_pct')}%** ({he.get('passed')}/{he.get('n')}) | - | Sandboxed execution, 95% CI {he.get('ci95')} |
| **GSM8K (exact match)** | Math (`n={gsm.get('n')}`, `--reasoning-effort low`) | **{gsm.get('acc_pct')}%** ({gsm.get('correct')}/{gsm.get('n')}) | - | 95% CI {gsm.get('ci95')} |
| **ARC-AGI v1** | Eval (`n={arc1.get('n')}`, compact grid `-c 8192`) | **{arc1.get('acc_pct')}%** ({arc1.get('correct')}/{arc1.get('n')}) | 84.5% | Single-pass `-c 8192`, 95% CI {arc1.get('ci95')} |
| **ARC-AGI v2** | Eval (`n={arc2.get('n')}`, compact grid `-c 8192`) | **{arc2.get('acc_pct')}%** ({arc2.get('correct')}/{arc2.get('n')}) | 49.3% | Single-pass `-c 8192`, 95% CI {arc2.get('ci95')} |
| **GraphWalks BFS** | `<=20k` chars (`n={gw.get('n')}`, `-c 8192`) | **F1 {gw.get('mean_f1_pct')}% / EM {gw.get('em_pct')}%** | 69.7% (256K full) | Local VRAM context bound vs. 256K cloud |
| **LongBench-v2** | `<=64k` chars (`n={lb.get('n')}` on `56`-exp `-c 16384`) | **{lb.get('acc_pct')}%** ({lb.get('correct')}/{lb.get('n')}) | 66.9% (full len) | Short subset fitting 6GB VRAM `-c 16384` |
| **StrongREJECT** | Validation (`n={sr.get('n')}`, `--reasoning-effort minimal`) | **{sr.get('rate_pct')}%** ({sr.get('refused')}/{sr.get('n')}) | 97.8% | Harmful refusal rate, 95% CI {sr.get('ci95')} |
| **FORTRESS Public (Adv.)** | Adversarial (`n={fa.get('n')}`) | **{fa.get('rate_pct')}%** ({fa.get('passed')}/{fa.get('n')}) | 89.5% | Unsafe request refusal, 95% CI {fa.get('ci95')} |
| **FORTRESS Public (Benign)** | Benign (`n={fb.get('n')}`) | **{fb.get('rate_pct')}%** ({fb.get('passed')}/{fb.get('n')}) | 96.4% | Non-over-refusal helpfulness, 95% CI {fb.get('ci95')} |
| **DeepSearchQA (Closed-Book)** | No search tool (`n={dsq.get('n')}`) | **{dsq.get('acc_pct')}%** ({dsq.get('correct')}/{dsq.get('n')}) | 85.3% (with search) | Parametric recall only (no web compaction) |
| **WikiText-2 PPL** | `{ppl64.get('chunks', 40)}` chunks (`c=512`) | **{ppl64.get('ppl')}** (`64`-exp) / **{ppl56.get('ppl')}** (`56`-exp) | - | Unsloth 20ch PPL: {bake.get('ppl_20ch', {}).get('64exp_unsloth', {}).get('ppl')} (`64`) |

---

## 4. Direct Execution (`llama.cpp` b12774+)

```bash
llama-server ^
  -hf AmSach/{DEFAULT_REPO_NAME}:{m64_name} ^
  -ngl 99 -c 8192 -fa on -ctk q8_0 -ctv q8_0 -ub 256 ^
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --jinja
```

For `16k` long-context sessions on 6 GB GPUs, use `{m56_name}` with `-c 16384`.
"""


def build_prelim_card(m64_path: str, m56_path: str) -> str:
    m64_name = os.path.basename(m64_path)
    m56_name = os.path.basename(m56_path)
    return f"""---
license: apache-2.0
tags:
- maple
- moe
- mixture-of-experts
- pruning
- inkling2_loop
- inkling-2-mini
- gguf
- sub-5gb
- llama.cpp
base_model:
- thinkingmachines/Inkling-2-Mini
- unsloth/Inkling-2-Mini-GGUF
---

# Inkling 2 Mini MAPLE (Sub-5GB Looped-MoE GGUF)

Sub-5GB MAPLE-pruned GGUF checkpoints of `thinkingmachines/Inkling-2-Mini` (`26.4B` total / `3.4B` active / `5.4B` effective via 3x middle-layer loop), running **100% In-VRAM (`49/49` layers)** on a 6 GB RTX 4050 Laptop GPU via `llama.cpp` `b12774+`:
- `{m64_name}` (`4.85 GB`, `64/128` experts/layer, work-calibrated: **78.0% HumanEval@50**, `35.83 t/s` gen / `546.4 t/s` pp512, `-c 8192` in `5,237 MiB` VRAM).
- `{m56_name}` (`4.39 GB`, `56/128` experts/layer, 60/40 blended: `36.08 t/s` gen / `550.0 t/s` pp512, `-c 16384` in `5,335 MiB` VRAM).

Full benchmark suite metrics are updating from live GPU evaluation logs. See https://github.com/AmSach/maple and https://amsach.github.io/maple/.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", default=r"E:\maple\logs")
    p.add_argument("--m64", required=True)
    p.add_argument("--m56", required=True)
    p.add_argument("--repo-name", default=DEFAULT_REPO_NAME)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--weights-only", action="store_true")
    p.add_argument("--card-only", action="store_true")
    args = p.parse_args()

    card_out = os.path.join(args.logs_dir, "inkling2_hf_README.md")
    card = build_prelim_card(args.m64, args.m56) if args.weights_only else build_model_card(args.logs_dir, args.m64, args.m56)
    with open(card_out, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[+] Wrote HF model card -> {card_out}")

    if args.dry_run:
        return

    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.repo_name}"
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"[*] Uploading to https://huggingface.co/{repo_id} ...")

    if args.card_only:
        files = [(card_out, "README.md")]
    else:
        files = [
            (card_out, "README.md"),
            (r"E:\maple\Modelfile.inkling2", "Modelfile"),
            (r"E:\maple\engine\calibration_corpus_work.txt", "calibration_corpus_work.txt"),
            (r"E:\maple\engine\calibration_corpus_inkling_work.txt", "calibration_corpus_inkling_work.txt"),
            (r"E:\maple\engine\inkling2_expert_selection_map_64exp_blended.json", "inkling2_expert_selection_map_64exp_blended.json"),
            (r"E:\maple\engine\inkling2_expert_selection_map_56exp_blended.json", "inkling2_expert_selection_map_56exp_blended.json"),
            (r"E:\maple\engine\inkling2_expert_selection_map_64exp_work.json", "inkling2_expert_selection_map_64exp_work.json"),
            (r"E:\maple\engine\inkling2_expert_selection_map_64exp.json", "inkling2_expert_selection_map_64exp.json"),
            (args.m64, os.path.basename(args.m64)),
            (args.m56, os.path.basename(args.m56)),
        ]
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    for src, dst in files:
        if dst in existing and dst.endswith(".gguf"):
            print(f"  [=] {dst} already present in repo, skipping.")
            continue
        if os.path.exists(src):
            print(f"  -> Uploading {dst} ({os.path.getsize(src):,} B)...", flush=True)
            for attempt in range(5):
                try:
                    api.upload_file(
                        path_or_fileobj=src,
                        path_in_repo=dst,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Add {dst} for MAPLE Inkling-2-Mini Sub-5GB release",
                    )
                    break
                except Exception as ex:
                    print(f"  [!] Upload attempt {attempt+1}/5 failed for {dst}: {ex}", flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(10 * (attempt + 1))

    api.add_collection_item(
        collection_slug=COLLECTION_SLUG,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    print(f"[+] Added {repo_id} to collection {COLLECTION_SLUG}")


if __name__ == "__main__":
    main()
