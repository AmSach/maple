"""
MAPLE Profiler for Inkling-2-Mini (24 Physical Layers, 128 Experts/Layer, Top-8 Active)
Computes composite frequency-energy saliency scores from llama-imatrix statistics:
  s_{e,l} = C_{e,l} * sqrt(mean_j(Sigma_{e,l,j}) + eps)
which is rank-equivalent to C_{e,l} * (E_bar_{e,l})^(1/3) (beta = 1/3).
"""

import argparse
import json
import os
import gguf
import numpy as np


def profile_inkling2(imatrix_path: str, output_map_path: str, target_k: int = 56, num_layers: int = 24, n_experts: int = 128):
    print(f"[*] Loading imatrix from {imatrix_path}...")
    reader = gguf.GGUFReader(imatrix_path, "r")
    tmap = {t.name: t for t in reader.tensors}

    selection_map = {}
    coverages = []
    freq_coverages = []
    overlaps = []

    for l in range(num_layers):
        c_key = f"blk.{l}.ffn_down_exps.weight.counts"
        s_key = f"blk.{l}.ffn_down_exps.weight.in_sum2"
        if c_key not in tmap or s_key not in tmap:
            raise KeyError(f"Missing expected imatrix keys for layer {l}: {c_key}, {s_key}")

        counts = np.array(tmap[c_key].data, dtype=np.float64).reshape(n_experts)
        in_sum2 = np.array(tmap[s_key].data, dtype=np.float64).reshape(n_experts, -1)
        sigma_bar = np.mean(in_sum2, axis=1)

        score = counts * np.sqrt(sigma_bar + 1e-8)
        top_maple = sorted([int(x) for x in np.argsort(-score)[:target_k]])
        top_freq = sorted([int(x) for x in np.argsort(-counts)[:target_k]])

        tot = max(float(counts.sum()), 1.0)
        cov_m = float(counts[top_maple].sum()) / tot * 100.0
        cov_f = float(counts[top_freq].sum()) / tot * 100.0
        ov = len(set(top_maple) & set(top_freq))

        selection_map[str(l)] = top_maple
        coverages.append(cov_m)
        freq_coverages.append(cov_f)
        overlaps.append(ov)

        if l in (0, 6, 12, 17, 18, 23):
            print(
                f"  Layer {l:02d}: calls={int(tot):,} | MAPLE cov={cov_m:.2f}% | "
                f"Freq cov={cov_f:.2f}% | overlap={ov}/{target_k}"
            )

    out_dir = os.path.dirname(os.path.abspath(output_map_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_map_path, "w", encoding="utf-8") as f:
        json.dump(selection_map, f, indent=2)

    print(
        f"[+] Saved {target_k}-expert selection map ({num_layers} layers) to {output_map_path}\n"
        f"    Mean call coverage: MAPLE={np.mean(coverages):.2f}% vs Freq={np.mean(freq_coverages):.2f}% | "
        f"Mean overlap={np.mean(overlaps):.2f}/{target_k}"
    )
    return selection_map


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--imatrix", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-experts", type=int, default=56)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--source-experts", type=int, default=128)
    args = p.parse_args()
    profile_inkling2(args.imatrix, args.output, args.target_experts, args.num_layers, args.source_experts)
