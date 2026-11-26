---
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
- `Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf` (`4.85 GB`, `64/128` experts/layer, work-calibrated: **78.0% HumanEval@50**, `35.83 t/s` gen / `546.4 t/s` pp512, `-c 8192` in `5,237 MiB` VRAM).
- `Inkling-2-Mini-MAPLE-56exp-Blended-IQ2_XXS.gguf` (`4.39 GB`, `56/128` experts/layer, 60/40 blended: `36.08 t/s` gen / `550.0 t/s` pp512, `-c 16384` in `5,335 MiB` VRAM).

Full benchmark suite metrics are updating from live GPU evaluation logs. See https://github.com/AmSach/maple and https://amsach.github.io/maple/.
