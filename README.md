---
language:
- en
license: apache-2.0
tags:
- moe
- mixture-of-experts
- pruning
- gemma-4
- llama.cpp
- gguf
- edge-ai
- model-compression
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
library_name: gguf
---

# MAPLE: Manifold-Activated Pruning of Latent Experts

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24042480.svg)](https://doi.org/10.5281/zenodo.24042480)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Aman%20Sachan-blue.svg)](https://scholar.google.com/citations?user=dfgiWLYAAAAJ&hl=en)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-AmSach%2Fmaple--4.75GB-yellow.svg)](https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF)
[![Direct Download](https://img.shields.io/badge/Direct%20Download-4.75%20GB%20GGUF-brightgreen.svg)](https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF/resolve/main/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf?download=true)
[![Format](https://img.shields.io/badge/Format-GGUF-green.svg)](https://github.com/ggerganov/llama.cpp)
[![Hardware](https://img.shields.io/badge/Target%20Hardware-RTX%204050%206GB%20(100%25%20VRAM)-orange.svg)]()
[![PitchHut](https://img.shields.io/badge/PitchHut-Featured%20Showcase-orange.svg)](https://www.pitchhut.com/project/maple-moe-pruning)
[![llama.cpp Showcase](https://img.shields.io/badge/llama.cpp-Showcase%20%2331894-purple.svg)](https://github.com/ggml-org/llama.cpp/discussions/31894)
[![Interactive Site](https://img.shields.io/badge/Official%20Site-amsach.github.io%2Fmaple-indigo.svg)](https://amsach.github.io/maple/)

> **Sub-2-BPW Structural Compression for Mixture-of-Experts (MoE) LLMs**  
> Compressing **Gemma 4 26B-A4B** from **9.92 GB to 4.75 GB** (10.95B physical container params at 3.47 BPW, 1.48 effective bits/param relative to unpruned 25.6B base), enabling **100% In-VRAM GPU Execution** on 6 GB laptop GPUs with **45.4 Tokens/Second** throughput.

**Author**: Aman Sachan ([Google Scholar](https://scholar.google.com/citations?user=dfgiWLYAAAAJ&hl=en) | [amansachan92905@gmail.com](mailto:amansachan92905@gmail.com))  
**Organization**: Independent Systems Research  

---

## Domain Specialization: Technical Reasoning and Code Synthesis

> **Model Focus:**  
> This 4.75 GB checkpoint is profiled and pruned specifically for **Python programming, algorithm synthesis, data structures, and formal STEM reasoning**.  
> By excising 80 dormant multilingual and general conversational experts, the active 48-expert manifold preserves deductive reasoning and syntax generation while fitting entirely within consumer 6 GB VRAM.

### Custom Domain MoE Pruning Pipeline
The MAPLE pruning pipeline is domain-agnostic and can be applied to any specialized field (such as medicine, finance, legal analysis, or creative writing) using the scripts in this repository:
1. **Calibration**: Run `engine/maple_profile.py` with your custom text corpus (e.g., `engine/calibration_corpus.txt` or a proprietary dataset) to profile routing frequency and activation energy via `llama-imatrix`.
2. **Selection**: MAPLE scores experts according to the composite energy metric and outputs `engine/expert_selection_map.json`.
3. **Container Assembly**: Run `engine/maple_build.py` to slice the unpruned base model and assemble an optimized sub-2-BPW GGUF container for 100% in-VRAM edge execution.

---

## Model Distribution and Direct Execution

The pre-built, verified 4.75 GB GGUF container is hosted on Hugging Face:

- **Hugging Face Model Repository**: [AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF](https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF)
- **Direct Binary Download**: [gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf (4.75 GB)](https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF/resolve/main/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf?download=true)

### Direct Execution via `llama.cpp`:
```bash
llama-cli \
  -hf AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF \
  -ngl 99 \
  -c 2048 \
  -n 512 \
  --temp 0.4 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  -r "<turn|>" \
  -p "<|turn>user\nExplain why the sky is blue using Rayleigh scattering.\n<turn|>\n<|turn>model\n"
```
*(llama-cli downloads the 4.75 GB GGUF container directly from Hugging Face and allocates all 30 layers into GPU VRAM at 45.4 TPS).*

---

## Overview

While global multilingual pre-training necessitates 128–256 experts to serve hundreds of human languages and encyclopedic domains, localized edge deployments (such as English technical reasoning and code generation) activate only a narrow subset of experts. 

MAPLE discovers this latent manifold by profiling live forward-pass expert activations via `llama-imatrix`, scoring experts via a composite frequency-energy metric:

$$s_{e,l} = C_{e,l} \cdot \sqrt{\bar{\sigma}_{e,l} + \epsilon}$$

where:
- **`s_{e,l}`**: Composite importance score for expert `e` in layer `l`
- **`C_{e,l}`**: Empirical token activation frequency (routing count)
- **`σ̄_{e,l}`**: Mean input channel activation energy
- **`ε`**: Numerical stability floor (10⁻⁸)

Applied to Gemma 4 26B-A4B (128 routed experts, 30 layers), MAPLE excises 80 dormant experts per layer to assemble an optimized 48-expert GGUF container capturing **79.88% of active forward-pass activation mass**.

---

## Performance Benchmark (RTX 4050 6GB Laptop GPU)

| Configuration | Model Size | Parameter Count | Bitrate | VRAM Footprint | Offloaded Layers | Generation Speed | Latency |
|---|---|---|---|---|---|---|---|
| **Original 128-Expert Base** | 9.92 GB | 25.6B total | ~3.10 BPW | 6,141 MiB (paged) | 16 / 30 | 6.9 Tokens/Sec | 144.9 ms / token |
| **MAPLE 48-Expert Container** | **4.75 GB** | **10.95B active** | **3.47 BPW (1.48 eff.)** | **4,820 MiB (100% VRAM)** | **30 / 30** | **45.4 Tokens/Sec** | **22.0 ms / token** |
| **Improvement** | **2.09x smaller** | **-57.2% params** | **1.48 eff. BPW** | **1.32 GB VRAM headroom** | **100% In-VRAM** | **6.57x speedup** | **6.58x faster** |

> **VRAM Accounting Note:**  
> The 4.75 GB GGUF container occupies 4,526 MiB of static weight memory in VRAM. With a 2048-token context window KV cache and compute scratch buffers (~294 MiB), total runtime memory allocation is 4,820 MiB, leaving 1,324 MiB (1.29 GiB) of free VRAM headroom on a 6 GB (6,144 MiB) RTX 4050 GPU.  
> **Bitrate Accounting Note:**  
> 4,745,987,168 bytes across the 10,952,640,446 physical tensor elements evaluates to 3.47 physical bits/param (blended between IQ2_XXS on experts and higher precision on shared layers). Relative to the unpruned 25.6B base architecture, it represents 1.48 effective bits per base parameter.

### Task-Level Semantic & Execution Fidelity Benchmark

| Task | Domain | Tested Capabilities | Keyword Hit Rate | Functional Execution |
|---|---|---|---|---|
| **T1: Fibonacci** | Code Synthesis | `def fibonacci`, `return`, `append`, `[0,1]` | **100%** | **PASSED** (`fibonacci(10)` assert) |
| **T2: Binary Search Tree** | Data Structures | `class Node`, `class BST`, `insert`, `search` | **75%** (100% @ ctx >= 500) | Valid syntax |
| **T3: Rayleigh Scattering** | Physics Reasoning | `Rayleigh`, `wavelength`, `scattering`, `blue` | **100%** | Coherent derivation |
| **T4: Syllogistic Logic** | Formal Deduction | `transitive`, `syllogism`, set containment | **100%** | Step-by-step logic |
| **Overall Mean** | | | **93.8%** | **Syntax & Logic Preserved** |

---

## Three Documented Failure Modes Overcome

1. **The Ghost Router (Proxy-Embedding Failure):** Estimating routing by projecting Layer-0 token embeddings through deep router gates failed because non-linear attention shifts rotate deep hidden representations ($l \ge 10$) into orthogonal spaces. MAPLE enforces true forward profiling via `llama-imatrix`.
2. **Cascaded Normalization Suppression:** Multiplying expert scales by $0.795$ to counteract softmax denominator shrinkage compounded across 30 layers ($0.795^{30} \approx 0.001$), blinding the residual stream. Scales must be preserved at $1.0\times$.
3. **DRY Sampler Interference:** The `--dry-allowed-length 2` penalty broke Python syntax by penalizing mandatory 2-character keywords (`def`, `if`, `in`, `+=`). Standard nucleus sampling restored syntax fidelity.

---

## Repository Structure

```
maple/
├── README.md                           # Documentation & benchmarks
├── LICENSE                             # Apache 2.0 License
├── requirements.txt                    # Python dependencies
├── run_maple.bat                       # Interactive Windows launcher
├── upload_to_hf.py                     # Hugging Face Hub upload utility
└── engine/
    ├── maple_profile.py                # Scores & extracts top-48 experts from imatrix
    ├── maple_build.py                  # Slices & builds 4.75 GB GGUF container
    ├── benchmark.py                    # Automated 4-domain semantic fidelity suite
    ├── chat.py                         # Streaming console with live telemetry
    ├── calibration_corpus.txt          # Multi-domain calibration corpus (1,979 tokens)
    └── expert_selection_map.json       # Extracted 30-layer empirical expert indices
```

---

## Local Setup & Interactive Console

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Chat Console
On Windows:
```cmd
run_maple.bat
```
Or via Python:
```bash
python engine/chat.py
```

---

## Model Download Links (Hugging Face Hub)

The compressed 4.75 GB GGUF model container is publicly hosted on Hugging Face:
- **Repository**: [https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF](https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF)
- **Direct Binary (4.75 GB)**: [gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf](https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF/resolve/main/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf?download=true)

---

## Citation

```bibtex
@article{sachan2026maple,
  title={MAPLE: Manifold-Activated Pruning of Latent Experts},
  author={Sachan, Aman},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.24042480},
  url={https://doi.org/10.5281/zenodo.24042480}
}
```
