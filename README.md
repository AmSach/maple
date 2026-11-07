# MAPLE: Manifold-Activated Pruning of Latent Experts

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/main.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Model-GGUF-green.svg)](https://github.com/ggerganov/llama.cpp)
[![Hardware](https://img.shields.io/badge/GPU-RTX%204050%206GB-orange.svg)]()

> **Sub-2-BPW Structural Compression for Mixture-of-Experts (MoE) LLMs**  
> Compressing **Gemma 4 26B-A4B** from **9.92 GB to 4.75 GB** (1.48 effective bits/param), enabling **100% In-VRAM GPU Execution** on 6 GB laptop GPUs with **45.4 Tokens/Second** throughput.

**Author**: Aman Sachan ([amansachan92905@gmail.com](mailto:amansachan92905@gmail.com))  
**Organization**: Independent Systems Research  

---

## Abstract

While global multilingual pre-training necessitates 128–256 experts to serve diverse languages and domains, localized edge deployments (e.g. English technical reasoning and code generation) activate only a narrow subset of experts. 

**MAPLE** discovers this latent manifold by profiling live forward-pass expert activations via `llama-imatrix`, scoring experts via a composite frequency-energy metric:

$$s_{e,l} = C_{e,l} \cdot \sqrt{\bar{\sigma}_{e,l} + \epsilon}$$

Applied to Gemma 4 26B-A4B (128 routed experts, 30 layers), MAPLE excises 80 dormant experts per layer to assemble an optimized 48-expert GGUF container capturing **79.88% of active forward-pass activation mass**.

---

## Performance Benchmark (RTX 4050 6GB Laptop GPU)

| Configuration | Model Size | VRAM Allocation | Offloaded Layers | Generation Speed | Latency |
|---|---|---|---|---|---|
| **Original 128-Expert Base** | 9.92 GB | 6,141 MiB (paged) | 16 / 30 | 6.9 Tokens/Sec | 144.9 ms / token |
| **MAPLE 48-Expert Container** | **4.75 GB** | **4,180 MiB (100% VRAM)** | **30 / 30** | **45.4 Tokens/Sec** | **22.0 ms / token** |
| **Improvement** | **2.09x smaller** | **-32% VRAM** | **100% In-VRAM** | **6.57x speedup** | **6.58x faster** |

### Task-Level Semantic Fidelity Benchmark

| Task | Domain | Tested Capabilities | Fidelity |
|---|---|---|---|
| **T1: Fibonacci** | Code Synthesis | `def fibonacci`, `return`, `append`, `[0,1]` | **100%** |
| **T2: Binary Search Tree** | Data Structures | `class Node`, `class BST`, `insert`, `search` | **75%** (100% with context >= 500) |
| **T3: Rayleigh Scattering** | Physics Reasoning | `Rayleigh`, `wavelength`, `scattering`, `blue` | **100%** |
| **T4: Syllogistic Logic** | Formal Deduction | `transitive`, `syllogism`, set containment | **100%** |
| **Overall Mean** | | | **93.8%** |

---

## Three Documented Failure Modes Overcome

1. **The Ghost Router (Proxy-Embedding Failure):** Estimating routing by projecting Layer-0 token embeddings through deep router gates failed because non-linear attention shifts rotate deep hidden representations ($l \ge 10$) into orthogonal spaces. MAPLE enforces true forward profiling via `llama-imatrix`.
2. **Cascaded Normalization Suppression:** Multiplying expert scales by $0.795$ to counteract softmax denominator shrinkage compounded across 30 layers ($0.795^{30} \approx 0.001$), blinding the residual stream. Scales must be preserved at $1.0\times$.
3. **DRY Sampler Interference:** The `--dry-allowed-length 2` penalty broke Python syntax by penalizing mandatory 2-character keywords (`def`, `if`, `in`, `+=`). Standard nucleus sampling restored syntax fidelity.

---

## Repository Structure

```
maple/
├── README.md                           # Comprehensive documentation
├── LICENSE                             # Apache 2.0 License
├── requirements.txt                    # Python dependencies
├── run_maple.bat                       # One-click Windows interactive launcher
├── upload_to_hf.py                     # Hugging Face Hub upload utility
├── engine/
│   ├── extract_empirical_experts.py    # Scores & extracts top-48 experts from imatrix
│   ├── stage2_build_48exp_empirical.py # Slices & builds 4.75 GB GGUF container
│   ├── run_quality_benchmark.py        # Automated 4-domain semantic fidelity suite
│   ├── interactive_cli.py              # Real-time streaming console with live telemetry
│   ├── comprehensive_calibration.txt   # Multi-domain calibration corpus (1,979 tokens)
│   └── layer_experts_empirical_48.json # Extracted 30-layer empirical expert indices
└── paper/
    ├── main.pdf                        # 13-page comprehensive reference paper
    ├── main.tex                        # LaTeX source
    ├── main_ieee.pdf                   # 4-page IEEE AccML 2027 edition
    ├── main_ieee.tex                   # IEEEtran LaTeX source
    ├── main_tmlr.pdf                   # 7-page TMLR double-blind edition
    ├── main_tmlr.tex                   # TMLR LaTeX source
    └── tmlr.sty                        # Official TMLR style file
```

---

## Quickstart

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Console
On Windows:
```cmd
run_maple.bat
```
Or with Python:
```bash
python engine/interactive_cli.py
```

### 3. Direct llama.cpp CLI Execution
```bash
llama-cli \
  -m gemma_4_26b_sub5gb_insitu.gguf \
  -ngl 99 \
  -c 2048 \
  -n 512 \
  --temp 0.4 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  -r "<turn|>" \
  -p "<|turn>user\nExplain why the sky is blue using Rayleigh scattering.\n<turn|>\n<|turn>model\n"
```

---

## Upload to Hugging Face Hub

To publish the 4.75 GB GGUF container on your Hugging Face account:

```bash
python upload_to_hf.py
```

---

## Citation

```bibtex
@article{sachan2026maple,
  title={MAPLE: Manifold-Activated Pruning of Latent Experts for Sub-2-BPW Mixture-of-Experts Language Models},
  author={Sachan, Aman},
  journal={arXiv preprint},
  year={2026}
}
```
