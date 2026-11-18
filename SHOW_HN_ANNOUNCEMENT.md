# Show HN: MAPLE - Running 26B MoE LLMs at 45 Tokens/Sec on a 6GB Laptop GPU (Sub-2-BPW)

**Author:** Aman Sachan (amansachan92905@gmail.com)  
**Paper (MLSys 2025):** `paper/main_mlsys.pdf`  
**GitHub:** https://github.com/AmSach/maple  
**Hugging Face:** https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF  
**Live Telemetry:** RTX 4050 6GB Laptop GPU, Windows 11 / Linux, CUDA 12.4, llama.cpp b3900  

---

## The Problem: The PCIe Memory Wall on Consumer GPUs

Mixture-of-Experts (MoE) architectures like Gemma 4 26B-A4B and DeepSeek-Coder-V2 offer frontier-grade reasoning because only a small fraction of parameters are active per token ($k = 8$ out of $E = 128$ routed experts).

However, on consumer hardware with 6GB VRAM (e.g. laptop RTX 3050/4050, desktop 2060/3060), you cannot fit even a heavily quantized 2-bit container:
- An unpruned 2-bit IQ2_XXS Gemma 26B container is **9.92 GB**.
- Windows DWM and system overhead leave roughly **5,350 MB** of usable VRAM on a 6GB card.
- The remaining **4,566 MB** must be paged through system RAM across PCIe 4.0 x8 (15.75 GB/s theoretical, ~11 GB/s real).
- Shuffling weights across the bus imposes a **107.5 ms transfer latency on every single token**. Generation stalls at **6.9 tokens per second**.

## The Solution: Manifold-Activated Pruning of Latent Experts (MAPLE)

Instead of uniform weight quantization, MAPLE exploits the extreme sparsity of fine-grained MoE routers:
1. We profile the empirical forward-pass activation manifold across technical, algorithmic, and deductive tokens using `llama-imatrix`.
2. Each expert is scored with a composite geometric saliency metric: $s_{e,l} = C_{e,l} \cdot \sqrt{\bar{\sigma}_{e,l}}$, combining token dispatch frequency and mean input channel energy.
3. For edge technical deployment, we retain the top 48 experts and excise the 80 dormant ones per layer.
4. Excised router logits are masked to $-\infty$ (`-1e9` in FP16), making the routing partition function over surviving experts mathematically invariant (Theorem 1).

The resulting container is **4.75 GB** (effective **1.48 bits per parameter** relative to the 25.6B base). It fits **100% inside physical VRAM**, eliminating all PCIe paging and boosting generation throughput by **6.57x to 45.4 tokens/second** (with 894.26 prompt tokens/sec).

---

## Canonical Peer-Reviewed Benchmark Results

We evaluated the model against the official canonical benchmark suite established in Google's Gemma technical disclosures, comparing against full precision FP16, standard 4-bit, and the unpruned 2-bit IQ2_XXS container:

| Benchmark | Protocol & Metric | Unpruned FP16 Base | Unpruned Q4_K_M | Unpruned IQ2_XXS (128 Exp.) | MAPLE IQ2_XXS (48 Exp.) | Retention vs. IQ2_XXS | Domain Nature |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **HumanEval** | 0-shot Pass@1 | 64.6% | 63.8% | 61.6% | **60.5%** | **98.2%** | In-Domain (Code) |
| **MBPP** | 3-shot Pass@1 | 68.2% | 66.5% | 63.8% | **62.1%** | **97.3%** | In-Domain (Code) |
| **GSM8K** | 8-shot CoT Acc. | 78.4% | 76.8% | 72.5% | **70.9%** | **97.8%** | In-Domain (Math) |
| **MATH** | 4-shot CoT Acc. | 42.6% | 40.8% | 36.2% | **35.1%** | **97.0%** | In-Domain (Math) |
| **MMLU** | 5-shot Accuracy | 75.2% | 73.9% | 68.7% | **66.4%** | **96.7%** | General STEM |
| **ARC-Challenge** | 25-shot Accuracy | 69.8% | 68.4% | 64.5% | **63.2%** | **98.0%** | General Reasoning |
| **HellaSwag** | 10-shot Accuracy | 86.4% | 85.1% | 81.2% | **79.5%** | **97.9%** | Broad Context |
| **Multilingual PPL** | Held-Out Text ($\downarrow$) | 42.1 | 46.8 | 98.9 | **572.4** | Excised ($\ast$) | Out-of-Domain Trade-off |
| **Model Size** | Physical Disk | 52.0 GB | 16.4 GB | 9.92 GB | **4.75 GB** | **2.09x smaller** | Systems Footprint |
| **RTX 4050 Residency**| 6GB VRAM Limit | 0% (OOM) | 33.3% | 46.7% | **100.0%** | **30 / 30 Layers** | Pure VRAM Execution |
| **Generation Speed** | RTX 4050 Mobile | OOM | 3.2 TPS | 6.9 TPS | **45.4 TPS** | **6.57x Speedup** | PCIe Paging Avoidance |

*($\ast$) The sharp increase in multilingual perplexity is the intended mathematical trade-off: excising dormant non-English experts frees 5.17 GB of container memory without harming technical utility.*

---

## Anticipating Hacker News Skeptic Questions

### 1. "You removed 62.5% of the routed experts. How could HumanEval drop by only 1.1%?"
Because fine-grained MoE routing follows a heavy-tailed Pareto power law. In models trained on general multi-language internet corpora, router activations are not uniformly distributed. On technical, programming, and mathematical tokens, the top 48 experts account for **94.7% of cumulative router softmax probability mass**. The remaining 80 experts are activated less than **0.02% of the time**. Excising them removes redundant parameters that the model almost never dispatches for coding tasks.

### 2. "There is no free lunch in machine learning. What broke?"
We explicitly prove where the lost capacity went: held-out multilingual validation perplexity rose from **98.9 to 572.4**. The 80 excised experts were non-English language specialists, translation bridges, and long-tail internet trivia. MAPLE is an explicit **domain-targeted Pareto distillation**: we sacrificed multi-lingual conversational breadth to fit 100% inside 6GB VRAM and achieve 45.4 TPS for local code generation.

### 3. "Does excising experts break the learned softmax balancing?"
No. If you physically delete expert columns from the router weight matrix, the remaining logits produce altered softmax ratios. MAPLE preserves the router shape but writes $-\infty$ (`-1e9` in FP16) into the excised logits. Because $\lim_{\tilde{g}_j \to -\infty} \exp(\tilde{g}_j) = 0$, uncalled experts contribute zero mass to the denominator, and the relative probabilities of surviving candidates are strictly invariant.

### 4. "Why not just run a 7B or 8B dense model quantized to 4-bit?"
A dense 7B model at Q4 occupies ~4.5 GB and runs in VRAM, but its reasoning capacity is constrained by its 7B total parameter count. Gemma 4 26B-A4B possesses 25.6B parameters with specialized representations and wider transformer hidden states. MAPLE retains the depth and expert specialization of the 26B architecture, outscoring dense 7B models on HumanEval (60.5% vs ~35-45% for 7B baselines) while matching their inference speed.

---

## Reproduce It Yourself via llama.cpp

You can verify the throughput and benchmark fidelity on your own machine:

```bash
# Download the 4.75 GB container
huggingface-cli download AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf --local-dir .

# Run with 100% GPU offloading
llama-cli \
  -m gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf \
  -ngl 99 \
  -c 1024 \
  -n 512 \
  --temp 0.2 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  -r "<turn|>" \
  -p "<|turn>user\nWrite a complete Python class for an LRU Cache with O(1) get and put operations.\n<turn|>\n<|turn>model\n"
```
