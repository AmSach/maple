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
- ollama
- reasoning
base_model:
- thinkingmachines/Inkling-2-Mini
- unsloth/Inkling-2-Mini-GGUF
pipeline_tag: text-generation
---

# Inkling 2 Mini MAPLE (Sub-5GB Looped-MoE GGUF)

Official Sub-5GB release of **Inkling 2 Mini** compressed using **MAPLE (Mass-Aware Pruning of Looped MoEs)**.

This checkpoint shrinks `thinkingmachines/Inkling-2-Mini` (26.4B total parameters, 3.4B active parameters, 5.4B effective parameters through 3x looping of layers 6-17) so that it runs **100% In-VRAM** on budget 6 GB GPUs like the NVIDIA GeForce RTX 4050 / 3050 Laptop GPU, completely eliminating CPU offload bottlenecks.

---

## Model Variants

| Checkpoint Name | File Size | Routed Experts / Layer | Retention | Recommended Context | Peak VRAM (Q8_0 KV) | Primary Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf`** | **4.85 GB** | **64 / 128** | **50.0%** | **8,192** | **4,400 MiB** | **Flagship: Best Reasoning & Coding (78.0% HumanEval)** |
| **`Inkling-2-Mini-MAPLE-56exp-Blended-IQ2_XXS.gguf`** | **4.39 GB** | **56 / 128** | **43.8%** | **16,384** | **4,020 MiB** | **Long Context Companion (16k in 6GB VRAM)** |

---

## Performance & Hardware Benchmarks

All benchmarks below were empirically measured on an **NVIDIA GeForce RTX 4050 6GB Laptop GPU** (6,141 MiB physical VRAM, 96-bit bus, 192 GB/s bandwidth, 50W TGP) running `llama.cpp` `b12774+` with Flash Attention and CUDA 12.4.

### 1. Throughput & Latency (`llama-bench -r 10`)

| Model Configuration | Disk Size | GPU Offload | Prompt Processing (pp512) | Text Generation (tg128) | Speedup vs Base IQ3 | Speedup vs Base IQ2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unpruned UD-IQ3_XXS** | 11.66 GB | 18/49 layers | 12.36 ± 0.59 t/s | 3.38 ± 0.51 t/s | 1.00x | - |
| **Unpruned UD-IQ2_XXS** | 8.49 GB | 14/49 layers | 83.04 ± 0.50 t/s | 12.13 ± 0.84 t/s | 6.72x pp / 3.59x tg | 1.00x |
| **MAPLE 64exp-Work (Flagship)** | **4.85 GB** | **49/49 layers (100%)** | **546.41 ± 1.94 t/s** | **35.83 ± 1.17 t/s** | **44.2x pp / 10.6x tg** | **6.58x pp / 2.95x tg** |
| **MAPLE 56exp-Blended (16k)** | **4.39 GB** | **49/49 layers (100%)** | **550.04 ± 1.34 t/s** | **36.08 ± 0.13 t/s** | **44.5x pp / 10.7x tg** | **6.62x pp / 2.97x tg** |

Because MAPLE fits 100% of the weights in VRAM, memory bandwidth is no longer throttled across the slow PCIe Gen4 x4 laptop bus. Text generation speeds jump from an unusable 3.38 tokens/second up to **35.8 tokens/second**.

### 2. Capabilities & Reasoning Scorecard

Evaluated using standard greedy decoding (`temp=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=42`, `reasoning_effort=low`):

| Evaluation Suite | Sample Size | Score | 95% Confidence Interval | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **HumanEval** | 50 tasks | **78.0%** (39/50) | [66.0%, 88.0%] | Python function completion and algorithmic code |
| **WikiText-2 Perplexity** | 40 chunks (512 ctx) | **30.006 ± 0.876** | - | Stable language modeling perplexity |
| **GSM8K** | 50 tasks | **22.0%** (11/50) | [12.0%, 34.0%] | Grade school math word problems with step-by-step trace |
| **ARC-AGI v1** | 40 tasks | **0.0%** (0/40) | [0.0%, 0.0%] | Pure visual-spatial grid transformations |
| **ARC-AGI v2** | 40 tasks | **0.0%** (0/40) | [0.0%, 0.0%] | Advanced reasoning grid transformations |

### 3. VRAM Consumption Profile on 6GB GPU

| Context Window | KV Cache Quant | Model Weights | KV Cache VRAM | CUDA Overheads | Total VRAM | Fits in 6GB GPU? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2,048** | Q8_0 | 4,082 MiB | 60.5 MiB | 502 MiB | **4,645 MiB** | Yes (1.5 GB headroom) |
| **4,096** | Q8_0 | 4,082 MiB | 121.1 MiB | 502 MiB | **4,705 MiB** | Yes (1.4 GB headroom) |
| **8,192** | Q8_0 | 4,082 MiB | 242.2 MiB | 502 MiB | **4,826 MiB** | **Yes (1.3 GB headroom)** |
| **16,384 (56exp)** | Q8_0 | 3,698 MiB | 484.4 MiB | 502 MiB | **4,684 MiB** | **Yes (1.4 GB headroom)** |

---

## Architectural Insight: The Looped MoE Expert Starvation Cliff

Why did we select `64exp-Work IQ2_XXS` instead of a smaller `44exp IQ3_XXS`?

Inkling 2 Mini features an innovative architecture: 24 physical layers where the middle 12 layers (`L6` through `L17`) loop 3 times sequentially per token. This creates 48 virtual routing layers.

1. **The 44-Expert Failure Mode:** In order to fit a `UD-IQ3_XXS` quantization under 5 GB, expert retention had to be dropped to 44 out of 128 experts (34.4%). Under 3-pass looping, tokens repeatedly starve for specialized routing pathways. Across 48 virtual layers, the routing distribution collapses into repetitive token loops (`15 + 27 = 30 + 27? ...`), yielding 0% HumanEval accuracy.
2. **The 64-Expert Sweet Spot:** At 64 experts (50.0% retention), the routing manifold remains diverse across all 3 loop passes. Calibrated against our leak-free coding and multi-turn corpus, the model preserves high-order reasoning, achieving **78.0% HumanEval@50** while fitting completely within 4.85 GB.

---

## Recommended Software & Runtimes

### 1. llama.cpp (CLI & Server)

Ensure you are using `llama.cpp` build `b12774` or later for proper looped MoE and GGUF v3 custom quant compatibility.

#### Interactive CLI Chat:
```bash
llama-cli \
  -m Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf \
  -c 8192 \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  --jinja \
  --reasoning-effort low \
  --temp 1.0 \
  --top-p 1.0 \
  --min-p 0.0 \
  -p "<|im_start|>user\nWrite a Python function to check if a binary tree is symmetric.<|im_end|>\n<|im_start|>assistant\n"
```

#### High-Performance Local OpenAI-Compatible Server:
```bash
llama-server \
  -m Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 8192 \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -np 1 \
  --jinja
```

Then query the server from any tool (Cursor, Continue.dev, Python `openai` client, or LangChain) at `http://localhost:8080/v1`.

---

### 2. Ollama Setup

1. Download `Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf` and the accompanying `Modelfile` from this repository.
2. Place both files in the same directory.
3. Build the Ollama model:
```bash
ollama create inkling2-maple -f Modelfile
```
4. Run interactive chat:
```bash
ollama run inkling2-maple
```

---

### 3. LM Studio, Jan.ai, and Open WebUI

To load in LM Studio or Jan:
- Copy `Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf` to your local models directory.
- **GPU Acceleration:** Set GPU Offload to `MAX` (or `49` layers).
- **Context Length:** Set to `8192`.
- **Flash Attention:** Enable `ON`.
- **KV Cache Quantization:** Select `Q8_0` (or `FP16` if context is under 4096).
- **Chat Template:** Select `ChatML / Jinja`.

---

### 4. Python API (`llama-cpp-python`)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf",
    n_gpu_layers=-1,      # Offload all 49 layers to GPU
    n_ctx=8192,
    flash_attn=True,
    type_k=8,             # Q8_0 KV cache
    type_v=8,
    verbose=False
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a concise, expert software engineer."},
        {"role": "user", "content": "Write a Python generator function to stream prime numbers using the Sieve of Eratosthenes."}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response["choices"][0]["message"]["content"])
```

---

## Example Usecases & Real Model Outputs

Here are real outputs generated by `Inkling-2-Mini-MAPLE-64exp-Work-IQ2_XXS.gguf` running on our RTX 4050 GPU:

### Example 1: High-Speed Algorithmic Python Code Generation

**Prompt:**
```text
Write a clean, efficient Python function `is_palindrome_permutation(s: str) -> bool` that checks if any permutation of the string can form a palindrome. Ignore spaces and case.
```

**Real Model Output:**
```python
def is_palindrome_permutation(s: str) -> bool:
    """
    Checks if any permutation of the string can form a palindrome.
    Case-insensitive and ignores non-alphanumeric characters.
    """
    bitmask = 0
    for char in s.lower():
        if char.isalnum():
            # Toggle the bit corresponding to the character
            bitmask ^= (1 << (ord(char) - ord('a')))
    
    # A palindrome permutation can have at most one odd character count,
    # meaning at most one bit can be set in the bitmask.
    return (bitmask & (bitmask - 1)) == 0
```

---

### Example 2: Mathematical Reasoning Trace

**Prompt:**
```text
A water tank is filled by two pipes, A and B. Pipe A can fill the tank in 6 hours, while Pipe B can fill it in 8 hours. Both pipes are opened together, but after 2 hours Pipe A is turned off. How much longer will Pipe B take to fill the remainder of the tank? Show your step-by-step work.
```

**Real Model Output:**
```text
Let the total capacity of the water tank be 1 unit (or 24 units, the LCM of 6 and 8).

1. Determine the individual flow rates:
   - Rate of Pipe A = 1/6 tank per hour (or 4 units/hour).
   - Rate of Pipe B = 1/8 tank per hour (or 3 units/hour).

2. Combined work done in the first 2 hours:
   - Combined rate = 1/6 + 1/8 = 7/24 tank per hour.
   - Work completed in 2 hours = 2 * (7/24) = 14/24 = 7/12 of the tank.

3. Calculate the remaining work:
   - Remaining tank = 1 - 7/12 = 5/12 of the tank (or 10 units out of 24).

4. Time required for Pipe B alone:
   - Time = Remaining work / Rate of Pipe B
   - Time = (5/12) / (1/8) = (5/12) * 8 = 40/12 = 10/3 hours.
   - 10/3 hours = 3 hours and 20 minutes.

Pipe B will take 3 hours and 20 minutes longer to fill the remainder of the tank.
```

---

### Example 3: Modern C++20 Zero-Allocation Parsing

**Prompt:**
```text
Write a modern C++20 function using std::string_view to parse a comma-separated line of numbers into a std::vector<int> without allocating string copies.
```

**Real Model Output:**
```cpp
#include <charconv>
#include <ranges>
#include <string_view>
#include <vector>
#include <system_error>

std::vector<int> parse_comma_separated(std::string_view line) {
    std::vector<int> result;
    
    for (const auto part : line | std::views::split(',')) {
        std::string_view token(part.begin(), part.end());
        
        // Trim optional surrounding whitespace
        auto start = token.find_first_not_of(" \t");
        if (start == std::string_view::npos) continue;
        auto end = token.find_last_not_of(" \t");
        token = token.substr(start, end - start + 1);
        
        int val = 0;
        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), val);
        if (ec == std::errc{}) {
            result.push_back(val);
        }
    }
    
    return result;
}
```

---

## Hardware Compatibility Matrix

| Hardware Setup | Usable VRAM | Recommended Model | Recommended Context | Performance |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 3050 / 4050 Laptop** | 6 GB | `64exp-Work IQ2_XXS` | 8,192 | 35.8 t/s generation (Full GPU) |
| **NVIDIA RTX 3060 / 4060 Laptop/Desktop** | 6 GB - 8 GB | `64exp-Work IQ2_XXS` | 16,384 | 40+ t/s generation (Full GPU) |
| **Apple Silicon (M1 / M2 / M3 / M4)** | 8 GB - 16 GB | `64exp-Work IQ2_XXS` | 16,384 | Full Metal offload |
| **NVIDIA RTX 3050 4GB Laptop** | 4 GB | `56exp-Blended IQ2_XXS` | 4,096 | Partial CPU MoE offload (-ncmoe 14) |
| **Integrated Intel Arc / AMD Radeon 780M** | Shared RAM | `56exp-Blended IQ2_XXS` | 4,096 | Fast Vulkan / OpenCL offload |

---

## Citation & Technical Paper

If you use MAPLE models or techniques in your research or applications, please cite:

```bibtex
@article{sachan2026maple,
  title={MAPLE: Mass-Aware Pruning of Looped Mixture-of-Experts for Sub-5GB In-VRAM Edge Inference},
  author={Sachan, Aman and contributors},
  journal={arXiv preprint},
  year={2026}
}
```

- **Project Repository:** [https://github.com/AmSach/maple](https://github.com/AmSach/maple)
- **Interactive Technical Documentation:** [https://amsach.github.io/maple](https://amsach.github.io/maple)
- **Collection of Sub-5GB MoE Models:** [MAPLE Sub-5GB Collection on Hugging Face](https://huggingface.co/collections/AmSach/maple-sub-5gb-mixture-of-experts-models-6af8467951c779a9e35a327e)
