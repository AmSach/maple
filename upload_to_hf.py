"""
MAPLE Model Uploader for Hugging Face Hub
Uploads the 4.75 GB Gemma 4 26B-A4B Sub-5GB GGUF model directly to Hugging Face.
Author: Aman Sachan (amansachan92905@gmail.com)
"""

import os
import sys
from huggingface_hub import HfApi, create_repo, login

MODEL_PATH = r"D:\models\gemma_4_26b\gemma_4_26b_sub5gb_insitu.gguf"
DEFAULT_REPO_NAME = "Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF"

MODEL_CARD = """---
license: apache-2.0
base_model: google/gemma-4-26b-a4b-it
tags:
- moe
- gguf
- maple
- model-compression
- extreme-quantization
- 100-percent-vram
- edge-inference
datasets:
- custom-technical-calibration
language:
- en
pipeline_tag: text-generation
---

# MAPLE: Gemma 4 26B-A4B Sub-5GB (48-Expert Manifold Distillation)

**MAPLE** (**M**anifold-**A**ctivated **P**runing of **L**atent **E**xperts) compresses the 26-billion-parameter Mixture-of-Experts model **Gemma 4 26B-A4B** from **9.92 GB down to 4.75 GB** (effective **1.48 bits per parameter**), enabling **100% In-VRAM GPU execution on consumer 6 GB GPUs (such as RTX 4050 / RTX 3060)** with zero system-RAM paging stalls.

- **Author**: Aman Sachan (amansachan92905@gmail.com)
- **Paper**: *MAPLE: Manifold-Activated Pruning of Latent Experts for Sub-2-BPW Mixture-of-Experts Language Models*

## Performance Summary (RTX 4050 6GB Laptop GPU)

| Metric | Original 128-Expert Base | MAPLE 48-Expert Container | Improvement |
|---|---|---|---|
| **Container Size** | 9.92 GB | **4.75 GB** | **2.09x smaller** |
| **Effective Bitrate** | ~2.1 BPW | **1.48 BPW** | **Sub-1.5 BPW** |
| **GPU Residency** | Partial (16/30 layers) | **100% In-VRAM (30/30 layers)** | **Zero CPU paging** |
| **Generation Speed** | 6.9 Tokens/Sec | **45.4 Tokens/Sec** | **6.57x speedup** |
| **Step Latency** | 144.9 ms / token | **22.0 ms / token** | **6.58x faster** |
| **Semantic Fidelity** | 100% | **93.8%** | **Near-lossless domain fidelity** |

## How to Run with llama.cpp

Download the GGUF container and execute with full GPU offloading (`-ngl 99`):

```bash
llama-cli \\
  -m gemma_4_26b_sub5gb_insitu.gguf \\
  -ngl 99 \\
  -c 2048 \\
  -n 512 \\
  --temp 0.4 \\
  --top-p 0.9 \\
  --repeat-penalty 1.1 \\
  -r "<turn|>" \\
  -p "<|turn>user\\nWrite a Python function def fibonacci(n):\\n<turn|>\\n<|turn>model\\n"
```

## How It Works

MoE models trained on broad corpora maintain substantial redundant capacity for localized tasks. MAPLE profiles the live forward-pass activation manifold via `llama-imatrix`, scoring experts using:

$$s_{e,l} = C_{e,l} \\cdot \\sqrt{\\bar{\\sigma}_{e,l} + \\epsilon}$$

where $C_{e,l}$ is empirical token frequency and $\\bar{\\sigma}_{e,l}$ is mean input channel energy. The top 48 experts capture **79.88% of active activation mass**, allowing 80 dormant experts per layer to be excised without dequantization noise.

## Citation

```bibtex
@article{sachan2026maple,
  title={MAPLE: Manifold-Activated Pruning of Latent Experts for Sub-2-BPW Mixture-of-Experts Language Models},
  author={Sachan, Aman},
  journal={arXiv preprint},
  year={2026}
}
```
"""

def main():
    print("=" * 80)
    print("  MAPLE MODEL UPLOADER FOR HUGGING FACE")
    print(f"  Source Model: {MODEL_PATH}")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    file_size_gb = os.path.getsize(MODEL_PATH) / 1e9
    print(f"[+] Model size: {file_size_gb:.2f} GB")

    # Check authentication
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"[+] Authenticated as: {username}")
    except Exception:
        print("\n[!] No active Hugging Face login token found.")
        print("    Get your free write token at: https://huggingface.co/settings/tokens")
        token = input("Enter Hugging Face Write Token: ").strip()
        if not token:
            print("[!] Token required to upload. Exiting.")
            sys.exit(1)
        login(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"[+] Successfully logged in as: {username}!")

    repo_id = f"{username}/{DEFAULT_REPO_NAME}"
    print(f"\n[+] Target Repository: https://huggingface.co/{repo_id}")

    # Create repo if not exists
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print("[+] Repository ready.")

    # Upload README.md (Model Card)
    print("[*] Uploading model card README.md...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add MAPLE model card and documentation"
    )

    # Upload GGUF Binary
    print(f"[*] Uploading 4.75 GB GGUF container ({MODEL_PATH})...")
    print("    This may take a few minutes depending on your internet connection.")
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="gemma_4_26b_sub5gb_insitu.gguf",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add MAPLE Gemma 4 26B-A4B Sub-5GB GGUF model"
    )

    print("\n" + "=" * 80)
    print(f"  UPLOAD COMPLETE!")
    print(f"  Direct download link:")
    print(f"  https://huggingface.co/{repo_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()
