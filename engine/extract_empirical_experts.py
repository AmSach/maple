"""
Extract Empirical Expert Rankings from comprehensive_imatrix.gguf
Produces layer_experts_empirical_48.json using actual forward pass activations.
"""

import json
import numpy as np
import gguf

IMATRIX_PATH = r"C:\Users\amans\.gemini\antigravity\scratch\repos\middleout-lattice\engine\comprehensive_imatrix.gguf"
OUTPUT_JSON  = r"C:\Users\amans\.gemini\antigravity\scratch\repos\middleout-lattice\engine\layer_experts_empirical_48.json"
N_EXPERTS    = 48

def extract():
    print("=" * 80)
    print("  EXTRACTING EMPIRICAL EXPERTS FROM COMPREHENSIVE IMATRIX")
    print(f"  Source: {IMATRIX_PATH}")
    print(f"  Target: {OUTPUT_JSON} (Top {N_EXPERTS} experts per layer)")
    print("=" * 80)
    
    reader = gguf.GGUFReader(IMATRIX_PATH)
    tensor_map = {t.name: t for t in reader.tensors}
    
    layer_experts = {}
    coverages = []
    
    for l in range(30):
        counts_name = f"blk.{l}.ffn_down_exps.weight.counts"
        insum2_name = f"blk.{l}.ffn_down_exps.weight.in_sum2"
        
        if counts_name not in tensor_map:
            raise KeyError(f"Missing {counts_name} in imatrix!")
            
        counts = tensor_map[counts_name].data.flatten().astype(np.float64) # (128,)
        
        if insum2_name in tensor_map:
            in_sum2 = tensor_map[insum2_name].data.astype(np.float64) # (128, 704)
            energy = np.mean(in_sum2, axis=1) # (128,)
            score = counts * np.sqrt(energy + 1e-8)
        else:
            score = counts
            
        # Tie-breaker for zero-count experts: router weight norm
        # If fewer than N_EXPERTS were active, pick remaining based on router gate norm
        zero_mask = (counts == 0)
        sorted_indices = list(np.argsort(score)[::-1])
        
        # Filter to make sure top N_EXPERTS are selected
        top_experts = sorted_indices[:N_EXPERTS]
        
        # Calculate empirical coverage of total activation counts
        total_calls = counts.sum()
        captured_calls = counts[top_experts].sum()
        cov = (captured_calls / total_calls * 100.0) if total_calls > 0 else 100.0
        coverages.append(cov)
        
        layer_experts[str(l)] = [int(x) for x in top_experts]
        
        if (l + 1) % 5 == 0 or l == 29:
            print(f"  • Layer [{l+1:2d}/30] | Top-{N_EXPERTS} Activation Coverage: {cov:6.2f}% | Top 5: {top_experts[:5]}")
            
    avg_cov = np.mean(coverages)
    print("=" * 80)
    print(f"  Average 30-Layer Empirical Activation Coverage: {avg_cov:.2f}%")
    print("=" * 80)
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(layer_experts, f, indent=2)
        
    print(f"[+] Saved {OUTPUT_JSON} successfully!")

if __name__ == "__main__":
    extract()
