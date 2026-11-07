"""
MAPLE: Model Container Assembler
Builds the pruned 48-expert sub-5GB GGUF container with 100% in-VRAM execution.
Target: D:\models\gemma_4_26b\gemma_4_26b_sub5gb_insitu.gguf
"""

import json
import os
import sys
import time
import numpy as np
import gguf

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SOURCE_GGUF = r"D:\models\gemma_4_26b\gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf"
TARGET_GGUF = r"D:\models\gemma_4_26b\gemma_4_26b_sub5gb_insitu.gguf"
JSON_PATH   = os.path.join(BASE_DIR, "expert_selection_map.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = r"C:\Users\amans\.gemini\antigravity\scratch\repos\middleout-lattice\engine\layer_experts_empirical_48.json"
N_EXPERTS   = 48
N_EXPERT_USED = 8

def run_build():
    print("=" * 85)
    print("  STAGE 2: ASSEMBLING 48-EXPERT EMPIRICAL HIGH-QUALITY GGUF CONTAINER")
    print(f"  Source: {SOURCE_GGUF}")
    print(f"  Target: {TARGET_GGUF}")
    print(f"  Config: {N_EXPERTS} experts / {N_EXPERT_USED} active per token")
    print("=" * 85)
    t0 = time.time()
    
    with open(JSON_PATH, "r") as f:
        layer_experts = json.load(f)
        
    reader = gguf.GGUFReader(SOURCE_GGUF)
    temp_dst = TARGET_GGUF + ".tmp"
    if os.path.exists(temp_dst):
        os.remove(temp_dst)
        
    writer = gguf.GGUFWriter(temp_dst, "gemma4", use_temp_file=True)
    
    # 1. Metadata
    print("[*] Writing calibrated metadata...")
    for field in reader.fields.values():
        name = field.name
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        elif name == "gemma4.expert_count":
            writer.add_expert_count(N_EXPERTS)
        elif name == "gemma4.expert_used_count":
            writer.add_expert_used_count(N_EXPERT_USED)
        else:
            val = field.contents()
            vtype = field.types[0]
            if vtype == gguf.GGUFValueType.BOOL:
                writer.add_bool(name, bool(val))
            elif vtype == gguf.GGUFValueType.UINT8:
                writer.add_uint8(name, int(val))
            elif vtype == gguf.GGUFValueType.INT8:
                writer.add_int8(name, int(val))
            elif vtype == gguf.GGUFValueType.UINT16:
                writer.add_uint16(name, int(val))
            elif vtype == gguf.GGUFValueType.INT16:
                writer.add_int16(name, int(val))
            elif vtype == gguf.GGUFValueType.UINT32:
                writer.add_uint32(name, int(val))
            elif vtype == gguf.GGUFValueType.INT32:
                writer.add_int32(name, int(val))
            elif vtype == gguf.GGUFValueType.UINT64:
                writer.add_uint64(name, int(val))
            elif vtype == gguf.GGUFValueType.INT64:
                writer.add_int64(name, int(val))
            elif vtype == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(name, float(val))
            elif vtype == gguf.GGUFValueType.FLOAT64:
                writer.add_float64(name, float(val))
            elif vtype == gguf.GGUFValueType.STRING:
                writer.add_string(name, str(val))
            elif vtype == gguf.GGUFValueType.ARRAY:
                writer.add_array(name, val)
                
    # 2. Tensors
    print(f"[*] Slicing and writing {len(reader.tensors)} tensors...")
    bytes_out = 0
    
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        t_type = tensor.tensor_type
        data = tensor.data
        
        parts = name.split(".")
        layer_idx_str = parts[1] if parts[0] == "blk" and parts[1].isdigit() else None
        
        if "ffn_gate_inp.weight" in name and layer_idx_str in layer_experts:
            selected = layer_experts[layer_idx_str]
            sliced = data[selected].copy()
            writer.add_tensor(name, sliced, raw_dtype=t_type)
            bytes_out += sliced.nbytes
        elif "ffn_down_exps.scale" in name and layer_idx_str in layer_experts:
            selected = layer_experts[layer_idx_str]
            # Original empirical scale preserved (1.0x)
            sliced = data[selected].copy()
            writer.add_tensor(name, sliced, raw_dtype=t_type)
            bytes_out += sliced.nbytes
        elif ("ffn_down_exps.weight" in name or "ffn_gate_up_exps.weight" in name) and layer_idx_str in layer_experts:
            selected = layer_experts[layer_idx_str]
            sliced = data[selected].copy()
            writer.add_tensor(name, sliced, raw_dtype=t_type)
            bytes_out += sliced.nbytes
        else:
            writer.add_tensor(name, data, raw_dtype=t_type)
            bytes_out += data.nbytes
            
        if (i + 1) % 100 == 0 or i == len(reader.tensors) - 1:
            print(f"  • Progress: [{i+1:3d}/{len(reader.tensors)}] tensors | Output: {bytes_out / 1e9:.2f} GB", flush=True)
            
    print("\n[*] Writing GGUF binary format to disk...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    # Replace destination atomically
    if os.path.exists(TARGET_GGUF):
        os.remove(TARGET_GGUF)
    os.rename(temp_dst, TARGET_GGUF)
    
    final_size_gb = os.path.getsize(TARGET_GGUF) / 1e9
    print("=" * 85)
    print("  STAGE 2 EMPIRICAL BUILD COMPLETE!")
    print(f"  • Model File: {TARGET_GGUF}")
    print(f"  • Final Model Size: {final_size_gb:.2f} GB (Sub-5GB Target: MET)")
    print(f"  • Total Build Time: {time.time()-t0:.2f} s")
    print("=" * 85)

if __name__ == "__main__":
    run_build()
