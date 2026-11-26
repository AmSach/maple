"""
MAPLE DeepSeek-Coder-V2-Lite Control Container Assembler
Cleanly slices DeepSeek-Coder-V2-Lite-Instruct.IQ2_XXS.gguf (64 routed + 2 shared experts)
to any target k' using MAPLE (beta=1/3), Beta sweep, Frequency-Only (beta=0), Pure Energy (E_bar),
Cumulative Energy (sigma_bar = C*E_bar), Router Column-Norm, Gate Prob Mass, REAP, or Random Drop.
Preserves anonymous general.* metadata (no personal identifiers).
"""

import os
import json
import time
import numpy as np
import gguf

DEFAULT_SRC = r"D:\models\deepseek_source\DeepSeek-Coder-V2-Lite-Instruct.IQ2_XXS.gguf"
DEFAULT_IMATRIX = r"C:\Users\amans\.gemini\antigravity\brain\6a0067ca-8615-42cc-948d-72d2d8f5f3dc\scratch\eval_data\cal_corpora\cal_mixed_2k.imatrix.gguf"


def compute_selection_map(
    method: str,
    target_k: int,
    imatrix_path: str = DEFAULT_IMATRIX,
    source_gguf: str = DEFAULT_SRC,
    seed: int = 42,
    tensor_key: str = "ffn_down_exps",
    custom_scores_json: str = None,
):
    """
    Compute per-layer expert indices (sorted ascending, monotonic mapping) for layers 1..26.
    """
    layer_experts = {}
    stats = {"method": method, "target_k": target_k, "imatrix": imatrix_path, "tensor_key": tensor_key, "layers": {}}

    if custom_scores_json is not None:
        with open(custom_scores_json, "r", encoding="utf-8") as f:
            cscores = json.load(f)
        for l in range(1, 27):
            sc = np.array(cscores[str(l)], dtype=np.float64)
            ranked = np.argsort(sc)[::-1]
            selected = sorted(int(x) for x in ranked[:target_k])
            layer_experts[str(l)] = selected
        return layer_experts, stats

    if method == "random":
        rng = np.random.default_rng(seed)
        for l in range(1, 27):
            chosen = rng.choice(64, size=target_k, replace=False)
            layer_experts[str(l)] = sorted(int(x) for x in chosen)
        return layer_experts, stats

    if method == "router_norm":
        reader = gguf.GGUFReader(source_gguf)
        tmap = {t.name: t for t in reader.tensors}
        for l in range(1, 27):
            w = tmap[f"blk.{l}.ffn_gate_inp.weight"].data.astype(np.float64)
            norms = np.linalg.norm(w, axis=1) if w.shape[0] == 64 else np.linalg.norm(w, axis=0)
            ranked = np.argsort(norms)[::-1]
            layer_experts[str(l)] = sorted(int(x) for x in ranked[:target_k])
        return layer_experts, stats

    if method == "layer0_proxy":
        import subprocess
        tok_exe = r"D:\tools\llama-cpp\llama-tokenize.exe"
        cal_txt = r"C:\Users\amans\.gemini\antigravity\brain\6a0067ca-8615-42cc-948d-72d2d8f5f3dc\scratch\eval_data\cal_corpora_clean\cal_mixed_2k.txt"
        p = subprocess.run([tok_exe, "-m", source_gguf, "-f", cal_txt, "--ids"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        tids = json.loads(p.stdout.strip().splitlines()[-1])
        reader = gguf.GGUFReader(source_gguf)
        tmap = {t.name: t for t in reader.tensors}
        emb_t = tmap["token_embd.weight"]
        emb = gguf.quants.dequantize(emb_t.data, emb_t.tensor_type).astype(np.float64)
        if emb.shape[0] != 102400:
            emb = emb.T
        x_emb = emb[np.array(tids, dtype=np.int64)]  # [T, 2048]
        for l in range(1, 27):
            w = tmap[f"blk.{l}.ffn_gate_inp.weight"].data.astype(np.float64)
            if w.shape[0] != 64:
                w = w.T  # [64, 2048]
            logits = x_emb @ w.T  # [T, 64]
            top6 = np.argsort(logits, axis=1)[:, -6:]
            counts = np.bincount(top6.reshape(-1), minlength=64).astype(np.float64)
            ranked = np.argsort(counts)[::-1]
            layer_experts[str(l)] = sorted(int(x) for x in ranked[:target_k])
        del emb
        return layer_experts, stats

    ir = gguf.GGUFReader(imatrix_path)
    tmap = {t.name: t for t in ir.tensors}
    covs = []
    for l in range(1, 27):
        ck = f"blk.{l}.{tensor_key}.weight.counts"
        ek = f"blk.{l}.{tensor_key}.weight.in_sum2"
        if ck not in tmap:
            ck = f"blk.{l}.ffn_gate_exps.weight.counts"
            ek = f"blk.{l}.ffn_gate_exps.weight.in_sum2"
        c = tmap[ck].data.flatten().astype(np.float64)
        s2 = tmap[ek].data.astype(np.float64)
        ax = 1 if s2.shape[0] == 64 else 0
        sigma_bar = np.mean(s2, axis=ax)  # cumulative mean channel energy = C * E_bar
        e_bar = np.where(c > 0, sigma_bar / np.maximum(c, 1.0), 0.0)

        if method == "maple":
            score = c * np.sqrt(sigma_bar + 1e-8)
        elif method == "freq":
            score = c.copy()
        elif method == "cum_energy":
            score = sigma_bar.copy()
        elif method == "energy_only":
            score = e_bar.copy()
        elif method.startswith("beta_"):
            beta = float(method.split("_", 1)[1])
            score = c * ((e_bar + 1e-8) ** beta)
        else:
            raise ValueError(f"Unknown method: {method}")

        ranked = np.argsort(score)[::-1]
        selected = sorted(int(x) for x in ranked[:target_k])
        layer_experts[str(l)] = selected
        cov = float(c[selected].sum() / c.sum() * 100.0) if c.sum() > 0 else 100.0
        covs.append(cov)
        stats["layers"][str(l)] = {"cov_pct": round(cov, 4), "calls": int(c.sum())}

    stats["mean_cov_pct"] = round(float(np.mean(covs)), 4)
    return layer_experts, stats


def slice_deepseek_container(
    source_gguf: str,
    target_gguf: str,
    layer_experts: dict,
    target_k: int,
    model_name: str,
    reader=None,
):
    t0 = time.time()
    if reader is None:
        reader = gguf.GGUFReader(source_gguf)
    arch = "deepseek2"

    os.makedirs(os.path.dirname(target_gguf), exist_ok=True)
    temp_dst = target_gguf + ".tmp"
    if os.path.exists(temp_dst):
        os.remove(temp_dst)

    writer = gguf.GGUFWriter(temp_dst, arch, use_temp_file=False)

    for field in reader.fields.values():
        name = field.name
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        elif name == f"{arch}.expert_count":
            writer.add_expert_count(target_k)
        elif name == f"{arch}.expert_used_count":
            writer.add_expert_used_count(6)
        elif name == "general.name":
            writer.add_name(model_name)
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

    for tensor in reader.tensors:
        name = tensor.name
        t_type = tensor.tensor_type
        data = tensor.data
        parts = name.split(".")
        l_str = parts[1] if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit() else None
        if l_str in layer_experts and any(k in name for k in ("ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_gate_inp.weight")):
            selected = layer_experts[l_str]
            sliced = data[selected].copy()
            writer.add_tensor(name, sliced, raw_dtype=t_type)
        else:
            writer.add_tensor(name, data, raw_dtype=t_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if os.path.exists(target_gguf):
        os.remove(target_gguf)
    os.rename(temp_dst, target_gguf)
    sz = os.path.getsize(target_gguf)
    print(f"[slice] Built {os.path.basename(target_gguf)} ({sz:,} B / {sz/1e9:.3f} GB) in {time.time()-t0:.1f}s", flush=True)
    return sz
