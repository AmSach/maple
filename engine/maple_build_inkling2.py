"""
MAPLE Binary GGUF Container Slicer for Inkling-2-Mini (inkling2_loop architecture).

Streams raw quantized tensor slices directly from the source GGUF without
dequantizing or re-encoding, supporting all GGML block types (including IQ1/IQ2/IQ4/K-quants)
regardless of Python `gguf` enum version. Slices all axis-0 expert tensors cleanly:
  - blk.{l}.ffn_down_exps.weight   [d_ff, d, E] -> [d_ff, d, k']
  - blk.{l}.ffn_down_exps.scale    [E]          -> [k']
  - blk.{l}.ffn_gate_up_exps.weight [d, 2*d_ff, E] -> [d, 2*d_ff, k']
  - blk.{l}.ffn_gate_up_exps.scale  [E]          -> [k']
  - blk.{l}.ffn_gate_inp.weight    [d, E]       -> [d, k']
  - blk.{l}.exp_probs_b.bias       [E]          -> [k']
Preserves all loop controllers, attention projections, conv layers, norms, and tokenizer verbatim.
"""

import argparse
import json
import os
import struct
import time

EXPERT_TENSOR_PATTERNS = (
    "ffn_down_exps.weight",
    "ffn_down_exps.scale",
    "ffn_gate_up_exps.weight",
    "ffn_gate_up_exps.scale",
    "ffn_gate_inp.weight",
    "exp_probs_b.bias",
)


class _Cursor:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def unpack(self, fmt: str):
        sz = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, self.buf, self.off)
        self.off += sz
        return val[0] if len(val) == 1 else val

    def read_str(self) -> str:
        n = self.unpack("<Q")
        s = self.buf[self.off : self.off + n].decode("utf-8", errors="replace")
        self.off += n
        return s

    def skip_val(self, vt: int):
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vt in fixed:
            self.off += fixed[vt]
        elif vt == 8:
            n = self.unpack("<Q")
            self.off += n
        elif vt == 9:
            et = self.unpack("<I")
            cnt = self.unpack("<Q")
            if et in fixed:
                self.off += fixed[et] * cnt
            else:
                for _ in range(cnt):
                    self.skip_val(et)
        else:
            raise ValueError(f"Unsupported GGUF value type: {vt}")

    def read_val(self, vt: int):
        fmt_map = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
        if vt in fmt_map:
            v = self.unpack(fmt_map[vt])
            return bool(v) if vt == 7 else v
        if vt == 8:
            return self.read_str()
        if vt == 9:
            et = self.unpack("<I")
            cnt = self.unpack("<Q")
            return [self.read_val(et) for _ in range(cnt)]
        raise ValueError(f"Unsupported GGUF value type: {vt}")


def _pack_kv_str(key: str, val: str) -> bytes:
    kb = key.encode("utf-8")
    vb = val.encode("utf-8")
    return struct.pack("<QI", len(kb), 8) + kb + struct.pack("<Q", len(vb)) + vb


def _pack_kv_uint32(key: str, val: int) -> bytes:
    kb = key.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<II", 4, val)


def _pack_tensor_info(name: str, dims: list[int], ttype: int, rel_off: int) -> bytes:
    nb = name.encode("utf-8")
    out = bytearray()
    out += struct.pack("<Q", len(nb)) + nb
    out += struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<IQ", ttype, rel_off)
    return bytes(out)


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def build_inkling2_maple(src_path: str, map_path: str, dst_path: str, source_experts: int = 128, target_experts: int = 56):
    t0 = time.time()
    file_size = os.path.getsize(src_path)
    with open(map_path, "r", encoding="utf-8") as f:
        layer_experts = json.load(f)

    # Read up to 64 MiB prefix containing header + KV + tensor info table
    read_sz = min(file_size, 64 * 1024 * 1024)
    with open(src_path, "rb") as f:
        hdr_buf = f.read(read_sz)

    cur = _Cursor(hdr_buf)
    magic = hdr_buf[:4]
    if magic != b"GGUF":
        raise ValueError(f"Invalid GGUF magic: {magic!r}")
    cur.off = 4
    version = cur.unpack("<I")
    n_tensors = cur.unpack("<Q")
    n_kv = cur.unpack("<Q")

    kv_entries = []
    alignment = 32
    arch = "inkling2_loop"
    for _ in range(n_kv):
        st = cur.off
        key = cur.read_str()
        vt = cur.unpack("<I")
        if key in ("general.alignment", "general.architecture"):
            val = cur.read_val(vt)
            if key == "general.alignment":
                alignment = int(val)
            else:
                arch = str(val)
        else:
            cur.skip_val(vt)
        en = cur.off
        kv_entries.append((key, hdr_buf[st:en]))

    tinfos = []
    for _ in range(n_tensors):
        name = cur.read_str()
        nd = cur.unpack("<I")
        dims = [cur.unpack("<Q") for _ in range(nd)]
        ttype = cur.unpack("<I")
        rel_off = cur.unpack("<Q")
        tinfos.append((name, dims, ttype, rel_off))

    src_data_start = _align_up(cur.off, alignment)
    by_off = sorted(tinfos, key=lambda x: x[3])
    src_spans = {}
    for idx, (nm, _, _, ro) in enumerate(by_off):
        nxt = by_off[idx + 1][3] if idx + 1 < len(by_off) else (file_size - src_data_start)
        src_spans[nm] = nxt - ro

    # Build output KV bytes
    out_kv = bytearray()
    for key, raw_kv in kv_entries:
        if key == f"{arch}.expert_count":
            kb = key.encode("utf-8")
            out_kv += struct.pack("<Q", len(kb)) + kb + struct.pack("<II", 4, target_experts)
        elif key == "general.name":
            kb = key.encode("utf-8")
            vb = f"Inkling-2-Mini-MAPLE-{target_experts}exp".encode("utf-8")
            out_kv += struct.pack("<Q", len(kb)) + kb + struct.pack("<IQ", 8, len(vb)) + vb
        elif key == "general.size_label":
            kb = key.encode("utf-8")
            vb = f"{target_experts}x1.8B".encode("utf-8")
            out_kv += struct.pack("<Q", len(kb)) + kb + struct.pack("<IQ", 8, len(vb)) + vb
        else:
            out_kv += raw_kv

    # Compute exact unpadded tensor byte size rules & output offsets
    out_tinfos = []
    cur_rel_off = 0
    for nm, dims, ttype, src_ro in tinfos:
        parts = nm.split(".")
        l_str = parts[1] if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit() else None
        is_exp_sliced = (
            l_str in layer_experts
            and len(dims) >= 1
            and dims[-1] == source_experts
            and any(pat in nm for pat in EXPERT_TENSOR_PATTERNS)
        )
        src_padded_bytes = src_spans[nm]
        if is_exp_sliced:
            # Verify whether unpadded byte count is smaller for 1D F32 tensors (e.g. [128] -> 512B, always multiple of 32)
            # For 2D/3D weights, source_experts divides evenly cleanly (e.g. dims[-1]=128)
            if src_padded_bytes % source_experts != 0:
                raise ValueError(f"Span {src_padded_bytes} not divisible by {source_experts} for {nm} ({dims})")
            exp_stride = src_padded_bytes // source_experts
            new_dims = list(dims[:-1]) + [target_experts]
            out_bytes = exp_stride * target_experts
            selected = layer_experts[l_str]
            if len(selected) != target_experts:
                raise ValueError(f"Layer {l_str} has {len(selected)} experts, expected {target_experts}")
            out_tinfos.append((nm, new_dims, ttype, cur_rel_off, src_ro, out_bytes, exp_stride, selected))
        else:
            out_tinfos.append((nm, dims, ttype, cur_rel_off, src_ro, src_padded_bytes, 0, None))
        cur_rel_off = _align_up(cur_rel_off + out_tinfos[-1][5], alignment)

    # Assemble header + tensor table
    out_hdr = bytearray()
    out_hdr += struct.pack("<4sIQQ", b"GGUF", version, n_tensors, n_kv)
    out_hdr += out_kv
    for nm, dims, ttype, ro, _, _, _, _ in out_tinfos:
        out_hdr += _pack_tensor_info(nm, dims, ttype, ro)

    out_data_start = _align_up(len(out_hdr), alignment)
    if len(out_hdr) < out_data_start:
        out_hdr += b"\x00" * (out_data_start - len(out_hdr))

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    tmp_dst = dst_path + ".tmp"
    if os.path.exists(tmp_dst):
        os.remove(tmp_dst)

    print(f"[*] Slicing {src_path} -> {dst_path} ({source_experts} -> {target_experts} experts/layer)...", flush=True)
    BUF = 16 * 1024 * 1024
    with open(src_path, "rb") as fin, open(tmp_dst, "wb") as fout:
        fout.write(out_hdr)
        written_rel = 0
        for idx, (nm, dims, ttype, ro, src_ro, out_bytes, exp_stride, selected) in enumerate(out_tinfos):
            if ro > written_rel:
                fout.write(b"\x00" * (ro - written_rel))
                written_rel = ro
            if selected is not None:
                base_off = src_data_start + src_ro
                for exp_id in selected:
                    fin.seek(base_off + exp_id * exp_stride)
                    data = fin.read(exp_stride)
                    if len(data) != exp_stride:
                        raise IOError(f"Short read on {nm} exp {exp_id}")
                    fout.write(data)
                written_rel += out_bytes
            else:
                fin.seek(src_data_start + src_ro)
                rem = out_bytes
                while rem > 0:
                    chunk = fin.read(min(BUF, rem))
                    if not chunk:
                        raise IOError(f"Unexpected EOF reading {nm}")
                    fout.write(chunk)
                    rem -= len(chunk)
                written_rel += out_bytes
            if (idx + 1) % 150 == 0 or idx == len(out_tinfos) - 1:
                print(f"  Progress: [{idx+1}/{len(out_tinfos)}] tensors | {(out_data_start + written_rel)/1e9:.3f} GB", flush=True)

    if os.path.exists(dst_path):
        os.remove(dst_path)
    os.rename(tmp_dst, dst_path)
    final_sz = os.path.getsize(dst_path)
    print(f"[+] Built MAPLE container: {dst_path} ({final_sz:,} B = {final_sz/1e9:.4f} GB = {final_sz/(1024**3):.4f} GiB) in {time.time()-t0:.1f}s")
    return final_sz


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-experts", type=int, default=128)
    p.add_argument("--target-experts", type=int, default=56)
    args = p.parse_args()
    build_inkling2_maple(args.source, args.map, args.output, args.source_experts, args.target_experts)
