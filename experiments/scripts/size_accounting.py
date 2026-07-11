"""Checkpoint residual compression size accounting.

Measures REAL serialized bytes of the residual stream under our pipeline:
  delta = ckpt_t - ckpt_{t-1}  ->  top-|score| mask at ratio p
  survivors quantized to INT4 (symmetric per-tensor)  ->  bitmap + packed nibbles
  optional zlib entropy coding on the serialized stream.

Reports compression ratio vs fp32 and fp16 raw state_dict.
Size is score-agnostic (same p + format => same bytes for ExCP/Ours);
quality at each p comes from the fault-tolerant experiments.
"""
import argparse
import io
import zlib
import torch
import numpy as np


def load_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def int4_pack(values):
    """Symmetric per-tensor INT4 quantization, packed 2 nibbles/byte."""
    if values.numel() == 0:
        return b'', 0.0
    scale = values.abs().max().item() / 7.0 or 1.0
    q = torch.clamp(torch.round(values / scale), -8, 7).to(torch.int8) + 8
    arr = q.numpy().astype(np.uint8)
    if len(arr) % 2:
        arr = np.append(arr, 0)
    packed = (arr[0::2] << 4) | arr[1::2]
    return packed.tobytes(), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_new', required=True)
    ap.add_argument('--ckpt_old', required=True)
    ap.add_argument('--ratios', default='0.5,0.7,0.9')
    ap.add_argument('--dtype', default='int4', choices=['int4', 'fp16'])
    args = ap.parse_args()

    new_s, old_s = load_state(args.ckpt_new), load_state(args.ckpt_old)
    deltas = {}
    for name, w in new_s.items():
        if not torch.is_tensor(w) or not w.is_floating_point():
            continue
        ref = old_s.get(name)
        if ref is None or ref.shape != w.shape:
            continue
        deltas[name] = (w.float() - ref.float()).flatten()

    n_params = sum(d.numel() for d in deltas.values())
    raw32, raw16 = n_params * 4, n_params * 2
    print(f"params covered: {n_params/1e6:.1f}M | raw fp32 {raw32/1e6:.1f} MB | fp16 {raw16/1e6:.1f} MB")

    for p in [float(x) for x in args.ratios.split(',')]:
        payload = io.BytesIO()
        for name, d in deltas.items():
            k = int(d.numel() * p)
            if k > 0:
                thr = torch.kthvalue(d.abs(), k).values
                mask = d.abs() > thr
            else:
                mask = torch.ones_like(d, dtype=torch.bool)
            bitmap = np.packbits(mask.numpy())
            if args.dtype == 'int4':
                vals, scale = int4_pack(d[mask])
            else:
                vals, scale = d[mask].to(torch.float16).numpy().tobytes(), 1.0
            payload.write(bitmap.tobytes())
            payload.write(vals)
            payload.write(np.float32(scale).tobytes())
        raw_bytes = payload.getbuffer().nbytes
        z_bytes = len(zlib.compress(payload.getvalue(), 6))
        print(f"p={p}: serialized {raw_bytes/1e6:.1f} MB ({raw32/raw_bytes:.1f}x vs fp32, "
              f"{raw16/raw_bytes:.1f}x vs fp16) | +zlib {z_bytes/1e6:.1f} MB "
              f"({raw32/z_bytes:.1f}x vs fp32, {raw16/z_bytes:.1f}x vs fp16)")


if __name__ == '__main__':
    main()
