"""Read a physical residual-gain bundle, counting and verifying its base payload."""
import io
import json
from pathlib import Path
import zlib

import torch
from experiments.lib.residual_runtime import sha256_file


def load_payload(path):
    path = Path(path)
    if path.suffix != ".json":
        return torch.load(io.BytesIO(zlib.decompress(path.read_bytes())), map_location="cpu", weights_only=True)
    manifest = json.loads(path.read_text())
    if manifest["format"] != "residual-gain-bundle-v1":
        raise ValueError("unsupported gain bundle")
    base = (path.parent / manifest["base_file"]).resolve()
    if base.parent != path.parent.resolve() or sha256_file(base) != manifest["base_sha256"]:
        raise ValueError("gain bundle base path or hash differs")
    payload = load_payload(base)
    for encoded in payload["model"].values():
        key = "raw" if "raw" in encoded else "codebook"
        encoded[key] = encoded[key].float() * manifest["gain"]
    return payload


def self_check():
    import tempfile
    from runpy import run_path
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        moment = torch.tensor([.25, .5], dtype=torch.bfloat16)
        payload = {"model": {"w": {"raw": torch.tensor([1., -2.])}}, "exp_avg": {"w": moment}}
        stream = io.BytesIO()
        torch.save(payload, stream)
        base = root / "base.zlib"
        base.write_bytes(zlib.compress(stream.getvalue()))
        manifest = root / "gain.json"
        manifest.write_text(json.dumps({"format": "residual-gain-bundle-v1", "base_file": base.name,
                                       "base_sha256": sha256_file(base), "gain": 1.25}))
        decoded = load_payload(manifest)
        assert torch.equal(decoded["model"]["w"]["raw"], torch.tensor([1.25, -2.5]))
        assert torch.equal(decoded["exp_avg"]["w"], moment)
        assert torch.equal(load_payload(base)["model"]["w"]["raw"], torch.tensor([1., -2.]))
    print("Gain bundle scales weights once, preserves moments and leaves its base intact")


if __name__ == "__main__":
    self_check()
