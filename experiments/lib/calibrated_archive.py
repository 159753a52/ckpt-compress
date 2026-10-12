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
    if manifest["format"] == "checkpoint-components-v1":
        components = {}
        for key in ("state", "weights"):
            item = manifest[key]
            component = (path.parent / item["file"]).resolve()
            if component.parent != path.parent.resolve() or sha256_file(component) != item["sha256"]:
                raise ValueError("checkpoint component path or hash differs")
            components[key] = load_payload(component)
        payload = components["state"]
        payload.update(components["weights"])
    elif manifest["format"] != "residual-gain-bundle-v1":
        raise ValueError("unsupported gain bundle")
    else:
        base = (path.parent / manifest["base_file"]).resolve()
        if base.parent != path.parent.resolve() or sha256_file(base) != manifest["base_sha256"]:
            raise ValueError("gain bundle base path or hash differs")
        payload = load_payload(base)
    targets = manifest.get("gain_parameter_names")
    if "gain" in manifest:
        for name, encoded in payload["model"].items():
            if targets is None or name in targets:
                key = "raw" if "raw" in encoded else "codebook"
                encoded[key] = encoded[key].float() * manifest["gain"]
    return payload


def self_check():
    import tempfile
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
