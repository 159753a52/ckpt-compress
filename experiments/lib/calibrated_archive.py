"""Read a physical residual-gain bundle, counting and verifying its base payload."""
import io
import json
from pathlib import Path
import zlib

import torch
from experiments.lib.residual_runtime import sha256_file


def _decode(encoded, shape):
    """Dense values of one excp_quant record (zero-reserved codebook, packed 4-bit or 8-bit codes)."""
    if "raw" in encoded:
        return encoded["raw"].float().view(shape)
    if "codes" in encoded:
        codes = encoded["codes"]
    else:
        packed = encoded["packed"]
        codes = torch.stack([packed // 16, packed % 16], dim=1).flatten()[: encoded["numel"]]
    table = torch.cat([torch.zeros(1), encoded["codebook"].float()])
    return table[codes.long()].view(shape)


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
    tiers = manifest.get("tiered_gain")
    if tiers is not None:
        # Coordinates whose stored second moment lies below the threshold get the slow gain.
        if "gain" in manifest or targets is None:
            raise ValueError("a tiered gain replaces the uniform gain and needs its target tensors")
        for name in targets:
            moment = payload["exp_avg_sq"][name]
            v = (moment["raw"] if isinstance(moment, dict) else moment).float()
            values = _decode(payload["model"][name], v.shape)
            payload["model"][name] = {"raw": torch.where(v < tiers["v_threshold"], values * tiers["slow"], values * tiers["fast"])}
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
        state_file = root / "state.zlib"
        weights_file = root / "weights.zlib"
        for file, data in [(state_file, {"exp_avg": {"w": moment}}),
                           (weights_file, {"model": {"w": {"raw": torch.tensor([1., -2.])},
                                                     "bias": {"raw": torch.tensor([.25])}}})]:
            stream = io.BytesIO()
            torch.save(data, stream)
            file.write_bytes(zlib.compress(stream.getvalue()))
        manifest.write_text(json.dumps({"format": "checkpoint-components-v1",
            "state": {"file": state_file.name, "sha256": sha256_file(state_file)},
            "weights": {"file": weights_file.name, "sha256": sha256_file(weights_file)},
            "gain": 1.25, "gain_parameter_names": ["w"]}))
        decoded = load_payload(manifest)
        assert torch.equal(decoded["model"]["w"]["raw"], torch.tensor([1.25, -2.5]))
        assert torch.equal(decoded["model"]["bias"]["raw"], torch.tensor([.25]))
        assert torch.equal(decoded["exp_avg"]["w"], moment)
        codebook = torch.tensor([.5, -1.], dtype=torch.float16)
        packed = torch.tensor([0 * 16 + 1, 2 * 16 + 0], dtype=torch.uint8)
        for file, data in [(state_file, {"exp_avg_sq": {"w": {"raw": torch.tensor([[1e-9, 1e-9], [1e-7, 1e-7]], dtype=torch.bfloat16)}}}),
                           (weights_file, {"model": {"w": {"packed": packed, "codebook": codebook, "numel": 4}}})]:
            stream = io.BytesIO()
            torch.save(data, stream)
            file.write_bytes(zlib.compress(stream.getvalue()))
        manifest.write_text(json.dumps({"format": "checkpoint-components-v1",
            "state": {"file": state_file.name, "sha256": sha256_file(state_file)},
            "weights": {"file": weights_file.name, "sha256": sha256_file(weights_file)},
            "tiered_gain": {"v_threshold": 1e-8, "slow": 3., "fast": 1.}, "gain_parameter_names": ["w"]}))
        decoded = load_payload(manifest)
        assert torch.equal(decoded["model"]["w"]["raw"], torch.tensor([[0., 1.5], [-1., 0.]]))
    print("Gain bundle scales weights once, preserves moments and leaves its base intact; tiered gain follows the stored second moment")


if __name__ == "__main__":
    self_check()
