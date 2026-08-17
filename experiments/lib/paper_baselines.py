"""Auditable method identities for paper-facing recovery experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from experiments.lib.residual_protocol import (
    NO_COMPRESSION_METHOD,
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD,
    TAYLOR_EXACT_GLOBAL_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_WEIBULL_MOM_METHOD,
)


@dataclass(frozen=True)
class MethodContract:
    """One public method name and the exact implementation it invokes."""

    name: str
    owner: str
    fidelity: str
    internal_method: str
    description: str

    def to_result_dict(self) -> dict[str, str]:
        return asdict(self)


METHOD_CONTRACTS = {
    "no_compression": MethodContract(
        name="no_compression",
        owner="control",
        fidelity="native",
        internal_method=NO_COMPRESSION_METHOD,
        description="Training trajectory without lossy checkpoint restoration.",
    ),
    "dacp": MethodContract(
        name="dacp",
        owner="dacp",
        fidelity="native",
        internal_method=TAYLOR_WEIBULL_MOM_METHOD,
        description="Taylor damage scoring with Weibull moment allocation.",
    ),
    "excp_style": MethodContract(
        name="excp_style",
        owner="excp",
        fidelity="style",
        internal_method="residual_magnitude_exact_global",
        description=(
            "Residual-magnitude scoring with exact global top-k selection under the "
            "same residual sparsity and recovery protocol; excludes ExCP optimizer "
            "compression and quantization."
        ),
    ),
    "inshrinkerator_style": MethodContract(
        name="inshrinkerator_style",
        owner="inshrinkerator",
        fidelity="style",
        internal_method="first_order_exact_global",
        description=(
            "Full-weight first-order sensitivity |g*theta| with exact global top-k "
            "selection, transferred to the same residual sparsity and recovery "
            "protocol; excludes the full partition, search, and encoding pipeline."
        ),
    ),
    "magnitude_uniform": MethodContract(
        name="magnitude_uniform",
        owner="dacp_ablation",
        fidelity="native",
        internal_method=RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
        description="Residual magnitude scoring with uniform per-layer allocation.",
    ),
    "magnitude_weibull": MethodContract(
        name="magnitude_weibull",
        owner="dacp_ablation",
        fidelity="native",
        internal_method=RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD,
        description="Residual magnitude scoring with Weibull moment allocation.",
    ),
    "taylor_uniform": MethodContract(
        name="taylor_uniform",
        owner="dacp_ablation",
        fidelity="native",
        internal_method=TAYLOR_UNIFORM_METHOD,
        description="Taylor HVP scoring with uniform per-layer allocation.",
    ),
    "taylor_exact_global": MethodContract(
        name="taylor_exact_global",
        owner="dacp_ablation",
        fidelity="native",
        internal_method=TAYLOR_EXACT_GLOBAL_METHOD,
        description="Taylor HVP scoring with exact global top-k allocation oracle.",
    ),
}


def resolve_method_contracts(names: Sequence[str]) -> list[MethodContract]:
    """Resolve unique method names and reject unsupported fidelity claims."""
    if not names:
        raise ValueError("At least one method is required")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate paper methods: {duplicates}")
    unknown = [name for name in names if name not in METHOD_CONTRACTS]
    if unknown:
        raise ValueError(f"Unknown paper methods {unknown}; available={sorted(METHOD_CONTRACTS)}")
    return [METHOD_CONTRACTS[name] for name in names]


def validate_claim_gate(
    contracts: Sequence[MethodContract],
    gate: Mapping[str, object],
) -> None:
    """Ensure a declared claim cannot silently upgrade style baselines to full."""
    required = gate.get("required_baseline_fidelity", "style")
    if required not in {"style", "full"}:
        raise ValueError(f"Unsupported required_baseline_fidelity: {required}")
    raw_owners = gate.get("baseline_owners", ())
    if not isinstance(raw_owners, list) or not all(
        isinstance(owner, str) and owner for owner in raw_owners
    ):
        raise ValueError("baseline_owners must be a list of non-empty strings")
    owners = set(raw_owners)
    if len(owners) != len(raw_owners):
        raise ValueError("baseline_owners must not contain duplicates")
    baselines = [contract for contract in contracts if contract.owner in owners]
    missing = owners - {contract.owner for contract in baselines}
    if missing:
        raise ValueError(f"Claim gate is missing baselines: {sorted(missing)}")
    if required == "full":
        incomplete = [contract.name for contract in baselines if contract.fidelity != "full"]
        if incomplete:
            raise ValueError(
                "Full-baseline claim requires full implementations; style adapters "
                f"cannot satisfy it: {incomplete}"
            )


__all__ = [
    "METHOD_CONTRACTS",
    "MethodContract",
    "resolve_method_contracts",
    "validate_claim_gate",
]
