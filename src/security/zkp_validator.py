"""Fail-closed snarkjs proof verification for claimant eligibility.

The circuit and verification key are deployment artifacts. ClaimGuard never
receives the witness (PAN/SSN/income); the caller submits a Groth16 proof and
public signals produced by the claimant-side prover. The circuit must bind the
identity commitment and requested policy thresholds to the public signals.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


class ZKPVerificationError(ValueError):
    """A proof could not be verified; callers must reject the claim."""


@dataclass(frozen=True)
class ZKPVerification:
    verified: bool
    zk_proof_hash: str
    public_signals: tuple[str, ...]


def _canonical_hash(proof: Mapping[str, Any], public_signals: list[str]) -> str:
    encoded = json.dumps(
        {"proof": proof, "publicSignals": public_signals},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_claimant_proof(
    proof: Mapping[str, Any],
    public_signals: list[str],
    *,
    verification_key: Optional[str] = None,
    snarkjs_bin: Optional[str] = None,
    timeout_seconds: int = 15,
) -> ZKPVerification:
    """Verify a snarkjs Groth16 proof; raises on missing config or invalid proof.

    Configure ``ZKP_VERIFICATION_KEY`` with the trusted circuit verification
    key and install ``snarkjs`` (or set ``SNARKJS_BIN``). No raw identity data
    should be supplied here. Use only the circuit's documented public signals.
    """
    key_path = Path(verification_key or os.environ.get("ZKP_VERIFICATION_KEY", ""))
    binary = snarkjs_bin or os.environ.get("SNARKJS_BIN") or shutil.which("snarkjs")
    if not key_path.is_file():
        raise ZKPVerificationError("Trusted ZKP verification key is not configured.")
    if not binary:
        raise ZKPVerificationError("snarkjs is unavailable; proof verification is disabled.")
    if not isinstance(proof, Mapping) or not isinstance(public_signals, list) or not public_signals:
        raise ZKPVerificationError("Proof and public signals are required.")

    # Temporary proof files contain only cryptographic proof/public signals.
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix="claimguard-zkp-") as temp_dir:
            proof_file = Path(temp_dir) / "proof.json"
            signals_file = Path(temp_dir) / "public.json"
            proof_file.write_text(json.dumps(proof), encoding="utf-8")
            signals_file.write_text(json.dumps(public_signals), encoding="utf-8")
            result = subprocess.run(
                [binary, "groth16", "verify", str(key_path), str(signals_file), str(proof_file)],
                capture_output=True, text=True, timeout=timeout_seconds, check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ZKPVerificationError("ZKP verifier execution failed.") from exc
    if result.returncode != 0 or "OK!" not in (result.stdout + result.stderr):
        raise ZKPVerificationError("Claimant proof is invalid.")
    return ZKPVerification(True, _canonical_hash(proof, public_signals), tuple(map(str, public_signals)))
