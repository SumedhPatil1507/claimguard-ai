"""Unit tests for the fail-closed snarkjs proof verification boundary."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.security.zkp_validator import ZKPVerificationError, verify_claimant_proof


def test_valid_proof_returns_stable_digest(tmp_path):
    key = tmp_path / "verification_key.json"
    key.write_text("{}", encoding="utf-8")
    proof = {"pi_a": ["1", "2"]}
    signals = ["1", "900000"]

    with patch(
        "src.security.zkp_validator.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="OK!", stderr=""),
    ) as run:
        result = verify_claimant_proof(
            proof, signals, verification_key=str(key), snarkjs_bin="snarkjs"
        )

    assert result.verified is True
    assert len(result.zk_proof_hash) == 64
    assert result.public_signals == tuple(signals)
    assert run.call_args.args[0][:3] == ["snarkjs", "groth16", "verify"]


def test_invalid_proof_fails_closed(tmp_path):
    key = tmp_path / "verification_key.json"
    key.write_text("{}", encoding="utf-8")
    with patch(
        "src.security.zkp_validator.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="Invalid proof"),
    ):
        with pytest.raises(ZKPVerificationError, match="invalid"):
            verify_claimant_proof(
                {"pi_a": ["1", "2"]}, ["1"],
                verification_key=str(key), snarkjs_bin="snarkjs",
            )


def test_missing_trusted_key_fails_closed(tmp_path):
    with pytest.raises(ZKPVerificationError, match="key is not configured"):
        verify_claimant_proof(
            {"pi_a": ["1", "2"]}, ["1"],
            verification_key=str(tmp_path / "missing.json"), snarkjs_bin="snarkjs",
        )

