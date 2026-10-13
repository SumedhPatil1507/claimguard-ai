-- Stores proof audit digests only. claimant_ref must be an opaque, non-PII ID.
CREATE TABLE IF NOT EXISTS claimant_zk_verifications (
    claimant_ref TEXT PRIMARY KEY CHECK (length(claimant_ref) BETWEEN 1 AND 128),
    zk_proof_hash CHAR(64) NOT NULL CHECK (zk_proof_hash ~ '^[0-9a-f]{64}$'),
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

REVOKE ALL ON claimant_zk_verifications FROM PUBLIC;
