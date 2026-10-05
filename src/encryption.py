"""
Field-level encryption utilities for ClaimGuard AI.

Key resolution order:
  1. ENCRYPTION_KEY environment variable (base64-urlsafe Fernet key)
  2. data/.encryption_key file on disk
  3. Auto-generate a new Fernet key and persist it to data/.encryption_key

Graceful fallback: if the `cryptography` package is absent the
encrypt_field / decrypt_field functions are transparent no-ops so the
rest of the codebase never crashes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    from cryptography.fernet import Fernet  # type: ignore

    HAS_FERNET = True
except ImportError:
    HAS_FERNET = False

# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------
_KEY_FILE = Path(__file__).resolve().parent.parent / "data" / ".encryption_key"

_fernet_instance: Optional["Fernet"] = None  # type: ignore[type-arg]


def _load_or_create_key() -> Optional["Fernet"]:  # type: ignore[type-arg]
    """Load or generate the Fernet key and return a Fernet instance."""
    if not HAS_FERNET:
        return None

    # 1. Environment variable takes highest priority
    env_key = os.environ.get("ENCRYPTION_KEY")
    if env_key:
        return Fernet(env_key.encode() if isinstance(env_key, str) else env_key)

    # 2. Key file on disk
    if _KEY_FILE.exists():
        key_bytes = _KEY_FILE.read_bytes().strip()
        return Fernet(key_bytes)

    # 3. Auto-generate and persist
    key_bytes = Fernet.generate_key()
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_bytes(key_bytes)
    return Fernet(key_bytes)


def _get_fernet() -> Optional["Fernet"]:  # type: ignore[type-arg]
    """Return the cached Fernet instance, initialising on first call."""
    global _fernet_instance
    if _fernet_instance is None:
        _fernet_instance = _load_or_create_key()
    return _fernet_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encrypt_field(value: str) -> str:
    """Encrypt *value* and return a URL-safe base64 token.

    Returns *value* unchanged when the ``cryptography`` package is absent.
    """
    fernet = _get_fernet()
    if fernet is None:
        return value
    return fernet.encrypt(value.encode()).decode()


def decrypt_field(encrypted: str) -> str:
    """Decrypt a token produced by :func:`encrypt_field`.

    Returns *encrypted* unchanged when the ``cryptography`` package is absent.
    """
    fernet = _get_fernet()
    if fernet is None:
        return encrypted
    return fernet.decrypt(encrypted.encode()).decode()
