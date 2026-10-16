"""Optional encryption at rest for voiceprints (biometric data, GDPR Art. 9).

The key lives in the OS keyring (Windows Credential Manager, macOS Keychain, Secret
Service on Linux) or, for headless setups, in ``TRANSCRIBE_VOICEPRINT_KEY``. Needs the
``secure`` extra (cryptography + keyring).
"""

from __future__ import annotations

import os
from typing import Protocol

ENV_KEY = "TRANSCRIBE_VOICEPRINT_KEY"
KEYRING_SERVICE = "transcribe"
KEYRING_USER = "voiceprint-key"


class Cipher(Protocol):
    encrypted: bool

    def encrypt(self, data: bytes) -> bytes: ...
    def decrypt(self, data: bytes) -> bytes: ...


class NullCipher:
    encrypted = False

    def encrypt(self, data: bytes) -> bytes:
        return data

    def decrypt(self, data: bytes) -> bytes:
        return data


class FernetCipher:
    encrypted = True

    def __init__(self, key: bytes) -> None:
        from cryptography.fernet import Fernet

        self._fernet = Fernet(key)

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return self._fernet.decrypt(data)


def load_or_create_key(create: bool = True) -> bytes | None:
    """Key from the environment, else from the OS keyring (generated on first use)."""
    if env := os.environ.get(ENV_KEY):
        return env.encode()
    try:
        import keyring
        from cryptography.fernet import Fernet
    except ImportError as e:
        raise RuntimeError("Voiceprint encryption needs the 'secure' extra: "
                           "pip install 'transcribe[secure]'") from e
    stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    if stored:
        return stored.encode()
    if not create:
        return None
    key = Fernet.generate_key()
    keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key.decode())
    return key


def make_cipher(encrypt: bool) -> Cipher:
    if not encrypt:
        return NullCipher()
    key = load_or_create_key()
    assert key is not None
    return FernetCipher(key)
