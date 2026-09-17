# multi_agent_generator/storage/secretbox.py
"""
Encrypting saved credentials at rest.

The LLM settings page lets a user save an API key so they do not have to re-enter it. That
convenience creates an obligation: the key ends up in a SQLite file on disk, and a plaintext
key in a database file is a credential leak waiting for a backup, a screen share or a
``sqlite3 magen.db .dump``.

So the rule this module enforces is simple and absolute: **a credential is either encrypted
or it is not stored.** If encryption is unavailable, saving a key fails with an explanation
telling the user to put it in ``.env`` instead. Falling back to plaintext "just this once"
would defeat the entire point, and a system that silently downgrades its own security is
worse than one that refuses, because the user believes they are protected.

The key material lives in a file next to the database, created readable only by its owner.
This protects against the realistic threat - a database file copied, committed or shared
without the surrounding directory - rather than against an attacker who already has
arbitrary read access to the user's home directory, which no local-first application can
defend against anyway. Being clear about which threat is covered matters more than implying
a stronger guarantee than exists.
"""
from __future__ import annotations

import base64
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

from ..errors import ConfigurationError

__all__ = ["SecretBox", "encryption_available"]


#: Filename of the local key, kept beside the database rather than inside it - a key stored
#: in the file it protects is not a key.
_KEY_FILENAME = ".keyring"


def encryption_available() -> bool:
    """Whether credential encryption can be used in this install."""
    try:
        import cryptography.fernet  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class SecretBox:
    """
    Symmetric encryption for stored credentials.

    One instance per data directory. The key is generated on first use and reused
    afterwards, so a key saved by one run can be read by the next - which is the entire
    point of persisting it.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._fernet = None

    # ------------------------------------------------------------------------ public API
    @property
    def available(self) -> bool:
        return encryption_available()

    def encrypt(self, plaintext: Optional[str]) -> Optional[str]:
        """
        Encrypt a credential for storage. Returns None for an empty input.

        Raises:
            ConfigurationError: if encryption is unavailable. This is a refusal, not a
                fallback - see the module docstring.
        """
        if plaintext is None or not str(plaintext).strip():
            return None
        fernet = self._require_fernet()
        return fernet.encrypt(str(plaintext).encode("utf-8")).decode("ascii")

    def decrypt(self, token: Optional[str]) -> Optional[str]:
        """
        Decrypt a stored credential, or return None if it cannot be read.

        Returns None rather than raising on a bad token. A key encrypted under a keyring that
        has since been deleted is unrecoverable, and the useful behaviour is to act as though
        no key was saved - the user re-enters it - rather than to make the settings page
        permanently unloadable.
        """
        if not token:
            return None
        if not self.available:
            return None
        try:
            return self._require_fernet().decrypt(token.encode("ascii")).decode("utf-8")
        except Exception:  # noqa: BLE001 - InvalidToken, bad padding, missing keyring
            return None

    # --------------------------------------------------------------------------- internals
    def _require_fernet(self):
        if self._fernet is not None:
            return self._fernet
        try:
            from cryptography.fernet import Fernet
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(
                "Saved API keys need the 'cryptography' package, which is not installed.",
                action=(
                    "Run `pip install cryptography` to save keys in the app, or put the key "
                    "in your .env file instead - the app reads it from there with no "
                    "storage at all."
                ),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        self._fernet = Fernet(self._load_or_create_key())
        return self._fernet

    def _load_or_create_key(self) -> bytes:
        """
        Read the local key, generating it on first use.

        Two properties matter here, and both come from how the file is opened.

        *Never world-readable.* The restrictive mode is passed to :func:`os.open` so the
        permissions are set by the same syscall that creates the file. Doing it as
        ``write_bytes()`` then ``chmod()`` would leave the key readable for the instant
        between the two calls.

        *Never silently replaced.* ``O_EXCL`` makes creation fail if the file already
        exists, which is what closes the race between the ``exists()`` check above and the
        open below. Without it, two processes starting at once could each generate a key and
        the second would overwrite the first - permanently destroying the ability to decrypt
        anything saved under the losing key. On ``FileExistsError`` the winner's key is read
        back and used, so both processes agree.

        On Windows the mode argument is largely a no-op and NTFS inheritance applies instead.
        That is a platform limitation rather than something this code can fix, and it is why
        the module docstring is specific about which threat model is covered.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)
        key_path = self._data_dir / _KEY_FILENAME

        existing = self._read_key(key_path)
        if existing is not None:
            return existing

        key = base64.urlsafe_b64encode(secrets.token_bytes(32))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(str(key_path), flags, stat.S_IRUSR | stat.S_IWUSR)
        except FileExistsError:
            # Someone created it between the read above and this open. Their key is now the
            # only correct one - anything encrypted since was encrypted with it.
            existing = self._read_key(key_path)
            if existing is not None:
                return existing
            # The file exists but is empty: a previous run died mid-creation. Truncating is
            # safe precisely because there is nothing encrypted under an empty key.
            descriptor = os.open(
                str(key_path),
                os.O_WRONLY | os.O_TRUNC,
                stat.S_IRUSR | stat.S_IWUSR,
            )
        try:
            os.write(descriptor, key)
        finally:
            os.close(descriptor)
        return key

    @staticmethod
    def _read_key(key_path: Path) -> Optional[bytes]:
        """The stored key, or None if the file is absent, empty or unreadable."""
        try:
            raw = key_path.read_bytes().strip()
        except OSError:
            return None
        return raw or None
