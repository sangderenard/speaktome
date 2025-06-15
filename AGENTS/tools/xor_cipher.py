#!/usr/bin/env python3
"""Simple XOR encrypt/decrypt utility for short strings."""
from __future__ import annotations

try:
    import base64
    import argparse
    import sys
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

def _xor_bytes(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def encrypt(text: str, passphrase: str) -> str:
    """Return base64-encoded XOR cipher text."""
    cipher = _xor_bytes(text.encode("utf-8"), passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(cipher).decode("ascii")


def decrypt(cipher_text: str, passphrase: str) -> str:
    """Decode text produced by :func:`encrypt`."""
    data = base64.urlsafe_b64decode(cipher_text.encode("ascii"))
    return _xor_bytes(data, passphrase.encode("utf-8")).decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decrypt", action="store_true", help="decode instead of encode")
    parser.add_argument("--passphrase", required=True)
    parser.add_argument("text")
    args = parser.parse_args(argv)

    if args.decrypt:
        print(decrypt(args.text, args.passphrase))
    else:
        print(encrypt(args.text, args.passphrase))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
