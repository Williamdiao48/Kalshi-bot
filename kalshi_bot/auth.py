import base64
import time
import logging
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()

KALSHI_KEY_ID: str = os.environ.get("KALSHI_KEY_ID", "")
_raw_key: str = os.environ.get("KALSHI_PRIVATE_KEY_STR", "")
if "\\n" in _raw_key:
    _raw_key = _raw_key.replace("\\n", "\n")
KALSHI_PRIVATE_KEY_STR: str = _raw_key

# Cached private key object — loaded once, reused on every request.
# Avoids re-parsing the PEM and constructing the RSA key on each call.
_private_key = None


def _get_private_key():
    global _private_key
    if _private_key is None:
        _private_key = serialization.load_pem_private_key(
            KALSHI_PRIVATE_KEY_STR.encode("utf-8"),
            password=None,
        )
    return _private_key


def generate_headers(method: str, path: str) -> dict:
    """Return RSA-PSS signed headers for a Kalshi V2 API request.

    Args:
        method: HTTP method in uppercase (e.g. "GET", "POST").
        path:   The API path starting with /trade-api/... (no query string).

    Returns:
        Dict of auth headers, or empty dict if credentials are missing/invalid.
    """
    if not KALSHI_KEY_ID or not KALSHI_PRIVATE_KEY_STR:
        logging.debug("Kalshi credentials not configured; skipping auth headers.")
        return {}

    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + method + path

    try:
        private_key = _get_private_key()
        signature = private_key.sign(
            msg_string.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }
    except Exception as exc:
        logging.error("Failed to generate Kalshi auth signature: %s", exc)
        return {}
