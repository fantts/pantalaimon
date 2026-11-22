import base64
import os

import pytest

from pantalaimon.floe import (
    FloeDecryptor,
    FloeEncryptor,
    PT_SEG_LEN,
    parse_header,
    decrypt_floe_attachment,
)

# 0 → empty last segment only; 4064 → one full segment;
# 4065 → intermediate + short last; 1 MiB → many segments.
ROUNDTRIP_SIZES = [0, 1, 100, PT_SEG_LEN - 1, PT_SEG_LEN, PT_SEG_LEN + 1,
                   2 * PT_SEG_LEN, 3 * PT_SEG_LEN + 7, 1024 * 1024]

FIXED_IV = bytes(range(32))


@pytest.fixture
def key_b64():
    return base64.b64encode(os.urandom(32)).rstrip(b"=").decode()


@pytest.mark.parametrize("size", ROUNDTRIP_SIZES)
def test_roundtrip(key_b64, size):
    pt = os.urandom(size)
    enc = FloeEncryptor(key_b64, floe_iv=FIXED_IV)
    blob, header = enc.encrypt(pt), enc.header_b64
    assert FloeDecryptor(key_b64, header).decrypt(blob) == pt


def test_attachment_helper(key_b64):
    pt = os.urandom(10_000)
    enc = FloeEncryptor(key_b64, floe_iv=FIXED_IV)
    blob, header = enc.encrypt(pt), enc.header_b64
    assert decrypt_floe_attachment(blob, key_b64, header) == pt


def test_header_params(key_b64):
    header = base64.b64encode(bytes(range(9)) + bytes(32) + bytes(32)).decode()
    with pytest.raises(ValueError):
        parse_header(header)


def test_header_tag_mismatch(key_b64):
    enc = FloeEncryptor(key_b64, floe_iv=FIXED_IV)
    blob = enc.encrypt(b"hello floe" * 100)

    # Wrong AAD → header tag must be rejected before any decryption happens.
    with pytest.raises(ValueError):
        FloeDecryptor(key_b64, enc.header_b64, aad=b"wrong").decrypt(blob)

    # Wrong key → same.
    wrong_key = base64.b64encode(os.urandom(32)).rstrip(b"=").decode()
    with pytest.raises(ValueError):
        FloeDecryptor(wrong_key, enc.header_b64).decrypt(blob)


def test_gcm_tamper_detection(key_b64):
    enc = FloeEncryptor(key_b64, floe_iv=FIXED_IV)
    blob = bytearray(enc.encrypt(os.urandom(10_000)))
    blob[len(blob) // 2] ^= 0x01  # flip one bit in the middle of a segment
    with pytest.raises(Exception):  # noqa: B017 — InvalidTag
        FloeDecryptor(key_b64, enc.header_b64).decrypt(bytes(blob))


def test_segment_sequence_enforced(key_b64):
    enc = FloeEncryptor(key_b64, floe_iv=FIXED_IV)
    blob = enc.encrypt(os.urandom(PT_SEG_LEN + 10))
    dec = FloeDecryptor(key_b64, enc.header_b64)
    dec.decrypt_segment(blob[:4096])          # intermediate
    dec.decrypt_segment(blob[4096:])          # last
    with pytest.raises(ValueError):           # decryptor is closed now
        dec.decrypt_segment(blob[4096:])
