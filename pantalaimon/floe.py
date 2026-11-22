"""floe (MSC4571) media decryption — Python port of the reference
implementation in mautrix-go `crypto/floe` (Snowflake, Apache-2.0).

Only the cipher suite in actual use by bridges is implemented:
``GCM256_IV256_4K`` (AES-256-GCM, SHA-384, 4096 byte encrypted segments).

Key material lives in the room event itself (``floe_key``/``floe_header``
fields of the ``file`` structure, mirroring how megolm events carry
``key``/``iv``/``hashes``): the matrix room E2EE (olm/megolm) is what
protects the key, floe is a symmetric content cipher layered on top of
it. Decryption therefore needs no HPKE and no key request machinery.

Header layout (74 bytes, base64 in ``floe_header``)::

    params_encoded(10) | floe_iv(32) | header_tag(32)

    params_encoded = aead(1) | hash(1) | enc_seg_len(u32BE) | iv_len(u32BE)
                   = 00 00 00 00 10 00 00 00 20 00  (for GCM256_IV256_4K)

Key schedule (pure HKDF-Expand, no extract step; ``floe_key`` is the PRK):

    header_tag  = HKDF-Expand(SHA384, key, params|floe_iv|"HEADER_TAG:"|aad, 32)
    message_key = HKDF-Expand(SHA384, key, params|floe_iv|"MESSAGE_KEY:"|aad, 48)
    aead_key    = HKDF-Expand(SHA384, message_key, "DEK:"|u64BE(counter & ~0xFFFFF), 32)

The encrypted blob stored at the mxc URL contains only segments (the
header is carried by the event). Each segment starts with a 4 byte
big-endian word: ``0xFFFFFFFF`` marks an intermediate segment (the rest
of the segment body is exactly ``ENC_SEG_LEN - 4`` bytes), anything else
is the last segment and its value is the length of the whole segment.

Segment body (Go ``cipher.NewGCMWithRandomNonce`` layout)::

    nonce(12) | ciphertext | tag(16)

with ``additional_data = u64BE(counter) | 0x00`` (``0x01`` for the final
segment). The raw (unmasked) segment counter is used for the AEAD AAD;
masking with ``~((1 << 20) - 1)`` only happens when deriving the DEK.
"""

import base64
import hashlib
import hmac
import struct

AEAD_AESGCM256 = 0
HASH_SHA384 = 0
ENC_SEG_LEN = 4096
FLOE_IV_LEN = 32
AEAD_NONCE_LEN = 12
AEAD_TAG_LEN = 16
SEG_MAGIC_LEN = 4
ROT_MASK = 20

PT_SEG_LEN = ENC_SEG_LEN - AEAD_NONCE_LEN - AEAD_TAG_LEN - SEG_MAGIC_LEN  # 4064


def hkdf_expand(prk, info, length, hash_alg=hashlib.sha384):
    """HKDF-Expand step (RFC 5869 §2.3)."""
    t = b""
    okm = b""
    i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hash_alg).digest()
        okm += t
        i += 1
    return okm[:length]


def _params_encoded():
    return (
        struct.pack(">B", AEAD_AESGCM256)
        + struct.pack(">B", HASH_SHA384)
        + struct.pack(">I", ENC_SEG_LEN)
        + struct.pack(">I", FLOE_IV_LEN)
    )


def _b64dec(s):
    return base64.b64decode(s + "=" * (-len(s) % 4))


def _kdf(prk, floe_iv, purpose, aad, length):
    """info = params_encoded | floe_iv | purpose | aad (see floe.go kdf())."""
    return hkdf_expand(prk, _params_encoded() + floe_iv + purpose + aad, length)


def parse_header(floe_header_b64):
    """Split a floe header into (floe_iv, header_tag), validating the params."""
    header = _b64dec(floe_header_b64)
    expected_len = 10 + FLOE_IV_LEN + 32
    if len(header) != expected_len:
        raise ValueError(f"header is {len(header)} bytes, expected {expected_len}")
    params, floe_iv, header_tag = (
        header[:10],
        header[10 : 10 + FLOE_IV_LEN],
        header[-32:],
    )
    if params != _params_encoded():
        raise ValueError(f"unexpected cipher suite params: {params.hex()}")
    return floe_iv, header_tag


class FloeDecryptor:
    """Streaming decryptor: feed segments in order, counter starts at 0."""

    def __init__(self, floe_key_b64, floe_header_b64, aad=b""):
        self.floe_key = _b64dec(floe_key_b64)
        if len(self.floe_key) != 32:
            raise ValueError("floe_key must be 32 bytes")
        self.floe_iv, self.header_tag = parse_header(floe_header_b64)
        expected = _kdf(self.floe_key, self.floe_iv, b"HEADER_TAG:", aad, 32)
        if not hmac.compare_digest(expected, self.header_tag):
            raise ValueError("header tag mismatch (wrong key or AAD)")
        self.message_key = _kdf(self.floe_key, self.floe_iv, b"MESSAGE_KEY:", aad, 48)
        self.aad = aad
        self.counter = 0
        self.closed = False
        self._aesgcm_cache = {}

    def _mask_counter(self, counter):
        return counter & ~((1 << ROT_MASK) - 1)

    def _derive_dek(self, counter):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        purpose = b"DEK:" + struct.pack(">Q", self._mask_counter(counter))
        # Same kdf() as the Go code: params_encoded|floe_iv|"DEK:"|u64BE|aad
        dek = _kdf(self.message_key, self.floe_iv, purpose, self.aad, 32)
        mkey = self._mask_counter(counter)  # DEK is constant within the mask window
        if mkey not in self._aesgcm_cache:
            self._aesgcm_cache[mkey] = AESGCM(dek)
        return self._aesgcm_cache[mkey]

    def _open(self, seg_body, final):
        aead_aad = struct.pack(">Q", self.counter) + (b"\x01" if final else b"\x00")
        aesgcm = self._derive_dek(self.counter)
        nonce, ct_and_tag = seg_body[:AEAD_NONCE_LEN], seg_body[AEAD_NONCE_LEN:]
        pt = aesgcm.decrypt(nonce, ct_and_tag, aead_aad)  # GCM tag authenticates
        self.counter += 1
        return pt

    def decrypt_segment(self, segment):
        """Decrypt one segment (intermediate or last, in order)."""
        if self.closed:
            raise ValueError("decryptor closed (last segment already consumed)")
        magic = struct.unpack(">I", segment[:SEG_MAGIC_LEN])[0]
        body = segment[SEG_MAGIC_LEN:]
        if magic == 0xFFFFFFFF:
            if len(body) != ENC_SEG_LEN - SEG_MAGIC_LEN:
                raise ValueError(
                    f"intermediate segment body {len(body)} != {ENC_SEG_LEN-4}"
                )
            return self._open(body, final=False)
        if magic != len(segment):
            raise ValueError(
                f"last segment magic {magic} != segment length {len(segment)}"
            )
        self.closed = True
        return self._open(body, final=True)

    def decrypt(self, blob):
        """Decrypt a whole blob (all segments, without the header) at once."""
        out = b""
        pos = 0
        while pos < len(blob):
            magic = struct.unpack(">I", blob[pos : pos + 4])[0]
            seg_len = ENC_SEG_LEN if magic == 0xFFFFFFFF else magic
            seg = blob[pos : pos + seg_len]
            if len(seg) < seg_len:
                raise ValueError("truncated blob")
            out += self.decrypt_segment(seg)
            pos += seg_len
        return out


class FloeEncryptor:
    """Mirror of the Go FloeEncryptor — used to generate synthetic test
    vectors; pantalaimon itself never encrypts floe media."""

    def __init__(self, key_b64, aad=b"", floe_iv=None):
        import os

        self.floe_key = _b64dec(key_b64)
        if len(self.floe_key) != 32:
            raise ValueError("floe_key must be 32 bytes")
        self.aad = aad
        # iv = RND(FLOE_IV_LEN); injectable for deterministic test vectors
        self.floe_iv = floe_iv if floe_iv is not None else os.urandom(FLOE_IV_LEN)
        header_prefix = _params_encoded() + self.floe_iv
        header_tag = _kdf(self.floe_key, self.floe_iv, b"HEADER_TAG:", aad, 32)
        self.header = header_prefix + header_tag
        self.header_b64 = base64.b64encode(self.header).rstrip(b"=").decode()
        self.message_key = _kdf(self.floe_key, self.floe_iv, b"MESSAGE_KEY:", aad, 48)
        self.counter = 0
        self.closed = False

    def _seal(self, pt, final):
        import os

        purpose = b"DEK:" + struct.pack(">Q", self.counter & ~((1 << ROT_MASK) - 1))
        dek = _kdf(self.message_key, self.floe_iv, purpose, self.aad, 32)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        flag = b"\x01" if final else b"\x00"
        aead_aad = struct.pack(">Q", self.counter) + flag
        nonce = os.urandom(AEAD_NONCE_LEN)
        ct_and_tag = AESGCM(dek).encrypt(nonce, pt, aead_aad)
        body = nonce + ct_and_tag  # Go NewGCMWithRandomNonce layout
        self.counter += 1
        if final:
            self.closed = True
            return struct.pack(">I", SEG_MAGIC_LEN + len(body)) + body
        return b"\xff\xff\xff\xff" + body

    def encrypt(self, pt):
        out = b""
        pos = 0
        while True:
            rest = pt[pos:]
            if len(rest) > PT_SEG_LEN:
                out += self._seal(rest[:PT_SEG_LEN], final=False)
                pos += PT_SEG_LEN
            else:
                out += self._seal(rest, final=True)  # empty pt → only the last one
                break
        return out


def decrypt_floe_attachment(body, floe_key_b64, floe_header_b64):
    """Module-level helper for the media proxy (picklable for the process pool)."""
    return FloeDecryptor(floe_key_b64, floe_header_b64, b"").decrypt(body)
