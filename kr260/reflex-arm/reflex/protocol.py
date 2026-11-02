"""Bounded UDP framing. CRC detects corruption; this is not authenticated control.

Network byte order: magic(4), version(u8), kind(u8), flags(u16), sequence(u64),
task(u64), sender monotonic ns(u64), TTL ms(u32), payload bytes(u32), JSON, CRC32(u32).
Receiver freshness uses its own monotonic clock and a challenge/handshake is
required before connecting this prototype to any actuator.
"""
from dataclasses import dataclass
import json
import math
import struct
import time
import zlib

HEADER = struct.Struct('!4sBBHQQQII')
MAGIC = b'RFAX'
MAX_PACKET = 4096


@dataclass
class Packet:
    sequence: int
    task: int
    payload: dict
    kind: int = 1
    ttl_ms: int = 100
    timestamp_ns: int = 0

    def encode(self):
        if self.sequence <= 0 or self.task < 0 or not 1 <= self.ttl_ms <= 1000 or self.kind not in (1, 2, 3):
            raise ValueError('Invalid packet metadata')
        body = json.dumps(self.payload, allow_nan=False, separators=(',', ':')).encode()
        header = HEADER.pack(MAGIC, 1, self.kind, 0, self.sequence, self.task,
                             self.timestamp_ns or time.monotonic_ns(), self.ttl_ms, len(body))
        frame = header+body
        frame += struct.pack('!I', zlib.crc32(frame))
        if len(frame) > MAX_PACKET:
            raise ValueError('Packet too large')
        return frame

    @classmethod
    def decode(cls, data):
        if not HEADER.size+4 <= len(data) <= MAX_PACKET:
            raise ValueError('Bad packet length')
        magic, version, kind, flags, seq, task, stamp, ttl, size = HEADER.unpack_from(data)
        if magic != MAGIC or version != 1 or flags or kind not in (1,2,3) or seq <= 0 or not 1 <= ttl <= 1000:
            raise ValueError('Bad header')
        if size != len(data)-HEADER.size-4:
            raise ValueError('Payload length mismatch')
        if struct.unpack('!I', data[-4:])[0] != zlib.crc32(data[:-4]):
            raise ValueError('CRC mismatch')
        def reject_constant(s):
            raise ValueError(f'Nonfinite JSON: {s}')
        def finite_float(s):
            value=float(s)
            if not math.isfinite(value): raise ValueError('Overflowed JSON number')
            return value
        try:
            payload = json.loads(data[HEADER.size:-4], parse_constant=reject_constant,parse_float=finite_float)
        except (UnicodeError, json.JSONDecodeError,RecursionError) as e:
            raise ValueError('Bad JSON') from e
        if not isinstance(payload, dict):
            raise ValueError('Object payload required')
        stack=[(payload,0)]
        while stack:
            value,depth=stack.pop()
            if depth>16: raise ValueError('JSON nesting exceeds protocol limit')
            if isinstance(value,dict): stack.extend((x,depth+1) for x in value.values())
            elif isinstance(value,list): stack.extend((x,depth+1) for x in value)
        return cls(seq, task, payload, kind, ttl, stamp)


class Receiver:
    def __init__(self):
        self.last_sequence = 0
        self.last_rx_ns = None
        self.ttl_ns = 0

    def accept(self, data, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        p = Packet.decode(data)
        if p.sequence <= self.last_sequence:
            raise ValueError('Duplicate/out-of-order sequence')
        # Sender timestamps are logged, never subtracted from an unrelated clock.
        self.last_sequence, self.last_rx_ns, self.ttl_ns = p.sequence, now, p.ttl_ms*1_000_000
        return p

    def expired(self, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        return self.last_rx_ns is None or not 0 <= now-self.last_rx_ns < self.ttl_ns
