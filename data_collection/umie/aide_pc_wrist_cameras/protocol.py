"""Bounded JSON + image framing. No ROS, SDK or pickle dependency."""
from __future__ import annotations

import json
import socket
import struct

PREFIX = struct.Struct('!4sII')
MAGIC = b'AWC1'
MAX_HEADER = 16384
MAX_PAYLOAD = 16 * 1024 * 1024
LEFT_SERIAL = '352122274013'
RIGHT_SERIAL = '261122274758'
CAMERAS = ('left_wrist', 'right_wrist')
DEFAULT_HOST = '192.168.50.20'
DEFAULT_PORT = 9800


def read_exact(connection, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = connection.recv_into(view[offset:])
        if not count:
            raise EOFError('camera connection closed')
        offset += count
    return bytes(data)


def send_packet(connection, header, payload=b''):
    encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if not 0 < len(encoded) <= MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ValueError('packet too large')
    connection.sendall(PREFIX.pack(MAGIC, len(encoded), len(payload)) + encoded)
    if payload:
        connection.sendall(payload)


def receive_packet(connection):
    magic, header_size, payload_size = PREFIX.unpack(read_exact(connection, PREFIX.size))
    if magic != MAGIC or not 0 < header_size <= MAX_HEADER or payload_size > MAX_PAYLOAD:
        raise ValueError('invalid camera packet prefix')
    header = json.loads(read_exact(connection, header_size))
    if not isinstance(header, dict):
        raise ValueError('camera packet header must be an object')
    return header, read_exact(connection, payload_size)


def connect(host, port):
    connection = socket.create_connection((host, port), timeout=3.)
    connection.settimeout(2.)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return connection
