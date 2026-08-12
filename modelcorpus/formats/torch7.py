"""Torch7 (.t7) serialization writer.

The legacy Torch7 format is a small tagged stream. Everything is little-endian.
An object is a type tag followed by a payload:

    int32 type      1 NUMBER, 2 STRING, 3 TABLE, 4 TORCH, 5 BOOLEAN, 0 NIL
    ...payload

A TORCH object is an index followed by a class name string, then the class's own
body. A string is an int32 length followed by that many bytes. That length is
signed on the wire, which is the interesting part for parser testing: nothing in
the format prevents it being negative.

Readers reach tensor bodies by matching the class name against ``torch.*Tensor``,
so ``torch.FloatTensor`` is the shortest route into dimension parsing.

Reference implementation consulted while writing this:
opencv/modules/dnn/src/torch/torch_importer.cpp
"""

from __future__ import annotations

import struct
from typing import Sequence

TYPE_NIL = 0
TYPE_NUMBER = 1
TYPE_STRING = 2
TYPE_TABLE = 3
TYPE_TORCH = 4
TYPE_BOOLEAN = 5
TYPE_FUNCTION = 6

# Element types accepted in a torch.<T>Tensor / torch.<T>Storage class name.
ELEMENT_TYPES = ("Double", "Float", "Byte", "Char", "Short", "Int", "Long", "Cuda")


def i32(v: int) -> bytes:
    """int32, little-endian. Accepts negatives, which is the whole point."""
    return struct.pack("<i", v)


def i64(v: int) -> bytes:
    return struct.pack("<q", v)


def f64(v: float) -> bytes:
    return struct.pack("<d", v)


def string(data: bytes, declared_length: int | None = None) -> bytes:
    """A Torch7 string.

    ``declared_length`` overrides the length field without changing the bytes
    that follow it. Leave it None for a well-formed string. Set it to disagree
    with ``len(data)`` to test what a reader does when the header lies, which is
    the single most productive knob in this format.
    """
    n = len(data) if declared_length is None else declared_length
    return i32(n) + data


def tensor_class(element: str = "Float") -> bytes:
    """Class name bytes for a tensor of the given element type."""
    if element not in ELEMENT_TYPES:
        raise ValueError(f"unknown element type {element!r}, expected one of {ELEMENT_TYPES}")
    return f"torch.{element}Tensor".encode()


def storage_class(element: str = "Float") -> bytes:
    if element not in ELEMENT_TYPES:
        raise ValueError(f"unknown element type {element!r}, expected one of {ELEMENT_TYPES}")
    return f"torch.{element}Storage".encode()


def tensor(
    *,
    index: int = 1,
    element: str = "Float",
    ndims: int = 1,
    sizes: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
    offset: int = 1,
    storage: bytes | None = None,
    declared_class_length: int | None = None,
) -> bytes:
    """A TORCH object holding a tensor.

    ``ndims`` is written verbatim and is deliberately not validated against
    ``sizes``. A reader is expected to allocate based on it, so leaving the two
    inconsistent is how you find out whether it bounds that allocation.

    ``sizes`` and ``steps`` default to matching ``ndims`` when it is positive,
    so the default call produces a well-formed object.
    """
    if sizes is None:
        sizes = [1] * max(ndims, 0)
    if steps is None:
        steps = [1] * max(ndims, 0)

    out = i32(TYPE_TORCH)
    out += i32(index)
    out += string(tensor_class(element), declared_class_length)
    out += i32(ndims)
    out += b"".join(i64(s) for s in sizes)
    out += b"".join(i64(s) for s in steps)
    out += i64(offset)

    if storage is None:
        out += i32(TYPE_NIL)
    else:
        out += storage
    return out


def storage(
    *,
    index: int = 2,
    element: str = "Float",
    count: int,
    payload: bytes = b"",
    declared_class_length: int | None = None,
) -> bytes:
    """A TORCH object holding a storage.

    ``count`` is the declared element count and ``payload`` is the raw bytes that
    follow. They are independent on purpose.
    """
    out = i32(TYPE_TORCH)
    out += i32(index)
    out += string(storage_class(element), declared_class_length)
    out += i64(count)
    out += payload
    return out


def standalone_string(data: bytes, declared_length: int | None = None) -> bytes:
    """A top-level STRING object. The shortest thing a reader will parse."""
    return i32(TYPE_STRING) + string(data, declared_length)


def number(value: float) -> bytes:
    return i32(TYPE_NUMBER) + f64(value)


def boolean(value: bool) -> bytes:
    return i32(TYPE_BOOLEAN) + i32(1 if value else 0)


def nil() -> bytes:
    return i32(TYPE_NIL)


def table(pairs: Sequence[tuple[bytes, bytes]], *, index: int = 1,
          declared_count: int | None = None) -> bytes:
    """A TABLE object.

    ``pairs`` are already-encoded (key, value) object pairs. ``declared_count``
    overrides the pair count in the header without changing what follows.
    """
    n = len(pairs) if declared_count is None else declared_count
    out = i32(TYPE_TABLE) + i32(index) + i32(n)
    for key, value in pairs:
        out += key + value
    return out
