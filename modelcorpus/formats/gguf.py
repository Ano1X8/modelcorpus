"""GGUF writer.

GGUF is the container used by llama.cpp and the projects that vendor its reader.
Layout, all little-endian:

    magic "GGUF" | uint32 version | uint64 tensor_count | uint64 metadata_kv_count
    metadata KV entries
    tensor info entries

    metadata KV:  uint64 key_len | key bytes | uint32 value_type | value
    tensor info:  uint64 name_len | name bytes | uint32 n_dims
                  int64 shape[n_dims] | uint32 tensor_type | uint64 offset

Note which counts are declared ahead of the data they describe: tensor_count,
metadata_kv_count, key_len, name_len and n_dims all precede their payloads. Each
one is a number a reader may size an allocation from before it has seen whether
the file can satisfy it. Every writer here lets you set those independently of
the bytes you actually append.

Nothing in this module targets a particular implementation. It emits files; what
a given reader does with them is yours to find out.
"""

from __future__ import annotations

import struct
from typing import Sequence

MAGIC = b"GGUF"

# Metadata value type tags.
T_UINT8 = 0
T_INT8 = 1
T_UINT16 = 2
T_INT16 = 3
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10
T_INT64 = 11
T_FLOAT64 = 12


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def i64(v: int) -> bytes:
    return struct.pack("<q", v)


def header(version: int = 3, tensor_count: int = 0, kv_count: int = 0) -> bytes:
    """File header.

    ``tensor_count`` and ``kv_count`` are declared counts. They are written as
    given and are not checked against however many entries you append.
    """
    return MAGIC + u32(version) + u64(tensor_count) + u64(kv_count)


def kv_string(key: bytes, value: bytes,
              declared_key_len: int | None = None,
              declared_value_len: int | None = None) -> bytes:
    """A string-valued metadata entry.

    Both lengths can be overridden to disagree with the bytes that follow.
    """
    klen = len(key) if declared_key_len is None else declared_key_len
    vlen = len(value) if declared_value_len is None else declared_value_len
    return u64(klen) + key + u32(T_STRING) + u64(vlen) + value


def kv_uint32(key: bytes, value: int, declared_key_len: int | None = None) -> bytes:
    klen = len(key) if declared_key_len is None else declared_key_len
    return u64(klen) + key + u32(T_UINT32) + u32(value)


def kv_array(key: bytes, element_type: int, elements: bytes,
             declared_count: int | None = None,
             declared_key_len: int | None = None) -> bytes:
    """An array-valued metadata entry.

    ``elements`` is raw encoded element data. ``declared_count`` is written as
    the element count and is independent of it.

    An array whose element type is itself T_ARRAY nests. Readers that recurse
    per level without a depth limit are the reason this is exposed rather than
    hidden behind a typed helper.
    """
    klen = len(key) if declared_key_len is None else declared_key_len
    count = declared_count if declared_count is not None else 0
    return u64(klen) + key + u32(T_ARRAY) + u32(element_type) + u64(count) + elements


def tensor_info(name: bytes, shape: Sequence[int], tensor_type: int = 0,
                offset: int = 0,
                declared_n_dims: int | None = None,
                declared_name_len: int | None = None) -> bytes:
    """One tensor info entry.

    ``declared_n_dims`` is written as the dimension count and defaults to
    ``len(shape)``. Setting it higher than the number of shape values you supply
    is the direct way to test whether a reader sizes its shape buffer from the
    header before reading.
    """
    n = len(shape) if declared_n_dims is None else declared_n_dims
    nlen = len(name) if declared_name_len is None else declared_name_len
    out = u64(nlen) + name + u32(n)
    out += b"".join(i64(d) for d in shape)
    out += u32(tensor_type) + u64(offset)
    return out


def file(*, version: int = 3, kvs: Sequence[bytes] = (),
         tensors: Sequence[bytes] = (),
         declared_kv_count: int | None = None,
         declared_tensor_count: int | None = None) -> bytes:
    """Assemble a complete file.

    By default the declared counts match what you passed, giving a well-formed
    file. Override either to make the header disagree with the body.
    """
    kv_count = len(kvs) if declared_kv_count is None else declared_kv_count
    t_count = len(tensors) if declared_tensor_count is None else declared_tensor_count
    return (header(version, t_count, kv_count)
            + b"".join(kvs)
            + b"".join(tensors))


def minimal() -> bytes:
    """A structurally valid single-tensor file. Useful as a corpus seed and as a
    control when you want to know a reader is not simply rejecting everything."""
    return file(tensors=[tensor_info(b"t", [1, 1])])
