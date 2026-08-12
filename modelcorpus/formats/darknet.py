"""Darknet .cfg and .weights writer.

Darknet models are two files. The .cfg is INI-ish text: bracketed section headers
and key=value lines, with the first section describing the network and each
subsequent section describing a layer. The .weights file is a small binary header
followed by raw float32 data, with no length prefixes and no structure of its own.

The consequence worth knowing: the .weights file carries no shape information at
all. Every allocation a loader performs is sized from integers in the .cfg text.
A few characters of text decide how much memory gets requested, which makes the
pairing the interesting thing to test rather than either file alone.

Weights header:
    int32 major | int32 minor | int32 revision
    then uint64 seen  if (major * 10 + minor) >= 2
         int32  seen  otherwise

Reference implementation consulted while writing this:
opencv/modules/dnn/src/darknet/darknet_io.cpp
"""

from __future__ import annotations

import struct
from typing import Iterable, Mapping


def weights_header(major: int = 0, minor: int = 1, revision: int = 0,
                   seen: int = 0) -> bytes:
    """The .weights preamble.

    The default (0, 1) selects the pre-v2 branch and its 32-bit ``seen`` field,
    which keeps the header at 16 bytes. Use (1, 0) and above for the 64-bit form.

    Note a major or minor above 1000 signals transposed weights, which some
    loaders reject outright. Left settable rather than clamped.
    """
    out = struct.pack("<iii", major, minor, revision)
    if (major * 10 + minor) >= 2:
        out += struct.pack("<Q", seen)
    else:
        out += struct.pack("<i", seen)
    return out


def weights(payload: bytes = b"", **header_kwargs) -> bytes:
    """A complete .weights file: header plus raw payload."""
    return weights_header(**header_kwargs) + payload


def floats(values: Iterable[float]) -> bytes:
    """Encode float32 payload data."""
    return b"".join(struct.pack("<f", v) for v in values)


def section(name: str, params: Mapping[str, object]) -> str:
    """One .cfg section."""
    lines = [f"[{name}]"]
    lines.extend(f"{k}={v}" for k, v in params.items())
    return "\n".join(lines) + "\n"


def net_section(width: int = 1, height: int = 1, channels: int = 3,
                **extra) -> str:
    """The leading [net] section.

    width, height and channels seed the running tensor shape that later layers
    compute their allocations from, so they are worth varying even though they
    look like boilerplate.
    """
    params = {"width": width, "height": height, "channels": channels}
    params.update(extra)
    return section("net", params)


def convolutional(filters: int = 1, size: int = 1, stride: int = 1,
                  pad: int = 0, batch_normalize: int = 0,
                  activation: str = "linear", **extra) -> str:
    """A [convolutional] section.

    ``filters`` and ``size`` both feed weight-blob dimensions. Loaders commonly
    check that they are positive and stop there, so large positive values are
    usually more productive than negative ones.
    """
    params = {
        "filters": filters,
        "size": size,
        "stride": stride,
        "pad": pad,
        "batch_normalize": batch_normalize,
        "activation": activation,
    }
    params.update(extra)
    return section("convolutional", params)


def connected(output: int = 1, batch_normalize: int = 0,
              activation: str = "linear", **extra) -> str:
    params = {
        "output": output,
        "batch_normalize": batch_normalize,
        "activation": activation,
    }
    params.update(extra)
    return section("connected", params)


def cfg(*sections: str) -> bytes:
    """Join sections into a .cfg file."""
    return "\n".join(sections).encode()


def minimal_cfg(**conv_kwargs) -> bytes:
    """The shortest .cfg that reaches convolutional weight loading."""
    return cfg(net_section(), convolutional(**conv_kwargs))


def split_buffer(cfg_bytes: bytes, weights_bytes: bytes) -> bytes:
    """Pack a cfg and weights pair into one buffer for a split-prefix harness.

    Fuzz harnesses for two-buffer APIs commonly take a single input and carve it
    in two. The convention here is a 4-byte little-endian prefix giving the cfg
    length, then cfg, then weights. Match this to your harness or skip it.
    """
    return struct.pack("<I", len(cfg_bytes)) + cfg_bytes + weights_bytes
