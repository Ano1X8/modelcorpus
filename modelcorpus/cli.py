"""Command line entry point.

`modelcorpus seeds <format> --out <dir>` writes a small set of structurally valid
files suitable as a fuzzing seed corpus. Seeds only. Nothing written here is
intended to crash anything; see the README on why the library ships no crash
inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .formats import darknet, gguf, torch7


def _torch7_seeds() -> dict[str, bytes]:
    return {
        "string": torch7.standalone_string(b"torch"),
        "number": torch7.number(1.0),
        "boolean": torch7.boolean(True),
        "tensor-1d": torch7.tensor(ndims=1, sizes=[4]),
        "tensor-2d": torch7.tensor(ndims=2, sizes=[3, 3]),
        "tensor-4d": torch7.tensor(ndims=4, sizes=[1, 3, 8, 8]),
        "tensor-double": torch7.tensor(element="Double", ndims=2, sizes=[2, 2]),
        "tensor-with-storage": torch7.tensor(
            ndims=1, sizes=[2],
            storage=torch7.storage(element="Float", count=2,
                                   payload=b"\x00\x00\x80\x3f" * 2),
        ),
        "table": torch7.table([(torch7.standalone_string(b"k"), torch7.number(2.0))]),
    }


def _darknet_seeds() -> dict[str, bytes]:
    seeds = {}
    for name, cfg in (
        ("conv-1", darknet.cfg(darknet.net_section(), darknet.convolutional())),
        ("conv-16", darknet.cfg(darknet.net_section(width=8, height=8),
                                darknet.convolutional(filters=16, size=3))),
        ("conv-bn", darknet.cfg(darknet.net_section(),
                                darknet.convolutional(filters=8, batch_normalize=1))),
        ("connected", darknet.cfg(darknet.net_section(), darknet.connected(output=10))),
        ("two-layer", darknet.cfg(darknet.net_section(width=8, height=8),
                                  darknet.convolutional(filters=4, size=3),
                                  darknet.convolutional(filters=8, size=1))),
    ):
        payload = darknet.floats([0.0] * 64)
        seeds[name] = darknet.split_buffer(cfg, darknet.weights(payload))
        seeds[name + "-v2hdr"] = darknet.split_buffer(
            cfg, darknet.weights(payload, major=1, minor=0))
    return seeds


def _gguf_seeds() -> dict[str, bytes]:
    return {
        "minimal": gguf.minimal(),
        "empty": gguf.file(),
        "one-kv": gguf.file(kvs=[gguf.kv_string(b"general.name", b"seed")]),
        "kv-uint32": gguf.file(kvs=[gguf.kv_uint32(b"general.version", 3)]),
        "one-tensor-1d": gguf.file(tensors=[gguf.tensor_info(b"a", [8])]),
        "one-tensor-4d": gguf.file(tensors=[gguf.tensor_info(b"a", [1, 3, 8, 8])]),
        "kv-and-tensor": gguf.file(
            kvs=[gguf.kv_string(b"general.architecture", b"llama")],
            tensors=[gguf.tensor_info(b"tok", [2, 2])],
        ),
        "array-kv": gguf.file(kvs=[
            gguf.kv_array(b"t.list", gguf.T_UINT32, gguf.u32(1) + gguf.u32(2),
                          declared_count=2)
        ]),
    }


BUILDERS = {
    "torch7": (_torch7_seeds, ".t7"),
    "darknet": (_darknet_seeds, ".bin"),
    "gguf": (_gguf_seeds, ".gguf"),
}


def cmd_seeds(args: argparse.Namespace) -> int:
    build, suffix = BUILDERS[args.format]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seeds = build()
    for name, data in sorted(seeds.items()):
        path = out / f"{name}{suffix}"
        path.write_bytes(data)
        print(f"{str(path):50s} {len(data):>8,} bytes")
    print(f"\n{len(seeds)} seeds written to {out}/")
    return 0


def cmd_formats(_: argparse.Namespace) -> int:
    for name, (build, suffix) in sorted(BUILDERS.items()):
        print(f"{name:10s} {suffix:8s} {len(build())} seeds")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelcorpus",
        description="Write ML model files for testing the parsers that read them.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_seeds = sub.add_parser("seeds", help="write a seed corpus for a format")
    p_seeds.add_argument("format", choices=sorted(BUILDERS))
    p_seeds.add_argument("--out", default="corpus", help="output directory")
    p_seeds.set_defaults(func=cmd_seeds)

    p_formats = sub.add_parser("formats", help="list supported formats")
    p_formats.set_defaults(func=cmd_formats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
