"""ONNX writer.

ONNX is protobuf, which makes it different from the other formats here. There is
no magic number and no fixed layout — a file is a serialised ModelProto, and
every field is a tag/value pair the reader may encounter in any order or not at
all. The counts still precede the data, they are just protobuf's counts:

    tag         varint, (field_number << 3) | wire_type
    varint      LEB128; a negative int64 is encoded as unsigned and costs 10 bytes
    length-delimited    tag | varint length | that many bytes

The length prefix is the first place a declared number and the bytes behind it
can disagree, and it applies to every string, every bytes field and every nested
message including the graph itself. `delimited()` takes `declared_length` for
exactly that reason.

The second place is specific to ONNX. A TensorProto declares `dims` as a repeated
field and its payload separately as `raw_data`:

    TensorProto   dims = 1 (repeated int64) | data_type = 2 | float_data = 4
                  int64_data = 7 | name = 8 | raw_data = 9

Nothing in the format requires the product of the dims to relate to the length of
raw_data. A reader that multiplies the dims to size a buffer and then fills it
from raw_data has trusted a number it did not check. This module never checks it
either — you pass dims and you pass raw_data, and they are two separate
decisions.

Field numbers used here (from onnx.proto3) and confirmed against files written by
the reference `onnx` Python package:

    ModelProto      ir_version = 1 | producer_name = 2 | graph = 7 | opset_import = 8
    GraphProto      node = 1 | name = 2 | initializer = 5 | input = 11 | output = 12
    NodeProto       input = 1 | output = 2 | name = 3 | op_type = 4 | attribute = 5
                    domain = 7
    ValueInfoProto  name = 1 | type = 2
    TypeProto       tensor_type = 1
    TypeProto.Tensor    elem_type = 1 | shape = 2
    TensorShapeProto    dim = 1
    Dimension       dim_value = 1 | dim_param = 2
    AttributeProto  name = 1 | f = 2 | i = 3 | s = 4 | t = 5 | g = 6
                    floats = 7 | ints = 8 | type = 20
    OperatorSetIdProto  domain = 1 | version = 2

Convention: every message builder returns the message *body*, without the tag and
length prefix that a containing message would put in front of it. The container
adds those, which is what lets you give a nested message a length that disagrees
with its contents. `model()` is the exception — it is the top level, so what it
returns is the file.

Nothing here targets a particular implementation. It emits files; what a given
reader does with them is yours to find out.
"""

from __future__ import annotations

import struct
from typing import Sequence

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LEN = 2
WIRE_FIXED32 = 5

# TensorProto.DataType
UNDEFINED = 0
FLOAT = 1
UINT8 = 2
INT8 = 3
UINT16 = 4
INT16 = 5
INT32 = 6
INT64 = 7
STRING = 8
BOOL = 9
FLOAT16 = 10
DOUBLE = 11
UINT32 = 12
UINT64 = 13

# AttributeProto.AttributeType
A_UNDEFINED = 0
A_FLOAT = 1
A_INT = 2
A_STRING = 3
A_TENSOR = 4
A_GRAPH = 5
A_FLOATS = 6
A_INTS = 7
A_STRINGS = 8
A_TENSORS = 9
A_GRAPHS = 10


# --- wire primitives -------------------------------------------------------

def varint(v: int) -> bytes:
    """LEB128.

    Negative values are encoded as their unsigned 64-bit reinterpretation, which
    is what protobuf does for int32 and int64 fields. That is why a dim of -1
    occupies ten bytes and reaches a reader as 18446744073709551615 if it is read
    back as unsigned.
    """
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def varint_padded(v: int, length: int) -> bytes:
    """A non-minimal encoding of ``v`` occupying exactly ``length`` bytes.

    Protobuf permits this and the reference decoders accept it; hand-written
    parsers and length-validating proxies frequently do not. Bits above
    ``7 * length`` are dropped, so a short length silently truncates.
    """
    if length < 1:
        raise ValueError("length must be at least 1")
    if v < 0:
        v += 1 << 64
    out = bytearray()
    for i in range(length):
        b = v & 0x7F
        v >>= 7
        out.append(b if i == length - 1 else b | 0x80)
    return bytes(out)


def tag(field: int, wire: int) -> bytes:
    return varint((field << 3) | wire)


def delimited(field: int, payload: bytes,
              declared_length: int | None = None) -> bytes:
    """A length-delimited field.

    ``declared_length`` is written as the length and is independent of the bytes
    that follow it. This is the general form of the lie for every string, bytes
    field and nested message in the format.
    """
    n = len(payload) if declared_length is None else declared_length
    return tag(field, WIRE_LEN) + varint(n) + payload


def varint_field(field: int, value: int) -> bytes:
    return tag(field, WIRE_VARINT) + varint(value)


def fixed32_field(field: int, raw: bytes) -> bytes:
    return tag(field, WIRE_FIXED32) + raw


def fixed64_field(field: int, raw: bytes) -> bytes:
    return tag(field, WIRE_FIXED64) + raw


def string_field(field: int, value: bytes,
                 declared_length: int | None = None) -> bytes:
    return delimited(field, value, declared_length)


def f32(v: float) -> bytes:
    return struct.pack("<f", v)


def f64(v: float) -> bytes:
    return struct.pack("<d", v)


# --- messages --------------------------------------------------------------

def tensor(*, name: bytes = b"", dims: Sequence[int] = (),
           data_type: int = FLOAT,
           raw_data: bytes | None = None,
           float_data: Sequence[float] = (),
           int64_data: Sequence[int] = (),
           declared_raw_length: int | None = None,
           declared_name_length: int | None = None) -> bytes:
    """A TensorProto body.

    ``dims`` and the payload are never compared. Passing ``dims=[1 << 30]`` with
    four bytes of ``raw_data`` is a well-formed TensorProto by the format's own
    rules; whether a reader agrees is the question you are asking.

    ``dims`` are written as unpacked repeated varints. Protobuf parsers are
    required to accept both packed and unpacked forms for repeated scalars, and
    the unpacked form keeps each dimension independently addressable.
    """
    out = b"".join(varint_field(1, d) for d in dims)
    out += varint_field(2, data_type)
    if float_data:
        out += delimited(4, b"".join(f32(v) for v in float_data))
    if int64_data:
        out += delimited(7, b"".join(varint(v) for v in int64_data))
    if name:
        out += string_field(8, name, declared_name_length)
    if raw_data is not None:
        out += delimited(9, raw_data, declared_raw_length)
    return out


def value_info(name: bytes, elem_type: int = FLOAT,
               shape: Sequence[int | bytes] = (),
               declared_name_length: int | None = None) -> bytes:
    """A ValueInfoProto body.

    An ``int`` entry in ``shape`` becomes a fixed ``dim_value``. A ``bytes`` entry
    becomes a symbolic ``dim_param``, which is how ONNX spells a dimension that is
    not known until run time — a reader that expects every dimension to carry a
    number has to do something when it does not.
    """
    dims = b""
    for d in shape:
        if isinstance(d, (bytes, bytearray)):
            dims += delimited(1, string_field(2, bytes(d)))
        else:
            dims += delimited(1, varint_field(1, d))
    tensor_type = varint_field(1, elem_type) + delimited(2, dims)
    return (string_field(1, name, declared_name_length)
            + delimited(2, delimited(1, tensor_type)))


def attribute_int(name: bytes, value: int) -> bytes:
    return string_field(1, name) + varint_field(3, value) + varint_field(20, A_INT)


def attribute_float(name: bytes, value: float) -> bytes:
    return string_field(1, name) + fixed32_field(2, f32(value)) + varint_field(20, A_FLOAT)


def attribute_string(name: bytes, value: bytes,
                     declared_length: int | None = None) -> bytes:
    return (string_field(1, name) + string_field(4, value, declared_length)
            + varint_field(20, A_STRING))


def attribute_ints(name: bytes, values: Sequence[int],
                   declared_length: int | None = None) -> bytes:
    packed = b"".join(varint(v) for v in values)
    return (string_field(1, name) + delimited(8, packed, declared_length)
            + varint_field(20, A_INTS))


def attribute_floats(name: bytes, values: Sequence[float],
                     declared_length: int | None = None) -> bytes:
    packed = b"".join(f32(v) for v in values)
    return (string_field(1, name) + delimited(7, packed, declared_length)
            + varint_field(20, A_FLOATS))


def attribute_tensor(name: bytes, t: bytes,
                     declared_length: int | None = None) -> bytes:
    return (string_field(1, name) + delimited(5, t, declared_length)
            + varint_field(20, A_TENSOR))


def attribute_graph(name: bytes, g: bytes,
                    declared_length: int | None = None) -> bytes:
    """A subgraph attribute, as used by If, Loop and Scan.

    A graph containing a node whose attribute is a graph containing a node is how
    you control the nesting depth of a reader that recurses per level. Nothing in
    the format bounds it.
    """
    return (string_field(1, name) + delimited(6, g, declared_length)
            + varint_field(20, A_GRAPH))


def node(op_type: bytes, inputs: Sequence[bytes] = (),
         outputs: Sequence[bytes] = (), name: bytes = b"",
         attributes: Sequence[bytes] = (),
         domain: bytes | None = None) -> bytes:
    """A NodeProto body.

    The input list is written as given. No operator's arity is consulted, because
    the number of inputs an operator receives and the number its implementation
    indexes are two different things, and the gap between them is worth being able
    to express.
    """
    out = b"".join(string_field(1, i) for i in inputs)
    out += b"".join(string_field(2, o) for o in outputs)
    if name:
        out += string_field(3, name)
    out += string_field(4, op_type)
    out += b"".join(delimited(5, a) for a in attributes)
    if domain is not None:
        out += string_field(7, domain)
    return out


def graph(*, nodes: Sequence[bytes] = (), name: bytes = b"g",
          inputs: Sequence[bytes] = (), outputs: Sequence[bytes] = (),
          initializers: Sequence[bytes] = (),
          declared_node_lengths: Sequence[int | None] = ()) -> bytes:
    """A GraphProto body.

    ``declared_node_lengths`` overrides the length prefix of the node at the same
    index, letting a node claim to be longer or shorter than it is.
    """
    out = b""
    for i, n in enumerate(nodes):
        dl = declared_node_lengths[i] if i < len(declared_node_lengths) else None
        out += delimited(1, n, dl)
    out += string_field(2, name)
    out += b"".join(delimited(5, t) for t in initializers)
    out += b"".join(delimited(11, v) for v in inputs)
    out += b"".join(delimited(12, v) for v in outputs)
    return out


def opset(version: int = 13, domain: bytes = b"") -> bytes:
    return string_field(1, domain) + varint_field(2, version)


def model(graph_body: bytes, *, ir_version: int = 8,
          opsets: Sequence[bytes] = (),
          producer: bytes = b"",
          declared_graph_length: int | None = None) -> bytes:
    """A complete file.

    ``declared_graph_length`` makes the outermost length prefix disagree with the
    graph behind it, which is the earliest point at which a reader can be given a
    number it should not trust.
    """
    out = varint_field(1, ir_version)
    if producer:
        out += string_field(2, producer)
    out += delimited(7, graph_body, declared_graph_length)
    if not opsets:
        opsets = (opset(),)
    out += b"".join(delimited(8, o) for o in opsets)
    return out


def minimal() -> bytes:
    """A structurally valid single-node model. Useful as a corpus seed and as a
    control when you want to know a reader is not simply rejecting everything."""
    return model(graph(
        nodes=[node(b"Relu", [b"x"], [b"y"])],
        inputs=[value_info(b"x", FLOAT, [1, 3, 4, 4])],
        outputs=[value_info(b"y", FLOAT, [1, 3, 4, 4])],
    ))
