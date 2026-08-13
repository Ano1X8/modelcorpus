"""Validation for the ONNX writer, using the reference `onnx` package as an oracle.

The library itself depends on nothing. This file does — that is the point of it.
A hand-rolled protobuf encoder is only worth trusting if something that did not
write it agrees on what it says.

    pip install -e '.[dev]'
    pytest tests/

The properties asserted here are deliberately stronger than "it parses":

* every generated seed satisfies `onnx.checker.check_model(full_check=True)`,
  which enforces graph-level invariants a byte-level writer will not satisfy by
  accident — every node input must resolve to a graph input, an initializer, or
  an earlier node's output;
* what the caller asked for is what a third-party decoder reads back;
* the shipped seed set stays well-formed as it grows. Those checks are properties
  rather than name greps precisely so that a seed added a year from now cannot
  quietly violate the repository's own no-crash-inputs policy.
"""

from __future__ import annotations

import struct

import pytest

onnx_ref = pytest.importorskip("onnx")
from onnx import checker  # noqa: E402

from modelcorpus.cli import _onnx_seeds  # noqa: E402
from modelcorpus.formats import onnx  # noqa: E402

SEEDS = _onnx_seeds()


# --- wire primitives -------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (0, b"\x00"), (1, b"\x01"), (127, b"\x7f"),
    (128, b"\x80\x01"), (300, b"\xac\x02"),
])
def test_varint_matches_spec(value, expected):
    assert onnx.varint(value) == expected


def test_negative_varint_is_ten_bytes():
    assert len(onnx.varint(-1)) == 10


def test_varint_padded_is_non_minimal_but_decodes_the_same():
    padded = onnx.varint_padded(1, 5)
    assert len(padded) == 5
    assert padded != onnx.varint(1)
    # A tensor carrying the padded form reads back as the value it encodes.
    body = onnx.tag(2, onnx.WIRE_VARINT) + padded
    t = onnx_ref.TensorProto()
    t.ParseFromString(body)
    assert t.data_type == 1


def test_declared_length_is_independent_of_payload():
    honest = onnx.delimited(1, b"ab")
    lying = onnx.delimited(1, b"ab", declared_length=99)
    assert honest == b"\x0a\x02ab"
    assert lying == b"\x0a\x63ab"


# --- round-trip ------------------------------------------------------------

def test_tensor_round_trips_dims_and_type():
    raw = struct.pack("<ff", 1.0, 2.0)
    body = onnx.tensor(name=b"w", dims=[1, 2], data_type=onnx.FLOAT, raw_data=raw)
    t = onnx_ref.TensorProto()
    t.ParseFromString(body)
    assert list(t.dims) == [1, 2]
    assert t.data_type == onnx.FLOAT
    assert t.name == "w"
    assert t.raw_data == raw


def test_dims_and_payload_are_not_reconciled():
    """The library states no relationship between dims and raw_data. Assert that."""
    body = onnx.tensor(dims=[1 << 20], data_type=onnx.FLOAT, raw_data=b"\x00" * 4)
    t = onnx_ref.TensorProto()
    t.ParseFromString(body)
    assert list(t.dims) == [1 << 20]
    assert len(t.raw_data) == 4


def test_node_does_not_enforce_operator_arity():
    body = onnx.node(b"Add", [b"only_one"], [b"y"])
    n = onnx_ref.NodeProto()
    n.ParseFromString(body)
    assert list(n.input) == ["only_one"]


# --- the shipped seed set --------------------------------------------------

@pytest.mark.parametrize("name", sorted(SEEDS))
def test_seed_parses(name):
    onnx_ref.load_model_from_string(SEEDS[name])


@pytest.mark.parametrize("name", sorted(SEEDS))
def test_seed_passes_full_check(name):
    model = onnx_ref.load_model_from_string(SEEDS[name])
    checker.check_model(model, full_check=True)


@pytest.mark.parametrize("name", sorted(SEEDS))
def test_seed_carries_no_subgraph(name):
    """Shipped seeds nest to depth zero.

    `attribute_graph` exists so a caller can choose the nesting depth of a reader
    that recurses per level. A seed that arrives with depth already stacked would
    make that choice on the caller's behalf, which is not this repository's job.
    """
    model = onnx_ref.load_model_from_string(SEEDS[name])
    for node in model.graph.node:
        for attr in node.attribute:
            assert attr.type not in (
                onnx_ref.AttributeProto.GRAPH,
                onnx_ref.AttributeProto.GRAPHS,
            ), f"{name}: {node.op_type} carries a subgraph attribute"


@pytest.mark.parametrize("name", sorted(SEEDS))
def test_seed_nodes_are_well_formed(name):
    """No seed presents an operator with fewer inputs than it is defined to take.

    This is the property behind the repository's no-crash-inputs policy, written
    as a property rather than a search for a particular helper name so that it
    still holds for seeds nobody has written yet. An importer that indexes an
    input it did not check is a real bug class; a seed corpus is not the place to
    hand out a trigger for it.
    """
    minimum_inputs = {
        "Gemm": 2, "Conv": 2, "MatMul": 2, "Add": 2, "Mul": 2, "Sub": 2,
        "Div": 2, "Concat": 2, "Reshape": 2, "BatchNormalization": 5,
    }
    model = onnx_ref.load_model_from_string(SEEDS[name])
    for node in model.graph.node:
        need = minimum_inputs.get(node.op_type, 1)
        assert len(node.input) >= need, (
            f"{name}: {node.op_type} has {len(node.input)} inputs, needs {need}")


def test_minimal_is_structurally_valid():
    model = onnx_ref.load_model_from_string(onnx.minimal())
    checker.check_model(model, full_check=True)
    assert [n.op_type for n in model.graph.node] == ["Relu"]


def test_minimal_matches_reference_serialisation():
    """Byte-identity with what the reference package writes for the same model.

    Protobuf permits field reordering, so this is an observation about the
    serialiser in a particular version of `onnx` rather than a guarantee the
    format owes us. It is worth asserting anyway: it is the check that pins the
    field numbers and wire types, and if it ever breaks the right response is to
    read the diff, not to delete the test.
    """
    reference = onnx_ref.helper.make_model(
        onnx_ref.helper.make_graph(
            [onnx_ref.helper.make_node("Relu", ["x"], ["y"])],
            "g",
            [onnx_ref.helper.make_tensor_value_info(
                "x", onnx_ref.TensorProto.FLOAT, [1, 3, 4, 4])],
            [onnx_ref.helper.make_tensor_value_info(
                "y", onnx_ref.TensorProto.FLOAT, [1, 3, 4, 4])],
        ),
        opset_imports=[onnx_ref.helper.make_opsetid("", 13)],
    )
    reference.ir_version = 8
    assert onnx.minimal() == reference.SerializeToString()
