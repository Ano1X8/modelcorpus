# modelcorpus

Writers for machine-learning model file formats, built for people testing the
parsers that read them.

Most format libraries exist to produce correct files and go out of their way to
stop you producing anything else. That is the wrong tool for testing a parser.
`modelcorpus` writes the header fields and the payload bytes as two separate
decisions, so a declared length, a dimension count or an element count can say
whatever you want regardless of what actually follows it.

That single property is the point of the library. In every format here, the
counts precede the data they describe, which means a reader has to decide
whether to trust a number before it can know if the file can satisfy it.

## Install

```
pip install -e .
```

No dependencies. Python 3.10+.

## Use

Generate a seed corpus for a fuzz target:

```
modelcorpus seeds gguf --out corpus/
```

Or build files directly:

```python
from modelcorpus.formats import torch7

# A well-formed tensor object.
good = torch7.tensor(element="Float", ndims=2, sizes=[3, 3])

# The same object claiming more dimensions than it carries.
skewed = torch7.tensor(element="Float", ndims=64, sizes=[3, 3])

# A string whose length field disagrees with its contents.
lying = torch7.standalone_string(b"abc", declared_length=-1)
```

```python
from modelcorpus.formats import darknet

cfg = darknet.minimal_cfg(filters=4096)
w = darknet.weights(darknet.floats([0.0] * 16))

# For a harness that takes one buffer and splits it in two.
buf = darknet.split_buffer(cfg, w)
```

```python
from modelcorpus.formats import gguf

# Header says four dimensions, body supplies two.
info = gguf.tensor_info(b"t", [1, 1], declared_n_dims=4)
data = gguf.file(tensors=[info])
```

```python
from modelcorpus.formats import onnx

# A tensor whose dims describe a gigabyte and whose payload is four bytes.
t = onnx.tensor(name=b"w", dims=[1 << 28], raw_data=b"\x00\x00\x00\x00")

# A graph whose length prefix disagrees with the graph behind it.
g = onnx.graph(nodes=[onnx.node(b"Relu", [b"x"], [b"y"])])
data = onnx.model(g, declared_graph_length=len(g) + 4096)
```

## Formats

| format | module | status |
|---|---|---|
| Torch7 `.t7` | `formats.torch7` | tensors, storages, strings, tables, scalars |
| Darknet `.cfg` + `.weights` | `formats.darknet` | net/convolutional/connected sections, both header versions |
| GGUF | `formats.gguf` | header, metadata KVs incl. nested arrays, tensor info |
| ONNX | `formats.onnx` | hand-rolled protobuf: models, graphs, nodes, tensors, value info, attributes incl. subgraphs |
| TFLite | planned | flatbuffers, so the useful approach is mutating real models rather than writing from scratch |

Each module documents the wire layout in its docstring and names the reference
implementation that was read while writing it. The formats are underspecified
enough that reading a real parser is the only way to get the details right, and
you will want that reference when a file behaves unexpectedly.

## What this does not do

**It does not ship crash inputs.** No file in this repository is known to crash
any specific software. The library gives you primitives; what you construct with
them is yours. This is deliberate: a repository of working denial-of-service
inputs for unfixed parsers is not a contribution.

**It does not assert anything about correctness.** There is no validation layer
and no "is this file valid" helper. Adding one would defeat the purpose.

**It does not test.** Pair it with libFuzzer, AFL++, or a plain loop calling a
loader. Seed corpus generation is the intended use.

## Why seeds matter

A parser fuzzed from an empty corpus spends its early budget rediscovering magic
bytes and header layout. Until a file parses far enough to reach a dispatch
table, none of the code behind that table is reachable at all.

Measured against OpenCV's DNN importers, comparing the same fuzz target with an
empty corpus and with the seeds this library generates. Coverage at `INITED`,
before any mutation:

| target | empty corpus | `modelcorpus` seeds |
|---|---|---|
| `readNetFromONNX` | 291 edges | **3,491 edges** (13 seeds) |
| `readNetFromDarknet` | 2 edges | **932 edges** (10 seeds) |
| `readNetFromTorch` | 225 edges | **436 edges** (9 seeds) |

The Darknet number is the striking ratio, and the reason is worth understanding
rather than quoting. That target takes one buffer and splits it into a `.cfg`
and a `.weights`, so random bytes are not merely an invalid config, they fail
before the config parser is reached at all. Two edges is the target rejecting
input at the door. The Torch gain is more modest because a single byte is
already a syntactically valid, if useless, tagged object.

ONNX is the largest absolute gain and shows what the seeds are actually buying.
The 291 baseline is protobuf refusing to decode; nothing past the deserialiser
runs. Each seed carries a different operator, and an importer dispatches on
operator type, so thirteen files reach thirteen layer handlers that no amount of
mutation on an empty corpus would arrive at in reasonable time. The number to
watch when you build your own seed set is not the file count, it is how many
distinct dispatch arms you cover.

Measured with the OpenCV oss-fuzz targets built with AddressSanitizer, run as
`<target> -runs=0 <corpus>` so no mutation has happened yet, comparing the
`INITED` line against an empty directory. `cov:` is the edge count.

## Validity

```
pip install -e '.[dev]'
pytest tests/
```

Every ONNX seed this library generates round-trips through the reference `onnx`
package (1.22.0) and passes `onnx.checker.check_model(..., full_check=True)`,
which is a real check rather than a formality — it requires every node input to
resolve to a graph input, an initializer, or an earlier node's output, and a
byte-level writer does not satisfy that by accident. The library itself depends
on nothing and never imports `onnx`; the package is a dev extra used as an
oracle, which is what `tests/` is for.

`onnx.minimal()` is additionally byte-for-byte identical to what the reference
package serialises for the same single-node model. Protobuf permits field
reordering, so that is an observation about a particular serialiser rather than
something the format owes anyone — but it is the check that pins the field
numbers and wire types, which is the part of a hand-rolled encoder most likely
to be quietly wrong.

The no-crash-inputs policy is enforced as a property, not a promise: `tests/`
asserts that no shipped seed presents an operator with fewer inputs than it is
defined to take, and that none arrives carrying a subgraph attribute. Those hold
for seeds nobody has written yet. The full seed set also runs clean through the
sanitised OpenCV targets above.

Reproduce with `modelcorpus seeds onnx --out corpus/`, zip it as
`<target>_seed_corpus.zip`, and compare `INITED` lines.

## License

Apache 2.0.
