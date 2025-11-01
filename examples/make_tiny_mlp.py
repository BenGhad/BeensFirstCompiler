import onnx, numpy as np
from onnx import helper, TensorProto
from pathlib import Path

# onnx: main library for creating / serializing ONNX models
# An onnx model is just a standardized format to make ts easy
# numpy: for creating numerical data
# helper: ONNX utility functions to build tensors, nodes, and graphs
# TensorProto: enum describing tensor data types (FLOAT, INT32, etc.)
# A tensor basically just represents math in dimensions.
# 0 -> () -> scalar, 1 -> (n) -> vector, 2 -> (n, m) -> matrix, 3+ TENSORS (a,b,c ) like amutidimesiinoal array.



# -----------------------------------------------------------
# Define model inputs and outputs (their names, types, and shapes)
# -----------------------------------------------------------

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1,4])
# Creates an ONNX "ValueInfoProto" describing the input:
# - name = "x" → must match node input names
# - type = FLOAT
# - shape = [1,4] → a 1×4 tensor (like a row vector)

y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1,3])
# Same, but for the output: a 1×3 tensor named "y"
# This defines the model’s visible I/O interface.

# -----------------------------------------------------------
# Create weights and bias parameters
# -----------------------------------------------------------

W = np.random.randn(4,3).astype("float32")
b = np.random.randn(3).astype("float32")
# W = weights for the layer (input_dim=4, output_dim=3)
# b = bias term (one per output neuron)
# Random initialization here for demo purposes.
# np.random.randn draws from a standard normal distribution.

# -----------------------------------------------------------
# Convert numpy arrays into ONNX constant tensors
# -----------------------------------------------------------

Wt = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten())
bt = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten())
# These "TensorProto" objects become graph initializers (constant data).
# flatten() ensures ONNX gets a 1D list of raw values.
# ONNX stores tensor data as a contiguous list, plus shape metadata.

# -----------------------------------------------------------
# Define computation nodes — each is one operation in the graph
# -----------------------------------------------------------

n1 = helper.make_node("MatMul", ["x","W"], ["z1"])
# "MatMul" performs matrix multiplication:
#   z1 = x @ W
# Inputs: "x" (runtime input), "W" (initializer)
# Output: intermediate tensor "z1"

n2 = helper.make_node("Add", ["z1","b"], ["z2"])
# "Add" adds two tensors elementwise:
#   z2 = z1 + b
# Here broadcasting applies automatically: b (shape [3]) is added to z1 (shape [1,3]).
# This matches how NumPy broadcasting works under the hood.

n3 = helper.make_node("Relu", ["z2"], ["y"])
# "Relu" applies the ReLU activation elementwise:
#   y = max(0, z2)
# No parameters, just an activation op.

# -----------------------------------------------------------
# Assemble the full computation graph
# -----------------------------------------------------------

graph = helper.make_graph(
    [n1, n2, n3],          # ordered list of nodes (computational steps)
    "tiny_mlp",            # graph name
    [x],                   # input tensors
    [y],                   # output tensors
    initializer=[Wt, bt]   # constant parameter tensors
)
# "make_graph" wires everything together:
# - nodes define the computation
# - inputs/outputs define the model interface
# - initializers store constant data (weights/bias)

# -----------------------------------------------------------
# Build the ONNX model object
# -----------------------------------------------------------

model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 13)]
    # gimme the default ops from versoin 13.
)

out_dir = Path("examples/builds/mlp_example")
out_dir.mkdir(parents=True, exist_ok=True)

onnx_path = out_dir / "tiny_mlp.onnx"
onnx.save(model, onnx_path)

# Serializes the model into a .onnx file on disk.

# -----------------------------------------------------------
# Save some example input and parameters for testing
# -----------------------------------------------------------

np.save(out_dir / "x.npy", np.random.randn(1,4).astype("float32"))
np.save(out_dir / "W.npy", W)
np.save(out_dir / "b.npy", b)

# -----------------------------------------------------------
# 🧩 Under the hood summary
# -----------------------------------------------------------
# This constructs a simple ONNX graph:
#
#     x ──▶ MatMul ──▶ Add ──▶ Relu ──▶ y
#           (W)         (b)
#
# Equivalent math:
#   y = ReLU(xW + b)
#
# The ONNX runtime (or any compatible engine) will:
#  1. Load the graph.
#  2. Allocate tensors for x, W, b. etc
#  3. Execute each node in topological order (those who do toposort for icpc :skull:).
#  4. Produce and return the output tensor y.
#
# This is effectively a 1-layer MLP.