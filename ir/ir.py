# IR Contract
# - Value: type info only (no actual data)
# - Op: pure node with op name, inputs, outputs, and attrs
# - Const ops define initial tensor data
# - Module.entry_inputs: user-fed tensors
# - Module.entry_outputs: final graph results


from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Value:  # tensor value
    id: str
    dtype: str
    shape: List[int]

@dataclass
class Op:
    id: str
    op: str
    inputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Module:
    values: Dict[str, Value]
    ops: List[Op]
    entry_inputs: List[str]
    entry_outputs: List[str]

    def to_json(self) -> dict:
        return {
            "values": {k: {"dtype":v.dtype,"shape":v.shape} for k,v in self.values.items()},
            "ops": [o.__dict__ for o in self.ops],
            "entry": {"inputs": self.entry_inputs, "outputs": self.entry_outputs},
        }
