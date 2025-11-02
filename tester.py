import argparse, time, ctypes, numpy as np
from pathlib import Path
import onnxruntime as ort
import json

# ---- ctypes ABI ----
class AiccTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("rank", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
    ]

def _to_tensor(arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    data_buf = (ctypes.c_float * arr.size)(*arr.ravel())
    shape_buf = (ctypes.c_size_t * arr.ndim)(*arr.shape)
    t = AiccTensor(data_buf, arr.ndim, shape_buf)
    # keep refs alive
    t._data_buf = data_buf
    t._shape_buf = shape_buf
    return t

# ---- runners ----
def run_onnx(onnx_path, inputs, warmup=1, reps=5):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])

    # Build ordered inputs
    ordered = {i.name: inputs[i.name] for i in sess.get_inputs()}

    # Warmup
    for _ in range(warmup):
        sess.run(None, ordered)

    t0 = time.perf_counter()
    for _ in range(reps):
        outs = sess.run(None, ordered)
    t1 = time.perf_counter()
    return outs, (t1 - t0) / reps, [o.name for o in sess.get_outputs()]

def run_aicc(lib_path, input_list, out_shapes, warmup=1, reps=5):
    import os, ctypes
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    lib = ctypes.CDLL(str(lib_path))
    try:
        set_thr = lib.openblas_set_num_threads
        set_thr.argtypes = [ctypes.c_int]
        set_thr(1)
    except AttributeError:
        pass
    lib.aicc_run.argtypes = [ctypes.POINTER(AiccTensor), ctypes.POINTER(AiccTensor)]
    lib.aicc_run.restype = ctypes.c_int

    in_ts = [_to_tensor(x) for x in input_list]
    InArr = AiccTensor * len(in_ts)
    cin = InArr(*in_ts)

    out_ts = [_to_tensor(np.zeros(s, np.float32)) for s in out_shapes]
    OutArr = AiccTensor * len(out_ts)
    cout = OutArr(*out_ts)

    # Warmup
    for _ in range(warmup):
        rc = lib.aicc_run(cin, cout)
        if rc != 0:
            raise RuntimeError(f"aicc_run failed (warmup): {rc}")

    t0 = time.perf_counter()
    for _ in range(reps):
        rc = lib.aicc_run(cin, cout)
        if rc != 0:
            raise RuntimeError(f"aicc_run failed: {rc}")
    t1 = time.perf_counter()

    outs = [np.ctypeslib.as_array(t.data, shape=tuple(s)).copy() for t, s in zip(out_ts, out_shapes)]
    return outs, (t1 - t0) / reps

# ---- FLOPs (optional; MatMul/Add only) ----
def estimate_flops(ir_path):
    try:
        ir = json.loads(Path(ir_path).read_text())
    except Exception:
        return None
    def numel(shp):
        return int(np.prod(shp)) if isinstance(shp, list) and all(isinstance(d,int) and d>0 for d in shp) else None
    flops = 0
    for op in ir.get("ops", []):
        k = op["op"].lower()
        if k == "matmul":
            A, B = op["inputs"]
            Ashp = ir["values"][A]["type"]["shape"]
            Bshp = ir["values"][B]["type"]["shape"]
            if len(Ashp)==2 and len(Bshp)==2 and all(isinstance(x,int) and x>0 for x in Ashp+Bshp):
                M,K = Ashp; K2,N = Bshp
                if K==K2:
                    flops += 2*M*K*N
        elif k == "add":
            out = op["outputs"][0]
            shp = ir["values"][out]["type"]["shape"]
            ne = numel(shp)
            if ne: flops += ne
    return flops or None

# ---- main harness ----
def test_model(name, reps=10, warmup=2, seed=0, atol=1e-5, rtol=1e-6):
    rng = np.random.default_rng(seed)


    candidates = [
        Path("examples/builds") / f"{name}.onnx",
        Path("examples/builds") / name / f"{name}.onnx",
        Path("examples/builds") / "mlp_example" / f"{name}.onnx",
        ]
    onnx_path = next((p for p in candidates if p.exists()), None)
    if not onnx_path:
        raise FileNotFoundError(f"could not find {name}.onnx in examples/builds/")

    base = Path("builds") / name
    lib_path = base / f"lib{name}.so"
    ir_path = base / "graph.ir.json"

    if not onnx_path.exists(): raise FileNotFoundError(onnx_path)
    if not lib_path.exists():  raise FileNotFoundError(lib_path)

    # Build inputs from ONNX IO schema in order
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {}
    input_list = []
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d>0 else 1 for d in i.shape]  # fill unknowns with 1
        arr = rng.standard_normal(shape, dtype=np.float32)
        inputs[i.name] = arr
        input_list.append(arr)

    ref_outs, t_ref, out_names = run_onnx(onnx_path, inputs, warmup=warmup, reps=reps)

    # Use concrete shapes from ref outputs
    out_shapes = [o.shape for o in ref_outs]
    my_outs, t_my = run_aicc(lib_path, input_list, out_shapes, warmup=warmup, reps=reps)

    # Compare all outputs
    diffs = []
    ok = True
    for ro, mo in zip(ref_outs, my_outs):
        diff = np.max(np.abs(ro - mo))
        rel = np.max(np.abs(ro - mo) / (np.abs(ro) + 1e-12))
        diffs.append((diff, rel))
        ok &= (diff <= atol + rtol * np.max(np.abs(ro)))

    ratio = (t_my / t_ref) * 100.0
    flops = estimate_flops(ir_path)
    flop_s = f", est_FLOPs={flops}" if flops is not None else ""

    # Report
    summary = (
        f"{name}: {'OK' if ok else 'FAIL'} | "
        f"max_abs={max(d[0] for d in diffs):.2e}, max_rel={max(d[1] for d in diffs):.2e} | "
        f"AICC={t_my*1e3:.2f} ms, ONNX={t_ref*1e3:.2f} ms ({ratio:.1f}%)"
        f"{flop_s}"
    )
    print(summary)
    return 0 if ok else 1

# defaults
DEFAULT_REPS   = 10
DEFAULT_WARMUP = 2
DEFAULT_SEED   = 0
DEFAULT_ATOL   = 2e-3   # tolerant enough for fp32 + ReLU boundaries
DEFAULT_RTOL   = 1e-5

if __name__ == "__main__":
    import os, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="model name (examples/builds/{name}.onnx, builds/{name}/lib{name}.so)")
    args = ap.parse_args()

    # Optional env overrides without changing CLI:
    reps   = int(os.getenv("AICC_REPS",   DEFAULT_REPS))
    warmup = int(os.getenv("AICC_WARMUP", DEFAULT_WARMUP))
    seed   = int(os.getenv("AICC_SEED",   DEFAULT_SEED))
    atol   = float(os.getenv("AICC_ATOL", DEFAULT_ATOL))
    rtol   = float(os.getenv("AICC_RTOL", DEFAULT_RTOL))

    raise SystemExit(test_model(args.name, reps, warmup, seed, atol, rtol))