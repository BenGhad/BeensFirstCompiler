#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frontends.onxx.importer import emit_ir_json
from passes.passes import run_all as run_passes
from codegen.c.emit_c import emit as emit_c

def resolve_onnx(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".onnx":
        if p.is_absolute(): return p
        q = (ROOT / p).resolve()
        if q.exists(): return q
        raise FileNotFoundError(q)
    name = name_or_path
    candidates = [
        ROOT / "examples" / "builds" / f"{name}.onnx",
        ROOT / "examples" / "builds" / name / f"{name}.onnx",
        ROOT / "examples" / "builds" / "mlp_example" / f"{name}.onnx",
    ]
    for c in candidates:
        if c.exists(): return c
    raise FileNotFoundError(f"could not find {name}.onnx in examples/builds/")

def build(onnx_path: str, out_root: Path, project: str, compile_c: bool = True):
    out_root.mkdir(parents=True, exist_ok=True)
    graph_path, weights_map = emit_ir_json(onnx_path, out_root=str(out_root), project=project)
    run_passes(graph_path)
    proj = Path(graph_path).parent.name  # should equal project

    c_out = out_root / proj / f"{proj}.c"
    emit_c(graph_path, str(c_out))

    built = {"graph": str(graph_path), "c": str(c_out), "weights": weights_map}
    if compile_c:
        so_out = out_root / proj / f"lib{proj}.so"   # tester expects lib prefix
        cmd = [
            "cc", "-O3", "-std=c11", "-fPIC", "-shared",
            "-I", str(ROOT / "runtime"),
            "-o", str(so_out),
            str(c_out),                 # object/source first
            "-Wl,--no-as-needed",       # keep lib even if symbols seen later
            "-lopenblas",               # libraries last
        ]
        subprocess.run(cmd, check=True)
        built["so"] = str(so_out)
    return built

def main():
    ap = argparse.ArgumentParser(description="name → ONNX → IR → passes → C → .so")
    ap.add_argument("name", help="model name (tiny_mlp) or path to .onnx")
    ap.add_argument("--no-cc", dest="no_cc", action="store_true", help="skip compiling the shared object")
    ap.add_argument("--test", action="store_true", help="run tester after build")
    args = ap.parse_args()

    onnx_path = resolve_onnx(args.name)
    built = build(str(onnx_path), ROOT / "builds",
                  project=Path(onnx_path).stem,      # safer if a .onnx path is passed
                  compile_c=not args.no_cc)
    print("graph:", built["graph"])
    print("c:", built["c"])
    if "so" in built: print("so:", built["so"])
    if built["weights"]:
        print("weights:")
        for k, v in built["weights"].items():
            print(" ", k, "→", v)

    if args.test:
        # import tester directly so CWD doesn’t matter
        import tester as T
        sys.exit(T.test_model(args.name))

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"error: cc failed with code {e.returncode}", file=sys.stderr); sys.exit(3)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
