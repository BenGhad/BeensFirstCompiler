#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frontends.onxx.importer import emit_ir_json
from passes.passes import run_all as run_passes
from codegen.c.emit_c import emit as emit_c

def build(onnx_path: str, out_root: str, project: str | None, compile_c: bool):
    graph_path, weights_map = emit_ir_json(onnx_path, out_root=out_root, project=project)
    run_passes(graph_path)  # in-place optimize + validate
    proj = Path(graph_path).parent.name

    c_out = Path(out_root) / proj / f"{proj}.c"
    emit_c(graph_path, str(c_out))

    built = {"graph": str(graph_path), "c": str(c_out), "weights": weights_map}
    if compile_c:
        so_out = Path(out_root) / proj / f"{proj}.so"
        cmd = [
            "cc", "-O3", "-std=c11", "-fPIC", "-shared",
            "-I", str(ROOT / "runtime"),
            "-o", str(so_out), str(c_out),
        ]
        subprocess.run(cmd, check=True)
        built["so"] = str(so_out)
    return built

def main():
    ap = argparse.ArgumentParser(description="ONNX → IR → passes → C [→ .so]")
    ap.add_argument("onnx", help="Path to .onnx")
    ap.add_argument("-o", "--out-root", default="builds")
    ap.add_argument("-p", "--project", default=None, help="Folder name; default = ONNX stem")
    ap.add_argument("--cc", action="store_true", help="Also compile C to shared object")
    args = ap.parse_args()

    built = build(args.onnx, args.out_root, args.project, args.cc)
    print("graph:", built["graph"])
    print("c:", built["c"])
    if "so" in built: print("so:", built["so"])
    if built["weights"]:
        print("weights:")
        for k, v in built["weights"].items():
            print(" ", k, "→", v)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"error: cc failed with code {e.returncode}", file=sys.stderr); sys.exit(3)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
