#!/usr/bin/env python3
# demo.py — simple interactive judge-facing harness
# Place next to tester.py at repo root.

import os, sys, json, subprocess, shutil, time
from pathlib import Path

# --- repo layout assumptions ---
EX_ONNX_DIR = Path("examples/builds")  # holds *.onnx or {name}/{name}.onnx
BUILDS_DIR  = Path("builds")           # holds {name}/lib{name}.so and {name}/{name}.c

# --- imports from tester ---
try:
    from tester import test_model  # uses your ONNXRuntime vs AICC harness
except Exception as e:
    print("Unable to import tester.test_model. Ensure demo.py is beside tester.py.")
    print(f"Import error: {e}")
    sys.exit(1)

# ---------- helpers ----------
def _unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def find_onnx_names():
    names = []
    if EX_ONNX_DIR.exists():
        # flat e.g. examples/builds/foo.onnx
        for p in EX_ONNX_DIR.glob("*.onnx"):
            names.append(p.stem)
        # nested e.g. examples/builds/foo/foo.onnx
        for p in EX_ONNX_DIR.glob("*/*"):
            if p.is_file() and p.suffix == ".onnx" and p.parent.name == p.stem:
                names.append(p.stem)
    return _unique(names)

def onnx_path_for(name: str) -> Path | None:
    cands = [
        EX_ONNX_DIR / f"{name}.onnx",
        EX_ONNX_DIR / name / f"{name}.onnx",
    ]
    for p in cands:
        if p.exists():
            return p
    return None

def find_c_names():
    names = []
    if BUILDS_DIR.exists():
        for d in BUILDS_DIR.iterdir():
            if d.is_dir():
                lib = d / f"lib{d.name}.so"
                if lib.exists():
                    names.append(d.name)
    return _unique(names)

def c_paths_for(name: str):
    base = BUILDS_DIR / name
    return {
        "dir": base,
        "c": base / f"{name}.c",
        "so": base / f"lib{name}.so",
        "ir": base / "graph.ir.json",
    }

def choose_compiler():
    for cc in ("cc", "clang", "gcc"):
        path = shutil.which(cc)
        if path:
            return path
    return None

def build_shared(name: str) -> Path:
    import os
    paths = c_paths_for(name)
    src = paths["c"]
    out = paths["so"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cc = choose_compiler()
    if not cc:
        raise RuntimeError("No C compiler found (cc/clang/gcc).")

    # Locate runtime include dir
    inc = os.getenv("AICC_INCLUDE")  # e.g., "codegen/c/runtime"
    guess_roots = [Path("codegen/c/runtime"), Path("runtime"), Path("include"), Path("src/runtime")]
    if not inc:
        for r in guess_roots:
            if (r / "aicc_runtime.h").exists():
                inc = str(r)
                break
    if not inc:
        raise FileNotFoundError("aicc_runtime.h not found. Set AICC_INCLUDE to its directory.")

    # If runtime has an implementation file, compile it too
    rt_c = Path(inc) / "aicc_runtime.c"
    sources = [str(src)]
    if rt_c.exists():
        sources.append(str(rt_c))


    cmd = [
        cc, "-O3", "-fPIC", "-shared", "-std=c11",
        "-ffast-math", "-march=native",
        "-I", inc,
        "-o", str(out),
        *sources,
        "-lm",
    ]
    print("Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not out.exists():
        raise RuntimeError("Compilation reported success but .so not found.")
    return out


def rerandomize_onnx(name: str, scale: float = 0.02) -> Path:
    """
    Re-randomize all FLOAT initializers in the ONNX model. Keeps shapes and names.
    Treat as quick 'retrain' for demo.
    """
    import onnx
    import numpy as np
    from onnx import numpy_helper, TensorProto

    path = onnx_path_for(name)
    if not path:
        raise FileNotFoundError(f"ONNX {name} not found in {EX_ONNX_DIR}/")
    model = onnx.load(str(path))

    changed = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            new = (np.random.randn(*arr.shape).astype(np.float32) * scale).astype(np.float32)
            # biases stay zeros if they were zeros originally; skip if 1-D and all zeros
            if not (arr.ndim == 1 and np.all(arr == 0)):
                new_t = numpy_helper.from_array(new, name=init.name)
                init.Clear()
                init.CopyFrom(new_t)
                changed += 1

    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"Re-randomized {changed} float initializers in {path}")
    return path

def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "quit"

# ---------- commands ----------
def cmd_list_onnx():
    names = find_onnx_names()
    if not names:
        print("No ONNX models found under examples/builds")
        return
    print("ONNX models:")
    for n in names:
        p = onnx_path_for(n)
        print(f"  {n:20s}  -> {p}")

def cmd_list_c():
    names = find_c_names()
    if not names:
        print("No compiled libs found under builds")
        return
    print("Compiled C libs:")
    for n in names:
        paths = c_paths_for(n)
        sz = paths["so"].stat().st_size if paths["so"].exists() else 0
        print(f"  {n:20s}  -> {paths['so']}  ({sz/1024:.1f} KiB)")

def cmd_retrain(name: str):
    rerandomize_onnx(name)

def cmd_build(name: str):
    paths = c_paths_for(name)
    if not paths["c"].exists():
        raise FileNotFoundError(f"Missing C source: {paths['c']}\n"
                                f"Generate IR+C first using your build pipeline.")
    so = build_shared(name)
    print(f"Built {so}")

def cmd_test(name: str, reps=10, warmup=2):
    # Uses tester.test_model which expects {onnx} in examples/builds and lib in builds/{name}/
    os.environ.setdefault("AICC_REPS", str(reps))
    os.environ.setdefault("AICC_WARMUP", str(warmup))
    rc = test_model(name)
    if rc != 0:
        print("Test finished: FAIL")
    else:
        print("Test finished: OK")

def print_help():
    print(
"""Commands:
  onnx                 List available ONNX models under examples/builds
  c                    List compiled C libs under builds
  retrain <name>       Re-randomize float weights in ONNX (quick demo 'train')
  build   <name>       Compile builds/<name>/<name>.c -> builds/<name>/lib<name>.so
  test    <name>       Run ONNX vs C comparison (uses tester.test_model)
  help                 Show commands
  quit                 Exit
"""
    )

# ---------- CLI ----------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Interactive demo harness")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("onnx")
    sub.add_parser("c")

    p_re = sub.add_parser("retrain")
    p_re.add_argument("name")

    p_b = sub.add_parser("build")
    p_b.add_argument("name")

    p_t = sub.add_parser("test")
    p_t.add_argument("name")
    p_t.add_argument("--reps", type=int, default=10)
    p_t.add_argument("--warmup", type=int, default=2)

    args = ap.parse_args()

    if args.cmd:
        if args.cmd == "onnx":   cmd_list_onnx()
        elif args.cmd == "c":    cmd_list_c()
        elif args.cmd == "retrain": cmd_retrain(args.name)
        elif args.cmd == "build":   cmd_build(args.name)
        elif args.cmd == "test":    cmd_test(args.name, reps=args.reps, warmup=args.warmup)
        return

    # Interactive REPL
    print("AICC Demo. Type 'help' for commands.")
    while True:
        s = safe_input("demo> ").strip()
        if not s:
            continue
        parts = s.split()
        cmd = parts[0].lower()
        if cmd in ("quit", "exit"):
            return
        elif cmd == "help":
            print_help()
        elif cmd == "onnx":
            cmd_list_onnx()
        elif cmd == "c":
            cmd_list_c()
        elif cmd == "retrain":
            if len(parts) < 2:
                print("usage: retrain <name>")
                continue
            cmd_retrain(parts[1])
        elif cmd == "build":
            if len(parts) < 2:
                print("usage: build <name>")
                continue
            try:
                cmd_build(parts[1])
            except subprocess.CalledProcessError as e:
                print("Compilation failed.")
                print("Command:", " ".join(e.cmd))
                print("Return code:", e.returncode)
            except Exception as e:
                print(f"Error: {e}")
        elif cmd == "test":
            if len(parts) < 2:
                print("usage: test <name>")
                continue
            reps = 10
            warm = 2
            if len(parts) >= 3:
                try: reps = int(parts[2])
                except: pass
            if len(parts) >= 4:
                try: warm = int(parts[3])
                except: pass
            cmd_test(parts[1], reps=reps, warmup=warm)
        else:
            print("Unknown command. Type 'help'.")

if __name__ == "__main__":
    main()
