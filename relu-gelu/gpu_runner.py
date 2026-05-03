from pathlib import Path

import modal

app = modal.App("cuda-relu-gelu-lab")

image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.11",
).apt_install("build-essential")


@app.function(gpu="T4", image=image, timeout=300)
def run_activation(cuda_code: str):
    import subprocess
    from pathlib import Path

    workdir = Path("/root/project")
    workdir.mkdir(parents=True, exist_ok=True)

    cu_file = workdir / "activation_gpu.cu"
    exe_file = workdir / "activation_gpu"

    cu_file.write_text(cuda_code)

    compile_result = subprocess.run(
        ["nvcc", str(cu_file), "-o", str(exe_file)],
        capture_output=True,
        text=True,
    )

    if compile_result.returncode != 0:
        return {
            "ok": False,
            "stage": "compile",
            "stdout": compile_result.stdout,
            "stderr": compile_result.stderr,
        }

    run_result = subprocess.run(
        [str(exe_file)],
        capture_output=True,
        text=True,
    )

    return {
        "ok": run_result.returncode == 0,
        "stage": "run",
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
    }


@app.local_entrypoint()
def main():
    # runner is inside relu-gelu/, so parent.parent = project root
    project_root = Path(__file__).resolve().parent.parent

    cuda_code = (project_root / "relu-gelu" / "activation_gpu.cu").read_text()

    result = run_activation.remote(cuda_code)

    if not result["ok"]:
        print(f"Error occurred during {result['stage']}:")
        print("STDOUT:")
        print(result["stdout"])
        print("STDERR:")
        print(result["stderr"])
    else:
        print(f"Output of {result['stage']}:")
        print(result["stdout"])

        if result["stderr"]:
            print("STDERR:")
            print(result["stderr"])
