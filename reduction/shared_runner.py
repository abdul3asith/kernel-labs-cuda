from pathlib import Path

import modal

app = modal.App("reduction-shared")

image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.11",
).apt_install("build-essential")


@app.function(gpu="T4", image=image)
def run_reduction_shared(cuda_code: str):
    import subprocess
    from pathlib import Path

    workdir = Path("/root/project")
    include_dir = workdir / "include"

    workdir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    cu_file = workdir / "reduction_shared.cu"
    exe_file = workdir / "reduction_shared"

    cu_file.write_text(cuda_code)

    compile_result = subprocess.run(
        [
            "nvcc",
            str(cu_file),
            "-o",
            str(exe_file),
        ],
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
    project_root = Path(__file__).resolve().parent.parent

    cuda_code = (project_root / "reduction" / "reduction_shared.cu").read_text()

    result = run_reduction_shared.remote(cuda_code)

    if not result["ok"]:
        print(f"Error occurred during {result['stage']}:\n{result['stderr']}")
    else:
        print(f"Output of {result['stage']}:\n{result['stdout']}")
