from pathlib import Path

import modal

app = modal.App("naive-cuda-softmax")

image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.11",
).apt_install("build-essential")


@app.function(
    gpu="T4",
    image=image,
    timeout=300,
)
def compile_and_run(
    cuda_code: str,
    timer_code: str,
) -> dict[str, object]:
    import subprocess
    from pathlib import Path

    workdir = Path("/root/project")
    include_dir = workdir / "include"

    workdir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    source_file = workdir / "naive_softmax.cu"
    timer_file = include_dir / "timer.hpp"
    executable_file = workdir / "naive_softmax"

    # Write local source files into the remote Modal container.
    source_file.write_text(cuda_code)
    timer_file.write_text(timer_code)

    compile_result = subprocess.run(
        [
            "nvcc",
            str(source_file),
            "-I",
            str(include_dir),
            "-O2",
            "-o",
            str(executable_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if compile_result.returncode != 0:
        return {
            "success": False,
            "stage": "compile",
            "stdout": compile_result.stdout,
            "stderr": compile_result.stderr,
        }

    run_result = subprocess.run(
        [str(executable_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "success": run_result.returncode == 0,
        "stage": "run",
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
    }


@app.local_entrypoint()
def main() -> None:
    # This points to the version2.0 folder because this script
    # lives inside version2.0/scripts/.
    version_root = Path(__file__).resolve().parent.parent

    cuda_file = version_root / "src" / "naive_softmax.cu"
    timer_file = version_root / "include" / "timer.hpp"

    if not cuda_file.exists():
        raise FileNotFoundError(f"CUDA file not found: {cuda_file}")

    if not timer_file.exists():
        raise FileNotFoundError(f"Timer header not found: {timer_file}")

    result = compile_and_run.remote(
        cuda_file.read_text(),
        timer_file.read_text(),
    )

    print(f"Stage: {result['stage']}")

    if result["stdout"]:
        print(result["stdout"])

    if result["stderr"]:
        print("Errors:")
        print(result["stderr"])

    if not result["success"]:
        raise RuntimeError(f"Softmax failed during {result['stage']}")
