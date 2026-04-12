import modal

app = modal.App("cuda-kernel-lab")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy")
    .apt_install("build-essential")
)

CUDA_CODE = r"""
#include <stdio.h>
__global__ void hello_kernel() {
    printf("Hello from GPU Kernel %d\n", threadIdx.x);
}

int main() {
    hello_kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
"""


@app.function(gpu="T4", image=image)
def compile_and_run():
    import subprocess
    from pathlib import Path

    workdir = Path("/root/project")
    workdir.mkdir(parents=True, exist_ok=True)

    cu_file = workdir / "hello.cu"
    exe_file = workdir / "hello"

    cu_file.write_text(CUDA_CODE)

    compile_result = subprocess.run(
        ["nvcc", str(cu_file), "-o", str(exe_file)],
        capture_output=True,
        text=True,
    )

    if compile_result.returncode != 0:
        return {
            "compile_stdout": compile_result.stdout,
            "compile_stderr": compile_result.stderr,
            "run_stdout": "",
            "run_stderr": "",
        }

    run_result = subprocess.run(
        [str(exe_file)],
        capture_output=True,
        text=True,
    )

    return {
        "compile_stdout": compile_result.stdout,
        "compile_stderr": compile_result.stderr,
        "run_stdout": run_result.stdout,
        "run_stderr": run_result.stderr,
    }


@app.local_entrypoint()
def main():
    result = compile_and_run.remote()
    print("=== COMPILE STDOUT ===")
    print(result["compile_stdout"])
    print("=== COMPILE STDERR ===")
    print(result["compile_stderr"])
    print("=== RUN STDOUT ===")
    print(result["run_stdout"])
    print("=== RUN STDERR ===")
    print(result["run_stderr"])
