# profile_cuda.py
import modal

# NGC PyTorch image — comes with nvcc, nsys, ncu, PyTorch, Triton preinstalled.
# First build is ~2-3 min; cached afterwards.
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.10-py3",
        add_python="3.11",
    )
    .run_commands(
        "which nsys && nsys --version",  # fail fast if nsys is missing
    )
    .add_local_dir(".", "/root/src")
)

# A Modal Volume to persist .nsys-rep report files between runs.
# This is what lets you download them to your laptop.
reports_volume = modal.Volume.from_name("nsys-reports", create_if_missing=True)

app = modal.App("cuda-profiler", image=image)


@app.function(
    gpu="T4",
    timeout=600,
    volumes={"/reports": reports_volume},
)
def profile(source_file: str = "intro.cu"):
    import os
    import subprocess

    src_dir = "/root/src"
    src_path = os.path.join(src_dir, source_file)
    binary_name = os.path.splitext(source_file)[0]
    binary_path = os.path.join(src_dir, binary_name)
    report_path = f"/reports/{binary_name}_report"

    # Diagnostic: check what nsys thinks of this environment
    print("=== nsys status ===")
    print(
        subprocess.run(["nsys", "status", "-e"], capture_output=True, text=True).stdout
    )

    print(f"\n=== Compiling {source_file} ===")
    subprocess.run(["nvcc", "-O3", "-o", binary_path, src_path], check=True)

    print(f"\n=== Profiling {binary_name} ===")
    env = os.environ.copy()
    # Force CUPTI to use the in-process injection method, which works
    # in containers without elevated privileges
    env["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"

    trace = subprocess.run(
        [
            "nsys",
            "profile",
            "-t",
            "cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace-fork-before-exec=true",
            "--cuda-flush-interval=100",
            "--force-overwrite=true",
            "--output",
            report_path,
            binary_path,
        ],
        capture_output=True,
        text=True,
        cwd=src_dir,
        env=env,
    )

    print(trace.stdout)
    print("--- stderr ---")
    print(trace.stderr)
    # ... rest of your stats calls
    # Stats — read the report and print human-friendly tables
    print("\n=== Kernel timing summary ===")
    print(
        subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "table",
                f"{report_path}.nsys-rep",
            ],
            capture_output=True,
            text=True,
        ).stdout
    )

    print("\n=== Memory transfer summary ===")
    print(
        subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_mem_time_sum",
                "--format",
                "table",
                f"{report_path}.nsys-rep",
            ],
            capture_output=True,
            text=True,
        ).stdout
    )

    # Persist the report file so we can download it locally
    reports_volume.commit()
    print(f"\nReport saved to volume: {report_path}.nsys-rep")
    print(
        f"Download with: modal volume get nsys-reports {binary_name}_report.nsys-rep ./reports/"
    )


@app.local_entrypoint()
def main(source: str = "intro.cu"):
    profile.remote(source)
