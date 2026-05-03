# profile_cuda.py
import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "curl", "git", "gnupg", "libxcb-cursor0")
    .run_commands(
        # Install Nsight Systems from the official .deb
        "wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb -O /tmp/nsys.deb || "
        "wget -q https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb -O /tmp/nsys.deb",
        "apt-get update",
        "apt-get install -y /tmp/nsys.deb",
        "rm /tmp/nsys.deb",
        "which nsys && nsys --version",  # build fails here if nsys is missing
    )
    .run_commands(
        "curl -L https://raw.githubusercontent.com/harrism/nsys_easy/main/nsys_easy "
        "-o /usr/local/bin/nsys_easy && chmod +x /usr/local/bin/nsys_easy"
    )
    .add_local_dir(".", "/root/src")
)
app = modal.App("cuda-profiler", image=image)


@app.function(gpu="T4", timeout=600)
def profile(source_file: str = "intro.cu"):
    import os
    import subprocess

    src_dir = "/root/src"
    src_path = os.path.join(src_dir, source_file)
    binary_name = os.path.splitext(source_file)[0]
    binary_path = os.path.join(src_dir, binary_name)
    report_path = "/tmp/report"  # nsys will create /tmp/report.nsys-rep

    print("=== nvidia-smi ===")
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    print(f"=== Compiling {source_file} ===")
    subprocess.run(
        ["nvcc", "-O3", "-o", binary_path, src_path],
        check=True,
    )

    # Step 1: run the trace (no stats yet — separate from analysis)
    print(f"=== Profiling {binary_name} ===")
    trace = subprocess.run(
        [
            "nsys",
            "profile",
            "-t",
            "cuda",
            "--cuda-memory-usage=false",
            "--force-overwrite=true",
            "--output",
            report_path,
            binary_path,
        ],
        capture_output=True,
        text=True,
        cwd=src_dir,
    )
    print("--- trace stdout ---")
    print(trace.stdout)
    print("--- trace stderr ---")
    print(trace.stderr)

    # Step 2: generate stats from the report
    print("=== Stats ===")
    stats = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum",
            "--report",
            "cuda_gpu_mem_time_sum",
            "--format",
            "table",
            f"{report_path}.nsys-rep",
        ],
        capture_output=True,
        text=True,
    )
    print("--- stats stdout ---")
    print(stats.stdout)
    print("--- stats stderr ---")
    print(stats.stderr)


@app.local_entrypoint()
def main(source: str = "intro.cu"):
    profile.remote(source)
