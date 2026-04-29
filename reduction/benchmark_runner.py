from pathlib import Path

import modal

app = modal.App("reduction-benchmark")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "python3-pip")
    .pip_install("matplotlib", "pandas")
)

# Create or reuse a persistent Modal Volume
results_vol = modal.Volume.from_name(
    "reduction-benchmark-results", create_if_missing=True
)


@app.function(
    gpu="T4",
    image=image,
    timeout=1800,
    volumes={"/root/out": results_vol},
)
def run_benchmark(naive_code: str, shared_code: str, warp_code: str, plot_code: str):
    import shutil
    import subprocess
    from pathlib import Path

    workdir = Path("/root/project")
    outdir = Path("/root/out")

    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    naive_file = workdir / "reduction_naive.cu"
    shared_file = workdir / "reduction_shared.cu"
    warp_file = workdir / "reduction_warp.cu"
    plot_file = workdir / "plot_reduction.py"

    naive_exe = workdir / "reduction_naive"
    shared_exe = workdir / "reduction_shared"
    warp_exe = workdir / "reduction_warp"

    naive_file.write_text(naive_code)
    shared_file.write_text(shared_code)
    warp_file.write_text(warp_code)
    plot_file.write_text(plot_code)

    def run_cmd(cmd, stage_name):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=workdir,
        )
        return {
            "ok": result.returncode == 0,
            "stage": stage_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    result = run_cmd(
        ["nvcc", str(naive_file), "-O2", "-o", str(naive_exe)],
        "compile_naive",
    )
    if not result["ok"]:
        return result

    result = run_cmd([str(naive_exe)], "run_naive")
    if not result["ok"]:
        return result

    result = run_cmd(
        ["nvcc", str(shared_file), "-O2", "-o", str(shared_exe)],
        "compile_shared",
    )
    if not result["ok"]:
        return result

    result = run_cmd([str(shared_exe)], "run_shared")
    if not result["ok"]:
        return result

    result = run_cmd(
        ["nvcc", str(warp_file), "-O2", "-o", str(warp_exe)],
        "compile_warp",
    )
    if not result["ok"]:
        return result

    result = run_cmd([str(warp_exe)], "run_warp")
    if not result["ok"]:
        return result

    result = run_cmd(["python3", str(plot_file)], "plot")
    if not result["ok"]:
        return result

    generated_files = []
    for f in sorted(workdir.iterdir()):
        if f.suffix in [".csv", ".png"]:
            dest = outdir / f.name
            shutil.copy2(f, dest)
            generated_files.append(f.name)

    # Make volume changes visible after the function finishes
    results_vol.commit()

    return {
        "ok": True,
        "stage": "done",
        "stdout": "Benchmark + plotting completed successfully.",
        "stderr": "",
        "generated_files": generated_files,
        "volume_name": "reduction-benchmark-results",
    }


@app.local_entrypoint()
def main():
    project_root = Path(__file__).resolve().parent

    naive_code = (project_root / "reduction_naive.cu").read_text()
    shared_code = (project_root / "reduction_shared.cu").read_text()
    warp_code = (project_root / "reduction_warp.cu").read_text()
    plot_code = (project_root / "plot_reduction.py").read_text()

    result = run_benchmark.remote(naive_code, shared_code, warp_code, plot_code)

    if not result["ok"]:
        print(f"Error occurred during {result['stage']}:\n")
        print("STDOUT:\n", result["stdout"])
        print("\nSTDERR:\n", result["stderr"])
    else:
        print(result["stdout"])
        print("\nGenerated files in Modal Volume:")
        for name in result.get("generated_files", []):
            print("-", name)

        print("\nDownload them locally with:")
        print("modal volume get reduction-benchmark-results / .")
