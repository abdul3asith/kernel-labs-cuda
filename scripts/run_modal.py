import modal

app = modal.App("cuda-kernel-lab")

image = modal.Image.debian_slim().pip_install("numpy")


@app.function(gpu="T4", image=image)
def gpu_check():
    import subprocess

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return result.stdout


@app.local_entrypoint()
def main():
    print(gpu_check.remote())
