from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "results.csv"
plots_dir = base_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# Common styling
FIG_BG = "black"
AX_BG = "black"
TEXT = "white"
RED = "red"
ORANGE = "orange"


def style_ax(fig, ax, title, xlabel, ylabel):
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color=TEXT)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(TEXT)
    ax.grid(True, alpha=0.2)


# 1. Kernel time vs input size
size_df = df[df["blockSize"] == 256].sort_values("n")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(size_df["n"], size_df["kernel_ms"], marker="o", color=RED, linewidth=2)
ax.set_xscale("log", base=2)
style_ax(fig, ax, "Kernel Time vs Input Size", "Input Size (n)", "Kernel Time Avg (ms)")
plt.tight_layout()
plt.savefig(plots_dir / "kernel_time_vs_n.png", facecolor=fig.get_facecolor())
plt.close()

# 2. Data size in MB vs kernel time
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    size_df["bytes"] / (1024 * 1024),
    size_df["kernel_ms"],
    marker="o",
    color=ORANGE,
    linewidth=2,
)
style_ax(
    fig,
    ax,
    "Kernel Time vs Data Size",
    "Data Size per Vector (MB)",
    "Kernel Time Avg (ms)",
)
plt.tight_layout()
plt.savefig(plots_dir / "size_mb_vs_kernel.png", facecolor=fig.get_facecolor())
plt.close()

# 3. Block size vs kernel time
block_df = df[df["n"] == (1 << 20)].sort_values("blockSize")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    block_df["blockSize"], block_df["kernel_ms"], marker="o", color=RED, linewidth=2
)
style_ax(fig, ax, "Kernel Time vs Block Size", "Block Size", "Kernel Time Avg (ms)")
plt.tight_layout()
plt.savefig(plots_dir / "kernel_time_vs_blocksize.png", facecolor=fig.get_facecolor())
plt.close()

# 4. Copy vs kernel time
rep = df[(df["n"] == (1 << 20)) & (df["blockSize"] == 256)].iloc[0]

labels = ["H2D Copy", "Kernel", "D2H Copy"]
values = [rep["h2d_ms"], rep["kernel_ms"], rep["d2h_ms"]]
colors = [RED, ORANGE, RED]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(labels, values, color=colors)
style_ax(fig, ax, "Copy Time vs Kernel Time", "Stage", "Time (ms)")
plt.tight_layout()
plt.savefig(plots_dir / "copy_vs_kernel.png", facecolor=fig.get_facecolor())
plt.close()

print(f"Plots saved to: {plots_dir}")
