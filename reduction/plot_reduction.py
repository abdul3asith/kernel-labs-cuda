import matplotlib.pyplot as plt
import pandas as pd

naive = pd.read_csv("naive_results.csv")
shared = pd.read_csv("shared_results.csv")
warp = pd.read_csv("warp_results.csv")

df = pd.concat([naive, shared, warp], ignore_index=True)


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


for n in sorted(df["n"].unique()):
    subset = df[df["n"] == n]

    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel in ["naive", "shared", "warp"]:
        kdf = subset[subset["kernel"] == kernel].sort_values("block_size")
        ax.plot(kdf["block_size"], kdf["kernel_ms"], marker="o", label=kernel)
    ax.set_title(f"Kernel Time vs Block Size (n={n})")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Kernel time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, f"kernel_time_vs_block_n{n}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel in ["naive", "shared", "warp"]:
        kdf = subset[subset["kernel"] == kernel].sort_values("block_size")
        ax.plot(
            kdf["block_size"],
            kdf["throughput_melems_per_sec"],
            marker="o",
            label=kernel,
        )
    ax.set_title(f"Throughput vs Block Size (n={n})")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Throughput (million elems/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, f"throughput_vs_block_n{n}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel in ["naive", "shared", "warp"]:
        kdf = subset[subset["kernel"] == kernel].sort_values("block_size")
        ax.plot(kdf["block_size"], kdf["abs_error"], marker="o", label=kernel)
    ax.set_title(f"Absolute Error vs Block Size (n={n})")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Absolute error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, f"abs_error_vs_block_n{n}.png")

best = df.sort_values("kernel_ms").groupby(["kernel", "n"], as_index=False).first()

fig, ax = plt.subplots(figsize=(8, 5))
for kernel in ["naive", "shared", "warp"]:
    kdf = best[best["kernel"] == kernel].sort_values("n")
    ax.plot(kdf["n"], kdf["kernel_ms"], marker="o", label=kernel)
ax.set_title("Best Kernel Time vs Input Size")
ax.set_xlabel("n")
ax.set_ylabel("Best kernel time (ms)")
ax.grid(True, alpha=0.3)
ax.legend()
save_plot(fig, "best_kernel_time_vs_n.png")

fig, ax = plt.subplots(figsize=(8, 5))
for kernel in ["naive", "shared", "warp"]:
    kdf = best[best["kernel"] == kernel].sort_values("n")
    ax.plot(kdf["n"], kdf["throughput_melems_per_sec"], marker="o", label=kernel)
ax.set_title("Best Throughput vs Input Size")
ax.set_xlabel("n")
ax.set_ylabel("Best throughput (million elems/s)")
ax.grid(True, alpha=0.3)
ax.legend()
save_plot(fig, "best_throughput_vs_n.png")

print("Saved all plots.")
