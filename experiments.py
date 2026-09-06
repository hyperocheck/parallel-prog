#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SIZES = [200, 400, 800, 1200, 1600, 2000]
TILE_SIZES = [8, 16, 32]

PYTHON = sys.executable


def parse_stats(path):
    stats = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            stats[key] = value
    return stats


def gen(n):
    subprocess.run([PYTHON, "gen_matrix.py", str(n)], check=True, capture_output=True, text=True)


def verify():
    res = subprocess.run([PYTHON, "verify.py", "data/A.txt", "data/B.txt", "data/C.txt"],
                          capture_output=True, text=True)
    return "PASS" if "VERIFICATION: PASS" in res.stdout else "FAIL"


def run_one(n, engine, tile):
    gen(n)
    cmd = ["./matmul", "data/A.txt", "data/B.txt", "data/C.txt", "data/stats.txt", engine]
    if engine == "gpu":
        cmd.append(str(tile))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    stats = parse_stats("data/stats.txt")
    return {
        "n": n,
        "engine": engine,
        "block_size": tile if engine == "gpu" else 0,
        "elapsed_seconds": float(stats["elapsed_seconds"]),
        "gflops": float(stats["gflops"]),
        "device": stats["device"],
        "verified": verify(),
    }


def main():
    subprocess.run(["make", "-s"], check=True)
    os.makedirs("results", exist_ok=True)

    rows = []
    t_start = time.time()
    for n in SIZES:
        row = run_one(n, "cpu", 0)
        rows.append(row)
        print(f"N={n:5d} engine=cpu              time={row['elapsed_seconds']:.4f}s  "
              f"GFLOPS={row['gflops']:.3f}  {row['verified']}")

        for tile in TILE_SIZES:
            row = run_one(n, "gpu", tile)
            rows.append(row)
            print(f"N={n:5d} engine=gpu block={tile:3d}  time={row['elapsed_seconds']:.4f}s  "
                  f"GFLOPS={row['gflops']:.3f}  {row['verified']}")

    print(f"total experiment time: {time.time() - t_start:.1f}s")

    csv_path = "results/results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    # график 1: время от N, CPU baseline + GPU по размерам тайла
    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs_cpu = [r["n"] for r in rows if r["engine"] == "cpu"]
    ys_cpu = [r["elapsed_seconds"] for r in rows if r["engine"] == "cpu"]
    ax.plot(xs_cpu, ys_cpu, marker="o", linewidth=1.5, markersize=4, label="CPU (1 поток)")
    for tile in TILE_SIZES:
        xs = [r["n"] for r in rows if r["engine"] == "gpu" and r["block_size"] == tile]
        ys = [r["elapsed_seconds"] for r in rows if r["engine"] == "gpu" and r["block_size"] == tile]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=f"GPU, тайл {tile}x{tile}")
    ax.set_xlabel("N (matrix size)")
    ax.set_ylabel("Time, s")
    ax.set_yscale("log")
    ax.set_title("Execution time vs matrix size (CUDA)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/time_vs_n.png", dpi=150)
    print("wrote results/time_vs_n.png")

    # график 2: ускорение GPU относительно однопоточного CPU, для каждого тайла
    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    for tile in TILE_SIZES:
        xs, ys = [], []
        for n in SIZES:
            cpu_t = next(r["elapsed_seconds"] for r in rows if r["engine"] == "cpu" and r["n"] == n)
            gpu_t = next(r["elapsed_seconds"] for r in rows
                         if r["engine"] == "gpu" and r["n"] == n and r["block_size"] == tile)
            xs.append(n)
            ys.append(cpu_t / gpu_t)
        ax2.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=f"тайл {tile}x{tile}")
    ax2.set_xlabel("N (matrix size)")
    ax2.set_ylabel("Speedup (CPU time / GPU time)")
    ax2.set_title("GPU speedup over single-thread CPU")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("results/speedup_vs_n.png", dpi=150)
    print("wrote results/speedup_vs_n.png")


if __name__ == "__main__":
    main()
