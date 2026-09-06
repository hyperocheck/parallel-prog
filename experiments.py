#!/usr/bin/env python3

import csv
import os
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREAD_COUNTS = [1, 2, 4, 8]


def parse_stats(path):
    stats = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            stats[key] = value
    return stats


def run_one(n, threads):
    subprocess.run(["python3", "gen_matrix.py", str(n)],
                   check=True, capture_output=True, text=True)

    subprocess.run(["./matmul", "data/A.txt", "data/B.txt", "data/C.txt", "data/stats.txt", str(threads)],
                    check=True, capture_output=True, text=True)
    stats = parse_stats("data/stats.txt")

    verify = subprocess.run(["./venv/bin/python", "verify.py",
                              "data/A.txt", "data/B.txt", "data/C.txt"],
                             capture_output=True, text=True)
    passed = "VERIFICATION: PASS" in verify.stdout

    return {
        "n": n,
        "threads": threads,
        "elapsed_seconds": float(stats["elapsed_seconds"]),
        "gflops": float(stats["gflops"]),
        "verified": "PASS" if passed else "FAIL",
    }


def main():
    os.makedirs("results", exist_ok=True)

    rows = []
    t_start = time.time()
    for n in SIZES:
        for threads in THREAD_COUNTS:
            row = run_one(n, threads)
            rows.append(row)
            print(f"N={n:5d} threads={threads}  time={row['elapsed_seconds']:.4f}s  "
                  f"GFLOPS={row['gflops']:.3f}  {row['verified']}")

    print(f"total experiment time: {time.time() - t_start:.1f}s")

    csv_path = "results/results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for threads in THREAD_COUNTS:
        xs = [r["n"] for r in rows if r["threads"] == threads]
        ys = [r["elapsed_seconds"] for r in rows if r["threads"] == threads]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=f"{threads} threads")
    ax.set_xlabel("N (matrix size)")
    ax.set_ylabel("Time, s")
    ax.set_title("Execution time vs matrix size (OpenMP)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = "results/time_vs_n.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
