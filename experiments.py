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


def parse_stats(path):
    stats = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            stats[key] = value
    return stats


def run_one(n, binary):
    subprocess.run(["python3", "gen_matrix.py", str(n)],
                   check=True, capture_output=True, text=True)

    subprocess.run([binary, "data/A.txt", "data/B.txt", "data/C.txt", "data/stats.txt"],
                    check=True, capture_output=True, text=True)
    stats = parse_stats("data/stats.txt")

    verify = subprocess.run(["./venv/bin/python", "verify.py",
                              "data/A.txt", "data/B.txt", "data/C.txt"],
                             capture_output=True, text=True)
    out = verify.stdout
    max_abs = next((l.split("=")[1] for l in out.splitlines() if l.startswith("max_abs_error")), "")
    max_rel = next((l.split("=")[1] for l in out.splitlines() if l.startswith("max_rel_error")), "")
    passed = "VERIFICATION: PASS" in out

    return {
        "n": n,
        "elapsed_seconds": float(stats["elapsed_seconds"]),
        "gflops": float(stats["gflops"]),
        "memory_mb": int(stats["memory_bytes"]) / (1024 * 1024),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "verified": "PASS" if passed else "FAIL",
    }


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    stop = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    binary = sys.argv[4] if len(sys.argv) > 4 else "./matmul"
    suffix = sys.argv[5] if len(sys.argv) > 5 else ""

    sizes = list(range(start, stop + 1, step))
    os.makedirs("results", exist_ok=True)

    rows = []
    t_start = time.time()
    for n in sizes:
        row = run_one(n, binary)
        rows.append(row)
        print(f"N={n:5d}  time={row['elapsed_seconds']:.4f}s  "
              f"GFLOPS={row['gflops']:.3f}  mem={row['memory_mb']:.2f}MB  "
              f"{row['verified']}  (max_rel_error={row['max_rel_error']})")

    print(f"total experiment time: {time.time() - t_start:.1f}s")

    csv_path = f"results/results{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    ns = [r["n"] for r in rows]
    times = [r["elapsed_seconds"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ns, times, marker="o", color="#2b6cb0", linewidth=1.5, markersize=4)
    ax.set_xlabel("N (matrix size)")
    ax.set_ylabel("Time, s")
    ax.set_title("Execution time vs matrix size")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = f"results/time_vs_n{suffix}.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
