#!/usr/bin/env python3

import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

RESULT_RE = re.compile(r"RESULT n=(\d+) processes=(\d+) time=([\d.]+)")


def collect():
    rows = {}
    for path in sorted(glob.glob("results/logs/*.log")):
        with open(path) as f:
            text = f.read()
        m = RESULT_RE.search(text)
        if not m:
            print(f"warning: no RESULT line in {path}, skipping")
            continue
        n, procs, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
        rows[(n, procs)] = t
    return rows


def main():
    raw = collect()
    if not raw:
        print("no results found in results/logs/")
        return

    sizes = sorted({n for n, _ in raw})
    proc_counts = sorted({p for _, p in raw})

    rows = []
    for n in sizes:
        base_time = raw.get((n, 1))
        for p in proc_counts:
            if (n, p) not in raw:
                continue
            t = raw[(n, p)]
            gflops = 2.0 * n ** 3 / t / 1e9 if t > 0 else 0.0
            speedup = base_time / t if base_time else None
            efficiency = speedup / p if speedup is not None else None
            rows.append({
                "n": n,
                "processes": p,
                "elapsed_seconds": t,
                "gflops": gflops,
                "speedup": speedup,
                "efficiency": efficiency,
            })

    os.makedirs("results", exist_ok=True)
    csv_path = "results/results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for p in proc_counts:
        xs = [r["n"] for r in rows if r["processes"] == p]
        ys = [r["elapsed_seconds"] for r in rows if r["processes"] == p]
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=f"{p} processes")
    ax.set_xlabel("N (matrix size)")
    ax.set_ylabel("Time, s")
    ax.set_title("Execution time vs matrix size (MPI, cluster)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = "results/time_vs_n.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
