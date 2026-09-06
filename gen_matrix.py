#!/usr/bin/env python3

import os
import random
import sys


def write_matrix(path, n, rng):
    with open(path, "w") as f:
        f.write(f"{n}\n")
        for _ in range(n):
            row = (rng.uniform(-10.0, 10.0) for _ in range(n))
            f.write(" ".join(f"{v:.10f}" for v in row))
            f.write("\n")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} N", file=sys.stderr)
        sys.exit(1)

    n = int(sys.argv[1])
    rng = random.Random(42)

    os.makedirs("data", exist_ok=True)
    write_matrix("data/A.txt", n, rng)
    write_matrix("data/B.txt", n, rng)
    print(f"generated {n}x{n} matrices: data/A.txt, data/B.txt")


if __name__ == "__main__":
    main()
