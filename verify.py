#!/usr/bin/env python3

import sys

import numpy as np


def read_matrix(path):
    with open(path) as f:
        n = int(f.readline())
        data = np.loadtxt(f, dtype=np.float64)
    return data.reshape(n, n)


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} A.txt B.txt C.txt", file=sys.stderr)
        sys.exit(2)

    a_path, b_path, c_path = sys.argv[1:4]
    a = read_matrix(a_path)
    b = read_matrix(b_path)
    c = read_matrix(c_path)

    expected = a @ b
    abs_err = np.abs(c - expected)
    denom = np.where(np.abs(expected) == 0, 1.0, np.abs(expected))
    rel_err = abs_err / denom

    ok = bool(np.allclose(c, expected, rtol=1e-8, atol=1e-6))

    print(f"n={a.shape[0]}")
    print(f"max_abs_error={abs_err.max():.3e}")
    print(f"max_rel_error={rel_err.max():.3e}")
    print("VERIFICATION: PASS" if ok else "VERIFICATION: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
