#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <stdint.h>

static double* read_matrix(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open file %s\n", path);
        perror(path);
        exit(1);
    }

    long long input_n;
    if (fscanf(f, "%lld", &input_n) != 1 || input_n <= 0) {
        fprintf(stderr, "bad matrix size\n");
        exit(1);
    }

    size_t n = (size_t)input_n;

    if (n > SIZE_MAX / n / sizeof(double)) {
        fprintf(stderr, "matrix size is too large\n");
        exit(1);
    }

    size_t total_elements = n * n;
    double *m = malloc(sizeof(double) * total_elements);
    if (!m) {
        perror("malloc failed");
        exit(1);
    }

    for (size_t i = 0; i < total_elements; i++) {
        if (fscanf(f, "%lf", &m[i]) != 1) {
            free(m);
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
    *n_out = n;
    return m;
}

static void write_matrix(const char *path, const double *m, int n) {
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }

    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%.10g ", m[(size_t)i * n + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

static void multiply(const double *a, const double *b, double *c, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double r = a[i * n + k];
            for (size_t j = 0; j < n; j++) {
                c[i * n + j] += r * b[k * n + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s <A.txt> <B.txt> <C_out.txt> <stats_out.txt> [num_threads]\n", argv[0]);
        return 1;
    }
    const char *path_mat1 = argv[1]; // matrix A path
    const char *path_mat2 = argv[2]; // matrix B path
    const char *path_mat_res = argv[3]; // result matrix file target (A x B)
    const char *path_stats = argv[4]; // stats file path

    if (argc == 6) {
        omp_set_num_threads(atoi(argv[5]));
    }

    size_t n_a, n_b;
    double *a = read_matrix(path_mat1, &n_a);
    double *b = read_matrix(path_mat2, &n_b);

    if (n_a != n_b) {
        free(a); free(b);
        return 1;
    }

    size_t n = n_a;

    double *c = calloc(n * n, sizeof(double));
    if (!c) {
        free(a); free(b);
        return 1;
    }

    double t0 = omp_get_wtime();
    multiply(a, b, c, n);
    double elapsed = omp_get_wtime() - t0;

    write_matrix(path_mat_res, c, n);

    // 2 * n^3
    double flops = 2.0 * (double)n * (double)n * (double)n;
    // flops / elapsed
    double gflops = elapsed > 0.0 ? flops / elapsed / 1e9 : 0.0;
    size_t bytes = 3 * n * n * sizeof(double);
    int threads = omp_get_max_threads();

    FILE *fs = fopen(path_stats, "w");
    if (!fs) {
        perror(path_stats);
        free(a); free(b); free(c);
        return 1;
    }
    fprintf(fs, "n=%zu\n", n);
    fprintf(fs, "threads=%d\n", threads);
    fprintf(fs, "elapsed_seconds=%.6f\n", elapsed);
    fprintf(fs, "flops=%.0f\n", flops);
    fprintf(fs, "gflops=%.4f\n", gflops);
    fprintf(fs, "memory_bytes=%zu\n", bytes);
    fclose(fs);

    printf("N=%zu  threads=%d  time=%.6f s  GFLOPS=%.4f  memory=%.2f MB\n",
           n, threads, elapsed, gflops, (double)bytes / (1024.0 * 1024.0));

    free(a);
    free(b);
    free(c);
    return 0;
}
