#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

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
    double *m = static_cast<double*>(malloc(sizeof(double) * total_elements));
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

static void multiply_local(const double *a_local, const double *b, double *c_local, size_t rows, size_t n) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t k = 0; k < n; k++) {
            double r = a_local[i * n + k];
            for (size_t j = 0; j < n; j++) {
                c_local[i * n + j] += r * b[k * n + j];
            }
        }
    }
}

static void row_distribution(size_t n, int size, int *sendcounts, int *displs) {
    size_t base = n / (size_t)size;
    size_t rem = n % (size_t)size;
    size_t offset = 0;
    for (int r = 0; r < size; r++) {
        size_t rows_r = base + ((size_t)r < rem ? 1 : 0);
        sendcounts[r] = (int)(rows_r * n);
        displs[r] = (int)(offset * n);
        offset += rows_r;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        MPI_Finalize();
        return 1;
    }
    const char *path_mat1 = argv[1]; // matrix A path
    const char *path_mat2 = argv[2]; // matrix B path
    const char *path_mat_res = argv[3]; // result matrix file target (A x B)
    const char *path_stats = argv[4]; // stats file path

    double *a = NULL, *b = NULL;
    size_t n = 0;

    if (rank == 0) {
        size_t n_a, n_b;
        a = read_matrix(path_mat1, &n_a);
        b = read_matrix(path_mat2, &n_b);
        if (n_a != n_b) {
            fprintf(stderr, "matrix size mismatch\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = n_a;
    }

    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        b = static_cast<double*>(malloc(sizeof(double) * n * n));
        if (!b) {
            perror("malloc failed");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int *sendcounts = static_cast<int*>(malloc(sizeof(int) * (size_t)size));
    int *displs = static_cast<int*>(malloc(sizeof(int) * (size_t)size));
    row_distribution(n, size, sendcounts, displs);
    size_t my_rows = (size_t)sendcounts[rank] / n;

    double *a_local = static_cast<double*>(malloc(sizeof(double) * my_rows * n));
    double *c_local = static_cast<double*>(calloc(my_rows * n, sizeof(double)));
    if (!a_local || !c_local) {
        perror("malloc failed");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Bcast(b, (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE,
                 a_local, (int)(my_rows * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    multiply_local(a_local, b, c_local, my_rows, n);

    double *c = NULL;
    if (rank == 0) {
        c = static_cast<double*>(malloc(sizeof(double) * n * n));
    }
    MPI_Gatherv(c_local, (int)(my_rows * n), MPI_DOUBLE,
                c, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double elapsed_local = MPI_Wtime() - t0;
    double elapsed;
    MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_matrix(path_mat_res, c, (int)n);

        // 2 * n^3
        double flops = 2.0 * (double)n * (double)n * (double)n;
        double gflops = elapsed > 0.0 ? flops / elapsed / 1e9 : 0.0;
        size_t bytes = 3 * n * n * sizeof(double);

        FILE *fs = fopen(path_stats, "w");
        if (!fs) {
            perror(path_stats);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fs, "n=%zu\n", n);
        fprintf(fs, "processes=%d\n", size);
        fprintf(fs, "elapsed_seconds=%.6f\n", elapsed);
        fprintf(fs, "flops=%.0f\n", flops);
        fprintf(fs, "gflops=%.4f\n", gflops);
        fprintf(fs, "memory_bytes=%zu\n", bytes);
        fclose(fs);

        printf("RESULT n=%zu processes=%d time=%.6f\n", n, size, elapsed);

        free(a);
        free(c);
    }

    free(b);
    free(a_local);
    free(c_local);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
