
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/stat.h>

static void write_matrix(const char *path, size_t n, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }

    fprintf(f, "%zu\n", n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            fprintf(f, "%.10f%c", dist(rng), (j + 1 < n) ? ' ' : '\n');
        }
    }

    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s N\n", argv[0]);
        return 1;
    }

    long long n_arg = atoll(argv[1]);
    if (n_arg <= 0) {
        fprintf(stderr, "bad matrix size\n");
        return 1;
    }
    size_t n = (size_t)n_arg;

    mkdir("data", 0755); 

    char a_path[64], b_path[64];
    snprintf(a_path, sizeof(a_path), "data/A_%zu.txt", n);
    snprintf(b_path, sizeof(b_path), "data/B_%zu.txt", n);

    std::mt19937_64 rng(42);
    write_matrix(a_path, n, rng);
    write_matrix(b_path, n, rng);

    printf("generated %zux%zu matrices: %s, %s\n", n, n, a_path, b_path);
    return 0;
}
