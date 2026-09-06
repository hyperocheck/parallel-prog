#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>   

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while (0)

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
    double *m = (double*)malloc(sizeof(double) * total_elements);
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

// cpu
static void multiply_cpu(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double r = a[i * n + k];
            for (size_t j = 0; j < n; j++) {
                c[i * n + j] += r * b[k * n + j];
            }
        }
    }
}


// тайловое умножение с разделяемой памятью: размер тайла = размер блока
__global__ void multiply_gpu_tiled(const double *a, const double *b, double *c, int n, int tile) {

    extern __shared__ double shared[]; 
    double *tile_a = shared; // кусок матрицы A
    double *tile_b = shared + (size_t)tile * tile; // кусок матрицы B

    // threadIdx - координаты потока внутри своего блока, blockIdx - координаты блока
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * tile + ty;
    int col = blockIdx.x * tile + tx;

    // сумма скалярного произведения копится в приватном регистре потока
    double sum = 0.0;
    int num_tiles = (n + tile - 1) / tile;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * tile + tx;
        int b_row = t * tile + ty;

        // совместная загрузка: каждый из tile*tile потоков блока тащит из медленной
        // глобальной памяти (VRAM) ровно один элемент A и один элемент B вместо того,
        // чтобы каждый поток сам заново читал одни и те же данные из VRAM для каждого k
        tile_a[ty * tile + tx] = (row < n && a_col < n) ? a[(size_t)row * n + a_col] : 0.0;
        tile_b[ty * tile + tx] = (b_row < n && col < n) ? b[(size_t)b_row * n + col] : 0.0;

        // sync 1: все потоки блока должны дождаться, пока каждый допишет свою часть
        // тайла в shared-память, иначе кто-то начнет читать еще не заполненные данные
        __syncthreads();

        for (int k = 0; k < tile; k++) {
            sum += tile_a[ty * tile + k] * tile_b[k * tile + tx];
        }

        // sync 2: нельзя начинать перезаписывать tile_a/tile_b данными следующего
        // тайла, пока все потоки не дочитали текущий
        __syncthreads();
    }

    // sum накопил полный C[row][col] (просуммировали по всем k кусками по tile)
    // каждый поток пишет свой единственный элемент обратно в глобальную память
    if (row < n && col < n) {
        c[(size_t)row * n + col] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: %s <A.txt> <B.txt> <C_out.txt> <stats_out.txt> <cpu|gpu> [block_size]\n", argv[0]);
        return 1;
    }
    const char *path_mat1 = argv[1]; // matrix A path
    const char *path_mat2 = argv[2]; // matrix B path
    const char *path_mat_res = argv[3]; // result matrix file target (A x B)
    const char *path_stats = argv[4]; // stats file path
    const char *engine = argv[5];
    int tile = (argc == 7) ? atoi(argv[6]) : 16;

    bool is_gpu;
    if (strcmp(engine, "cpu") == 0) {
        is_gpu = false;
    } else if (strcmp(engine, "gpu") == 0) {
        is_gpu = true;
        if (tile <= 0 || tile > 32) {
            fprintf(stderr, "block_size must be in 1..32 (32*32=1024 is the CUDA per-block thread limit)\n");
            return 1;
        }
    } else {
        fprintf(stderr, "engine must be 'cpu' or 'gpu', got '%s'\n", engine);
        return 1;
    }

    size_t n_a, n_b;
    double *a = read_matrix(path_mat1, &n_a);
    double *b = read_matrix(path_mat2, &n_b);

    if (n_a != n_b) {
        fprintf(stderr, "matrix size mismatch\n");
        free(a); free(b);
        return 1;
    }

    size_t n = n_a;

    double *c = (double*)calloc(n * n, sizeof(double));
    if (!c) {
        perror("calloc failed");
        free(a); free(b);
        return 1;
    }

    double elapsed = 0.0;
    char device_name[256] = "cpu";

    if (!is_gpu) {
        auto t0 = std::chrono::steady_clock::now();
        multiply_cpu(a, b, c, n);
        auto t1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } else {
        // выделение памяти в VRAM видеокарты 
        double *d_a, *d_b, *d_c;
        size_t bytes_n2 = n * n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_a, bytes_n2));
        CUDA_CHECK(cudaMalloc(&d_b, bytes_n2));
        CUDA_CHECK(cudaMalloc(&d_c, bytes_n2));
        CUDA_CHECK(cudaMemcpy(d_a, a, bytes_n2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b, bytes_n2, cudaMemcpyHostToDevice));

        // конфигурация запуска: block - сколько потоков в одном блоке (tile x tile),
        // grid - сколько блоков нужно, чтобы покрыть всю матрицу C
        // shmem_bytes - сколько байт динамической shared-памяти запросить на блок,
        // ровно под два тайла tile_a+tile_b внутри кернел
        dim3 block(tile, tile);
        dim3 grid((unsigned)((n + tile - 1) / tile), (unsigned)((n + tile - 1) / tile));
        size_t shmem_bytes = 2 * (size_t)tile * tile * sizeof(double);


        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        multiply_gpu_tiled<<<grid, block, shmem_bytes>>>(d_a, d_b, d_c, (int)n, tile);
        CUDA_CHECK(cudaGetLastError());       
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop)); 

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        elapsed = (double)ms / 1000.0;

        CUDA_CHECK(cudaMemcpy(c, d_c, bytes_n2, cudaMemcpyDeviceToHost));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        snprintf(device_name, sizeof(device_name), "%s", prop.name);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    write_matrix(path_mat_res, c, (int)n);

    // 2 * n^3
    double flops = 2.0 * (double)n * (double)n * (double)n;
    double gflops = elapsed > 0.0 ? flops / elapsed / 1e9 : 0.0;
    size_t bytes = 3 * n * n * sizeof(double);

    FILE *fs = fopen(path_stats, "w");
    if (!fs) {
        perror(path_stats);
        free(a); free(b); free(c);
        return 1;
    }
    fprintf(fs, "n=%zu\n", n);
    fprintf(fs, "engine=%s\n", engine);
    fprintf(fs, "block_size=%d\n", is_gpu ? tile : 0);
    fprintf(fs, "device=%s\n", device_name);
    fprintf(fs, "elapsed_seconds=%.6f\n", elapsed);
    fprintf(fs, "flops=%.0f\n", flops);
    fprintf(fs, "gflops=%.4f\n", gflops);
    fprintf(fs, "memory_bytes=%zu\n", bytes);
    fclose(fs);

    printf("N=%zu  engine=%s  block=%d  device=%s  time=%.6f s  GFLOPS=%.4f  memory=%.2f MB\n",
           n, engine, is_gpu ? tile : 0, device_name, elapsed, gflops, (double)bytes / (1024.0 * 1024.0));

    free(a);
    free(b);
    free(c);
    return 0;
}
