/*
 * =============================================================================
 *  P-Chase v2 — Multi-Work-Item Microbenchmark
 * =============================================================================
 *
 *  Extensao do P-Chase v1 com dois modos de execucao:
 *
 *  Modo 1 (Single): 1 work-item, mede latencia pura por nivel de cache.
 *                    Reproduz v1 como baseline.
 *
 *  Modo 2 (Multi):  1 work-item por CU, cada um com cadeia independente.
 *                   Cria contencao real no L2 compartilhado, revelando
 *                   a transicao L2 -> VRAM com mais clareza.
 *
 *  Motivacao: na v1 (single work-item), a diferenca L2 vs VRAM foi de
 *  apenas ~2% (133 vs 136 ns), pois um unico work-item nao gera
 *  contencao de bandwidth. Com N work-items competindo pelo L2,
 *  a latencia VRAM deve ser significativamente maior.
 *
 *  Compilar (MinGW):
 *    g++ -Wall -O2 -D__USE_MINGW_ANSI_STDIO=1
 *        "-Wl,--enable-stdcall-fixup"
 *        -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *        experimentos\pchase_v2.cpp -o output\pchase_v2.exe
 *        "C:\Windows\System32\OpenCL.dll"
 *
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

/* -------------------------------------------------------------------------- */
/*  Kernels                                                                   */
/* -------------------------------------------------------------------------- */

static const char* kernel_source =
"__kernel void pchase_single(\n"
"    __global const uint* chain,\n"
"    __global uint* result,\n"
"    const uint iterations)\n"
"{\n"
"    uint idx = 0;\n"
"    for (uint i = 0; i < iterations; i++) {\n"
"        idx = chain[idx];\n"
"    }\n"
"    result[0] = idx;\n"
"}\n"
"\n"
"__kernel void pchase_multi(\n"
"    __global const uint* chain,\n"
"    __global uint* result,\n"
"    const uint iterations,\n"
"    const uint chain_size)\n"
"{\n"
"    uint wid = get_global_id(0);\n"
"    uint base = wid * chain_size;\n"
"    uint idx = 0;\n"
"    for (uint i = 0; i < iterations; i++) {\n"
"        idx = chain[base + idx];\n"
"    }\n"
"    result[wid] = idx;\n"
"}\n";

/* -------------------------------------------------------------------------- */

#define CHECK_CL(e, m) do { \
    if ((e) != CL_SUCCESS) { \
        fprintf(stderr, "[ERRO] %s: %d\n", (m), (e)); \
        exit(1); \
    } \
} while(0)

static double get_ns(cl_event ev) {
    cl_ulong t0, t1;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (double)(t1 - t0);
}

/* -------------------------------------------------------------------------- */
/*  Gera ciclo Hamiltoniano aleatorio de tamanho N                            */
/* -------------------------------------------------------------------------- */

static void generate_random_chain(unsigned int* chain, int N) {
    unsigned int* perm = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
    if (!perm) { fprintf(stderr, "malloc falhou\n"); exit(1); }

    for (int i = 0; i < N; i++) perm[i] = (unsigned int)i;

    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        unsigned int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    for (int i = 0; i < N - 1; i++) {
        chain[perm[i]] = perm[i + 1];
    }
    chain[perm[N - 1]] = perm[0];

    free(perm);
}

/* -------------------------------------------------------------------------- */

typedef struct { int elements; const char* label; } TestSize;

static TestSize sizes[] = {
    /* Dentro do L1 (16 KB = 4096 uint) */
    {       256,     "1 KB" },
    {      1024,     "4 KB" },
    {      2048,     "8 KB" },
    {      4096,    "16 KB" },  /* ← limite L1 */

    /* Transicao L1 → L2 */
    {      5120,    "20 KB" },
    {      8192,    "32 KB" },
    {     16384,    "64 KB" },
    {     65536,   "256 KB" },
    {    131072,   "512 KB" },
    {    262144,     "1 MB" },
    {    524288,     "2 MB" },  /* ← limite L2 */

    /* Transicao L2 → VRAM */
    {   1048576,     "4 MB" },
    {   2097152,     "8 MB" },
};
#define NUM_SIZES (int)(sizeof(sizes) / sizeof(sizes[0]))

/* Limite do multi: 8 MB por WI × 36 CUs = 288 MB total */
#define MULTI_MAX_ELEMS 2097152

static const char* infer_level(double lat) {
    if (lat < 45.0)        return "L1";
    else if (lat < 65.0)   return "L1->L2";
    else if (lat < 135.0)  return "L2";
    else                   return "VRAM";
}

/* -------------------------------------------------------------------------- */
/*  Benchmark: single work-item                                               */
/* -------------------------------------------------------------------------- */

static void bench_single(cl_context ctx, cl_command_queue q, cl_kernel kern,
                         double* latencies) {
    cl_int err;
    cl_mem d_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_uint), NULL, &err);
    CHECK_CL(err, "buf result single");

    const unsigned int TARGET_ITERS = 1024 * 1024;
    const int WARMUP = 2, RUNS = 5;

    for (int s = 0; s < NUM_SIZES; s++) {
        int N = sizes[s].elements;

        unsigned int iters = TARGET_ITERS;
        if (N > 2097152) iters = 512 * 1024;
        if (N > 4194304) iters = 256 * 1024;

        unsigned int* h_chain = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
        generate_random_chain(h_chain, N);

        cl_mem d_chain = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         (size_t)N * sizeof(unsigned int), h_chain, &err);
        CHECK_CL(err, "buf chain single");

        clSetKernelArg(kern, 0, sizeof(cl_mem), &d_chain);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &d_result);
        clSetKernelArg(kern, 2, sizeof(cl_uint), &iters);

        size_t global = 1, local = 1;

        for (int w = 0; w < WARMUP; w++) {
            cl_event ev;
            clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }

        double total = 0;
        for (int r = 0; r < RUNS; r++) {
            cl_event ev;
            clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            total += get_ns(ev) / (double)iters;
            clReleaseEvent(ev);
        }

        latencies[s] = total / RUNS;
        clReleaseMemObject(d_chain);
        free(h_chain);
    }

    clReleaseMemObject(d_result);
}

/* -------------------------------------------------------------------------- */
/*  Benchmark: multi work-item (1 por CU)                                     */
/* -------------------------------------------------------------------------- */

static void bench_multi(cl_context ctx, cl_command_queue q, cl_kernel kern,
                        int num_cus, double* latencies) {
    cl_int err;
    cl_mem d_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                      (size_t)num_cus * sizeof(cl_uint), NULL, &err);
    CHECK_CL(err, "buf result multi");

    const unsigned int TARGET_ITERS = 1024 * 1024;
    const int WARMUP = 2, RUNS = 5;

    for (int s = 0; s < NUM_SIZES; s++) {
        int N = sizes[s].elements;

        if (N > MULTI_MAX_ELEMS) {
            latencies[s] = -1.0;
            continue;
        }

        unsigned int iters = TARGET_ITERS;
        if (N > 524288)  iters = 512 * 1024;
        if (N > 1048576) iters = 256 * 1024;

        size_t total_elems = (size_t)num_cus * N;
        unsigned int* h_chain = (unsigned int*)malloc(total_elems * sizeof(unsigned int));
        if (!h_chain) {
            fprintf(stderr, "malloc falhou para multi N=%d (%zu bytes)\n",
                    N, total_elems * sizeof(unsigned int));
            latencies[s] = -1.0;
            continue;
        }

        for (int cu = 0; cu < num_cus; cu++) {
            generate_random_chain(h_chain + (size_t)cu * N, N);
        }

        cl_mem d_chain = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         total_elems * sizeof(unsigned int), h_chain, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clCreateBuffer falhou para multi N=%d: %d\n", N, err);
            free(h_chain);
            latencies[s] = -1.0;
            continue;
        }

        cl_uint chain_size = (cl_uint)N;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &d_chain);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &d_result);
        clSetKernelArg(kern, 2, sizeof(cl_uint), &iters);
        clSetKernelArg(kern, 3, sizeof(cl_uint), &chain_size);

        size_t global = (size_t)num_cus;
        size_t local  = 1;

        for (int w = 0; w < WARMUP; w++) {
            cl_event ev;
            clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }

        double total = 0;
        for (int r = 0; r < RUNS; r++) {
            cl_event ev;
            clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            total += get_ns(ev) / (double)iters;
            clReleaseEvent(ev);
        }

        latencies[s] = total / RUNS;
        clReleaseMemObject(d_chain);
        free(h_chain);
    }

    clReleaseMemObject(d_result);
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */

int main(void) {
    cl_int err;
    srand((unsigned int)time(NULL));

    printf("=============================================================\n");
    printf("  P-Chase v2 — Single vs Multi Work-Item\n");
    printf("  Tecnica: Wong et al. (2010), Mei & Chu (2017)\n");
    printf("=============================================================\n\n");
    fflush(stdout);

    /* --- Setup OpenCL --- */
    cl_platform_id plat;
    clGetPlatformIDs(1, &plat, NULL);

    cl_device_id dev;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    CHECK_CL(err, "GPU nao encontrada");

    char name[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);

    cl_uint num_cus;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_cus), &num_cus, NULL);

    cl_ulong global_mem, local_mem, cache_size;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE,  sizeof(local_mem),  &local_mem,  NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cache_size), &cache_size, NULL);

    cl_uint clock_mhz;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_mhz), &clock_mhz, NULL);

    printf("GPU: %s\n", name);
    printf("Compute Units: %u | Clock: %u MHz\n", num_cus, clock_mhz);
    printf("VRAM: %lu MB | LDS/CU: %lu KB | L1 reportado: %lu KB\n",
           (unsigned long)(global_mem / (1024*1024)),
           (unsigned long)(local_mem / 1024),
           (unsigned long)(cache_size / 1024));
    printf("Modo multi: %u work-items (1 por CU)\n\n", num_cus);
    fflush(stdout);

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK_CL(err, "context");

    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "queue");

    /* --- Compilar kernels --- */
    const char* src[] = { kernel_source };
    cl_program prog = clCreateProgramWithSource(ctx, 1, src, NULL, &err);
    CHECK_CL(err, "source");

    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Build:\n%s\n", log);
        exit(1);
    }

    cl_kernel k_single = clCreateKernel(prog, "pchase_single", &err);
    CHECK_CL(err, "kernel single");
    cl_kernel k_multi = clCreateKernel(prog, "pchase_multi", &err);
    CHECK_CL(err, "kernel multi");

    /* --- Fase 1: Single work-item (baseline) --- */
    printf("=== FASE 1: Single work-item (baseline) ===\n\n");
    fflush(stdout);

    double lat_single[NUM_SIZES];
    bench_single(ctx, q, k_single, lat_single);

    printf("%-10s  %10s  %s\n", "Working", "Latencia", "Nivel");
    printf("%-10s  %10s  %s\n", "Set", "(ns)", "Inferido");
    printf("----------  ----------  ----------\n");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf("%-10s  %10.2f  %s\n",
               sizes[s].label, lat_single[s], infer_level(lat_single[s]));
    }
    printf("\n");
    fflush(stdout);

    /* --- Fase 2: Multi work-item (1 por CU) --- */
    printf("=== FASE 2: Multi work-item (%u CUs) ===\n\n", num_cus);
    fflush(stdout);

    double lat_multi[NUM_SIZES];
    bench_multi(ctx, q, k_multi, (int)num_cus, lat_multi);

    printf("%-10s  %10s  %s\n", "Working", "Latencia", "Nivel");
    printf("%-10s  %10s  %s\n", "Set/CU", "(ns)", "Inferido");
    printf("----------  ----------  ----------\n");
    for (int s = 0; s < NUM_SIZES; s++) {
        if (lat_multi[s] < 0) {
            printf("%-10s  %10s  %s\n", sizes[s].label, "---", "skip");
        } else {
            printf("%-10s  %10.2f  %s\n",
                   sizes[s].label, lat_multi[s], infer_level(lat_multi[s]));
        }
    }
    printf("\n");
    fflush(stdout);

    /* --- Comparacao --- */
    printf("=== COMPARACAO: Single vs Multi ===\n\n");
    printf("%-10s  %10s  %10s  %10s  %s\n",
           "Working", "Single", "Multi", "Ratio", "Footprint");
    printf("%-10s  %10s  %10s  %10s  %s\n",
           "Set/CU", "(ns)", "(ns)", "(M/S)", "Total Multi");
    printf("----------  ----------  ----------  ----------  ----------\n");

    for (int s = 0; s < NUM_SIZES; s++) {
        size_t footprint = (size_t)sizes[s].elements * sizeof(unsigned int) * num_cus;
        char fp_str[32];

        if (footprint < 1024 * 1024) {
            snprintf(fp_str, sizeof(fp_str), "%lu KB",
                     (unsigned long)(footprint / 1024));
        } else {
            snprintf(fp_str, sizeof(fp_str), "%lu MB",
                     (unsigned long)(footprint / (1024*1024)));
        }

        if (lat_multi[s] < 0) {
            printf("%-10s  %10.2f  %10s  %10s  %s\n",
                   sizes[s].label, lat_single[s], "---", "---", fp_str);
        } else {
            double ratio = lat_multi[s] / lat_single[s];
            printf("%-10s  %10.2f  %10.2f  %10.2fx  %s\n",
                   sizes[s].label, lat_single[s], lat_multi[s], ratio, fp_str);
        }
    }

    /* --- Resumo --- */
    printf("\n=============================================================\n");
    printf("  Interpretacao:\n");
    printf("  Single: mede latencia pura (sem contencao)\n");
    printf("  Multi:  mede latencia com contencao no L2\n");
    printf("  Ratio > 1.0 indica contencao no nivel de cache\n");
    printf("  Salto grande no ratio marca transicao para nivel inferior\n");
    printf("=============================================================\n\n");

    /* Cleanup */
    clReleaseKernel(k_single);
    clReleaseKernel(k_multi);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    printf("=== Fim do P-Chase v2 ===\n");
    return 0;
}
