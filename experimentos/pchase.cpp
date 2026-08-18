/*
 * =============================================================================
 *  P-Chase Microbenchmark — Mapeamento da Hierarquia de Memoria da GPU
 * =============================================================================
 *
 *  Tecnica baseada em Wong et al. (2010) e Mei & Chu (2016):
 *  Uma lista encadeada (pointer chasing) e percorrida N vezes.
 *  Variando o tamanho do working set, a latencia media por acesso
 *  revela os tamanhos e latencias de cada nivel de cache.
 *
 *  Esperado para AMD Ellesmere (Polaris / GFX8):
 *    - Working set <=  16 KB  →  L1 vCache (baixa latencia)
 *    - Working set <=   2 MB  →  L2 cache  (latencia intermediaria)
 *    - Working set >    2 MB  →  VRAM GDDR5 (alta latencia)
 *
 *  A cadeia e uma permutacao aleatoria (ciclo Hamiltoniano) para
 *  derrotar prefetchers e garantir que cada acesso depende do anterior
 *  (dependencia de dados impede reordenacao/otimizacao).
 *
 *  Compilar (MinGW):
 *    g++ -Wall -O2 -D__USE_MINGW_ANSI_STDIO=1
 *        "-Wl,--enable-stdcall-fixup"
 *        -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *        pchase.cpp -o output\pchase.exe
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
/*  Kernel: single work-item pointer chasing                                  */
/*  Cada iteracao: idx = chain[idx]                                           */
/*  A dependencia de dados serializa os acessos (mede latencia, nao banda)    */
/* -------------------------------------------------------------------------- */

static const char* kernel_source =
"__kernel void pchase(\n"
"    __global const uint* chain,\n"
"    __global uint* result,\n"
"    const uint iterations)\n"
"{\n"
"    uint idx = 0;\n"
"    for (uint i = 0; i < iterations; i++) {\n"
"        idx = chain[idx];\n"
"    }\n"
"    /* Impede o compilador de eliminar o loop */\n"
"    result[0] = idx;\n"
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
/*  Gera um ciclo Hamiltoniano aleatorio de tamanho N                         */
/*  chain[perm[0]] = perm[1], chain[perm[1]] = perm[2], ...                  */
/*  chain[perm[N-1]] = perm[0]   (fecha o ciclo)                             */
/*                                                                            */
/*  Isso garante que TODOS os N elementos sao visitados em ordem aleatoria,   */
/*  derrotando prefetchers de hardware e stride detectors.                    */
/* -------------------------------------------------------------------------- */

static void generate_random_chain(unsigned int* chain, int N) {
    unsigned int* perm = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
    if (!perm) { fprintf(stderr, "malloc falhou\n"); exit(1); }

    /* Inicializa identidade */
    for (int i = 0; i < N; i++) perm[i] = (unsigned int)i;

    /* Fisher-Yates shuffle */
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        unsigned int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    /* Monta a cadeia: perm[i] aponta para perm[i+1] */
    for (int i = 0; i < N - 1; i++) {
        chain[perm[i]] = perm[i + 1];
    }
    chain[perm[N - 1]] = perm[0]; /* fecha o ciclo */

    free(perm);
}

/* -------------------------------------------------------------------------- */

int main(void) {
    cl_int err;

    srand((unsigned int)time(NULL));

    printf("=============================================================\n");
    printf("  P-Chase Microbenchmark — Hierarquia de Memoria GPU\n");
    printf("  Tecnica: Wong et al. (2010), Mei & Chu (2016)\n");
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

    cl_ulong global_mem, local_mem, cache_size;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE,  sizeof(local_mem),  &local_mem,  NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cache_size), &cache_size, NULL);

    printf("GPU: %s\n", name);
    printf("VRAM: %lu MB | LDS por CU: %lu KB | Cache reportado: %lu KB\n\n",
           (unsigned long)(global_mem / (1024*1024)),
           (unsigned long)(local_mem / 1024),
           (unsigned long)(cache_size / 1024));
    fflush(stdout);

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK_CL(err, "context");

    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "queue");

    /* --- Compilar kernel --- */
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

    cl_kernel kern = clCreateKernel(prog, "pchase", &err);
    CHECK_CL(err, "kernel");

    /* --- Definir working set sizes a testar --- */
    /* Pontos densos ao redor dos limites esperados:                          */
    /*   L1 = 16 KB per CU  →  4096 uint (cada uint = 4 bytes)              */
    /*   L2 = 2 MB shared   →  524288 uint                                   */
    typedef struct { int elements; const char* label; } TestSize;

    TestSize sizes[] = {
        /* Dentro do L1 (16 KB = 4096 uint) */
        {       64,   "256 B"  },
        {      128,   "512 B"  },
        {      256,     "1 KB" },
        {      512,     "2 KB" },
        {     1024,     "4 KB" },
        {     2048,     "8 KB" },
        {     3072,    "12 KB" },
        {     4096,    "16 KB" },  /* ← limite L1 */

        /* Transicao L1 → L2 */
        {     5120,    "20 KB" },
        {     6144,    "24 KB" },
        {     8192,    "32 KB" },
        {    12288,    "48 KB" },
        {    16384,    "64 KB" },
        {    32768,   "128 KB" },
        {    65536,   "256 KB" },
        {   131072,   "512 KB" },
        {   262144,     "1 MB" },
        {   393216,   "1.5 MB" },
        {   524288,     "2 MB" },  /* ← limite L2 */

        /* Transicao L2 → VRAM */
        {   655360,   "2.5 MB" },
        {   786432,     "3 MB" },
        {  1048576,     "4 MB" },
        {  2097152,     "8 MB" },
        {  4194304,    "16 MB" },
        {  8388608,    "32 MB" },
    };
    const int NUM_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    /* --- Parametros de benchmark --- */
    const unsigned int MIN_ITERS   = 1024 * 256;  /* 256K minimo */
    const unsigned int TARGET_ITERS = 1024 * 1024; /* 1M alvo */
    const int WARMUP = 2;
    const int RUNS   = 5;

    /* Buffer de resultado (1 uint para impedir dead code elimination) */
    cl_mem d_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_uint), NULL, &err);
    CHECK_CL(err, "buf result");

    /* --- Header da tabela --- */
    printf("%-10s  %8s  %10s  %10s  %10s  %s\n",
           "Working", "Elems", "Latencia", "Latencia", "Latencia",
           "Nivel");
    printf("%-10s  %8s  %10s  %10s  %10s  %s\n",
           "Set", "(uint)", "Media(ns)", "Min(ns)", "Max(ns)",
           "Inferido");
    printf("----------  --------  ----------  ----------  ----------  ----------\n");
    fflush(stdout);

    double prev_lat = 0.0;

    for (int s = 0; s < NUM_SIZES; s++) {
        int N = sizes[s].elements;

        /* Ajustar iteracoes: para arrays pequenos, mais traversals */
        unsigned int iters = TARGET_ITERS;
        if ((unsigned int)N < MIN_ITERS) {
            /* Garantir pelo menos 4 traversals completos do ciclo */
            unsigned int min_traversals = 4;
            unsigned int from_traversals = (unsigned int)N * min_traversals;
            if (from_traversals > iters) iters = from_traversals;
        }
        /* Para arrays muito grandes, reduzir para manter tempo razoavel */
        if (N > 2097152) iters = 512 * 1024;   /* 512K para > 8 MB */
        if (N > 4194304) iters = 256 * 1024;   /* 256K para > 16 MB */

        /* Gerar cadeia aleatoria */
        unsigned int* h_chain = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
        if (!h_chain) { fprintf(stderr, "malloc falhou para N=%d\n", N); exit(1); }
        generate_random_chain(h_chain, N);

        /* Upload para GPU */
        cl_mem d_chain = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         (size_t)N * sizeof(unsigned int), h_chain, &err);
        CHECK_CL(err, "buf chain");

        /* Configurar argumentos */
        clSetKernelArg(kern, 0, sizeof(cl_mem), &d_chain);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &d_result);
        clSetKernelArg(kern, 2, sizeof(cl_uint), &iters);

        size_t global = 1; /* Single work-item para medir latencia pura */
        size_t local  = 1;

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            cl_event ev;
            clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }

        /* Benchmark */
        double total_ns = 0, mn = 1e18, mx = 0;

        for (int r = 0; r < RUNS; r++) {
            cl_event ev;
            clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
            clWaitForEvents(1, &ev);

            double ns = get_ns(ev);
            double lat = ns / (double)iters; /* latencia por acesso */

            total_ns += lat;
            if (lat < mn) mn = lat;
            if (lat > mx) mx = lat;

            clReleaseEvent(ev);
        }

        double avg_lat = total_ns / RUNS;

        /* Inferir nivel de cache baseado na latencia medida.
         * Calibrado para Ellesmere (GFX8):
         *   L1 ~39 ns | transicao ~58 ns | L2 ~80-130 ns | VRAM ~150+ ns */
        const char* nivel;
        if (avg_lat < 45.0)        nivel = "L1 vCache";
        else if (avg_lat < 65.0)   nivel = "L1->L2";
        else if (avg_lat < 135.0)  nivel = "L2 Cache";
        else                       nivel = "VRAM";

        printf("%-10s  %8d  %10.2f  %10.2f  %10.2f  %s",
               sizes[s].label, N, avg_lat, mn, mx, nivel);

        /* Marcar transicoes significativas */
        if (prev_lat > 0 && avg_lat > prev_lat * 1.5) {
            printf("  ← salto %.1fx", avg_lat / prev_lat);
        }

        printf("\n");
        fflush(stdout);

        prev_lat = avg_lat;

        clReleaseMemObject(d_chain);
        free(h_chain);
    }

    /* --- Resumo --- */
    printf("\n=============================================================\n");
    printf("  Interpretacao:\n");
    printf("  - Latencia estavel em faixa baixa  → dados no L1 (16 KB/CU)\n");
    printf("  - Salto de latencia ~16-32 KB      → transicao L1 → L2\n");
    printf("  - Latencia estavel intermediaria    → dados no L2 (2 MB)\n");
    printf("  - Salto de latencia ~2 MB           → transicao L2 → VRAM\n");
    printf("  - Latencia alta e estavel           → dados na VRAM GDDR5\n");
    printf("=============================================================\n\n");

    /* Ler resultado para verificar que o kernel realmente executou */
    cl_uint final_idx;
    clEnqueueReadBuffer(q, d_result, CL_TRUE, 0, sizeof(cl_uint), &final_idx, 0, NULL, NULL);
    printf("[DEBUG] Indice final retornado: %u\n", final_idx);

    /* Cleanup */
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseMemObject(d_result);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    printf("\n=== Fim do P-Chase ===\n");
    return 0;
}
