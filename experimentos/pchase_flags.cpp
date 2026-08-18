/*
 * =============================================================================
 *  pchase_flags.cpp — P-chase com controle explicito do nivel de cache
 * =============================================================================
 *
 *  PERGUNTA
 *  --------
 *  O experimento `niveis_cache.cpp` mostrou que as flags GLC/SLC produzem
 *  encodings distintos (DC500000 / DC510000 / DC530000) e tempos distintos.
 *  Mas ficou uma limitacao: pular o L2 alem do L1 acrescentava apenas 3-8%,
 *  o que e compativel com DUAS hipoteses que o throughput nao distingue:
 *
 *     (H1) nessas cargas o L2 contribui pouco alem do que o L1 ja captura
 *     (H2) SLC=1 nao e bypass rigido do L2, e sim dica de nao-alocacao
 *
 *  METODO
 *  ------
 *  P-chase (Wong et al. 2010; Mei & Chu 2017): uma lista encadeada em ciclo
 *  aleatorio percorrida por `idx = chain[idx]`. Cada load depende do anterior,
 *  entao o tempo por iteracao E a latencia de acesso -- nao ha sobreposicao
 *  possivel. Variando o tamanho do conjunto de trabalho, a latencia salta
 *  quando o conjunto deixa de caber em cada nivel.
 *
 *  Este experimento repete a curva de latencia sob as tres configuracoes de
 *  flags. As previsoes sao falsificaveis:
 *
 *     sem flags   -> patamares em L1 (~39 ns), L2 (~86-140 ns), VRAM
 *     glc         -> o patamar de L1 DESAPARECE; mesmo com 1 KB de conjunto
 *                    de trabalho a latencia deve ser a do L2
 *     glc slc     -> se H1, a curva fica na latencia de VRAM desde 1 KB;
 *                    se H2, ela estaciona na latencia de L2
 *
 *  A MEDIDA DECISIVA e o ponto de 1 KB: um conjunto de trabalho desse tamanho
 *  cabe folgadamente em L1 e em L2, entao a latencia observada ali identifica
 *  sem ambiguidade qual nivel esta servindo o acesso.
 *
 *  POR QUE ESTE EXPERIMENTO E IMUNE AO CONFUNDIDOR DA SERIALIZACAO
 *  --------------------------------------------------------------
 *  Nos experimentos anteriores, o `s_waitcnt vmcnt(0)` do bloco asm eliminava
 *  a sobreposicao de latencia e se confundia com o efeito das flags. Aqui a
 *  cadeia de dependencias JA forca a serializacao: o endereco do proximo load
 *  so existe depois que o anterior retorna. O waitcnt nao muda nada, e o
 *  confundidor desaparece por construcao.
 *
 *  Single work-item: mede latencia pura, sem contencao entre CUs.
 *
 *  REQUISITOS
 *  ----------
 *  Runtime OpenCL Clang/LLVM com inline asm GCN (ROCm) e ROC_ENABLE_PRE_VEGA=1.
 *
 *  Build:
 *    g++ -Wall -O2 experimentos/pchase_flags.cpp -o output/pchase_flags -lOpenCL
 *  Execucao:
 *    source setup/env_rocm.sh && ./output/pchase_flags
 *
 *  Saida: resultados/pchase_flags.csv + isa/rocm_gfx803_pchase_flags.bin
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

/* ------------------------------------------------------------------------- */
/*  Kernels — as tres variantes diferem apenas no modificador do load         */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
"#define PCHASE(NOME, F)                                            \\\n"
"__kernel void NOME(__global const uint* chain,                     \\\n"
"                   __global uint* result,                          \\\n"
"                   const uint iterations)                          \\\n"
"{                                                                  \\\n"
"    uint idx = 0;                                                  \\\n"
"    for (uint i = 0; i < iterations; i++) {                        \\\n"
"        __global const uint* p = &chain[idx];                      \\\n"
"        __asm__ volatile(                                          \\\n"
"            \"flat_load_dword %0, %1\" F \"\\n\"                       \\\n"
"            \"s_waitcnt vmcnt(0)\\n\"                                \\\n"
"            : \"=&v\"(idx) : \"v\"(p) : \"memory\");                    \\\n"
"    }                                                              \\\n"
"    result[0] = idx;                                               \\\n"
"}\n"
"PCHASE(pchase_L1,   \"\")\n"
"PCHASE(pchase_L2,   \" glc\")\n"
"PCHASE(pchase_VRAM, \" glc slc\")\n";

/* ------------------------------------------------------------------------- */

#define CHECK(err, msg) do {                                                   \
    if ((err) != CL_SUCCESS) {                                                 \
        fprintf(stderr, "ERRO %d em %s (linha %d)\n", (int)(err), (msg), __LINE__); \
        exit(1);                                                               \
    }                                                                          \
} while (0)

static double get_ns(cl_event ev)
{
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (double)(t1 - t0);
}

/*
 * Cadeia em ciclo aleatorio cobrindo todos os N elementos exatamente uma vez.
 * O embaralhamento impede que o prefetcher de hardware antecipe o proximo
 * endereco -- essencial para que a latencia medida seja a do nivel de cache
 * e nao a de um acesso ja pre-buscado.
 */
static void generate_random_chain(unsigned int* chain, int N)
{
    unsigned int* perm = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
    for (int i = 0; i < N; i++) perm[i] = (unsigned int)i;

    /* Fisher-Yates com gerador proprio, para a cadeia ser reproduzivel */
    unsigned int seed = 12345u;
    for (int i = N - 1; i > 0; i--) {
        seed = seed * 1103515245u + 12345u;
        int j = (int)((seed >> 16) % (unsigned int)(i + 1));
        unsigned int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    for (int i = 0; i < N - 1; i++) chain[perm[i]] = perm[i + 1];
    chain[perm[N - 1]] = perm[0];

    free(perm);
}

typedef struct { int elements; const char* label; } TestSize;

/* Mesmos tamanhos do pchase_v2.cpp, para a curva ser comparavel */
static TestSize sizes[] = {
    {       256,   "1 KB" },     /* <<< medida decisiva: cabe em L1 e em L2 */
    {      1024,   "4 KB" },
    {      2048,   "8 KB" },
    {      4096,  "16 KB" },     /* limite do L1                            */
    {      5120,  "20 KB" },
    {      8192,  "32 KB" },
    {     16384,  "64 KB" },
    {     65536, "256 KB" },
    {    131072, "512 KB" },
    {    262144,   "1 MB" },
    {    524288,   "2 MB" },     /* limite do L2                            */
    {   1048576,   "4 MB" },
    {   2097152,   "8 MB" },
};
#define NUM_SIZES (int)(sizeof(sizes) / sizeof(sizes[0]))

#define NUM_VAR 3
static const char* VAR_FN[NUM_VAR]  = { "pchase_L1", "pchase_L2", "pchase_VRAM" };
static const char* VAR_LAB[NUM_VAR] = { "sem flags", "glc", "glc slc" };
static const char* VAR_DESC[NUM_VAR] = {
    "L1 -> L2 -> VRAM", "contorna L1", "contorna L1 e L2"
};

int main(void)
{
    printf("================================================================\n");
    printf(" P-chase com controle explicito do nivel de cache\n");
    printf(" Single work-item (latencia pura, sem contencao)\n");
    printf("================================================================\n\n");

    cl_int err;
    cl_uint np = 0;
    clGetPlatformIDs(0, NULL, &np);
    cl_platform_id* plats = (cl_platform_id*)malloc(np * sizeof(cl_platform_id));
    clGetPlatformIDs(np, plats, NULL);
    cl_platform_id plat = NULL;
    cl_device_id dev = NULL;
    for (cl_uint i = 0; i < np; i++)
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) {
            plat = plats[i];
            break;
        }
    if (!dev) { fprintf(stderr, "Nenhuma GPU OpenCL encontrada.\n"); return 1; }

    char buf[512];
    clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
    printf("Plataforma: %s\n", buf);
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
    printf("Dispositivo: %s\n", buf);
    cl_ulong l1 = 0;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(l1), &l1, NULL);
    printf("Cache L1 reportado: %llu KB\n\n", (unsigned long long)(l1 >> 10));

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK(err, "clCreateCommandQueue");

    const char* srcs[] = { kernel_source };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, NULL, &err);
    CHECK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t ls = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ls);
        char* log = (char*)malloc(ls + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, log, NULL);
        log[ls] = '\0';
        fprintf(stderr, "FALHA no build (%d):\n%s\n", err, log);
        fprintf(stderr, "\nExige runtime OpenCL Clang/LLVM com inline asm GCN (ROCm).\n");
        return 1;
    }

    {
        size_t bs = 0;
        clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bs, NULL);
        unsigned char* bin = (unsigned char*)malloc(bs);
        unsigned char* bins[] = { bin };
        clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), bins, NULL);
        FILE* f = fopen("isa/rocm_gfx803_pchase_flags.bin", "wb");
        if (f) {
            fwrite(bin, 1, bs, f);
            fclose(f);
            printf("[ISA] isa/rocm_gfx803_pchase_flags.bin (%lu bytes)\n\n",
                   (unsigned long)bs);
        }
        free(bin);
    }

    cl_kernel kern[NUM_VAR];
    for (int v = 0; v < NUM_VAR; v++) {
        kern[v] = clCreateKernel(prog, VAR_FN[v], &err);
        CHECK(err, VAR_FN[v]);
    }

    cl_mem d_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_uint), NULL, &err);
    CHECK(err, "buffer result");

    static double lat[NUM_VAR][NUM_SIZES];
    const unsigned int TARGET_ITERS = 1024 * 1024;
    const int WARMUP = 2, RUNS = 5;

    printf("%10s %14s %14s %14s\n", "conj. trab.",
           "sem flags(ns)", "glc(ns)", "glc slc(ns)");
    printf("--------------------------------------------------------------\n");

    for (int s = 0; s < NUM_SIZES; s++) {
        int N = sizes[s].elements;
        unsigned int iters = TARGET_ITERS;
        if (N > 524288)  iters = 512 * 1024;
        if (N > 1048576) iters = 256 * 1024;

        unsigned int* h_chain =
            (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
        generate_random_chain(h_chain, N);
        cl_mem d_chain = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        (size_t)N * sizeof(unsigned int), h_chain, &err);
        CHECK(err, "buffer chain");

        size_t global = 1, local = 1;
        for (int v = 0; v < NUM_VAR; v++) {
            clSetKernelArg(kern[v], 0, sizeof(cl_mem), &d_chain);
            clSetKernelArg(kern[v], 1, sizeof(cl_mem), &d_result);
            clSetKernelArg(kern[v], 2, sizeof(cl_uint), &iters);

            for (int w = 0; w < WARMUP; w++) {
                cl_event ev;
                err = clEnqueueNDRangeKernel(q, kern[v], 1, NULL, &global, &local,
                                             0, NULL, &ev);
                CHECK(err, "warmup");
                clWaitForEvents(1, &ev);
                clReleaseEvent(ev);
            }
            double total = 0.0;
            for (int r = 0; r < RUNS; r++) {
                cl_event ev;
                err = clEnqueueNDRangeKernel(q, kern[v], 1, NULL, &global, &local,
                                             0, NULL, &ev);
                CHECK(err, "enqueue");
                clWaitForEvents(1, &ev);
                total += get_ns(ev) / (double)iters;
                clReleaseEvent(ev);
            }
            lat[v][s] = total / RUNS;
        }

        printf("%10s %14.2f %14.2f %14.2f\n", sizes[s].label,
               lat[0][s], lat[1][s], lat[2][s]);

        clReleaseMemObject(d_chain);
        free(h_chain);
    }

    /* ------------------------------------------------------------------ */
    /*  A medida decisiva: conjunto de trabalho de 1 KB                    */
    /* ------------------------------------------------------------------ */
    printf("\n================================================================\n");
    printf(" MEDIDA DECISIVA — conjunto de trabalho de 1 KB\n");
    printf(" (cabe folgadamente em L1 e em L2; a latencia identifica o nivel)\n");
    printf("================================================================\n");
    for (int v = 0; v < NUM_VAR; v++)
        printf("  %-10s (%-17s) %8.2f ns   %5.2fx sobre o L1\n",
               VAR_LAB[v], VAR_DESC[v], lat[v][0], lat[v][0] / lat[0][0]);

    printf("\n Leitura:\n");
    printf("  - se `glc` subiu para perto da latencia de L2 medida nos tamanhos\n");
    printf("    grandes, o bypass do L1 esta confirmado;\n");
    printf("  - se `glc slc` subiu ALEM disso, aproximando-se da latencia de VRAM,\n");
    printf("    entao SLC=1 de fato contorna o L2 (hipotese H1);\n");
    printf("  - se `glc slc` estacionou junto de `glc`, entao SLC=1 e apenas dica\n");
    printf("    de nao-alocacao e nao um bypass rigido (hipotese H2).\n");

    /* Patamares medidos nos tamanhos grandes, como referencia */
    printf("\n Referencias da curva sem flags:\n");
    printf("   1 KB  (L1)   = %8.2f ns\n", lat[0][0]);
    printf("   256 KB (L2)  = %8.2f ns\n", lat[0][7]);
    printf("   8 MB (VRAM)  = %8.2f ns\n", lat[0][NUM_SIZES - 1]);
    printf("================================================================\n");

    FILE* csv = fopen("resultados/pchase_flags.csv", "w");
    if (csv) {
        fprintf(csv, "elementos,bytes,rotulo,variante,flags,latencia_ns\n");
        for (int s = 0; s < NUM_SIZES; s++)
            for (int v = 0; v < NUM_VAR; v++)
                fprintf(csv, "%d,%zu,\"%s\",%s,\"%s\",%.4f\n",
                        sizes[s].elements,
                        (size_t)sizes[s].elements * sizeof(unsigned int),
                        sizes[s].label, VAR_FN[v], VAR_LAB[v], lat[v][s]);
        fclose(csv);
        printf("\nCSV: resultados/pchase_flags.csv\n");
    }

    for (int v = 0; v < NUM_VAR; v++) clReleaseKernel(kern[v]);
    clReleaseMemObject(d_result);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(plats);
    return 0;
}
