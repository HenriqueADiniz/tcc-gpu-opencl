/*
 * =============================================================================
 *  glc_scaling.cpp — Como o custo de contornar o L1 escala com o tamanho do
 *                    problema (varredura de N sobre o fatorial 2x2)
 * =============================================================================
 *
 *  Continuacao de glc_factorial.cpp, que mediu um unico tamanho (N=1024) e
 *  obteve, para o bypass do L1 isolado, o fator (3)/(1) = 1.36x.
 *
 *  PERGUNTA
 *  --------
 *  Esse fator e constante ou cresce com N?
 *
 *  A multiplicacao de matrizes naive faz 2N^3 loads globais para 2N^3 flops:
 *  a intensidade aritmetica e constante em N. O que muda com N e o tamanho do
 *  conjunto de trabalho por workgroup em relacao aos 16 KB do L1 por CU.
 *
 *    - Se o ganho do L1 vem de reuso dentro do workgroup, o fator deve ser
 *      aproximadamente estavel, pois o reuso e limitado pelo TILE, nao por N.
 *    - Se parte do ganho vem de reuso entre workgroups (linhas de A ainda
 *      residentes no L1 de execucoes vizinhas), o fator deve CAIR quando o
 *      conjunto de trabalho ultrapassa o L1.
 *
 *  MODELO
 *  ------
 *  Para cada celula ajusta-se  T(N) = a*N^3 + b*N^2  por minimos quadrados.
 *  O termo N^3 domina o custo assintotico; a razao entre os coeficientes `a`
 *  das celulas (3) e (1) da o custo assintotico de operar sem o L1, livre dos
 *  efeitos de borda (lancamento do kernel, prologo) que vivem no termo N^2.
 *
 *  Reporta-se ainda a largura de banda efetiva de load:
 *      bytes emitidos = 2 * N^3 * sizeof(float)
 *      BW_efetiva     = bytes / tempo
 *  Para GLC=1 essa e a taxa vista pelo L2 (SLC=0, o L2 continua ativo);
 *  para GLC=0 e a taxa apos a filtragem do L1.
 *
 *  Build:
 *    g++ -Wall -O2 experimentos/glc_scaling.cpp -o output/glc_scaling -lOpenCL
 *  Execucao:
 *    source setup/env_rocm.sh && ./output/glc_scaling
 *    ./output/glc_scaling 256 512 1024 2048        # tamanhos personalizados
 *
 *  Saida: resultados/glc_scaling.csv
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define TILE    16
#define WARMUP  3
#define RUNS    10

/* Tamanhos padrao: multiplos de TILE, cobrindo de bem menor que o L2 (2 MB)
   ate muito maior. Uma matriz NxN de float ocupa 4*N^2 bytes. */
static const int DEFAULT_SIZES[] = { 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072 };

/* ------------------------------------------------------------------------- */
/*  Kernels — identicos aos de glc_factorial.cpp                             */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
"#define PROLOGO                                    \\\n"
"    int row = get_global_id(1);                    \\\n"
"    int col = get_global_id(0);                    \\\n"
"    if (row >= N || col >= N) return;              \\\n"
"    float sum = 0.0f;\n"
"\n"
/* (1) asm, GLC=0, loads sobrepostos — baseline do fatorial */
"__kernel void mm_glc0_overlap(__global const float* restrict A,\n"
"                              __global const float* restrict B,\n"
"                              __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) {\n"
"        float a, b;\n"
"        __global const float* pa = &A[row * N + k];\n"
"        __global const float* pb = &B[k * N + col];\n"
"        __asm__ volatile(\n"
"            \"flat_load_dword %0, %2\\n\"\n"
"            \"flat_load_dword %1, %3\\n\"\n"
"            \"s_waitcnt vmcnt(0)\\n\"\n"
"            : \"=v\"(a), \"=v\"(b) : \"v\"(pa), \"v\"(pb) : \"memory\");\n"
"        sum += a * b;\n"
"    }\n"
"    C[row * N + col] = sum;\n"
"}\n"
"\n"
/* (2) asm, GLC=0, loads serializados */
"__kernel void mm_glc0_serial(__global const float* restrict A,\n"
"                             __global const float* restrict B,\n"
"                             __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) {\n"
"        float a, b;\n"
"        __global const float* pa = &A[row * N + k];\n"
"        __global const float* pb = &B[k * N + col];\n"
"        __asm__ volatile(\n"
"            \"flat_load_dword %0, %2\\n\"\n"
"            \"s_waitcnt vmcnt(0)\\n\"\n"
"            \"flat_load_dword %1, %3\\n\"\n"
"            \"s_waitcnt vmcnt(0)\\n\"\n"
"            : \"=v\"(a), \"=v\"(b) : \"v\"(pa), \"v\"(pb) : \"memory\");\n"
"        sum += a * b;\n"
"    }\n"
"    C[row * N + col] = sum;\n"
"}\n"
"\n"
/* (3) asm, GLC=1, loads sobrepostos — isola o bypass do L1 */
"__kernel void mm_glc1_overlap(__global const float* restrict A,\n"
"                              __global const float* restrict B,\n"
"                              __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) {\n"
"        float a, b;\n"
"        __global const float* pa = &A[row * N + k];\n"
"        __global const float* pb = &B[k * N + col];\n"
"        __asm__ volatile(\n"
"            \"flat_load_dword %0, %2 glc\\n\"\n"
"            \"flat_load_dword %1, %3 glc\\n\"\n"
"            \"s_waitcnt vmcnt(0)\\n\"\n"
"            : \"=v\"(a), \"=v\"(b) : \"v\"(pa), \"v\"(pb) : \"memory\");\n"
"        sum += a * b;\n"
"    }\n"
"    C[row * N + col] = sum;\n"
"}\n"
"\n"
/* (4) asm, GLC=1, loads serializados */
"__kernel void mm_glc1_serial(__global const float* restrict A,\n"
"                             __global const float* restrict B,\n"
"                             __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) {\n"
"        float a, b;\n"
"        __global const float* pa = &A[row * N + k];\n"
"        __global const float* pb = &B[k * N + col];\n"
"        __asm__ volatile(\n"
"            \"flat_load_dword %0, %2 glc\\n\"\n"
"            \"s_waitcnt vmcnt(0)\\n\"\n"
"            \"flat_load_dword %1, %3 glc\\n\"\n"
"            \"s_waitcnt vmcnt(0)\\n\"\n"
"            : \"=v\"(a), \"=v\"(b) : \"v\"(pa), \"v\"(pb) : \"memory\");\n"
"        sum += a * b;\n"
"    }\n"
"    C[row * N + col] = sum;\n"
"}\n"
"\n"
/* (0) C puro — referencia externa ao fatorial */
"__kernel void mm_plain_c(__global const float* restrict A,\n"
"                         __global const float* restrict B,\n"
"                         __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];\n"
"    C[row * N + col] = sum;\n"
"}\n"
"\n"
/* (5) volatile puro — controle */
"__kernel void mm_volatile(__global volatile const float* A,\n"
"                          __global volatile const float* B,\n"
"                          __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];\n"
"    C[row * N + col] = sum;\n"
"}\n";

/* ------------------------------------------------------------------------- */

#define CHECK(err, msg) do {                                                   \
    if ((err) != CL_SUCCESS) {                                                 \
        fprintf(stderr, "ERRO %d em %s (linha %d)\n", (int)(err), (msg), __LINE__); \
        exit(1);                                                               \
    }                                                                          \
} while (0)

static double ms_of(cl_event ev)
{
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (double)(t1 - t0) / 1e6;
}

static const char* KFN[] = {
    "mm_glc0_overlap", "mm_glc0_serial", "mm_glc1_overlap",
    "mm_glc1_serial",  "mm_plain_c",     "mm_volatile"
};
static const char* KLABEL[] = {
    "(1) asm GLC=0 sobreposto", "(2) asm GLC=0 serializado",
    "(3) asm GLC=1 sobreposto", "(4) asm GLC=1 serializado",
    "(0) C puro",               "(5) volatile"
};
#define NK 6

int main(int argc, char** argv)
{
    int sizes[64];
    int ns = 0;
    if (argc > 1) {
        for (int i = 1; i < argc && ns < 64; i++) {
            int n = atoi(argv[i]);
            if (n <= 0 || n % TILE) {
                fprintf(stderr, "tamanho invalido '%s' (precisa ser multiplo de %d)\n",
                        argv[i], TILE);
                return 2;
            }
            sizes[ns++] = n;
        }
    } else {
        ns = (int)(sizeof(DEFAULT_SIZES) / sizeof(DEFAULT_SIZES[0]));
        memcpy(sizes, DEFAULT_SIZES, sizeof(DEFAULT_SIZES));
    }

    /* -- Dispositivo -- */
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
    cl_ulong l1cache = 0;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(l1cache), &l1cache, NULL);
    printf("Cache global reportado: %llu KB | workgroup %dx%d | warmup %d | runs %d\n\n",
           (unsigned long long)(l1cache >> 10), TILE, TILE, WARMUP, RUNS);

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK(err, "clCreateCommandQueue");

    const char* srcs[] = { kernel_source };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, NULL, &err);
    CHECK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, "-cl-mad-enable", NULL, NULL);
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

    cl_kernel ks[NK];
    for (int i = 0; i < NK; i++) {
        ks[i] = clCreateKernel(prog, KFN[i], &err);
        CHECK(err, KFN[i]);
    }

    FILE* csv = fopen("resultados/glc_scaling.csv", "w");
    if (!csv) { fprintf(stderr, "nao abriu resultados/glc_scaling.csv\n"); return 1; }
    fprintf(csv, "N,kernel,rotulo,media_ms,dp_ms,min_ms,gflops,bytes_load,bw_efetiva_GBs,maxdiff\n");

    printf("%6s  %-26s %10s %8s %10s %12s %10s\n",
           "N", "celula", "media(ms)", "dp", "GFLOPS", "BW_ef(GB/s)", "check");
    printf("%s\n", "----------------------------------------------------------------------------------------------");

    for (int si = 0; si < ns; si++) {
        const int N = sizes[si];
        const size_t elems = (size_t)N * N;
        const size_t msz   = elems * sizeof(float);
        const double flops = 2.0 * (double)N * N * N;
        /* loads globais emitidos pelo kernel naive: 2 por iteracao de k, por work-item */
        const double bytes = 2.0 * (double)N * N * N * sizeof(float);

        float* hA = (float*)malloc(msz);
        float* hB = (float*)malloc(msz);
        float* hC = (float*)malloc(msz);
        float* hRef = (float*)malloc(msz);
        for (size_t i = 0; i < elems; i++) {
            hA[i] = (float)((i * 7919) % 1000) / 1000.0f;
            hB[i] = (float)((i * 6271) % 1000) / 1000.0f;
        }

        cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, msz, hA, &err);
        CHECK(err, "buffer A");
        cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, msz, hB, &err);
        CHECK(err, "buffer B");
        cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, msz, NULL, &err);
        CHECK(err, "buffer C");

        size_t global[2] = { (size_t)N, (size_t)N };
        size_t local[2]  = { TILE, TILE };
        const int n_arg = N;

        for (int i = 0; i < NK; i++) {
            clSetKernelArg(ks[i], 0, sizeof(cl_mem), &dA);
            clSetKernelArg(ks[i], 1, sizeof(cl_mem), &dB);
            clSetKernelArg(ks[i], 2, sizeof(cl_mem), &dC);
            clSetKernelArg(ks[i], 3, sizeof(int),    &n_arg);

            for (int w = 0; w < WARMUP; w++) {
                err = clEnqueueNDRangeKernel(q, ks[i], 2, NULL, global, local, 0, NULL, NULL);
                CHECK(err, "warmup");
            }
            clFinish(q);

            double t[RUNS], sum = 0.0, mn = 1e30;
            for (int r = 0; r < RUNS; r++) {
                cl_event ev;
                err = clEnqueueNDRangeKernel(q, ks[i], 2, NULL, global, local, 0, NULL, &ev);
                CHECK(err, "enqueue");
                clWaitForEvents(1, &ev);
                t[r] = ms_of(ev);
                clReleaseEvent(ev);
                sum += t[r];
                if (t[r] < mn) mn = t[r];
            }
            double mean = sum / RUNS, var = 0.0;
            for (int r = 0; r < RUNS; r++) var += (t[r] - mean) * (t[r] - mean);
            double sd = sqrt(var / (RUNS - 1));

            /* Verificacao: a celula (1) define a referencia; as demais sao comparadas
               contra ela. Evita um matmul O(N^3) na CPU para N grande. */
            clEnqueueReadBuffer(q, dC, CL_TRUE, 0, msz, hC, 0, NULL, NULL);
            double maxdiff = 0.0;
            if (i == 0) {
                memcpy(hRef, hC, msz);
            } else {
                for (size_t e = 0; e < elems; e++) {
                    double d = fabs((double)hC[e] - (double)hRef[e]);
                    if (d > maxdiff) maxdiff = d;
                }
            }

            double gflops = flops / (mean / 1000.0) / 1e9;
            double bw     = bytes / (mean / 1000.0) / 1e9;

            printf("%6d  %-26s %10.3f %8.3f %10.1f %12.1f %10s\n",
                   N, KLABEL[i], mean, sd, gflops, bw,
                   (i == 0) ? "ref" : (maxdiff < 1e-2 ? "OK" : "DIVERGIU"));
            fprintf(csv, "%d,%s,\"%s\",%.6f,%.6f,%.6f,%.3f,%.0f,%.3f,%.8f\n",
                    N, KFN[i], KLABEL[i], mean, sd, mn, gflops, bytes, bw, maxdiff);
            fflush(csv);
        }
        printf("\n");

        clReleaseMemObject(dA);
        clReleaseMemObject(dB);
        clReleaseMemObject(dC);
        free(hA); free(hB); free(hC); free(hRef);
    }

    fclose(csv);
    printf("CSV: resultados/glc_scaling.csv\n");
    printf("Grafico e ajuste do modelo: python3 analise/plot_glc_scaling.py\n");

    for (int i = 0; i < NK; i++) clReleaseKernel(ks[i]);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(plats);
    return 0;
}
