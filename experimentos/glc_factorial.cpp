/*
 * =============================================================================
 *  glc_factorial.cpp — Isolamento do efeito do GLC=1 na multiplicacao de matrizes
 * =============================================================================
 *
 *  MOTIVACAO
 *  ---------
 *  No ROCm (Linux), o qualificador `volatile` no gfx803 gera GLC=1 nos loads
 *  (encoding DC510000), ao contrario do driver AMDOCL do Windows, que gera
 *  GLC=0 (DC500000) com e sem volatile.
 *
 *  Porem, a inspecao do ISA mostrou que o codegen do volatile no ROCm altera
 *  DUAS coisas ao mesmo tempo:
 *
 *    1. emite a flag `glc` nos loads  -> contorna o L1
 *    2. insere `s_waitcnt vmcnt(0)` apos CADA load -> serializa os acessos,
 *       eliminando a sobreposicao de latencia (memory-level parallelism)
 *
 *  Logo, a diferenca de tempo entre o kernel normal e o kernel volatile NAO
 *  pode ser atribuida ao bypass do L1 isoladamente: os dois efeitos estao
 *  confundidos.
 *
 *  DESENHO EXPERIMENTAL
 *  --------------------
 *  Fatorial 2x2 completo, com as celulas construidas via inline assembly GCN
 *  (possivel no ROCm, impossivel no driver Windows):
 *
 *                     | loads sobrepostos      | loads serializados
 *      ---------------+------------------------+------------------------
 *      GLC=0 (usa L1) | (1) baseline           | (2) so serializacao
 *      GLC=1 (sem L1) | (3) so bypass de L1    | (4) ambos
 *
 *  Com as 4 celulas medidas, os efeitos se separam:
 *      efeito da serializacao = (2)/(1)
 *      efeito do bypass do L1 = (3)/(1)
 *      efeito combinado       = (4)/(1)
 *  e verifica-se a interacao comparando (4)/(1) com [(2)/(1)] x [(3)/(1)].
 *
 *  O kernel (5) usa `volatile` puro e serve de controle: seu ISA e seu tempo
 *  devem coincidir com os da celula (4).
 *
 *  REQUISITOS
 *  ----------
 *  Runtime OpenCL baseado em Clang/LLVM (ROCm). Exige ROC_ENABLE_PRE_VEGA=1
 *  para que o ROCm 5.2 enumere o gfx803. Ver setup/README.md.
 *
 *  Build:
 *    g++ -Wall -O2 experimentos/glc_factorial.cpp -o output/glc_factorial -lOpenCL
 *  Execucao:
 *    source setup/env_rocm.sh && ./output/glc_factorial
 *
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define N          1024   /* matrizes NxN, mesmo tamanho dos experimentos anteriores */
#define TILE       16     /* workgroup 16x16 = 256 threads, idem                     */
#define WARMUP     3
#define RUNS       10

/* ------------------------------------------------------------------------- */
/*  Kernels — as 4 celulas do fatorial + o controle com volatile             */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
/* --- prologo comum: calcula row/col e faz o guard --- */
"#define PROLOGO                                    \\\n"
"    int row = get_global_id(1);                    \\\n"
"    int col = get_global_id(0);                    \\\n"
"    if (row >= N || col >= N) return;              \\\n"
"    float sum = 0.0f;\n"
"\n"
/* ===================================================================== */
/* (1) GLC=0, loads sobrepostos, VIA ASM — baseline do fatorial          */
/*     Celula de referencia: mesma construcao por inline asm das demais, */
/*     mudando apenas os fatores em estudo. Comparar as celulas asm      */
/*     entre si evita confundir o efeito do GLC com o efeito de usar     */
/*     inline asm no lugar do codegen normal do compilador.              */
/* ===================================================================== */
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
/* ===================================================================== */
/* (0) REFERENCIA EXTERNA: C puro, sem asm e sem volatile                */
/*     Fica fora do fatorial. Serve para medir quanto custa construir os */
/*     loads por inline asm em vez de deixar o compilador agendar.       */
/* ===================================================================== */
"__kernel void mm_plain_c(__global const float* restrict A,\n"
"                         __global const float* restrict B,\n"
"                         __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) {\n"
"        sum += A[row * N + k] * B[k * N + col];\n"
"    }\n"
"    C[row * N + col] = sum;\n"
"}\n"
"\n"
/* ===================================================================== */
/* (2) GLC=0, loads serializados — isola o custo da perda de sobreposicao */
/*     Mesma flag de cache do baseline, mas com waitcnt apos cada load.   */
/* ===================================================================== */
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
/* ===================================================================== */
/* (3) GLC=1, loads sobrepostos — isola o custo do bypass do L1           */
/*     Os dois loads sao emitidos antes de um unico waitcnt, preservando  */
/*     a sobreposicao de latencia do baseline.                            */
/* ===================================================================== */
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
/* ===================================================================== */
/* (4) GLC=1, loads serializados — reproduz o codegen do volatile         */
/* ===================================================================== */
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
/* ===================================================================== */
/* (5) CONTROLE: volatile puro, sem asm                                   */
/*     Deve produzir ISA e tempo equivalentes a celula (4).               */
/* ===================================================================== */
"__kernel void mm_volatile(__global volatile const float* A,\n"
"                          __global volatile const float* B,\n"
"                          __global float* restrict C, const int N)\n"
"{\n"
"    PROLOGO\n"
"    for (int k = 0; k < N; k++) {\n"
"        sum += A[row * N + k] * B[k * N + col];\n"
"    }\n"
"    C[row * N + col] = sum;\n"
"}\n";

/* ------------------------------------------------------------------------- */

#define CHECK(err, msg) do {                                              \
    if ((err) != CL_SUCCESS) {                                            \
        fprintf(stderr, "ERRO %d em %s (linha %d)\n", (int)(err), (msg), __LINE__); \
        exit(1);                                                          \
    }                                                                     \
} while (0)

static double ms_of(cl_event ev)
{
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (double)(t1 - t0) / 1e6;
}

typedef struct {
    const char* nome;
    const char* celula;
    double mean, sd, min;
    double gflops;
    int    ok;
} Resultado;

static void matmul_cpu(const float* A, const float* B, float* C, int n)
{
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float aik = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += aik * B[k * n + j];
        }
}

int main(void)
{
    const size_t elems    = (size_t)N * N;
    const size_t mat_size = elems * sizeof(float);
    const double flops    = 2.0 * (double)N * N * N;

    printf("================================================================\n");
    printf(" Isolamento do efeito GLC=1 — desenho fatorial 2x2\n");
    printf(" Matriz %dx%d | workgroup %dx%d | warmup %d | runs %d\n",
           N, N, TILE, TILE, WARMUP, RUNS);
    printf("================================================================\n\n");

    /* -- Dados -- */
    float* hA = (float*)malloc(mat_size);
    float* hB = (float*)malloc(mat_size);
    float* hC = (float*)malloc(mat_size);
    float* ref = (float*)calloc(elems, sizeof(float));
    for (size_t i = 0; i < elems; i++) {
        hA[i] = (float)((i * 7919) % 1000) / 1000.0f;
        hB[i] = (float)((i * 6271) % 1000) / 1000.0f;
    }

    /* -- Dispositivo: primeira plataforma com GPU -- */
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
    if (!dev) { fprintf(stderr, "Nenhum dispositivo GPU OpenCL encontrado.\n"); return 1; }

    char buf[512];
    clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
    printf("Plataforma: %s\n", buf);
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
    printf("Dispositivo: %s\n", buf);
    cl_ulong l1 = 0, lds = 0;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(l1), &l1, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lds), &lds, NULL);
    printf("Cache global reportado: %llu KB | LDS: %llu KB\n\n",
           (unsigned long long)(l1 >> 10), (unsigned long long)(lds >> 10));

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK(err, "clCreateCommandQueue");

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, mat_size, hA, &err);
    CHECK(err, "buffer A");
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, mat_size, hB, &err);
    CHECK(err, "buffer B");
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, mat_size, NULL, &err);
    CHECK(err, "buffer C");

    /* -- Compilar -- */
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
        fprintf(stderr, "\nEste experimento exige um runtime OpenCL Clang/LLVM com suporte a\n"
                        "inline assembly GCN (ROCm). O driver AMDOCL do Windows nao compila.\n");
        return 1;
    }

    /* Dump do ISA para verificacao das flags GLC/SLC */
    {
        size_t bs = 0;
        clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bs, NULL);
        unsigned char* bin = (unsigned char*)malloc(bs);
        unsigned char* bins[] = { bin };
        clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), bins, NULL);
        FILE* f = fopen("isa/rocm_gfx803_glc_factorial.bin", "wb");
        if (f) {
            fwrite(bin, 1, bs, f);
            fclose(f);
            printf("[ISA] isa/rocm_gfx803_glc_factorial.bin (%lu bytes)\n", (unsigned long)bs);
            printf("      verificar com:\n"
                   "      llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 \\\n"
                   "        isa/rocm_gfx803_glc_factorial.bin > isa/glc_factorial_disasm.txt\n"
                   "      python3 isa/analyze_disasm.py isa/glc_factorial_disasm.txt\n\n");
        }
        free(bin);
    }

    struct { const char* fn; const char* celula; } specs[] = {
        { "mm_glc0_overlap", "(1) asm GLC=0, sobreposto  [baseline fatorial]" },
        { "mm_glc0_serial",  "(2) asm GLC=0, serializado [so serializacao]"   },
        { "mm_glc1_overlap", "(3) asm GLC=1, sobreposto  [so bypass L1]"      },
        { "mm_glc1_serial",  "(4) asm GLC=1, serializado [ambos fatores]"     },
        { "mm_plain_c",      "(0) C puro, sem asm        [ref. externa]"      },
        { "mm_volatile",     "(5) volatile puro          [controle]"          },
    };
    const int NK = (int)(sizeof(specs) / sizeof(specs[0]));
    Resultado res[8];

    size_t global[2] = { N, N };
    size_t local[2]  = { TILE, TILE };
    const int n_arg = N;

    for (int i = 0; i < NK; i++) {
        cl_kernel k = clCreateKernel(prog, specs[i].fn, &err);
        CHECK(err, specs[i].fn);
        clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
        clSetKernelArg(k, 1, sizeof(cl_mem), &dB);
        clSetKernelArg(k, 2, sizeof(cl_mem), &dC);
        clSetKernelArg(k, 3, sizeof(int),    &n_arg);

        printf("--- %s  (%s) ---\n", specs[i].celula, specs[i].fn);

        for (int w = 0; w < WARMUP; w++) {
            err = clEnqueueNDRangeKernel(q, k, 2, NULL, global, local, 0, NULL, NULL);
            CHECK(err, "warmup");
        }
        clFinish(q);

        double t[RUNS], sum = 0.0, mn = 1e30;
        for (int r = 0; r < RUNS; r++) {
            cl_event ev;
            err = clEnqueueNDRangeKernel(q, k, 2, NULL, global, local, 0, NULL, &ev);
            CHECK(err, "enqueue");
            clWaitForEvents(1, &ev);
            t[r] = ms_of(ev);
            clReleaseEvent(ev);
            sum += t[r];
            if (t[r] < mn) mn = t[r];
        }
        double mean = sum / RUNS;
        double var = 0.0;
        for (int r = 0; r < RUNS; r++) var += (t[r] - mean) * (t[r] - mean);
        double sd = sqrt(var / (RUNS - 1));

        /* Verificacao numerica contra a CPU (so na primeira vez, e reutilizada) */
        clEnqueueReadBuffer(q, dC, CL_TRUE, 0, mat_size, hC, 0, NULL, NULL);
        if (i == 0) matmul_cpu(hA, hB, ref, N);
        double maxdiff = 0.0;
        for (size_t e = 0; e < elems; e++) {
            double d = fabs((double)hC[e] - (double)ref[e]);
            if (d > maxdiff) maxdiff = d;
        }
        int ok = (maxdiff < 1e-2);

        res[i].nome   = specs[i].fn;
        res[i].celula = specs[i].celula;
        res[i].mean   = mean;
        res[i].sd     = sd;
        res[i].min    = mn;
        res[i].gflops = flops / (mean / 1000.0) / 1e9;
        res[i].ok     = ok;

        printf("    media %.3f ms  (dp %.3f)  min %.3f ms  |  %.1f GFLOPS  |  %s (maxdiff %.5f)\n\n",
               mean, sd, mn, res[i].gflops, ok ? "OK" : "DIVERGIU", maxdiff);
    }

    /* ---------------------------------------------------------------- */
    /*  Sintese                                                         */
    /* ---------------------------------------------------------------- */
    printf("================================================================\n");
    printf(" TABELA — tempos absolutos\n");
    printf("================================================================\n");
    printf(" %-38s %10s %8s %10s\n", "celula", "media(ms)", "dp", "GFLOPS");
    for (int i = 0; i < NK; i++)
        printf(" %-38s %10.3f %8.3f %10.1f\n",
               res[i].celula, res[i].mean, res[i].sd, res[i].gflops);

    double base = res[0].mean;
    double f_ser = res[1].mean / base;   /* (2)/(1) */
    double f_glc = res[2].mean / base;   /* (3)/(1) */
    double f_both = res[3].mean / base;  /* (4)/(1) */

    printf("\n================================================================\n");
    printf(" DECOMPOSICAO DOS EFEITOS\n");
    printf(" (todas as celulas do fatorial usam inline asm; so os fatores mudam)\n");
    printf("================================================================\n");
    printf(" serializacao isolada  (2)/(1) = %.2fx\n", f_ser);
    printf(" bypass do L1 isolado  (3)/(1) = %.2fx   <-- efeito do L1\n", f_glc);
    printf(" combinado             (4)/(1) = %.2fx\n", f_both);
    printf(" produto dos isolados          = %.2fx\n", f_ser * f_glc);
    printf(" interacao (combinado / produto) = %.2fx", f_both / (f_ser * f_glc));
    if (fabs(f_both / (f_ser * f_glc) - 1.0) < 0.15)
        printf("  -> efeitos aproximadamente multiplicativos\n");
    else
        printf("  -> efeitos NAO multiplicativos (ha interacao)\n");

    printf("\n custo de construir os loads por asm, sem mudar fator nenhum:\n");
    printf("   (1) asm GLC=0 sobreposto / (0) C puro = %.3f / %.3f = %.2fx\n",
           res[0].mean, res[4].mean, res[0].mean / res[4].mean);
    printf(" controle: (5) volatile / (4) asm GLC=1 serializado = %.3f / %.3f = %.2fx\n",
           res[5].mean, res[3].mean, res[5].mean / res[3].mean);

    printf("\n Leitura: o unico numero atribuivel ao bypass do L1, com todos os demais\n");
    printf(" fatores fixos, e (3)/(1) = %.2fx. Ou seja, nesta carga o L1 responde por\n", f_glc);
    printf(" uma reducao de tempo de %.0f%% em relacao a executar sem ele.\n",
           100.0 * (1.0 - 1.0 / f_glc));
    printf("================================================================\n");

    free(hA); free(hB); free(hC); free(ref); free(plats);
    return 0;
}
