/*
 * =============================================================================
 *  niveis_cache.cpp — Os tres niveis da hierarquia, medidos diretamente
 * =============================================================================
 *
 *  O QUE ESTE EXPERIMENTO RESOLVE
 *  ------------------------------
 *  Desde a Etapa 1 o trabalho persegue tres cenarios de acesso:
 *      (a) via L1 -> L2 -> VRAM      (caminho normal)
 *      (b) contornando o L1          (servido pelo L2)
 *      (c) contornando L1 e L2       (direto da VRAM)
 *
 *  Nenhuma tentativa anterior alcancou (c). Verificacao no ISA:
 *      - `volatile` no driver Windows (AMDOCL): GLC=0 SLC=0 sempre
 *      - `volatile` no ROCm/Linux:              GLC=1 SLC=0
 *      - o kernel `matmul_direto_VRAM` gera ISA BYTE-IDENTICA ao
 *        `matmul_via_L2` -- o nome sempre foi enganoso, ele nunca
 *        chegou a VRAM
 *
 *  O nivel de cache NAO depende do algoritmo: depende das flags da instrucao.
 *  Trocar de algoritmo (scan, crivo, FFT) nao mudaria nada, porque `volatile`
 *  nao emite SLC em nenhum deles. O que faltava era o modificador `slc`, que
 *  so o inline assembly produz -- e que so compila em runtime no ROCm.
 *
 *  As quatro combinacoes, confirmadas no ISA do gfx803:
 *      flat_load_dword           -> DC500000  GLC=0 SLC=0  (L1 -> L2 -> VRAM)
 *      flat_load_dword ... glc   -> DC510000  GLC=1 SLC=0  (L2 -> VRAM)
 *      flat_load_dword ... slc   -> DC520000  GLC=0 SLC=1  (L1, pula L2)
 *      flat_load_dword ... glc slc -> DC530000 GLC=1 SLC=1 (VRAM direto)
 *
 *  DESENHO
 *  -------
 *  Fatorial nivel_de_cache x carga:
 *
 *      3 niveis  {L1, L2, VRAM}
 *    x 3 cargas  {stream (reuso 1), stencil 3x3 (reuso 9), matmul (reuso N)}
 *
 *  A hipotese e que o custo de subir na hierarquia cresce com o reuso temporal
 *  do algoritmo: um cache so ajuda se o dado for tocado mais de uma vez
 *  enquanto ainda esta residente.
 *
 *  NOTA SOBRE `slc` SOZINHO
 *  ------------------------
 *  A combinacao GLC=0 SLC=1 (pular o L2 mas usar o L1) e medida por
 *  completude do fatorial de flags, mas nao corresponde a um cenario util:
 *  no GFX8 o L1 e write-through e nao inclusivo, entao um miss no L1 vai ao
 *  L2 de qualquer forma. E incluida para documentar o encoding, nao como
 *  cenario de interesse.
 *
 *  CUIDADO DE IMPLEMENTACAO: earlyclobber
 *  --------------------------------------
 *  As saidas do bloco asm usam "=&v" (earlyclobber), NAO "=v". Sem o `&`, o
 *  compilador pode alocar um registrador de saida sobre um registrador de
 *  endereco que ainda sera lido por um load posterior do mesmo bloco,
 *  corrompendo o endereco. Com 4 loads por bloco isso causa falha de pagina
 *  na GPU de forma reproduzivel.
 *
 *  REQUISITOS
 *  ----------
 *  Runtime OpenCL Clang/LLVM com inline asm GCN (ROCm) e ROC_ENABLE_PRE_VEGA=1.
 *  Nao funciona no driver AMDOCL do Windows.
 *
 *  Build:
 *    g++ -Wall -O2 experimentos/niveis_cache.cpp -o output/niveis_cache -lOpenCL
 *  Execucao:
 *    source setup/env_rocm.sh && ./output/niveis_cache
 *
 *  Saida: resultados/niveis_cache.csv  +  isa/rocm_gfx803_niveis_cache.bin
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define WARMUP 3
#define RUNS   10

#define STREAM_ITEMS 262144   /* work-items                                   */
#define STREAM_PER   64       /* elementos por work-item (multiplo de 4)      */
                              /* total 16M floats = 64 MB, cada um lido 1 vez */
#define STENCIL_W 4096
#define STENCIL_H 4096        /* 16M elementos = 64 MB                        */

#define MM_N    1024
#define MM_TILE 16

/* ------------------------------------------------------------------------- */
/*  Kernels — gerados por macro, para que as variantes de nivel de cache      */
/*  difiram EXCLUSIVAMENTE no modificador da instrucao de load.               */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
/* ---- blocos de load parametrizados pelo modificador de cache ---- */
"#define LOAD4(a,b,c,d, p0,p1,p2,p3, F)                    \\\n"
"    __asm__ volatile(                                     \\\n"
"        \"flat_load_dword %0, %4\" F \"\\n\"                 \\\n"
"        \"flat_load_dword %1, %5\" F \"\\n\"                 \\\n"
"        \"flat_load_dword %2, %6\" F \"\\n\"                 \\\n"
"        \"flat_load_dword %3, %7\" F \"\\n\"                 \\\n"
"        \"s_waitcnt vmcnt(0)\\n\"                           \\\n"
"        : \"=&v\"(a), \"=&v\"(b), \"=&v\"(c), \"=&v\"(d)        \\\n"
"        : \"v\"(p0), \"v\"(p1), \"v\"(p2), \"v\"(p3) : \"memory\")\n"
"\n"
"#define LOAD3(a,b,c, p0,p1,p2, F)                         \\\n"
"    __asm__ volatile(                                     \\\n"
"        \"flat_load_dword %0, %3\" F \"\\n\"                 \\\n"
"        \"flat_load_dword %1, %4\" F \"\\n\"                 \\\n"
"        \"flat_load_dword %2, %5\" F \"\\n\"                 \\\n"
"        \"s_waitcnt vmcnt(0)\\n\"                           \\\n"
"        : \"=&v\"(a), \"=&v\"(b), \"=&v\"(c)                   \\\n"
"        : \"v\"(p0), \"v\"(p1), \"v\"(p2) : \"memory\")\n"
"\n"
"#define LOAD2(a,b, p0,p1, F)                              \\\n"
"    __asm__ volatile(                                     \\\n"
"        \"flat_load_dword %0, %2\" F \"\\n\"                 \\\n"
"        \"flat_load_dword %1, %3\" F \"\\n\"                 \\\n"
"        \"s_waitcnt vmcnt(0)\\n\"                           \\\n"
"        : \"=&v\"(a), \"=&v\"(b)                             \\\n"
"        : \"v\"(p0), \"v\"(p1) : \"memory\")\n"
"\n"
/* ===================================================================== */
/* CARGA A — streaming puro, reuso = 1                                    */
/* Grid-stride coalescido: cada elemento e lido exatamente uma vez em     */
/* todo o grid. Nao ha reuso algum para um cache explorar.                */
/* ===================================================================== */
"#define K_STREAM(NOME, F)                                                    \\\n"
"__kernel void NOME(__global const float* restrict in,                        \\\n"
"                   __global float* restrict out,                             \\\n"
"                   const int per_item, const int total)                      \\\n"
"{                                                                            \\\n"
"    int gid = get_global_id(0);                                              \\\n"
"    float sum = 0.0f;                                                        \\\n"
"    for (int i = 0; i < per_item; i += 4) {                                  \\\n"
"        float a, b, c, d;                                                    \\\n"
"        __global const float* p0 = &in[gid + (i    ) * total];               \\\n"
"        __global const float* p1 = &in[gid + (i + 1) * total];               \\\n"
"        __global const float* p2 = &in[gid + (i + 2) * total];               \\\n"
"        __global const float* p3 = &in[gid + (i + 3) * total];               \\\n"
"        LOAD4(a, b, c, d, p0, p1, p2, p3, F);                                \\\n"
"        sum += a + b + c + d;                                                \\\n"
"    }                                                                        \\\n"
"    out[gid] = sum;                                                          \\\n"
"}\n"
"K_STREAM(stream_L1,   \"\")\n"
"K_STREAM(stream_L2,   \" glc\")\n"
"K_STREAM(stream_VRAM, \" glc slc\")\n"
"\n"
/* ===================================================================== */
/* CARGA B — stencil 2D blur 3x3, reuso = 9                               */
/* Cada elemento de entrada participa de 9 saidas distintas.              */
/* ===================================================================== */
"#define K_STENCIL(NOME, F)                                                   \\\n"
"__kernel void NOME(__global const float* restrict in,                        \\\n"
"                   __global float* restrict out,                             \\\n"
"                   const int W, const int H)                                 \\\n"
"{                                                                            \\\n"
"    int x = get_global_id(0);                                                \\\n"
"    int y = get_global_id(1);                                                \\\n"
"    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) {                            \\\n"
"        if (x < W && y < H) out[y*W+x] = in[y*W+x];                          \\\n"
"        return;                                                              \\\n"
"    }                                                                        \\\n"
"    float sum = 0.0f;                                                        \\\n"
"    for (int dy = -1; dy <= 1; dy++) {                                       \\\n"
"        float v0, v1, v2;                                                    \\\n"
"        __global const float* q0 = &in[(y+dy)*W + (x-1)];                    \\\n"
"        __global const float* q1 = &in[(y+dy)*W + (x  )];                    \\\n"
"        __global const float* q2 = &in[(y+dy)*W + (x+1)];                    \\\n"
"        LOAD3(v0, v1, v2, q0, q1, q2, F);                                    \\\n"
"        sum += v0 + v1 + v2;                                                 \\\n"
"    }                                                                        \\\n"
"    out[y*W+x] = sum / 9.0f;                                                 \\\n"
"}\n"
"K_STENCIL(stencil_L1,   \"\")\n"
"K_STENCIL(stencil_L2,   \" glc\")\n"
"K_STENCIL(stencil_VRAM, \" glc slc\")\n"
"\n"
/* ===================================================================== */
/* CARGA C — matmul naive, reuso = N                                      */
/* ===================================================================== */
"#define K_MM(NOME, F)                                                        \\\n"
"__kernel void NOME(__global const float* restrict A,                         \\\n"
"                   __global const float* restrict B,                         \\\n"
"                   __global float* restrict C, const int N)                  \\\n"
"{                                                                            \\\n"
"    int row = get_global_id(1);                                              \\\n"
"    int col = get_global_id(0);                                              \\\n"
"    if (row >= N || col >= N) return;                                        \\\n"
"    float sum = 0.0f;                                                        \\\n"
"    for (int k = 0; k < N; k++) {                                            \\\n"
"        float a, b;                                                          \\\n"
"        __global const float* pa = &A[row * N + k];                          \\\n"
"        __global const float* pb = &B[k * N + col];                          \\\n"
"        LOAD2(a, b, pa, pb, F);                                              \\\n"
"        sum += a * b;                                                        \\\n"
"    }                                                                        \\\n"
"    C[row * N + col] = sum;                                                  \\\n"
"}\n"
"K_MM(mm_L1,   \"\")\n"
"K_MM(mm_L2,   \" glc\")\n"
"K_MM(mm_VRAM, \" glc slc\")\n";

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

static void medir(cl_command_queue q, cl_kernel k, cl_uint dim,
                  const size_t* global, const size_t* local,
                  double* mean, double* sd)
{
    cl_int err;
    for (int w = 0; w < WARMUP; w++) {
        err = clEnqueueNDRangeKernel(q, k, dim, NULL, global, local, 0, NULL, NULL);
        CHECK(err, "warmup");
    }
    clFinish(q);

    double t[RUNS], sum = 0.0;
    for (int r = 0; r < RUNS; r++) {
        cl_event ev;
        err = clEnqueueNDRangeKernel(q, k, dim, NULL, global, local, 0, NULL, &ev);
        CHECK(err, "enqueue");
        clWaitForEvents(1, &ev);
        t[r] = ms_of(ev);
        clReleaseEvent(ev);
        sum += t[r];
    }
    *mean = sum / RUNS;
    double var = 0.0;
    for (int r = 0; r < RUNS; r++) var += (t[r] - *mean) * (t[r] - *mean);
    *sd = sqrt(var / (RUNS - 1));
}

static const char* NIVEL[3] = { "L1 (normal)", "L2 (pula L1)", "VRAM (pula L1+L2)" };

typedef struct {
    const char* carga;
    double reuso;
    double t[3], sd[3];
} Res;

static void imprime_carga(const Res* r, const char* const* fns)
{
    for (int v = 0; v < 3; v++)
        printf("    %-14s %-20s %8.3f ms (dp %.3f)   %6.3fx\n",
               fns[v], NIVEL[v], r->t[v], r->sd[v], r->t[v] / r->t[0]);
    printf("\n");
}

int main(void)
{
    printf("================================================================\n");
    printf(" Os tres niveis da hierarquia x o reuso do algoritmo\n");
    printf(" warmup %d | runs %d\n", WARMUP, RUNS);
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
    printf("Dispositivo: %s\n\n", buf);

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

    {
        size_t bs = 0;
        clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bs, NULL);
        unsigned char* bin = (unsigned char*)malloc(bs);
        unsigned char* bins[] = { bin };
        clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), bins, NULL);
        FILE* f = fopen("isa/rocm_gfx803_niveis_cache.bin", "wb");
        if (f) {
            fwrite(bin, 1, bs, f);
            fclose(f);
            printf("[ISA] isa/rocm_gfx803_niveis_cache.bin (%lu bytes)\n"
                   "      confira as flags com:\n"
                   "      llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 \\\n"
                   "        isa/rocm_gfx803_niveis_cache.bin > isa/niveis_cache_disasm.txt\n"
                   "      python3 isa/analyze_disasm.py isa/niveis_cache_disasm.txt\n\n",
                   (unsigned long)bs);
        }
        free(bin);
    }

    Res res[3];
    int nres = 0;

    /* ---------------------- CARGA A: stream ---------------------- */
    {
        const size_t n_elem = (size_t)STREAM_ITEMS * STREAM_PER;
        const size_t bytes  = n_elem * sizeof(float);
        printf("--- CARGA A: streaming (reuso = 1) — %.0f MB ---\n", bytes / 1048576.0);

        float* h = (float*)malloc(bytes);
        for (size_t i = 0; i < n_elem; i++) h[i] = (float)(i % 997) * 0.001f;
        cl_mem din = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h, &err);
        CHECK(err, "buffer stream in");
        cl_mem dout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                     STREAM_ITEMS * sizeof(float), NULL, &err);
        CHECK(err, "buffer stream out");

        size_t global = STREAM_ITEMS, local = 256;
        const int per_item = STREAM_PER, total = STREAM_ITEMS;
        static const char* fns[3] = { "stream_L1", "stream_L2", "stream_VRAM" };
        Res r; r.carga = "stream"; r.reuso = 1.0;
        for (int v = 0; v < 3; v++) {
            cl_kernel k = clCreateKernel(prog, fns[v], &err);
            CHECK(err, fns[v]);
            clSetKernelArg(k, 0, sizeof(cl_mem), &din);
            clSetKernelArg(k, 1, sizeof(cl_mem), &dout);
            clSetKernelArg(k, 2, sizeof(int), &per_item);
            clSetKernelArg(k, 3, sizeof(int), &total);
            medir(q, k, 1, &global, &local, &r.t[v], &r.sd[v]);
            clReleaseKernel(k);
        }
        imprime_carga(&r, fns);
        res[nres++] = r;
        clReleaseMemObject(din); clReleaseMemObject(dout); free(h);
    }

    /* ---------------------- CARGA B: stencil --------------------- */
    {
        const size_t n_elem = (size_t)STENCIL_W * STENCIL_H;
        const size_t bytes  = n_elem * sizeof(float);
        printf("--- CARGA B: stencil 3x3 (reuso = 9) — %dx%d, %.0f MB ---\n",
               STENCIL_W, STENCIL_H, bytes / 1048576.0);

        float* h = (float*)malloc(bytes);
        for (size_t i = 0; i < n_elem; i++) h[i] = (float)(i % 997) * 0.001f;
        cl_mem din = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h, &err);
        CHECK(err, "buffer stencil in");
        cl_mem dout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
        CHECK(err, "buffer stencil out");

        size_t global[2] = { STENCIL_W, STENCIL_H };
        size_t local[2]  = { 16, 16 };
        const int W = STENCIL_W, H = STENCIL_H;
        static const char* fns[3] = { "stencil_L1", "stencil_L2", "stencil_VRAM" };
        Res r; r.carga = "stencil3x3"; r.reuso = 9.0;
        for (int v = 0; v < 3; v++) {
            cl_kernel k = clCreateKernel(prog, fns[v], &err);
            CHECK(err, fns[v]);
            clSetKernelArg(k, 0, sizeof(cl_mem), &din);
            clSetKernelArg(k, 1, sizeof(cl_mem), &dout);
            clSetKernelArg(k, 2, sizeof(int), &W);
            clSetKernelArg(k, 3, sizeof(int), &H);
            medir(q, k, 2, global, local, &r.t[v], &r.sd[v]);
            clReleaseKernel(k);
        }
        imprime_carga(&r, fns);
        res[nres++] = r;
        clReleaseMemObject(din); clReleaseMemObject(dout); free(h);
    }

    /* ---------------------- CARGA C: matmul ---------------------- */
    {
        const size_t n_elem = (size_t)MM_N * MM_N;
        const size_t bytes  = n_elem * sizeof(float);
        printf("--- CARGA C: matmul naive %dx%d (reuso = %d) ---\n", MM_N, MM_N, MM_N);

        float* hA = (float*)malloc(bytes);
        float* hB = (float*)malloc(bytes);
        for (size_t i = 0; i < n_elem; i++) {
            hA[i] = (float)((i * 7919) % 1000) / 1000.0f;
            hB[i] = (float)((i * 6271) % 1000) / 1000.0f;
        }
        cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, hA, &err);
        CHECK(err, "buffer A");
        cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, hB, &err);
        CHECK(err, "buffer B");
        cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
        CHECK(err, "buffer C");

        size_t global[2] = { MM_N, MM_N };
        size_t local[2]  = { MM_TILE, MM_TILE };
        const int N = MM_N;
        static const char* fns[3] = { "mm_L1", "mm_L2", "mm_VRAM" };
        Res r; r.carga = "matmul"; r.reuso = (double)MM_N;
        for (int v = 0; v < 3; v++) {
            cl_kernel k = clCreateKernel(prog, fns[v], &err);
            CHECK(err, fns[v]);
            clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
            clSetKernelArg(k, 1, sizeof(cl_mem), &dB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &dC);
            clSetKernelArg(k, 3, sizeof(int), &N);
            medir(q, k, 2, global, local, &r.t[v], &r.sd[v]);
            clReleaseKernel(k);
        }
        imprime_carga(&r, fns);
        res[nres++] = r;
        clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC);
        free(hA); free(hB);
    }

    /* ---------------------------- Sintese ---------------------------- */
    printf("================================================================\n");
    printf(" CUSTO DE SUBIR NA HIERARQUIA (razao sobre o nivel L1)\n");
    printf("================================================================\n");
    printf(" %-12s %7s %12s %12s %12s\n",
           "carga", "reuso", "L1", "L2/L1", "VRAM/L1");
    for (int i = 0; i < nres; i++)
        printf(" %-12s %7.0f %10.3f ms %11.3fx %11.3fx\n",
               res[i].carga, res[i].reuso, res[i].t[0],
               res[i].t[1] / res[i].t[0], res[i].t[2] / res[i].t[0]);

    printf("\n Tempos absolutos NAO sao comparaveis entre cargas (a largura de\n");
    printf(" sobreposicao de loads difere: 4, 3 e 2). Apenas as razoes dentro\n");
    printf(" de cada carga sao comparaveis.\n");
    printf("================================================================\n");

    FILE* csv = fopen("resultados/niveis_cache.csv", "w");
    if (csv) {
        fprintf(csv, "carga,reuso,nivel,tempo_ms,dp_ms,razao_sobre_L1\n");
        for (int i = 0; i < nres; i++)
            for (int v = 0; v < 3; v++)
                fprintf(csv, "%s,%.0f,\"%s\",%.6f,%.6f,%.6f\n",
                        res[i].carga, res[i].reuso, NIVEL[v],
                        res[i].t[v], res[i].sd[v], res[i].t[v] / res[i].t[0]);
        fclose(csv);
        printf("\nCSV: resultados/niveis_cache.csv\n");
    }

    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(plats);
    return 0;
}
