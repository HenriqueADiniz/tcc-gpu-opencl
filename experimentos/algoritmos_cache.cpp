/*
 * =============================================================================
 *  algoritmos_cache.cpp — Scan, primos e crivo sob os niveis de cache do GFX8
 * =============================================================================
 *
 *  Estende o estudo de niveis de cache (ver niveis_cache.cpp e pchase_flags.cpp)
 *  a tres algoritmos classicos, cada um com um perfil de memoria distinto:
 *
 *    A) SCAN (soma de prefixos)   — reuso global ~1; o reuso interno acontece
 *                                    no LDS, nao no cache. Tres fases.
 *    B) PRIMOS por divisao        — CONTROLE NEGATIVO: um load por work-item e
 *                                    centenas de operacoes de ALU. Se o nivel
 *                                    de cache nao alterar o tempo aqui, fica
 *                                    confirmado que os efeitos medidos nas
 *                                    outras cargas vem do trafego de memoria.
 *    C) CRIVO segmentado          — a tabela de primos pequenos (poucos KB) e
 *                                    lida por todos os work-items: reuso
 *                                    temporal altissimo sobre conjunto que
 *                                    cabe no L1.
 *
 *  Cada algoritmo e medido em tres variantes que diferem EXCLUSIVAMENTE no
 *  modificador da instrucao de load (ver algoritmos_cache.cl).
 *
 *  Todos os resultados sao verificados contra uma implementacao de referencia
 *  em CPU antes de qualquer tempo ser reportado.
 *
 *  REQUISITOS
 *  ----------
 *  Runtime OpenCL Clang/LLVM com inline asm GCN (ROCm) e ROC_ENABLE_PRE_VEGA=1.
 *
 *  Build:
 *    g++ -Wall -O2 experimentos/algoritmos_cache.cpp -o output/algoritmos_cache -lOpenCL
 *  Execucao (a partir da raiz do projeto, pois le o .cl por caminho relativo):
 *    source setup/env_rocm.sh && ./output/algoritmos_cache
 *
 *  Saida: resultados/algoritmos_cache.csv + isa/rocm_gfx803_algoritmos_cache.bin
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define CL_PATH "experimentos/algoritmos_cache.cl"

#define WARMUP 5
#define RUNS   15
#define BLOCO  256          /* work-items por workgroup (casa com TAM_BLOCO no .cl) */

/* --- A) Scan: 256^3 elementos, exatamente tres niveis de reducao de blocos --- */
#define SCAN_N    16777216u              /* 16 M uint = 64 MB                     */
#define SCAN_N1   (SCAN_N / BLOCO)       /* 65536 somas de bloco                  */
#define SCAN_N2   (SCAN_N1 / BLOCO)      /* 256                                   */
#define SCAN_N3   (SCAN_N2 / BLOCO)      /* 1                                     */

/* --- B) Primos: candidatos consecutivos a partir de uma base --- */
#define PRIMOS_N     131072u
#define PRIMOS_BASE  1000000u

/* --- C) Crivo segmentado --- */
#define CRIVO_N      16777216u           /* limite superior do intervalo          */
#define CRIVO_BASE   (CRIVO_N / 2u)      /* peneira [N/2, N)                      */
#define CRIVO_SEG    512u                /* elementos por work-item               */
#define CRIVO_ITEMS  ((CRIVO_N - CRIVO_BASE) / CRIVO_SEG)  /* 4096 work-items     */
/* Peneirar a metade superior mantem a carga uniforme: para todo segmento,
   fim > 4096^2, de modo que TODOS os primos da tabela sao aplicados. Sobre
   [0, N) os segmentos iniciais descartam quase todos os primos pela condicao
   p*p > fim, o que produz desbalanceamento severo e desvio padrao de ~30%. */

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

static char* read_file(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "nao abriu '%s'\n", path);
        fprintf(stderr, "execute a partir da raiz do projeto.\n");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(n + 1);
    size_t rd = fread(buf, 1, n, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

static const char* NIVEL[3] = { "L1 (normal)", "L2 (glc)", "VRAM (glc slc)" };

typedef struct {
    const char* algoritmo;
    const char* perfil;
    double t[3], sd[3], med[3];
    int ok[3];
} Res;

/* Estatistica de uma serie de medicoes. */
static int cmp_double(const void* a, const void* b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* Devolve media, desvio padrao e MEDIANA. As razoes deste experimento sao
   calculadas sobre a mediana: as medicoes apresentam outliers esporadicos em
   ambas as direcoes (execucoes lentas por variacao de frequencia da GPU, e
   execucoes rapidas isoladas que mudam de variante entre execucoes, logo nao
   sao efeito das flags). A mediana e robusta aos dois casos. */
static void estat(const double* t, int n, double* mean, double* sd, double* med)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += t[i];
    *mean = s / n;
    double var = 0.0;
    for (int i = 0; i < n; i++) var += (t[i] - *mean) * (t[i] - *mean);
    *sd = sqrt(var / (n - 1));

    double* c = (double*)malloc((size_t)n * sizeof(double));
    memcpy(c, t, (size_t)n * sizeof(double));
    qsort(c, (size_t)n, sizeof(double), cmp_double);
    *med = (n % 2) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
}

/* ------------------------------------------------------------------------- */
/*  Referencias em CPU, para verificacao                                      */
/* ------------------------------------------------------------------------- */

static void scan_cpu(const unsigned* in, unsigned* out, unsigned n)
{
    unsigned acc = 0;
    for (unsigned i = 0; i < n; i++) { acc += in[i]; out[i] = acc; }
}

static int eh_primo(unsigned v)
{
    if (v < 2u) return 0;
    for (unsigned d = 2u; d * d <= v; d++) if (v % d == 0u) return 0;
    return 1;
}

/* Crivo simples em CPU sobre [0, n), devolve o numero de primos. */
static unsigned crivo_cpu(unsigned n, unsigned char* marcas)
{
    memset(marcas, 0, n);
    marcas[0] = marcas[1] = 1;
    for (unsigned p = 2; (unsigned long long)p * p < n; p++)
        if (!marcas[p])
            for (unsigned long long m = (unsigned long long)p * p; m < n; m += p)
                marcas[(size_t)m] = 1;
    unsigned c = 0;
    for (unsigned i = 0; i < n; i++) if (!marcas[i]) c++;
    return c;
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("================================================================\n");
    printf(" Scan, primos e crivo sob os tres niveis de cache\n");
    printf(" warmup %d | runs %d | workgroup %d\n", WARMUP, RUNS, BLOCO);
    printf("================================================================\n\n");

    char* src = read_file(CL_PATH);
    if (!src) return 1;

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

    const char* srcs[] = { src };
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
        FILE* f = fopen("isa/rocm_gfx803_algoritmos_cache.bin", "wb");
        if (f) { fwrite(bin, 1, bs, f); fclose(f);
                 printf("[ISA] isa/rocm_gfx803_algoritmos_cache.bin (%lu bytes)\n\n",
                        (unsigned long)bs); }
        free(bin);
    }

    Res res[3];
    int nres = 0;

    /* ================================================================== */
    /*  A) SCAN                                                           */
    /* ================================================================== */
    {
        printf("--- ALGORITMO A: scan (soma de prefixos), %u elementos = %.0f MB ---\n",
               SCAN_N, SCAN_N * 4.0 / 1048576.0);

        unsigned* h_in  = (unsigned*)malloc((size_t)SCAN_N * 4);
        unsigned* h_out = (unsigned*)malloc((size_t)SCAN_N * 4);
        unsigned* h_ref = (unsigned*)malloc((size_t)SCAN_N * 4);
        for (unsigned i = 0; i < SCAN_N; i++) h_in[i] = i % 7u;   /* soma cabe em uint */
        scan_cpu(h_in, h_ref, SCAN_N);

        cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)SCAN_N * 4, h_in, &err);
        CHECK(err, "scan d_in");
        cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)SCAN_N * 4, NULL, &err);
        CHECK(err, "scan d_out");
        cl_mem d_s1  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)SCAN_N1 * 4, NULL, &err);
        cl_mem d_s1s = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)SCAN_N1 * 4, NULL, &err);
        cl_mem d_s2  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)SCAN_N2 * 4, NULL, &err);
        cl_mem d_s2s = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)SCAN_N2 * 4, NULL, &err);
        cl_mem d_s3  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)(SCAN_N3 ? SCAN_N3 : 1) * 4,
                                      NULL, &err);
        CHECK(err, "scan buffers");

        static const char* fn_scan[3] = { "scan_bloco_L1", "scan_bloco_L2", "scan_bloco_VRAM" };
        static const char* fn_add[3]  = { "add_offset_L1", "add_offset_L2", "add_offset_VRAM" };

        Res r; r.algoritmo = "scan"; r.perfil = "reuso global ~1, LDS interno";

        for (int v = 0; v < 3; v++) {
            cl_kernel ks = clCreateKernel(prog, fn_scan[v], &err); CHECK(err, fn_scan[v]);
            cl_kernel ka = clCreateKernel(prog, fn_add[v],  &err); CHECK(err, fn_add[v]);

            /* Uma passada completa do scan em tres fases. */
            #define SCAN_FASE(K, IN, OUT, SOMAS, N)                                     \
                do {                                                                    \
                    unsigned n_ = (unsigned)(N);                                        \
                    size_t g_ = ((n_ + BLOCO - 1) / BLOCO) * (size_t)BLOCO, l_ = BLOCO; \
                    clSetKernelArg(K, 0, sizeof(cl_mem), &(IN));                        \
                    clSetKernelArg(K, 1, sizeof(cl_mem), &(OUT));                       \
                    clSetKernelArg(K, 2, sizeof(cl_mem), &(SOMAS));                     \
                    clSetKernelArg(K, 3, sizeof(cl_uint) * BLOCO, NULL);                \
                    clSetKernelArg(K, 4, sizeof(cl_uint), &n_);                         \
                    err = clEnqueueNDRangeKernel(q, K, 1, NULL, &g_, &l_, 0, NULL, NULL);\
                    CHECK(err, "scan fase");                                            \
                } while (0)

            #define ADD_FASE(K, DADOS, PREF, N)                                         \
                do {                                                                    \
                    unsigned n_ = (unsigned)(N);                                        \
                    size_t g_ = ((n_ + BLOCO - 1) / BLOCO) * (size_t)BLOCO, l_ = BLOCO; \
                    clSetKernelArg(K, 0, sizeof(cl_mem), &(DADOS));                     \
                    clSetKernelArg(K, 1, sizeof(cl_mem), &(PREF));                       \
                    clSetKernelArg(K, 2, sizeof(cl_uint), &n_);                         \
                    err = clEnqueueNDRangeKernel(q, K, 1, NULL, &g_, &l_, 0, NULL, NULL);\
                    CHECK(err, "add fase");                                             \
                } while (0)

            #define SCAN_COMPLETO()                                  \
                do {                                                 \
                    SCAN_FASE(ks, d_in,  d_out, d_s1,  SCAN_N);      \
                    SCAN_FASE(ks, d_s1,  d_s1s, d_s2,  SCAN_N1);     \
                    SCAN_FASE(ks, d_s2,  d_s2s, d_s3,  SCAN_N2);     \
                    ADD_FASE(ka, d_s1s, d_s2s, SCAN_N1);             \
                    ADD_FASE(ka, d_out, d_s1s, SCAN_N);              \
                } while (0)

            for (int w = 0; w < WARMUP; w++) SCAN_COMPLETO();
            clFinish(q);

            /* Cronometra a sequencia inteira com eventos de fronteira. */
            double t[RUNS];
            for (int rr = 0; rr < RUNS; rr++) {
                cl_event ev0, ev1;
                clEnqueueMarkerWithWaitList(q, 0, NULL, &ev0);
                SCAN_COMPLETO();
                clEnqueueMarkerWithWaitList(q, 0, NULL, &ev1);
                clWaitForEvents(1, &ev1);
                cl_ulong a = 0, b = 0;
                clGetEventProfilingInfo(ev0, CL_PROFILING_COMMAND_END, sizeof(a), &a, NULL);
                clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_END, sizeof(b), &b, NULL);
                t[rr] = (double)(b - a) / 1e6;
                clReleaseEvent(ev0); clReleaseEvent(ev1);
            }
            estat(t, RUNS, &r.t[v], &r.sd[v], &r.med[v]);

            clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, (size_t)SCAN_N * 4, h_out, 0, NULL, NULL);
            r.ok[v] = (memcmp(h_out, h_ref, (size_t)SCAN_N * 4) == 0);

            printf("    %-16s %-16s media %8.3f (dp %5.3f)  mediana %8.3f  %6.3fx  %s\n",
                   fn_scan[v], NIVEL[v], r.t[v], r.sd[v], r.med[v],
                   r.med[v] / r.med[0], r.ok[v] ? "OK" : "DIVERGIU");

            clReleaseKernel(ks); clReleaseKernel(ka);
        }
        printf("\n");
        res[nres++] = r;

        clReleaseMemObject(d_in);  clReleaseMemObject(d_out);
        clReleaseMemObject(d_s1);  clReleaseMemObject(d_s1s);
        clReleaseMemObject(d_s2);  clReleaseMemObject(d_s2s);
        clReleaseMemObject(d_s3);
        free(h_in); free(h_out); free(h_ref);
    }

    /* ================================================================== */
    /*  B) PRIMOS por divisao sucessiva — controle negativo               */
    /* ================================================================== */
    {
        printf("--- ALGORITMO B: primos por divisao sucessiva (controle negativo) ---\n");
        printf("    %u candidatos a partir de %u\n", PRIMOS_N, PRIMOS_BASE);

        unsigned* h_c   = (unsigned*)malloc((size_t)PRIMOS_N * 4);
        unsigned* h_r   = (unsigned*)malloc((size_t)PRIMOS_N * 4);
        unsigned* h_ref = (unsigned*)malloc((size_t)PRIMOS_N * 4);
        for (unsigned i = 0; i < PRIMOS_N; i++) h_c[i] = PRIMOS_BASE + i;
        unsigned n_primos_ref = 0;
        for (unsigned i = 0; i < PRIMOS_N; i++) {
            h_ref[i] = (unsigned)eh_primo(h_c[i]);
            n_primos_ref += h_ref[i];
        }
        printf("    referencia em CPU: %u primos no intervalo\n", n_primos_ref);

        cl_mem d_c = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    (size_t)PRIMOS_N * 4, h_c, &err);
        CHECK(err, "primos d_c");
        cl_mem d_r = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)PRIMOS_N * 4, NULL, &err);
        CHECK(err, "primos d_r");

        static const char* fn[3] = { "primos_L1", "primos_L2", "primos_VRAM" };
        Res r; r.algoritmo = "primos"; r.perfil = "compute-bound, 1 load/work-item";

        size_t g = ((PRIMOS_N + BLOCO - 1) / BLOCO) * (size_t)BLOCO, l = BLOCO;
        const unsigned n_arg = PRIMOS_N;

        for (int v = 0; v < 3; v++) {
            cl_kernel k = clCreateKernel(prog, fn[v], &err); CHECK(err, fn[v]);
            clSetKernelArg(k, 0, sizeof(cl_mem), &d_c);
            clSetKernelArg(k, 1, sizeof(cl_mem), &d_r);
            clSetKernelArg(k, 2, sizeof(cl_uint), &n_arg);

            for (int w = 0; w < WARMUP; w++) {
                err = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
                CHECK(err, "primos warmup");
            }
            clFinish(q);

            double t[RUNS];
            for (int rr = 0; rr < RUNS; rr++) {
                cl_event ev;
                err = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &ev);
                CHECK(err, "primos enqueue");
                clWaitForEvents(1, &ev);
                t[rr] = ms_of(ev);
                clReleaseEvent(ev);
            }
            estat(t, RUNS, &r.t[v], &r.sd[v], &r.med[v]);

            clEnqueueReadBuffer(q, d_r, CL_TRUE, 0, (size_t)PRIMOS_N * 4, h_r, 0, NULL, NULL);
            r.ok[v] = (memcmp(h_r, h_ref, (size_t)PRIMOS_N * 4) == 0);

            printf("    %-16s %-16s media %8.3f (dp %5.3f)  mediana %8.3f  %6.3fx  %s\n",
                   fn[v], NIVEL[v], r.t[v], r.sd[v], r.med[v],
                   r.med[v] / r.med[0], r.ok[v] ? "OK" : "DIVERGIU");
            clReleaseKernel(k);
        }
        printf("\n");
        res[nres++] = r;

        clReleaseMemObject(d_c); clReleaseMemObject(d_r);
        free(h_c); free(h_r); free(h_ref);
    }

    /* ================================================================== */
    /*  C) CRIVO segmentado                                               */
    /* ================================================================== */
    {
        printf("--- ALGORITMO C: crivo de Eratostenes segmentado ---\n");
        printf("    intervalo [%u, %u) | segmento %u | %u work-items\n",
               CRIVO_BASE, CRIVO_N, CRIVO_SEG, CRIVO_ITEMS);

        /* Referencia em CPU e tabela de primos pequenos ate sqrt(N) */
        unsigned char* h_ref = (unsigned char*)malloc(CRIVO_N);
        crivo_cpu(CRIVO_N, h_ref);
        unsigned n_primos_ref = 0;
        for (unsigned i = CRIVO_BASE; i < CRIVO_N; i++) if (!h_ref[i]) n_primos_ref++;

        unsigned lim = 1u;
        while ((unsigned long long)(lim + 1) * (lim + 1) <= CRIVO_N) lim++;
        unsigned n_pp = 0;
        for (unsigned p = 2; p <= lim; p++) if (!h_ref[p]) n_pp++;
        unsigned* h_pp = (unsigned*)malloc((size_t)n_pp * 4);
        {
            unsigned j = 0;
            for (unsigned p = 2; p <= lim; p++) if (!h_ref[p]) h_pp[j++] = p;
        }
        printf("    %u primos pequenos ate %u (tabela de %.1f KB)\n",
               n_pp, lim, n_pp * 4.0 / 1024.0);
        printf("    referencia em CPU: %u primos em [%u, %u)\n",
               n_primos_ref, CRIVO_BASE, CRIVO_N);

        const unsigned faixa = CRIVO_N - CRIVO_BASE;
        unsigned char* h_m = (unsigned char*)malloc(faixa);

        cl_mem d_pp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)n_pp * 4, h_pp, &err);
        CHECK(err, "crivo d_pp");
        cl_mem d_m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, faixa, NULL, &err);
        CHECK(err, "crivo d_m");

        static const char* fn[3] = { "crivo_L1", "crivo_L2", "crivo_VRAM" };
        Res r; r.algoritmo = "crivo"; r.perfil = "tabela de primos com reuso alto";

        size_t g = ((CRIVO_ITEMS + BLOCO - 1) / BLOCO) * (size_t)BLOCO, l = BLOCO;
        const unsigned base = CRIVO_BASE, seg = CRIVO_SEG, npp = n_pp;
        const unsigned char zero = 0;

        for (int v = 0; v < 3; v++) {
            cl_kernel k = clCreateKernel(prog, fn[v], &err); CHECK(err, fn[v]);
            clSetKernelArg(k, 0, sizeof(cl_mem), &d_pp);
            clSetKernelArg(k, 1, sizeof(cl_uint), &npp);
            clSetKernelArg(k, 2, sizeof(cl_mem), &d_m);
            clSetKernelArg(k, 3, sizeof(cl_uint), &base);
            clSetKernelArg(k, 4, sizeof(cl_uint), &seg);

            for (int w = 0; w < WARMUP; w++) {
                clEnqueueFillBuffer(q, d_m, &zero, 1, 0, faixa, 0, NULL, NULL);
                err = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
                CHECK(err, "crivo warmup");
            }
            clFinish(q);

            double t[RUNS];
            for (int rr = 0; rr < RUNS; rr++) {
                clEnqueueFillBuffer(q, d_m, &zero, 1, 0, faixa, 0, NULL, NULL);
                cl_event ev;
                err = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &ev);
                CHECK(err, "crivo enqueue");
                clWaitForEvents(1, &ev);
                t[rr] = ms_of(ev);
                clReleaseEvent(ev);
            }
            estat(t, RUNS, &r.t[v], &r.sd[v], &r.med[v]);

            clEnqueueReadBuffer(q, d_m, CL_TRUE, 0, faixa, h_m, 0, NULL, NULL);
            unsigned c = 0;
            for (unsigned i = 0; i < faixa; i++) if (!h_m[i]) c++;
            r.ok[v] = (c == n_primos_ref);

            printf("    %-16s %-16s media %8.3f (dp %5.3f)  mediana %8.3f  %6.3fx  %s (%u primos)\n",
                   fn[v], NIVEL[v], r.t[v], r.sd[v], r.med[v],
                   r.med[v] / r.med[0], r.ok[v] ? "OK" : "DIVERGIU", c);
            clReleaseKernel(k);
        }
        printf("\n");
        res[nres++] = r;

        clReleaseMemObject(d_pp); clReleaseMemObject(d_m);
        free(h_ref); free(h_pp); free(h_m);
    }

    /* ------------------------------- Sintese ------------------------------- */
    printf("================================================================\n");
    printf(" CUSTO DE SUBIR NA HIERARQUIA, POR ALGORITMO\n");
    printf("================================================================\n");
    printf(" (razoes sobre a MEDIANA, robusta a outliers em ambas as direcoes)\n\n");
    printf(" %-10s %-28s %9s %8s %9s %8s\n",
           "algoritmo", "perfil de memoria", "L1 mediana", "dp rel.", "L2/L1", "VRAM/L1");
    for (int i = 0; i < nres; i++)
        printf(" %-10s %-28s %8.3f %7.1f%% %8.3fx %8.3fx\n",
               res[i].algoritmo, res[i].perfil, res[i].med[0],
               100.0 * res[i].sd[0] / res[i].t[0],
               res[i].med[1] / res[i].med[0], res[i].med[2] / res[i].med[0]);

    printf("\n Todos os resultados foram verificados contra referencia em CPU.\n");
    printf(" O algoritmo `primos` e o controle negativo: por ser compute-bound,\n");
    printf(" espera-se razao proxima de 1,00x em ambos os niveis.\n");
    printf("================================================================\n");

    FILE* csv = fopen("resultados/algoritmos_cache.csv", "w");
    if (csv) {
        fprintf(csv, "algoritmo,perfil,nivel,media_ms,dp_ms,mediana_ms,razao_mediana_sobre_L1,verificado\n");
        for (int i = 0; i < nres; i++)
            for (int v = 0; v < 3; v++)
                fprintf(csv, "%s,\"%s\",\"%s\",%.6f,%.6f,%.6f,%.6f,%d\n",
                        res[i].algoritmo, res[i].perfil, NIVEL[v],
                        res[i].t[v], res[i].sd[v], res[i].med[v],
                        res[i].med[v] / res[i].med[0], res[i].ok[v]);
        fclose(csv);
        printf("\nCSV: resultados/algoritmos_cache.csv\n");
    }

    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(plats); free(src);
    return 0;
}
