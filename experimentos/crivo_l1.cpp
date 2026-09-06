/*
 * =============================================================================
 *  crivo_l1.cpp — Recuperando a sensibilidade ao L1 num algoritmo de primos
 * =============================================================================
 *
 *  O Experimento 8 (algoritmos_cache) mediu 1,00x para o crivo segmentado em
 *  todos os niveis de cache. Este experimento testa a hipotese de que a causa
 *  nao e o problema (primos), e sim a FORMULACAO: o crivo segmentado tem poucos
 *  loads, uniformes, e e dominado por escrita.
 *
 *  Tres kernels, cada um em 3 variantes de flag de cache:
 *
 *    crivo      — crivo de Eratostenes segmentado (linha de base do Exp. 8)
 *    tdiv_unif  — divisao sucessiva tabelada, acesso UNIFORME  a tabela
 *    tdiv_rot   — divisao sucessiva tabelada, acesso DISPERSO  a tabela
 *
 *  tdiv_unif e tdiv_rot executam exatamente o mesmo trabalho e o mesmo numero
 *  de loads; diferem apenas em quantos enderecos distintos os 64 lanes de uma
 *  wavefront tocam por instrucao. A diferenca entre eles isola o fator (i) do
 *  diagnostico; a diferenca entre tdiv e crivo isola o fator (ii).
 *
 *  Teste de divisibilidade (Granlund & Montgomery, 1994):
 *      para p impar,  p | v  <=>  v * (p^-1 mod 2^32)  <=  floor((2^32-1)/p)
 *  Custo: 1 multiplicacao e 1 comparacao — sem divisao inteira, que no GCN
 *  custa ~25 instrucoes e tornaria o kernel compute-bound.
 *
 *  Todos os resultados sao verificados contra referencia em CPU.
 *
 *  Build:
 *    g++ -Wall -O2 experimentos/crivo_l1.cpp -o output/crivo_l1 -lOpenCL
 *  Execucao (a partir da raiz do projeto):
 *    source setup/env_rocm.sh && ./output/crivo_l1
 *
 *  Saida: resultados/crivo_l1.csv + isa/rocm_gfx803_crivo_l1.bin
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define CL_PATH "experimentos/crivo_l1.cl"

#define WARMUP 5
#define RUNS   21
#define BLOCO  256

/* --- A) Divisao sucessiva tabelada --------------------------------------- *
 * A tabela util vai ate sqrt(vmax). Manter vmax abaixo de ~4,2 M mantem a
 * tabela em ~2,4 KB, bem dentro dos 16 KB do L1 por CU, com reuso de leitura
 * igual ao numero de candidatos.                                            */
#define TD_BASE   3000000u
#define TD_N      1048576u               /* candidatos: [3,0 M ; 4,05 M)      */

/* --- B) Crivo segmentado (parametros identicos aos do Experimento 8) ------ */
#define CRIVO_N      16777216u
#define CRIVO_BASE   (CRIVO_N / 2u)
#define CRIVO_SEG    512u
#define CRIVO_ITEMS  ((CRIVO_N - CRIVO_BASE) / CRIVO_SEG)

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
        fprintf(stderr, "nao abriu '%s' — execute a partir da raiz do projeto.\n", path);
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

static int cmp_double(const void* a, const void* b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

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

static const char* NIVEL[3] = { "L1 (normal)", "L2 (glc)", "VRAM (glc slc)" };

typedef struct {
    const char* kernel;
    const char* perfil;
    double t[3], sd[3], med[3];
    int ok[3];
    double loads;        /* loads globais emitidos pelo kernel inteiro */
} Res;

/* Inverso multiplicativo de p (impar) modulo 2^32, por iteracao de Newton.
   x_{k+1} = x_k * (2 - p*x_k) dobra o numero de bits corretos a cada passo;
   partindo de x_0 = p (correto em 3 bits), 5 passos cobrem os 32 bits. */
static unsigned inv32(unsigned p)
{
    unsigned x = p;
    for (int i = 0; i < 5; i++) x *= 2u - p * x;
    return x;
}

/* Crivo simples em CPU sobre [0, n): marcas[i] = 1 se i for composto. */
static void crivo_cpu(unsigned n, unsigned char* marcas)
{
    memset(marcas, 0, n);
    marcas[0] = marcas[1] = 1;
    for (unsigned p = 2; (unsigned long long)p * p < n; p++)
        if (!marcas[p])
            for (unsigned long long m = (unsigned long long)p * p; m < n; m += p)
                marcas[(size_t)m] = 1;
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("================================================================\n");
    printf(" Sensibilidade ao L1 em algoritmos de primos: formulacao importa\n");
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
    for (cl_uint i = 0; i < np; i++) {
        char pn[256] = {0};
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(pn), pn, NULL);
        if (strstr(pn, "AMD Accelerated") == NULL) continue;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) {
            plat = plats[i];
            break;
        }
    }
    if (!dev) { fprintf(stderr, "Plataforma AMD APP nao encontrada (ROC_ENABLE_PRE_VEGA=1?).\n"); return 1; }

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
        return 1;
    }
    {
        size_t bs = 0;
        clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bs, NULL);
        unsigned char* bin = (unsigned char*)malloc(bs);
        unsigned char* bins[] = { bin };
        clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), bins, NULL);
        FILE* f = fopen("isa/rocm_gfx803_crivo_l1.bin", "wb");
        if (f) { fwrite(bin, 1, bs, f); fclose(f);
                 printf("[ISA] isa/rocm_gfx803_crivo_l1.bin (%lu bytes)\n\n", (unsigned long)bs); }
        free(bin);
    }

    Res res[3];
    int nres = 0;

    /* ================================================================== */
    /*  Tabelas de primos e referencias em CPU                            */
    /* ================================================================== */

    const unsigned TD_MAX = TD_BASE + TD_N;          /* exclusivo */
    unsigned char* h_sieve = (unsigned char*)malloc(TD_MAX);
    crivo_cpu(TD_MAX, h_sieve);

    /* Tabela para a divisao tabelada: primos IMPARES ate sqrt(TD_MAX-1). */
    unsigned lim_td = 1u;
    while ((unsigned long long)(lim_td + 1) * (lim_td + 1) < TD_MAX) lim_td++;
    unsigned n_td = 0;
    for (unsigned p = 3; p <= lim_td; p += 2) if (!h_sieve[p]) n_td++;

    unsigned* h_inv = (unsigned*)malloc((size_t)n_td * 4);
    unsigned* h_lim = (unsigned*)malloc((size_t)n_td * 4);
    {
        unsigned j = 0;
        for (unsigned p = 3; p <= lim_td; p += 2) {
            if (h_sieve[p]) continue;
            unsigned iv = inv32(p);
            if (p * iv != 1u) { fprintf(stderr, "inv32 falhou para p=%u\n", p); return 1; }
            h_inv[j] = iv;
            h_lim[j] = 0xFFFFFFFFu / p;
            j++;
        }
    }
    printf("Tabela da divisao tabelada: %u primos impares ate %u\n", n_td, lim_td);
    printf("  working set reusado = 2 x %u x 4 B = %.1f KB  (L1 = 16 KB/CU)\n",
           n_td, 2.0 * n_td * 4.0 / 1024.0);
    printf("  reuso de leitura    = %u candidatos\n", TD_N);

    /* Sanidade do teste de Granlund-Montgomery contra o crivo em CPU. */
    {
        unsigned checados = 0;
        for (unsigned v = TD_BASE; v < TD_BASE + 4096u; v++) {
            unsigned comp = ((v & 1u) == 0u) ? 1u : 0u;
            for (unsigned i = 0; i < n_td; i++)
                if (v * h_inv[i] <= h_lim[i]) { comp = 1u; break; }
            if (comp != (unsigned)h_sieve[v]) {
                fprintf(stderr, "divergencia GM em v=%u (gm=%u crivo=%u)\n",
                        v, comp, (unsigned)h_sieve[v]);
                return 1;
            }
            checados++;
        }
        printf("  teste de divisibilidade validado em %u valores\n", checados);
    }

    unsigned n_primos_td = 0;
    for (unsigned v = TD_BASE; v < TD_MAX; v++) if (!h_sieve[v]) n_primos_td++;
    printf("  referencia em CPU: %u primos em [%u, %u)\n\n",
           n_primos_td, TD_BASE, TD_MAX);

    /* ================================================================== */
    /*  A) DIVISAO SUCESSIVA TABELADA — uniforme vs. disperso             */
    /* ================================================================== */
    {
        unsigned char* h_m = (unsigned char*)malloc(TD_N);

        cl_mem d_inv = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (size_t)n_td * 4, h_inv, &err); CHECK(err, "d_inv");
        cl_mem d_lim = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (size_t)n_td * 4, h_lim, &err); CHECK(err, "d_lim");
        cl_mem d_m   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, TD_N, NULL, &err);
        CHECK(err, "d_m");

        static const char* fn[2][3] = {
            { "tdiv_unif_L1", "tdiv_unif_L2", "tdiv_unif_VRAM" },
            { "tdiv_rot_L1",  "tdiv_rot_L2",  "tdiv_rot_VRAM"  }
        };
        static const char* rotulo[2] = { "tdiv_unif", "tdiv_rot" };
        static const char* perfil[2] = { "tabela 2,4 KB, acesso uniforme",
                                         "tabela 2,4 KB, acesso disperso" };

        size_t g = ((TD_N + BLOCO - 1) / BLOCO) * (size_t)BLOCO, l = BLOCO;
        const unsigned base = TD_BASE, ncand = TD_N, nprim = n_td;

        for (int k = 0; k < 2; k++) {
            printf("--- %s (%s) ---\n", rotulo[k], perfil[k]);
            Res r;
            r.kernel = rotulo[k];
            r.perfil = perfil[k];
            r.loads = 2.0 * (double)TD_N * (double)n_td;

            for (int v = 0; v < 3; v++) {
                cl_kernel kk = clCreateKernel(prog, fn[k][v], &err); CHECK(err, fn[k][v]);
                clSetKernelArg(kk, 0, sizeof(cl_mem), &d_inv);
                clSetKernelArg(kk, 1, sizeof(cl_mem), &d_lim);
                clSetKernelArg(kk, 2, sizeof(cl_uint), &nprim);
                clSetKernelArg(kk, 3, sizeof(cl_mem), &d_m);
                clSetKernelArg(kk, 4, sizeof(cl_uint), &base);
                clSetKernelArg(kk, 5, sizeof(cl_uint), &ncand);

                for (int w = 0; w < WARMUP; w++) {
                    err = clEnqueueNDRangeKernel(q, kk, 1, NULL, &g, &l, 0, NULL, NULL);
                    CHECK(err, "tdiv warmup");
                }
                clFinish(q);

                double t[RUNS];
                for (int rr = 0; rr < RUNS; rr++) {
                    cl_event ev;
                    err = clEnqueueNDRangeKernel(q, kk, 1, NULL, &g, &l, 0, NULL, &ev);
                    CHECK(err, "tdiv enqueue");
                    clWaitForEvents(1, &ev);
                    t[rr] = ms_of(ev);
                    clReleaseEvent(ev);
                }
                estat(t, RUNS, &r.t[v], &r.sd[v], &r.med[v]);

                clEnqueueReadBuffer(q, d_m, CL_TRUE, 0, TD_N, h_m, 0, NULL, NULL);
                unsigned c = 0;
                for (unsigned i = 0; i < TD_N; i++) if (!h_m[i]) c++;
                r.ok[v] = (c == n_primos_td);

                printf("    %-16s %-16s media %8.3f (dp %5.3f)  mediana %8.3f  %6.3fx  %s (%u primos)\n",
                       fn[k][v], NIVEL[v], r.t[v], r.sd[v], r.med[v],
                       r.med[v] / r.med[0], r.ok[v] ? "OK" : "DIVERGIU", c);
                clReleaseKernel(kk);
            }
            printf("    %.0f M loads globais | %.1f G loads/s no nivel L1\n\n",
                   r.loads / 1e6, r.loads / (r.med[0] * 1e6));
            res[nres++] = r;
        }

        clReleaseMemObject(d_inv); clReleaseMemObject(d_lim); clReleaseMemObject(d_m);
        free(h_m);
    }

    /* ================================================================== */
    /*  B) CRIVO SEGMENTADO — linha de base                               */
    /* ================================================================== */
    {
        printf("--- crivo segmentado (linha de base do Experimento 8) ---\n");

        unsigned char* h_ref = (unsigned char*)malloc(CRIVO_N);
        crivo_cpu(CRIVO_N, h_ref);
        unsigned n_ref = 0;
        for (unsigned i = CRIVO_BASE; i < CRIVO_N; i++) if (!h_ref[i]) n_ref++;

        unsigned lim = 1u;
        while ((unsigned long long)(lim + 1) * (lim + 1) <= CRIVO_N) lim++;
        unsigned n_pp = 0;
        for (unsigned p = 2; p <= lim; p++) if (!h_ref[p]) n_pp++;
        unsigned* h_pp = (unsigned*)malloc((size_t)n_pp * 4);
        {
            unsigned j = 0;
            for (unsigned p = 2; p <= lim; p++) if (!h_ref[p]) h_pp[j++] = p;
        }

        const unsigned faixa = CRIVO_N - CRIVO_BASE;
        unsigned char* h_m = (unsigned char*)malloc(faixa);

        cl_mem d_pp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)n_pp * 4, h_pp, &err); CHECK(err, "d_pp");
        cl_mem d_m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, faixa, NULL, &err);
        CHECK(err, "d_m crivo");

        static const char* fn[3] = { "crivo_L1", "crivo_L2", "crivo_VRAM" };
        Res r;
        r.kernel = "crivo";
        r.perfil = "tabela 2,2 KB, uniforme, dominado por escrita";
        /* loads uniformes: 1 requisicao por wavefront, nao por work-item */
        r.loads = (double)CRIVO_ITEMS * (double)n_pp;

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
            r.ok[v] = (c == n_ref);

            printf("    %-16s %-16s media %8.3f (dp %5.3f)  mediana %8.3f  %6.3fx  %s (%u primos)\n",
                   fn[v], NIVEL[v], r.t[v], r.sd[v], r.med[v],
                   r.med[v] / r.med[0], r.ok[v] ? "OK" : "DIVERGIU", c);
            clReleaseKernel(k);
        }
        printf("    %.0f M loads globais | %.1f G loads/s no nivel L1\n\n",
               r.loads / 1e6, r.loads / (r.med[0] * 1e6));
        res[nres++] = r;

        clReleaseMemObject(d_pp); clReleaseMemObject(d_m);
        free(h_ref); free(h_pp); free(h_m);
    }

    /* ------------------------------- Sintese ------------------------------- */
    printf("================================================================\n");
    printf(" CUSTO DE CONTORNAR O L1, POR FORMULACAO\n");
    printf("================================================================\n\n");
    printf(" %-12s %-34s %10s %8s %9s %9s\n",
           "kernel", "perfil", "L1 mediana", "dp rel.", "L2/L1", "VRAM/L1");
    for (int i = 0; i < nres; i++)
        printf(" %-12s %-34s %9.3f %7.1f%% %8.3fx %8.3fx\n",
               res[i].kernel, res[i].perfil, res[i].med[0],
               100.0 * res[i].sd[0] / res[i].t[0],
               res[i].med[1] / res[i].med[0], res[i].med[2] / res[i].med[0]);
    printf("\n Referencia (Experimento 6): matmul naive 1,273x | stencil 3x3 1,796x\n");
    printf("================================================================\n");

    FILE* csv = fopen("resultados/crivo_l1.csv", "w");
    if (csv) {
        fprintf(csv, "kernel,perfil,nivel,media_ms,dp_ms,mediana_ms,razao_mediana_sobre_L1,loads_globais,verificado\n");
        for (int i = 0; i < nres; i++)
            for (int v = 0; v < 3; v++)
                fprintf(csv, "%s,\"%s\",\"%s\",%.6f,%.6f,%.6f,%.6f,%.0f,%d\n",
                        res[i].kernel, res[i].perfil, NIVEL[v],
                        res[i].t[v], res[i].sd[v], res[i].med[v],
                        res[i].med[v] / res[i].med[0], res[i].loads, res[i].ok[v]);
        fclose(csv);
        printf("\nCSV: resultados/crivo_l1.csv\n");
    }

    free(h_sieve); free(h_inv); free(h_lim);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(plats); free(src);
    return 0;
}
