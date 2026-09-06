/*
 * =============================================================================
 *  tdiv_ws.cpp — O custo de contornar o L1 em funcao do conjunto de trabalho
 * =============================================================================
 *
 *  O Experimento 9 (crivo_l1) mediu 7,01x para a divisao sucessiva tabelada com
 *  acesso disperso, sobre uma tabela de 2,4 KB. A previsao que falta testar e a
 *  do Experimento 7 (P-chase): o efeito deve COLAPSAR quando o conjunto de
 *  trabalho reusado ultrapassar os 16 KB do L1 por CU, porque a partir dai o
 *  nivel "com L1" tambem passa a errar.
 *
 *  Este experimento varre o tamanho da tabela mantendo o ALGORITMO intacto:
 *
 *    - a tabela contem os k primeiros primos IMPARES, ate lim = p_k
 *    - os candidatos ficam em [lim^2 - n_cand, lim^2)
 *
 *  Assim todo fator primo de qualquer candidato e <= sqrt(vmax) = lim e esta na
 *  tabela: nenhum primo da tabela e supérfluo e o teste continua exato. O limite
 *  superior da varredura vem da aritmetica de 32 bits — lim < 65536 para que
 *  lim^2 caiba em uint — o que da uma tabela maxima de 6541 primos = 51 KB.
 *
 *  O numero total de loads e mantido constante (~640 M) ajustando n_cand, de
 *  modo que os pontos da curva sejam comparaveis em tempo.
 *
 *  Reaproveita os kernels de experimentos/crivo_l1.cl.
 *
 *  Build:
 *    g++ -Wall -O2 experimentos/tdiv_ws.cpp -o output/tdiv_ws -lOpenCL
 *  Execucao (a partir da raiz do projeto):
 *    source setup/env_rocm.sh && ./output/tdiv_ws
 *
 *  Saida: resultados/tdiv_ws.csv
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

#define WARMUP 3
#define RUNS   15
#define BLOCO  256

#define LOADS_ALVO  640000000.0     /* loads globais por ponto da curva */
#define N_CAND_MIN  32768u
#define PRIMO_MAX   65536u          /* lim^2 tem de caber em uint32 */

/* Tamanhos de tabela (numero de primos impares). O working set reusado e
   2 * k * 4 bytes, pois ha dois arrays paralelos (inverso e limite). */
static const unsigned ALVO[] = { 256, 512, 1024, 2048, 3072, 4096, 6541 };
#define N_PONTOS ((int)(sizeof(ALVO) / sizeof(ALVO[0])))

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
    if (!f) { fprintf(stderr, "nao abriu '%s' — rode a partir da raiz.\n", path); return NULL; }
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

static unsigned inv32(unsigned p)
{
    unsigned x = p;
    for (int i = 0; i < 5; i++) x *= 2u - p * x;
    return x;
}

static const char* NIVEL[3] = { "L1 (normal)", "L2 (glc)", "VRAM (glc slc)" };

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("================================================================\n");
    printf(" Custo de contornar o L1 vs. conjunto de trabalho reusado\n");
    printf(" warmup %d | runs %d | workgroup %d | ~%.0f M loads por ponto\n",
           WARMUP, RUNS, BLOCO, LOADS_ALVO / 1e6);
    printf("================================================================\n\n");

    /* --- Primos impares ate 65536, base de todas as tabelas --------------- */
    unsigned char* comp = (unsigned char*)calloc(PRIMO_MAX + 1, 1);
    for (unsigned p = 2; p * p <= PRIMO_MAX; p++)
        if (!comp[p]) for (unsigned m = p * p; m <= PRIMO_MAX; m += p) comp[m] = 1;
    unsigned n_all = 0;
    for (unsigned p = 3; p < PRIMO_MAX; p += 2) if (!comp[p]) n_all++;
    unsigned* pr = (unsigned*)malloc((size_t)n_all * 4);
    { unsigned j = 0; for (unsigned p = 3; p < PRIMO_MAX; p += 2) if (!comp[p]) pr[j++] = p; }
    printf("%u primos impares abaixo de %u (maior: %u)\n\n", n_all, PRIMO_MAX, pr[n_all - 1]);

    char* src = read_file(CL_PATH);
    if (!src) return 1;

    cl_int err;
    cl_uint np = 0;
    clGetPlatformIDs(0, NULL, &np);
    cl_platform_id* plats = (cl_platform_id*)malloc(np * sizeof(cl_platform_id));
    clGetPlatformIDs(np, plats, NULL);
    cl_device_id dev = NULL;
    for (cl_uint i = 0; i < np; i++) {
        char pn[256] = {0};
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(pn), pn, NULL);
        if (strstr(pn, "AMD Accelerated") == NULL) continue;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) break;
    }
    if (!dev) { fprintf(stderr, "Plataforma AMD APP nao encontrada.\n"); return 1; }
    { char b[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(b), b, NULL);
      printf("Dispositivo: %s\n\n", b); }

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err, "ctx");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK(err, "queue");
    const char* srcs[] = { src };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, NULL, &err); CHECK(err, "program");
    err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t ls = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ls);
        char* log = (char*)malloc(ls + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, log, NULL);
        log[ls] = '\0';
        fprintf(stderr, "FALHA no build:\n%s\n", log);
        return 1;
    }

    static const char* fn[3][3] = {
        { "tdiv_unif_L1", "tdiv_unif_L2", "tdiv_unif_VRAM" },
        { "tdiv_rot_L1",  "tdiv_rot_L2",  "tdiv_rot_VRAM"  },
        { "tdiv_dsyn_L1", "tdiv_dsyn_L2", "tdiv_dsyn_VRAM" }
    };
    static const char* rotulo[3] = { "tdiv_unif", "tdiv_rot", "tdiv_dsyn" };

    FILE* csv = fopen("resultados/tdiv_ws.csv", "w");
    if (csv) fprintf(csv, "kernel,n_primos,working_set_KB,lim,base,n_cand,loads,"
                          "nivel,media_ms,dp_ms,mediana_ms,razao_sobre_L1,"
                          "linhas_por_wave,req_por_ciclo_por_CU,verificado\n");

    printf(" %-10s %6s %8s %10s %10s %10s %10s %9s %9s\n",
           "kernel", "k", "WS (KB)", "n_cand", "L1 (ms)", "L2 (ms)", "VRAM (ms)",
           "L2/L1", "VRAM/L1");
    printf(" ---------------------------------------------------------------"
           "-------------------------\n");

    for (int pi = 0; pi < N_PONTOS; pi++) {
        unsigned k = ALVO[pi];
        if (k > n_all) k = n_all;
        unsigned lim = pr[k - 1];
        double ws_kb = 2.0 * k * 4.0 / 1024.0;

        /* Candidatos imediatamente abaixo de lim^2, para que todo fator primo
           de qualquer candidato esteja na tabela. */
        unsigned long long vmax = (unsigned long long)lim * lim;
        unsigned n_cand = (unsigned)(LOADS_ALVO / (2.0 * k));
        n_cand = (n_cand / BLOCO) * BLOCO;
        if (n_cand < N_CAND_MIN) n_cand = N_CAND_MIN;
        if ((unsigned long long)n_cand > vmax - (unsigned long long)lim - 1)
            n_cand = (unsigned)(vmax - lim - 1);
        n_cand = (n_cand / BLOCO) * BLOCO;
        unsigned base = (unsigned)(vmax - n_cand);

        /* Tabelas de Granlund-Montgomery. */
        unsigned* h_inv = (unsigned*)malloc((size_t)k * 4);
        unsigned* h_lim = (unsigned*)malloc((size_t)k * 4);
        for (unsigned i = 0; i < k; i++) {
            h_inv[i] = inv32(pr[i]);
            h_lim[i] = 0xFFFFFFFFu / pr[i];
        }

        /* Referencia em CPU por divisao inteira — independente da tecnica GM. */
        unsigned n_ref = 0;
        for (unsigned c = 0; c < n_cand; c++) {
            unsigned v = base + c;
            if ((v & 1u) == 0u) continue;
            int primo = 1;
            for (unsigned i = 0; i < k; i++) {
                unsigned p = pr[i];
                if ((unsigned long long)p * p > v) break;
                if (v % p == 0u) { primo = 0; break; }
            }
            n_ref += (unsigned)primo;
        }

        unsigned char* h_m = (unsigned char*)malloc(n_cand);
        cl_mem d_inv = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (size_t)k * 4, h_inv, &err); CHECK(err, "d_inv");
        cl_mem d_lim = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (size_t)k * 4, h_lim, &err); CHECK(err, "d_lim");
        cl_mem d_m = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n_cand, NULL, &err);
        CHECK(err, "d_m");

        size_t g = ((n_cand + BLOCO - 1) / BLOCO) * (size_t)BLOCO, l = BLOCO;
        double loads = 2.0 * (double)n_cand * (double)k;

        for (int kk = 0; kk < 3; kk++) {
            double med[3], mean[3], sd[3];
            int ok[3];
            for (int v = 0; v < 3; v++) {
                cl_kernel ker = clCreateKernel(prog, fn[kk][v], &err); CHECK(err, fn[kk][v]);
                clSetKernelArg(ker, 0, sizeof(cl_mem), &d_inv);
                clSetKernelArg(ker, 1, sizeof(cl_mem), &d_lim);
                clSetKernelArg(ker, 2, sizeof(cl_uint), &k);
                clSetKernelArg(ker, 3, sizeof(cl_mem), &d_m);
                clSetKernelArg(ker, 4, sizeof(cl_uint), &base);
                clSetKernelArg(ker, 5, sizeof(cl_uint), &n_cand);

                for (int w = 0; w < WARMUP; w++) {
                    err = clEnqueueNDRangeKernel(q, ker, 1, NULL, &g, &l, 0, NULL, NULL);
                    CHECK(err, "warmup");
                }
                clFinish(q);

                double t[RUNS];
                for (int rr = 0; rr < RUNS; rr++) {
                    cl_event ev;
                    err = clEnqueueNDRangeKernel(q, ker, 1, NULL, &g, &l, 0, NULL, &ev);
                    CHECK(err, "enqueue");
                    clWaitForEvents(1, &ev);
                    t[rr] = ms_of(ev);
                    clReleaseEvent(ev);
                }
                estat(t, RUNS, &mean[v], &sd[v], &med[v]);

                clEnqueueReadBuffer(q, d_m, CL_TRUE, 0, n_cand, h_m, 0, NULL, NULL);
                unsigned c = 0;
                for (unsigned i = 0; i < n_cand; i++) if (!h_m[i]) c++;
                ok[v] = (c == n_ref);
                clReleaseKernel(ker);
            }

            /* Linhas de cache distintas tocadas por uma wavefront por load:
               no modo uniforme sempre 1; no disperso, min(64, linhas da tabela). */
            double linhas_tab = ceil((double)k * 4.0 / 64.0);
            double linhas = (kk == 0) ? 1.0 : (linhas_tab < 64.0 ? linhas_tab : 64.0);
            double req = (loads / 64.0) * linhas;
            double req_ciclo_cu = req / (med[0] * 1e-3) / (36.0 * 1.35e9);

            printf(" %-10s %6u %8.1f %10u %10.3f %10.3f %10.3f %8.3fx %8.3fx%s\n",
                   rotulo[kk], k, ws_kb, n_cand, med[0], med[1], med[2],
                   med[1] / med[0], med[2] / med[0],
                   (ok[0] && ok[1] && ok[2]) ? "" : "  <DIVERGIU>");

            if (csv)
                for (int v = 0; v < 3; v++)
                    fprintf(csv, "%s,%u,%.2f,%u,%u,%u,%.0f,\"%s\",%.6f,%.6f,%.6f,%.6f,"
                                 "%.0f,%.4f,%d\n",
                            rotulo[kk], k, ws_kb, lim, base, n_cand, loads, NIVEL[v],
                            mean[v], sd[v], med[v], med[v] / med[0],
                            linhas, req_ciclo_cu, ok[v]);
        }
        printf("\n");

        clReleaseMemObject(d_inv); clReleaseMemObject(d_lim); clReleaseMemObject(d_m);
        free(h_inv); free(h_lim); free(h_m);
    }

    printf("================================================================\n");
    printf(" L1 do gfx803 = 16 KB por CU. A previsao do Experimento 7 e de que\n");
    printf(" a razao L2/L1 colapse para ~1,0 quando o working set a ultrapassa.\n");
    printf("================================================================\n");
    if (csv) { fclose(csv); printf("\nCSV: resultados/tdiv_ws.csv\n"); }

    free(comp); free(pr);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(plats); free(src);
    return 0;
}
