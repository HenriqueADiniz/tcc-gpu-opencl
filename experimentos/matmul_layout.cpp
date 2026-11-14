/*
 * =============================================================================
 *  matmul_layout.cpp — efeito do layout de B na multiplicação de matrizes
 * =============================================================================
 *
 *  Compara C = A*B com B em ordem normal (percorrida por coluna) e com B
 *  transposta no anfitrião (percorrida por linha). O produto é o mesmo, os
 *  FLOPs são os mesmos e os dados são os mesmos --- muda apenas o padrão de
 *  acesso do segundo operando.
 *
 *  Transpor B é a otimização canônica de matmul em CPU. A hipótese aqui é que
 *  na GPU ela não compensa: ganha-se contiguidade dentro de cada lane e
 *  perde-se coalescência entre lanes, e o Experimento de localidade mostrou que
 *  o acesso espalhado sustenta ~185 GB/s contra ~650 GB/s do coalescido.
 *
 *  Validação amostrada, como nos demais programas da varredura: a referência
 *  completa é O(N^3) e dominaria o tempo total.
 *
 *  Compilar:
 *    source setup/env_rocm.sh
 *    g++ -Wall -Wextra -O2 experimentos/matmul_layout.cpp -o output/matmul_layout -lOpenCL
 *
 *  Uso:  ./output/matmul_layout ["N1 N2 ..."] [ORCAMENTO_MS] [LINHAS_POR_LOTE]
 * =============================================================================
 */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <CL/cl.h>

#define N_AMOSTRA 4096
#define RUNS_MIN  5
#define RUNS_MAX  100

static char* ler(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "nao abriu '%s' (rode a partir da raiz do projeto)\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc(n + 1); size_t k = fread(b, 1, n, f); b[k] = 0; fclose(f); return b;
}
static int cmp_d(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y);
}
static double ms_ev(cl_event e) {
    cl_ulong a, b;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(a), &a, NULL);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(b), &b, NULL);
    return (double)(b - a) / 1e6;
}
/* Uma medição completa, em um ou vários lotes de linhas. */
static double executa(cl_command_queue fila, cl_kernel kern, int N, size_t* ls, int lote)
{
    double total = 0.0;
    for (int r0 = 0; r0 < N; r0 += lote) {
        size_t linhas = (size_t)((r0 + lote <= N) ? lote : (N - r0));
        size_t off[2] = { 0, (size_t)r0 }, gs[2] = { (size_t)N, linhas };
        cl_event ev;
        cl_int rc = clEnqueueNDRangeKernel(fila, kern, 2, off, gs, ls, 0, NULL, &ev);
        if (rc != CL_SUCCESS) { fprintf(stderr, "NDRange falhou (%d)\n", rc); exit(1); }
        clWaitForEvents(1, &ev);
        total += ms_ev(ev);
        clReleaseEvent(ev);
    }
    return total;
}

int main(int argc, char** argv)
{
    const char* lista = (argc > 1) ? argv[1] : "1024 2048 4096";
    const double ORC = (argc > 2) ? atof(argv[2]) : 4000.0;
    const int LOTE_ARG = (argc > 3) ? atoi(argv[3]) : 0;
    int Ns[16], nN = 0;
    { char* c = strdup(lista); for (char* t = strtok(c, " ,"); t && nN < 16; t = strtok(NULL, " ,")) Ns[nN++] = atoi(t); free(c); }

    cl_uint np = 0; clGetPlatformIDs(0, NULL, &np);
    cl_platform_id* pl = (cl_platform_id*)malloc(np * sizeof(*pl)); clGetPlatformIDs(np, pl, NULL);
    cl_device_id dev = NULL;
    for (cl_uint i = 0; i < np; i++) if (clGetDeviceIDs(pl[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) break;
    if (!dev) { fprintf(stderr, "nenhuma GPU encontrada\n"); return 1; }
    char nome[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(nome), nome, NULL);

    cl_int e;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    cl_command_queue fila = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
    char* fonte = ler("experimentos/matmul_layout.cl");
    const char* fontes[] = { fonte };
    cl_program prog = clCreateProgramWithSource(ctx, 1, fontes, NULL, &e);
    if (clBuildProgram(prog, 1, &dev, "", NULL, NULL) != CL_SUCCESS) {
        size_t ls = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ls);
        char* log = (char*)malloc(ls + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, log, NULL);
        log[ls] = 0; fprintf(stderr, "--- build log ---\n%s\n", log); return 1;
    }

    const char* KN[6]  = { "padrao_noflag", "padrao_glc", "padrao_glc_slc",
                           "transp_noflag", "transp_glc", "transp_glc_slc" };
    const char* ROT[6] = { "B normal  GLC=0", "B normal  glc", "B normal  glc slc",
                           "B transp  GLC=0", "B transp  glc", "B transp  glc slc" };
    const char* LAY[6] = { "normal","normal","normal","transposta","transposta","transposta" };
    const char* FL[3]  = { "GLC=0", "glc", "glc slc" };

    printf("=== Layout de B na multiplicação de matrizes — %s ===\n\n", nome);
    int novo = (access("resultados/matmul_layout.csv", F_OK) != 0);
    FILE* csv = fopen("resultados/matmul_layout.csv", "a");
    if (csv && novo) fprintf(csv, "N,layout,flags,runs,media_ms,mediana_ms,cv_pct,gflops,razao_sobre_noflag,validacao\n");

    for (int a = 0; a < nN; a++) {
        const int N = Ns[a];
        const size_t tam = (size_t)N * N * sizeof(float);
        const double gflop = 2.0 * (double)N * N * N / 1e9;
        int LOTE = (LOTE_ARG > 0 && LOTE_ARG < N) ? (LOTE_ARG / 16) * 16 : N;
        if (LOTE < 16) LOTE = 16;

        printf("--- N = %d  (%.2f GFLOP, %.1f MB por matriz)", N, gflop, (double)tam / 1048576.0);
        if (LOTE < N) printf("  [lotes de %d linhas]", LOTE);
        printf(" ---\n"); fflush(stdout);

        float* hA  = (float*)malloc(tam);
        float* hB  = (float*)malloc(tam);
        float* hBt = (float*)malloc(tam);
        float* hC  = (float*)malloc(tam);
        if (!hA || !hB || !hBt || !hC) { fprintf(stderr, "  sem memória, pulando\n"); continue; }
        for (size_t i = 0; i < (size_t)N * N; i++) {
            hA[i] = (float)(i % 251) * 0.125f + 1.0f;
            hB[i] = (float)(i % 173) * 0.0625f + 1.0f;
        }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) hBt[(size_t)j*N+i] = hB[(size_t)i*N+j];

        /* referência amostrada: C[i][j] = soma_k A[i][k]*B[k][j] */
        int*   idx = (int*)  malloc(N_AMOSTRA * sizeof(int));
        float* ref = (float*)malloc(N_AMOSTRA * sizeof(float));
        srand(2027);
        for (int t = 0; t < N_AMOSTRA; t++) {
            long r1 = ((long)rand() << 15) ^ (long)rand();
            idx[t] = (int)(((unsigned long)r1) % ((unsigned long)N * N));
        }
        for (int t = 0; t < N_AMOSTRA; t++) {
            int i = idx[t] / N, j = idx[t] % N;
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += hA[(size_t)i*N+k] * hB[(size_t)k*N+j];
            ref[t] = s;
        }

        cl_mem dA  = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tam, hA,  &e);
        cl_mem dB  = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tam, hB,  &e);
        cl_mem dBt = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tam, hBt, &e);
        cl_mem dC  = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, tam, NULL, &e);
        size_t ls[2] = { 16, 16 };
        double base[2] = { 0.0, 0.0 };

        for (int k = 0; k < 6; k++) {
            cl_kernel kern = clCreateKernel(prog, KN[k], &e);
            cl_mem segundo = (k < 3) ? dB : dBt;
            clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &segundo);
            clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
            cl_int n = N; clSetKernelArg(kern, 3, sizeof(cl_int), &n);

            double t_aq = executa(fila, kern, N, ls, LOTE);
            int R = (int)(ORC / t_aq); if (R < RUNS_MIN) R = RUNS_MIN; if (R > RUNS_MAX) R = RUNS_MAX;
            double* t = (double*)malloc((size_t)R * sizeof(double));
            for (int r = 0; r < R; r++) t[r] = executa(fila, kern, N, ls, LOTE);

            double soma = 0.0; for (int r = 0; r < R; r++) soma += t[r];
            double media = soma / R, sq = 0.0;
            for (int r = 0; r < R; r++) sq += (t[r]-media)*(t[r]-media);
            double dp = (R > 1) ? sqrt(sq/(R-1)) : 0.0;
            qsort(t, (size_t)R, sizeof(double), cmp_d);
            double med = (R % 2) ? t[R/2] : 0.5*(t[R/2-1]+t[R/2]);

            clEnqueueReadBuffer(fila, dC, CL_TRUE, 0, tam, hC, 0, NULL, NULL);
            int erros = 0;
            const float tol = 0.02f * ((float)N / 1024.0f);
            for (int t2 = 0; t2 < N_AMOSTRA; t2++)
                if (fabsf(hC[idx[t2]] - ref[t2]) > tol * fabsf(ref[t2]) + tol) erros++;

            if (k % 3 == 0) base[k / 3] = med;
            double raz = med / base[k / 3];
            printf("  %-18s %10.3f ms  cv %5.2f%%  %7.1f GFLOPS  %6.3fx  n=%-4d %s\n",
                   ROT[k], med, 100.0*dp/media, gflop / (med/1000.0), raz, R,
                   erros ? "FALHOU" : "OK");
            fflush(stdout);
            if (csv) {
                fprintf(csv, "%d,%s,%s,%d,%.4f,%.4f,%.4f,%.3f,%.4f,%s\n",
                        N, LAY[k], FL[k%3], R, media, med, 100.0*dp/media,
                        gflop/(med/1000.0), raz, erros ? "FALHOU" : "OK");
                fflush(csv);
            }
            free(t); clReleaseKernel(kern);
        }
        printf("\n");
        clReleaseMemObject(dA); clReleaseMemObject(dB);
        clReleaseMemObject(dBt); clReleaseMemObject(dC);
        free(hA); free(hB); free(hBt); free(hC); free(idx); free(ref);
    }
    if (csv) fclose(csv);
    printf("Resultados: resultados/matmul_layout.csv\n");
    return 0;
}
