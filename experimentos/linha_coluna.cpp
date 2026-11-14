/*
 * =============================================================================
 *  linha_coluna.cpp — localidade de acesso isolada dentro de um mesmo kernel
 * =============================================================================
 *
 *  Pergunta: quanto o custo de contornar o L1 depende da localidade do padrão
 *  de acesso? Os experimentos anteriores respondiam trocando de algoritmo, o
 *  que muda muitas variáveis ao mesmo tempo. Aqui os dois padrões saem do
 *  próprio matmul e diferem apenas na direção do percurso (ver o .cl).
 *
 *  Desenho 2 x 3: {linha, coluna} x {GLC=0, glc, glc slc}. As seis variantes
 *  são construídas do mesmo modo, em inline assembly, de modo que a razão entre
 *  flags não carregue custo de construção.
 *
 *  Validação O(N^2): a saída do padrão linha é a soma da linha, repetida em
 *  toda a linha; a do padrão coluna é a soma da coluna. Basta N somas de N
 *  termos para conferir as N^2 posições.
 *
 *  Compilar:
 *    source setup/env_rocm.sh
 *    g++ -Wall -Wextra -O2 experimentos/linha_coluna.cpp -o output/linha_coluna -lOpenCL
 *
 *  Uso:  ./output/linha_coluna ["N1 N2 ..."] [ORCAMENTO_MS] [PADRAO] [CSV]
 *        PADRAO: linha | dispersa | coluna | todos (padrão: todos)
 *        CSV:    arquivo de saída (padrão: resultados/linha_coluna.csv). Uma
 *                remedição deve ir para arquivo próprio, para não misturar
 *                campanhas no mesmo CSV.
 *
 *  Cada medição é UM lançamento da NDRange inteira, sem fatiamento. Em
 *  N=16384 o kernel mais lento leva ~218 s por lançamento e terminou sem
 *  reset da GPU (28/09/2026).
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

#define RUNS_MIN 5
#define RUNS_MAX 200

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

int main(int argc, char** argv)
{
    const char* lista = (argc > 1) ? argv[1] : "1024 2048 4096";
    const double ORC = (argc > 2) ? atof(argv[2]) : 3000.0;
    const char* FILTRO = (argc > 3) ? argv[3] : "todos";
    const char* SAIDA  = (argc > 4) ? argv[4] : "resultados/linha_coluna.csv";
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
    char* fonte = ler("experimentos/linha_coluna.cl");
    const char* fontes[] = { fonte };
    cl_program prog = clCreateProgramWithSource(ctx, 1, fontes, NULL, &e);
    if (clBuildProgram(prog, 1, &dev, "", NULL, NULL) != CL_SUCCESS) {
        size_t ls = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ls);
        char* log = (char*)malloc(ls + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, log, NULL);
        log[ls] = 0; fprintf(stderr, "--- build log ---\n%s\n", log); return 1;
    }

    const char* KN[9] = { "linha_noflag", "linha_glc", "linha_glc_slc",
                          "dispersa_noflag", "dispersa_glc", "dispersa_glc_slc",
                          "coluna_noflag", "coluna_glc", "coluna_glc_slc" };
    const char* ROT[9] = { "linha    GLC=0", "linha    glc", "linha    glc slc",
                           "dispersa GLC=0", "dispersa glc", "dispersa glc slc",
                           "coluna   GLC=0", "coluna   glc", "coluna   glc slc" };
    const char* PAD[9] = { "linha","linha","linha","dispersa","dispersa","dispersa",
                           "coluna","coluna","coluna" };

    printf("=== Localidade de acesso x flags de cache — %s ===\n\n", nome);
    int novo = (access(SAIDA, F_OK) != 0);
    FILE* csv = fopen(SAIDA, "a");
    if (csv && novo) fprintf(csv, "N,padrao,flags,runs,media_ms,mediana_ms,cv_pct,gbps,razao_sobre_noflag,validacao\n");

    for (int a = 0; a < nN; a++) {
        const int N = Ns[a];
        const size_t tam = (size_t)N * N * sizeof(float);
        printf("--- N = %d  (matriz de %.1f MB; linha de %.1f KB, coluna toca %d linhas de cache) ---\n",
               N, (double)tam / 1048576.0, N * 4 / 1024.0, N);

        float* hA = (float*)malloc(tam);
        float* hC = (float*)malloc(tam);
        for (size_t i = 0; i < (size_t)N * N; i++) hA[i] = (float)(i % 251) * 0.125f + 1.0f;

        /* referências O(N^2): soma de cada linha e de cada coluna */
        double* sl = (double*)calloc(N, sizeof(double));
        double* sc = (double*)calloc(N, sizeof(double));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) { sl[i] += hA[(size_t)i*N+j]; sc[j] += hA[(size_t)i*N+j]; }

        cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tam, hA, &e);
        cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, tam, NULL, &e);
        size_t gs[2] = { (size_t)N, (size_t)N }, ls[2] = { 16, 16 };
        double base[3] = { 0.0, 0.0, 0.0 };

        for (int k = 0; k < 9; k++) {
            if (strcmp(FILTRO, "todos") != 0 && strcmp(FILTRO, PAD[k]) != 0) continue;
            cl_kernel kern = clCreateKernel(prog, KN[k], &e);
            clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &dC);
            cl_int n = N; clSetKernelArg(kern, 2, sizeof(cl_int), &n);

            cl_event ev;
            clEnqueueNDRangeKernel(fila, kern, 2, NULL, gs, ls, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            double t_aq = ms_ev(ev); clReleaseEvent(ev);
            int R = (int)(ORC / t_aq); if (R < RUNS_MIN) R = RUNS_MIN; if (R > RUNS_MAX) R = RUNS_MAX;

            double* t = (double*)malloc((size_t)R * sizeof(double));
            for (int r = 0; r < R; r++) {
                clEnqueueNDRangeKernel(fila, kern, 2, NULL, gs, ls, 0, NULL, &ev);
                clWaitForEvents(1, &ev); t[r] = ms_ev(ev); clReleaseEvent(ev);
            }
            double soma = 0.0; for (int r = 0; r < R; r++) soma += t[r];
            double media = soma / R, sq = 0.0;
            for (int r = 0; r < R; r++) sq += (t[r]-media)*(t[r]-media);
            double dp = (R > 1) ? sqrt(sq/(R-1)) : 0.0;
            qsort(t, (size_t)R, sizeof(double), cmp_d);
            double med = (R % 2) ? t[R/2] : 0.5*(t[R/2-1]+t[R/2]);

            clEnqueueReadBuffer(fila, dC, CL_TRUE, 0, tam, hC, 0, NULL, NULL);
            int erros = 0;
            const double tol = 0.02 * ((double)N / 1024.0);
            for (int i = 0; i < N && erros == 0; i++)
                for (int j = 0; j < N; j++) {
                    double esp = (k < 3) ? sl[i] : ((k < 6) ? sl[(i+j) % N] : sc[j]);
                    if (fabs((double)hC[(size_t)i*N+j] - esp) > tol * fabs(esp) + tol) { erros++; break; }
                }

            if (k % 3 == 0) base[k / 3] = med;
            double raz = med / base[k / 3];
            double gbps = ((double)N*N*N*4.0) / (med/1000.0) / 1e9;

            printf("  %-17s %9.3f ms  cv %5.2f%%  %7.1f GB/s  %6.3fx  n=%-4d %s\n",
                   ROT[k], med, 100.0*dp/media, gbps, raz, R, erros ? "FALHOU" : "OK");
            fflush(stdout);
            if (csv) {
                fprintf(csv, "%d,%s,%s,%d,%.4f,%.4f,%.4f,%.2f,%.4f,%s\n",
                        N, PAD[k],
                        (k%3==0)?"GLC=0":((k%3==1)?"glc":"glc slc"),
                        R, media, med, 100.0*dp/media, gbps, raz, erros?"FALHOU":"OK");
                fflush(csv);
            }
            free(t); clReleaseKernel(kern);
        }
        printf("\n");
        clReleaseMemObject(dA); clReleaseMemObject(dC);
        free(hA); free(hC); free(sl); free(sc);
    }
    if (csv) fclose(csv);
    printf("Resultados: %s\n", SAIDA);
    return 0;
}
