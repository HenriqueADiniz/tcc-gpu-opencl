/*
 * =============================================================================
 *  matmul_cpu.cpp — referência em CPU para a multiplicação de matrizes
 * =============================================================================
 *
 *  Mede na CPU o MESMO algoritmo dos cenários de GPU: laço triplo ingênuo, sem
 *  blocagem de cache e sem vetorização manual. A intenção é comparar
 *  DISPOSITIVOS com a implementação mantida constante --- e não comparar a
 *  melhor GEMM de CPU com a melhor de GPU, que seria outra pergunta.
 *
 *  Usa TODAS as threads disponíveis. Medir uma única thread contra 36 Compute
 *  Units seria um desenho viciado: a GPU é uma máquina paralela, e a referência
 *  justa é a CPU também operando em paralelo. O modo sequencial permanece
 *  acessível (terceiro argumento) apenas como diagnóstico de escalabilidade.
 *
 *  Validação amostrada: a verificação completa seria O(N^3) e, em N=16384,
 *  custaria horas --- mais que a própria medição. Confere-se uma amostra
 *  aleatória de posições recalculadas sequencialmente, o que detecta as falhas
 *  em questão, que não são pontuais.
 *
 *  Repetições adaptativas: fixa-se um orçamento de tempo por tamanho e o número
 *  de execuções é derivado da duração medida no aquecimento. Sem isso a
 *  varredura é inviável --- o laço é O(N^3) e uma execução em N=16384 leva
 *  cerca de 19 minutos.
 *
 *  Compilar:
 *    g++ -Wall -Wextra -O2 -march=native -fopenmp \
 *        experimentos/matmul_cpu.cpp -o output/matmul_cpu
 *
 *  Uso:  ./output/matmul_cpu ["N1 N2 ..."] [ORCAMENTO_S] [SEQUENCIAL=0|1]
 * =============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define N_AMOSTRA 4096
#define RUNS_MIN  3
#define RUNS_MAX  25

static double agora_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static void matmul(const float* A, const float* B, float* C, int N, int paralelo)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(paralelo)
#else
    (void)paralelo;
#endif
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float soma = 0.0f;
            for (int k = 0; k < N; k++) soma += A[i * N + k] * B[k * N + j];
            C[i * N + j] = soma;
        }
    }
}

/* Recalcula sequencialmente apenas as posicoes sorteadas. */
static void referencia_amostra(const float* A, const float* B, int N,
                               const int* idx, float* saida, int n)
{
    for (int t = 0; t < n; t++) {
        int i = idx[t] / N, j = idx[t] % N;
        float s = 0.0f;
        for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
        saida[t] = s;
    }
}

static void hms(double s, char* buf, size_t n) {
    int h = (int)(s / 3600), m = (int)((s - 3600 * h) / 60), q = (int)(s - 3600 * h - 60 * m);
    if (h) snprintf(buf, n, "%dh%02dm%02ds", h, m, q);
    else if (m) snprintf(buf, n, "%dm%02ds", m, q);
    else snprintf(buf, n, "%.1fs", s);
}

int main(int argc, char** argv)
{
    const char* lista = (argc > 1) ? argv[1] : "1024 2048 4096 8192 16384";
    const double ORC_S = (argc > 2) ? atof(argv[2]) : 60.0;
    const int SEQ = (argc > 3) ? atoi(argv[3]) : 0;

    int Ns[32], nN = 0;
    { char* c = strdup(lista); for (char* t = strtok(c, " ,"); t && nN < 32; t = strtok(NULL, " ,")) Ns[nN++] = atoi(t); free(c); }

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    printf("################################################################\n");
    printf("#  Referência em CPU — multiplicação de matrizes                #\n");
    printf("#  Mesmo algoritmo dos kernels de GPU: laço triplo ingênuo      #\n");
    printf("################################################################\n");
    printf("Threads: %d | Orçamento por tamanho: %.0f s | Tamanhos:", nthreads, ORC_S);
    for (int i = 0; i < nN; i++) printf(" %d", Ns[i]);
    printf("\n\n");
    fflush(stdout);

    int novo = (access("resultados/matmul_cpu_sweep.csv", F_OK) != 0);
    FILE* csv = fopen("resultados/matmul_cpu_sweep.csv", "a");
    if (csv && novo) fprintf(csv, "N,threads,runs,media_ms,mediana_ms,dp_ms,cv_pct,gflops,validacao\n");

    double t_ini = agora_s();
    for (int a = 0; a < nN; a++) {
        const int N = Ns[a];
        const size_t tam = (size_t)N * N * sizeof(float);
        const double gflop = 2.0 * (double)N * N * N / 1e9;

        printf("=== [%d/%d]  N = %d  (%.2f GFLOP, %.2f GB em 3 matrizes) ===\n",
               a + 1, nN, N, gflop, 3.0 * (double)tam / (1 << 30));
        printf("    alocando e inicializando...\n"); fflush(stdout);

        float* A = (float*)malloc(tam);
        float* B = (float*)malloc(tam);
        float* C = (float*)malloc(tam);
        if (!A || !B || !C) { fprintf(stderr, "    sem memória para N=%d, pulando\n", N); free(A); free(B); free(C); continue; }
        for (size_t i = 0; i < (size_t)N * N; i++) {
            A[i] = (float)(i % 1000) * 0.5f + 1.0f;
            B[i] = (float)(i % 777)  * 0.25f + 1.0f;
        }

        printf("    aquecimento...\n"); fflush(stdout);
        double t0 = agora_s();
        matmul(A, B, C, N, !SEQ);
        double t_aq = agora_s() - t0;

        int R = (int)(ORC_S / t_aq);
        if (R < RUNS_MIN) R = RUNS_MIN;
        if (R > RUNS_MAX) R = RUNS_MAX;
        char b1[32], b2[32]; hms(t_aq, b1, sizeof b1); hms(t_aq * R, b2, sizeof b2);
        printf("    %s por execução -> %d execuções (~%s)\n", b1, R, b2); fflush(stdout);

        double* t = (double*)malloc((size_t)R * sizeof(double));
        for (int r = 0; r < R; r++) {
            double u0 = agora_s();
            matmul(A, B, C, N, !SEQ);
            t[r] = agora_s() - u0;
            char b3[32]; hms(t[r], b3, sizeof b3);
            printf("      execução %d/%d: %s\r", r + 1, R, b3); fflush(stdout);
        }
        printf("                                             \r");

        double soma = 0.0;
        for (int r = 0; r < R; r++) soma += t[r];
        double media = soma / R, sq = 0.0;
        for (int r = 0; r < R; r++) sq += (t[r] - media) * (t[r] - media);
        double dp = (R > 1) ? sqrt(sq / (R - 1)) : 0.0;
        qsort(t, (size_t)R, sizeof(double), cmp_double);
        double med = (R % 2) ? t[R/2] : 0.5 * (t[R/2 - 1] + t[R/2]);

        printf("    validando %d posições amostradas...\n", N_AMOSTRA); fflush(stdout);
        int* idx = (int*)malloc(N_AMOSTRA * sizeof(int));
        float* ref = (float*)malloc(N_AMOSTRA * sizeof(float));
        srand(12345);
        for (int i = 0; i < N_AMOSTRA; i++) {
            long r1 = ((long)rand() << 15) ^ (long)rand();
            idx[i] = (int)(((unsigned long)r1) % ((unsigned long)N * N));
        }
        referencia_amostra(A, B, N, idx, ref, N_AMOSTRA);
        const float tol = 0.01f * ((float)N / 1024.0f);
        int erros = 0;
        for (int i = 0; i < N_AMOSTRA; i++) if (fabsf(C[idx[i]] - ref[i]) > tol) erros++;

        printf("    ------------------------------------------------------------\n");
        printf("    N=%-6d  mediana %10.1f ms   média %10.1f ms   cv %5.2f%%\n",
               N, med * 1000.0, media * 1000.0, 100.0 * dp / media);
        printf("               %10.2f GFLOPS   n=%d   validação: %s\n",
               gflop / med, R, erros ? "FALHOU" : "OK");
        printf("    ------------------------------------------------------------\n\n");
        fflush(stdout);

        if (csv) {
            fprintf(csv, "%d,%d,%d,%.3f,%.3f,%.3f,%.4f,%.4f,%s\n",
                    N, SEQ ? 1 : nthreads, R, media * 1000.0, med * 1000.0,
                    dp * 1000.0, 100.0 * dp / media, gflop / med, erros ? "FALHOU" : "OK");
            fflush(csv);
        }
        free(idx); free(ref); free(t); free(A); free(B); free(C);
    }
    char bt[32]; hms(agora_s() - t_ini, bt, sizeof bt);
    printf("################ CONCLUÍDO em %s ################\n", bt);
    printf("Resultados: resultados/matmul_cpu_sweep.csv\n");
    if (csv) fclose(csv);
    return 0;
}
