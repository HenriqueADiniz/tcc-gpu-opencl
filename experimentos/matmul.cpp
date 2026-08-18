/*
 * =============================================================================
 *  AMD GPU Cache Bypass — Multiplicacao de Matrizes — 3 Cenarios
 * =============================================================================
 *
 *  Cenario 1: Acesso normal via L1 -> L2 -> VRAM  (padrao)
 *  Cenario 2: Bypass L1, usar L2 apenas            (volatile = GLC=1)
 *  Cenario 3: Bypass L1 + L2, direto a VRAM        (volatile = GLC=1, SLC via ISA)
 *
 *  Na multiplicacao de matrizes, cada elemento de A e B e reutilizado N vezes.
 *  Isso torna o cache MUITO relevante — sem cache, cada acesso vai a VRAM,
 *  e a diferenca de performance entre os cenarios fica evidente.
 *
 *  Compilar (MinGW):
 *    g++ -Wall -Wextra -O2 -D__USE_MINGW_ANSI_STDIO=1
 *        -Wl,--enable-stdcall-fixup
 *        -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *        matmul.cpp -o output\matmul.exe "C:\Windows\System32\OpenCL.dll"
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

/* ------------------------------------------------------------------------- */
/*  3 Kernels de multiplicacao de matrizes C = A * B                         */
/*  Cada um com nivel diferente de cache                                     */
/*  Matrizes quadradas NxN, row-major                                        */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
"/* ====================================================================== */\n"
"/* Cenario 1: MatMul NORMAL — usa L1 -> L2 -> VRAM (padrao)               */\n"
"/* Cada work-item calcula um elemento C[row][col]                          */\n"
"/* Os acessos a A e B passam pelo cache L1 normalmente                     */\n"
"/* ISA: buffer_load_dword (GLC=0, SLC=0)                                   */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_via_L1(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Cenario 2: MatMul com volatile — HIPOTESE: bypass L1 via GLC=1          */\n"
"/* RESULTADO: volatile NAO gera GLC=1 no GFX8 (encoding identico ao C1)   */\n"
"/* Efeito real: inibe reuso de registradores (11 vs 10 VGPRs)             */\n"
"/* ISA: buffer_load_dword/flat_load_dword com GLC=0 (sem bypass)          */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_via_L2(\n"
"    __global volatile const float* A,\n"
"    __global volatile const float* B,\n"
"    __global volatile float*       C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Cenario 3: MatMul com volatile (identico ao C2 na pratica)              */\n"
"/* RESULTADO: ISA identica ao Cenario 1 e 2 no GFX8                       */\n"
"/* volatile atua no modelo de memória (impede otimizacoes do compilador)  */\n"
"/* mas NAO emite flags de cache — caminho de memoria identico             */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_direto_VRAM(\n"
"    __global volatile const float* A,\n"
"    __global volatile const float* B,\n"
"    __global volatile float*       C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* BONUS: MatMul com LOCAL MEMORY (tiled) — melhor uso de cache            */\n"
"/* Usa __local (LDS/shared memory na GPU) para tiles de 16x16              */\n"
"/* Isso reduz acessos a global memory de N para N/TILE_SIZE por tile       */\n"
"/* ====================================================================== */\n"
"#define TILE_SIZE 16\n"
"__kernel void matmul_tiled(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    int localRow = get_local_id(1);\n"
"    int localCol = get_local_id(0);\n"
"\n"
"    __local float tileA[TILE_SIZE][TILE_SIZE];\n"
"    __local float tileB[TILE_SIZE][TILE_SIZE];\n"
"\n"
"    float sum = 0.0f;\n"
"    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;\n"
"\n"
"    for (int t = 0; t < numTiles; t++) {\n"
"        int tiledCol = t * TILE_SIZE + localCol;\n"
"        int tiledRow = t * TILE_SIZE + localRow;\n"
"\n"
"        if (row < N && tiledCol < N)\n"
"            tileA[localRow][localCol] = A[row * N + tiledCol];\n"
"        else\n"
"            tileA[localRow][localCol] = 0.0f;\n"
"\n"
"        if (tiledRow < N && col < N)\n"
"            tileB[localRow][localCol] = B[tiledRow * N + col];\n"
"        else\n"
"            tileB[localRow][localCol] = 0.0f;\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"        for (int k = 0; k < TILE_SIZE; k++) {\n"
"            sum += tileA[localRow][k] * tileB[k][localCol];\n"
"        }\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    if (row < N && col < N) {\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n";


/* ------------------------------------------------------------------------- */
/*  Utilitarios                                                              */
/* ------------------------------------------------------------------------- */

#define CHECK_CL(err, msg) do {                                         \
    if ((err) != CL_SUCCESS) {                                          \
        fprintf(stderr, "[ERRO OpenCL] %s (codigo: %d)\n", (msg), (err)); \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static double get_time_ms(cl_event ev) {
    cl_ulong t_start, t_end;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                           sizeof(cl_ulong), &t_start, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                           sizeof(cl_ulong), &t_end, NULL);
    return (double)(t_end - t_start) / 1e6;
}

/* Converte o nome do dispositivo num componente de nome de arquivo seguro:
   "Radeon RX 580 Series (POLARIS10...)" -> "Radeon_RX_580_Series_POLARIS10" */
static void slug_dispositivo(const char* nome, char* out, size_t out_sz) {
    size_t j = 0;
    int ultimo_foi_sep = 1;   /* evita '_' no inicio e repetidos */
    for (size_t i = 0; nome[i] && j + 1 < out_sz; i++) {
        unsigned char c = (unsigned char)nome[i];
        int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                 (c >= '0' && c <= '9') || c == '.' || c == '-';
        if (ok) {
            out[j++] = (char)c;
            ultimo_foi_sep = 0;
        } else if (!ultimo_foi_sep) {
            out[j++] = '_';
            ultimo_foi_sep = 1;
        }
    }
    while (j > 0 && out[j - 1] == '_') j--;   /* tira '_' final */
    if (j == 0) { const char* f = "desconhecido"; for (; *f && j + 1 < out_sz; f++) out[j++] = *f; }
    out[j] = '\0';
}

/*
 * Grava o binario ISA compilado pelo driver.
 *
 * O nome do arquivo inclui o dispositivo, e o destino e a pasta isa/. Sem isso,
 * rodar este programa em plataformas diferentes sobrescreve o dump anterior --
 * o binario do driver Windows (AMDOCL) e o do ROCm/Linux sao evidencias
 * distintas e precisam coexistir para a comparacao multiplataforma.
 */
static void dump_isa_info(cl_program program) {
    size_t bin_size = 0;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t), &bin_size, NULL);
    if (bin_size == 0) return;

    /* Nome do dispositivo para o qual este binario foi compilado */
    cl_device_id dev = NULL;
    char dev_nome[256] = "desconhecido";
    if (clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                         sizeof(cl_device_id), &dev, NULL) == CL_SUCCESS && dev) {
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dev_nome), dev_nome, NULL);
    }
    char slug[128];
    slug_dispositivo(dev_nome, slug, sizeof(slug));

    char caminho[320];
    snprintf(caminho, sizeof(caminho), "isa/matmul_%s.bin", slug);

    unsigned char* binary = (unsigned char*)malloc(bin_size);
    unsigned char* binaries[] = { binary };
    clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                     sizeof(unsigned char*), binaries, NULL);

    FILE* fp = fopen(caminho, "wb");
    if (!fp) {
        /* isa/ pode nao existir se o programa for executado de outra pasta */
        snprintf(caminho, sizeof(caminho), "matmul_%s.bin", slug);
        fp = fopen(caminho, "wb");
    }
    if (fp) {
        fwrite(binary, 1, bin_size, fp);
        fclose(fp);
        printf("[INFO] Binario ISA salvo em '%s' (%lu bytes)\n",
               caminho, (unsigned long)bin_size);
        printf("       Dispositivo: %s\n", dev_nome);
        printf("       Analise: llvm-objdump -d --triple=amdgcn-amd-amdhsa \\\n"
               "                  --mcpu=gfx803 %s   (Linux/ROCm)\n", caminho);
        printf("                clrxdisasm %s                       (binario do driver Windows)\n\n",
               caminho);
    } else {
        fprintf(stderr, "[AVISO] nao consegui gravar o binario ISA em '%s'\n", caminho);
    }
    free(binary);
}

/* Verificacao CPU simples (para matrizes pequenas) */
static void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


/* ------------------------------------------------------------------------- */
/*  Main                                                                     */
/* ------------------------------------------------------------------------- */

int main(void) {
    cl_int err;

    /*
     * Tamanho da matriz: N x N
     * 1024 = bom para ver diferenca de cache (~12MB de dados)
     * 2048 = mais pesado (~48MB), diferenca mais obvia
     * 512  = rapido para teste
     */
    const int N = 1024;
    const size_t mat_size = (size_t)N * N * sizeof(float);
    const double gflops_per_run = 2.0 * (double)N * N * N / 1e9;

    printf("=== AMD GPU Cache Bypass — Multiplicacao de Matrizes ===\n");
    printf("Matriz: %d x %d (%.1f MB por matriz, %.1f MB total)\n",
           N, N, (double)mat_size / (1024.0*1024.0),
           3.0 * (double)mat_size / (1024.0*1024.0));
    printf("FLOPs por execucao: %.2f GFLOP\n\n", gflops_per_run);

    /* -- Alocar e inicializar matrizes -- */
    float* h_A = (float*)malloc(mat_size);
    float* h_B = (float*)malloc(mat_size);
    float* h_C = (float*)malloc(mat_size);
    float* h_C_ref = (float*)malloc(mat_size);  /* referencia CPU */
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        return 1;
    }

    /* Valores pequenos para evitar overflow */
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i + 37) % 100) / 100.0f;
    }

    /* -- Setup OpenCL -- */
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    CHECK_CL(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL(err, "clGetDeviceIDs");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("GPU: %s\n", dev_name);

    char vendor[256];
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    printf("Vendor: %s\n\n", vendor);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(
        ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    /* -- Buffers GPU -- */
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 mat_size, h_A, &err);
    CHECK_CL(err, "clCreateBuffer A");

    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 mat_size, h_B, &err);
    CHECK_CL(err, "clCreateBuffer B");

    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 mat_size, NULL, &err);
    CHECK_CL(err, "clCreateBuffer C");

    /* -- Compilar -- */
    const char* sources[] = { kernel_source };
    cl_program program = clCreateProgramWithSource(ctx, 1, sources, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        exit(1);
    }

    dump_isa_info(program);

    /* -- Criar kernels -- */
    cl_kernel k_L1    = clCreateKernel(program, "matmul_via_L1",      &err);
    CHECK_CL(err, "clCreateKernel L1");
    cl_kernel k_L2    = clCreateKernel(program, "matmul_via_L2",      &err);
    CHECK_CL(err, "clCreateKernel L2");
    cl_kernel k_VRAM  = clCreateKernel(program, "matmul_direto_VRAM", &err);
    CHECK_CL(err, "clCreateKernel VRAM");
    cl_kernel k_tiled = clCreateKernel(program, "matmul_tiled",       &err);
    CHECK_CL(err, "clCreateKernel tiled");

    /* -- Configurar argumentos -- */
    cl_kernel kernels[] = { k_L1, k_L2, k_VRAM, k_tiled };
    const char* nomes[] = {
        "Cenario 1: via L1 (normal, GLC=0 SLC=0)",
        "Cenario 2: via L2 (volatile — GLC=0, sem bypass real)",
        "Cenario 3: direto VRAM (volatile — GLC=0, ISA identica)",
        "BONUS: Tiled com local memory (melhor aproveitamento de cache)"
    };
    const int NUM_KERNELS = 4;

    for (int k = 0; k < NUM_KERNELS; k++) {
        clSetKernelArg(kernels[k], 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(kernels[k], 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(kernels[k], 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(kernels[k], 3, sizeof(int),    &N);
    }

    /* -- Executar cada cenario -- */
    size_t global_size[2] = { (size_t)N, (size_t)N };
    size_t local_size_normal[2] = { 16, 16 };   /* 256 threads */
    size_t local_size_tiled[2]  = { 16, 16 };   /* TILE_SIZE = 16 */

    const int WARMUP = 2;
    const int RUNS   = 5;

    printf("Matriz: %dx%d | Workgroup: 16x16 (256 threads)\n", N, N);
    printf("Warmup: %d | Runs: %d\n", WARMUP, RUNS);
    printf("================================================================\n\n");

    for (int k = 0; k < NUM_KERNELS; k++) {
        printf("--- %s ---\n", nomes[k]);

        size_t* ls = (k == 3) ? local_size_tiled : local_size_normal;

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            cl_event ev;
            clEnqueueNDRangeKernel(queue, kernels[k], 2, NULL,
                                   global_size, ls, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }

        /* Medicoes */
        double total_ms = 0.0;
        double min_ms = 1e9, max_ms = 0.0;

        for (int r = 0; r < RUNS; r++) {
            cl_event ev;
            clEnqueueNDRangeKernel(queue, kernels[k], 2, NULL,
                                   global_size, ls, 0, NULL, &ev);
            clWaitForEvents(1, &ev);

            double ms = get_time_ms(ev);
            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;

            printf("  Run %d: %.3f ms\n", r + 1, ms);
            clReleaseEvent(ev);
        }

        double avg_ms = total_ms / RUNS;
        double gflops = gflops_per_run / (avg_ms / 1000.0);

        printf("  Media: %.3f ms | Min: %.3f ms | Max: %.3f ms\n",
               avg_ms, min_ms, max_ms);
        printf("  Performance: %.2f GFLOPS\n\n", gflops);
    }

    /* -- Verificacao: comparar ultimo resultado GPU com CPU -- */
    printf("--- Verificacao (CPU vs GPU) ---\n");
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_size, h_C, 0, NULL, NULL);
    matmul_cpu(h_A, h_B, h_C_ref, N);

    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
        /* Tolerancia maior para matmul (acumula erros de FP) */
        if (diff > 0.01f) {
            if (errors < 5) {
                printf("  [ERRO] C[%d] GPU=%.6f CPU=%.6f diff=%.6f\n",
                       i, h_C[i], h_C_ref[i], diff);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  OK! Max diferenca: %.8f\n", max_diff);
    } else {
        printf("  %d erros encontrados. Max diferenca: %.8f\n", errors, max_diff);
    }

    /* -- Cleanup -- */
    clReleaseKernel(k_L1);
    clReleaseKernel(k_L2);
    clReleaseKernel(k_VRAM);
    clReleaseKernel(k_tiled);
    clReleaseProgram(program);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    printf("\n=== Fim ===\n");
    return 0;
}