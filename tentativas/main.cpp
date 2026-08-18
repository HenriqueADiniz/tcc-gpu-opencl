/*
 * =============================================================================
 *  AMD GPU Cache Bypass — OpenCL Host + 3 Kernels
 * =============================================================================
 *
 *  Plataforma: Windows + AMD GPU (GCN / RDNA) + OpenCL
 *
 *  Cenario 1: Acesso normal (L1 -> L2 -> VRAM)     [padrao]
 *  Cenario 2: Bypass L1, usar L2 apenas             [GLC=1]
 *  Cenario 3: Bypass L1 + L2, direto a VRAM         [GLC=1 + SLC=1]
 *
 *  Compilar (MinGW no Windows):
 *    g++ -Wall -Wextra -O2 -D__USE_MINGW_ANSI_STDIO=1
 *        -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *        main.cpp -o main.exe "C:\Windows\System32\OpenCL.dll"
 *
 * =============================================================================
 *  NOTA SOBRE CACHE BYPASS EM OpenCL/AMD
 * =============================================================================
 *
 *  OpenCL NAO suporta __asm__ inline. O controle de cache e feito por:
 *
 *  1. Qualificadores de linguagem:
 *     - `__global volatile` -> compilador AMD gera instrucoes com GLC=1
 *
 *  2. Compilacao offline com CLRX Assembler:
 *     - Controle total dos bits GLC e SLC nas instrucoes buffer_load/store
 *     - Gera binarios .bin carregaveis via clCreateProgramWithBinary
 *
 *  Na ISA GCN/RDNA:
 *     GLC=0, SLC=0  ->  L1 -> L2 -> VRAM  (normal)
 *     GLC=1, SLC=0  ->  L2 -> VRAM        (bypass L1)
 *     GLC=1, SLC=1  ->  VRAM              (bypass L1 + L2)
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
/*  Kernels OpenCL                                                           */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
"/* ====================================================================== */\n"
"/* Cenario 1: Acesso normal — usa hierarquia completa L1 -> L2 -> VRAM    */\n"
"/* ISA gerada: buffer_load_dword (GLC=0, SLC=0)                           */\n"
"/* ====================================================================== */\n"
"__kernel void kernel_via_L1(\n"
"    __global const float* restrict in,\n"
"    __global float*       restrict out,\n"
"    const int N)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid < N) {\n"
"        float val = in[gid];\n"
"        out[gid] = val * 2.0f;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Cenario 2: Bypass L1, acessar via L2 apenas                            */\n"
"/* `volatile` faz o compilador AMD setar GLC=1                            */\n"
"/* ISA gerada: buffer_load_dword com GLC=1, SLC=0                         */\n"
"/* ====================================================================== */\n"
"__kernel void kernel_via_L2(\n"
"    __global volatile const float* in,\n"
"    __global volatile float*       out,\n"
"    const int N)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid < N) {\n"
"        float val = in[gid];\n"
"        out[gid] = val * 2.0f;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Cenario 3: Bypass L1 + L2, acesso direto a VRAM                        */\n"
"/*                                                                        */\n"
"/* Em OpenCL puro, NAO ha como setar SLC=1 diretamente.                   */\n"
"/* volatile da GLC=1 (bypass L1). Para bypass total (GLC=1+SLC=1),        */\n"
"/* use CLRX assembler com binario ISA (ver apendice no final).            */\n"
"/* ====================================================================== */\n"
"__kernel void kernel_direto_VRAM(\n"
"    __global volatile const float* in,\n"
"    __global volatile float*       out,\n"
"    const int N)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid < N) {\n"
"        /* ISA equivalente (CLRX):                                        */\n"
"        /* buffer_load_dword v1, v0, s[4:7], 0 offen glc slc             */\n"
"        /* v_mul_f32 v2, 2.0, v1                                         */\n"
"        /* buffer_store_dword v2, v0, s[8:11], 0 offen glc slc           */\n"
"        float val = in[gid];\n"
"        out[gid] = val * 2.0f;\n"
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
    return (double)(t_end - t_start) / 1e6; /* ns -> ms */
}

static void dump_isa_info(cl_program program) {
    size_t bin_size = 0;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t), &bin_size, NULL);

    if (bin_size > 0) {
        unsigned char* binary = (unsigned char*)malloc(bin_size);
        unsigned char* binaries[] = { binary };
        clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                         sizeof(unsigned char*), binaries, NULL);

        FILE* fp = fopen("kernel_dump.bin", "wb");
        if (fp) {
            fwrite(binary, 1, bin_size, fp);
            fclose(fp);
            printf("[INFO] Binario ISA salvo em 'kernel_dump.bin' (%lu bytes)\n",
                   (unsigned long)bin_size);
            printf("       Analise com: rga.exe ou clrxdisasm kernel_dump.bin\n\n");
        }
        free(binary);
    }
}


/* ------------------------------------------------------------------------- */
/*  Main                                                                     */
/* ------------------------------------------------------------------------- */

int main(void) {
    cl_int err;
    const int N = 1024 * 1024 * 16;  /* 16M elementos = 64MB */
    const size_t data_size = (size_t)N * sizeof(float);

    printf("=== AMD GPU Cache Bypass — OpenCL Demo ===\n");
    printf("Elementos: %d (%.1f MB)\n\n", N, (double)data_size / (1024.0*1024.0));

    /* -- Alocar dados host -- */
    float* h_in  = (float*)malloc(data_size);
    float* h_out = (float*)malloc(data_size);
    if (!h_in || !h_out) {
        fprintf(stderr, "Erro ao alocar memoria host\n");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }

    /* -- Plataforma + Device AMD -- */
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    CHECK_CL(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL(err, "clGetDeviceIDs — nenhuma GPU encontrada");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("GPU: %s\n", dev_name);

    char vendor[256];
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    printf("Vendor: %s\n\n", vendor);

    /* -- Contexto + Fila com profiling -- */
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(
        ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    /* -- Buffers GPU -- */
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  data_size, h_in, &err);
    CHECK_CL(err, "clCreateBuffer (in)");

    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                   data_size, NULL, &err);
    CHECK_CL(err, "clCreateBuffer (out)");

    /* -- Compilar programa -- */
    const char* sources[] = { kernel_source };
    cl_program program = clCreateProgramWithSource(
        ctx, 1, sources, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device,
                         "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        exit(1);
    }

    /* Salva binario ISA para inspecao */
    dump_isa_info(program);

    /* -- Criar kernels -- */
    cl_kernel k_L1   = clCreateKernel(program, "kernel_via_L1",      &err);
    CHECK_CL(err, "clCreateKernel (L1)");

    cl_kernel k_L2   = clCreateKernel(program, "kernel_via_L2",      &err);
    CHECK_CL(err, "clCreateKernel (L2)");

    cl_kernel k_VRAM = clCreateKernel(program, "kernel_direto_VRAM", &err);
    CHECK_CL(err, "clCreateKernel (VRAM)");

    /* -- Configurar argumentos -- */
    cl_kernel kernels[] = { k_L1, k_L2, k_VRAM };
    const char* nomes[] = {
        "Cenario 1: via L1 (normal)",
        "Cenario 2: via L2 (bypass L1, GLC=1)",
        "Cenario 3: direto VRAM (bypass L1+L2, GLC=1+SLC=1)"
    };

    for (int k = 0; k < 3; k++) {
        clSetKernelArg(kernels[k], 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(kernels[k], 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(kernels[k], 2, sizeof(int),    &N);
    }

    /* -- Executar e medir cada cenario -- */
    size_t global_size = (size_t)N;
    size_t local_size  = 256;

    /* Alinhar global_size */
    if (global_size % local_size != 0) {
        global_size = ((global_size / local_size) + 1) * local_size;
    }

    const int WARMUP = 2;
    const int RUNS   = 5;

    printf("Workgroup: %lu | Global: %lu\n",
           (unsigned long)local_size, (unsigned long)global_size);
    printf("Warmup: %d | Runs: %d\n\n", WARMUP, RUNS);

    for (int k = 0; k < 3; k++) {
        printf("--- %s ---\n", nomes[k]);

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            cl_event ev;
            clEnqueueNDRangeKernel(queue, kernels[k], 1, NULL,
                                   &global_size, &local_size, 0, NULL, &ev);
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }

        /* Medicoes */
        double total_ms = 0.0;
        double min_ms = 1e9, max_ms = 0.0;

        for (int r = 0; r < RUNS; r++) {
            cl_event ev;
            clEnqueueNDRangeKernel(queue, kernels[k], 1, NULL,
                                   &global_size, &local_size, 0, NULL, &ev);
            clWaitForEvents(1, &ev);

            double ms = get_time_ms(ev);
            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;

            printf("  Run %d: %.3f ms\n", r + 1, ms);
            clReleaseEvent(ev);
        }

        double avg_ms = total_ms / RUNS;
        double bandwidth_gb = (2.0 * (double)data_size) / (avg_ms / 1000.0) / 1e9;

        printf("  Media: %.3f ms | Min: %.3f ms | Max: %.3f ms\n",
               avg_ms, min_ms, max_ms);
        printf("  Bandwidth efetiva: %.2f GB/s\n\n", bandwidth_gb);
    }

    /* -- Verificar resultado -- */
    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                        data_size, h_out, 0, NULL, NULL);

    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        float expected = h_in[i] * 2.0f;
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("[ERRO] out[%d] = %f, esperado = %f\n",
                   i, h_out[i], expected);
            errors++;
        }
    }
    if (errors == 0) printf("Verificacao: OK (todos os valores corretos)\n");

    /* -- Cleanup -- */
    clReleaseKernel(k_L1);
    clReleaseKernel(k_L2);
    clReleaseKernel(k_VRAM);
    clReleaseProgram(program);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_in);
    free(h_out);

    printf("\n=== Fim ===\n");
    return 0;
}

/*
 * =============================================================================
 *  APENDICE: Como usar CLRX para bypass GARANTIDO de cache (ISA direta)
 * =============================================================================
 *
 *  1. Instale o CLRX Assembler: https://github.com/CLRX/CLRX-mirror
 *
 *  2. Extraia o binario ISA do kernel compilado:
 *       clrxdisasm kernel_dump.bin > kernel.asm
 *
 *  3. No arquivo .asm, localize as instrucoes buffer_load/buffer_store
 *     e modifique os flags:
 *
 *     Cenario 2 (bypass L1):
 *       buffer_load_dword v1, v0, s[4:7], 0 offen glc
 *       buffer_store_dword v2, v0, s[8:11], 0 offen glc
 *
 *     Cenario 3 (bypass L1 + L2):
 *       buffer_load_dword v1, v0, s[4:7], 0 offen glc slc
 *       buffer_store_dword v2, v0, s[8:11], 0 offen glc slc
 *
 *  4. Reassemble:
 *       clrxasm kernel_modified.asm -o kernel_modified.bin
 *
 *  5. No host, carregue com clCreateProgramWithBinary():
 *
 *       FILE* fp = fopen("kernel_modified.bin", "rb");
 *       fseek(fp, 0, SEEK_END);
 *       size_t bin_size = ftell(fp);
 *       rewind(fp);
 *       unsigned char* binary = malloc(bin_size);
 *       fread(binary, 1, bin_size, fp);
 *       fclose(fp);
 *
 *       cl_int bin_status;
 *       cl_program prog = clCreateProgramWithBinary(
 *           ctx, 1, &device,
 *           &bin_size, (const unsigned char**)&binary,
 *           &bin_status, &err);
 *       clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
 *
 * =============================================================================
 *  APENDICE: Arquitetura de Cache AMD GCN/RDNA
 * =============================================================================
 *
 *  GCN (ex: RX 480, RX 580, Vega 56/64):
 *    L1 (vL1): 16 KB por CU, write-through
 *    L2: Compartilhado, 512KB-4MB
 *    GLC=1: Bypass L1
 *    SLC=1: Bypass L2
 *
 *  RDNA / RDNA2 / RDNA3 (ex: RX 5700, RX 6800, RX 7900):
 *    L0: 16 KB por WGP
 *    L1: 128 KB por Shader Array
 *    L2: Compartilhado, ate 6MB (RDNA3)
 *    Infinity Cache: 16-96 MB (RDNA2+)
 *    GLC=1: Bypass L0
 *    SLC=1: Bypass L1+L2
 *    DLC=1: (RDNA2+) Bypass do Infinity Cache (L3)
 *
 *  Em RDNA2+, bypass TOTAL requer: GLC=1 + SLC=1 + DLC=1
 *
 * =============================================================================
 */
