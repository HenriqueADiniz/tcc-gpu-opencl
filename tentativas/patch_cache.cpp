/*
 * =============================================================================
 *  AMD Ellesmere (GFX8) — Cache Bypass via Binary ISA Patching
 * =============================================================================
 *
 *  Este programa:
 *  1. Compila um kernel OpenCL normalmente
 *  2. Extrai o binario ISA compilado pelo driver
 *  3. Localiza as instrucoes MUBUF (buffer_load/store)
 *  4. Cria 3 versoes do binario com diferentes flags de cache:
 *     - Original:  GLC=0, SLC=0 (usa L1 -> L2 -> VRAM)
 *     - Patch GLC: GLC=1, SLC=0 (bypass L1, usa L2)
 *     - Patch SLC: GLC=1, SLC=1 (bypass L1 + L2, direto VRAM)
 *  5. Recarrega cada versao via clCreateProgramWithBinary
 *  6. Benchmarka os 3 cenarios
 *
 *  Compilar:
 *    g++ -Wall -Wextra -O2 -D__USE_MINGW_ANSI_STDIO=1
 *        "-Wl,--enable-stdcall-fixup"
 *        -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *        patch_cache.cpp -o output\patch_cache.exe
 *        "C:\Windows\System32\OpenCL.dll"
 *
 * =============================================================================
 *  GFX8 MUBUF Encoding (64 bits):
 *
 *  Word 0 [31:0]:
 *    [31:26] = 111000 (0x38) = MUBUF
 *    [25:18] = OP (opcode)
 *    [17]    = SLC  <--- System Level Coherent (bypass L2)
 *    [16]    = TFE
 *    [15]    = OFFEN
 *    [14]    = IDXEN
 *    [13]    = GLC  <--- Globally Coherent (bypass L1)
 *    [12]    = LDS
 *    [11:0]  = OFFSET
 *
 *  Opcodes relevantes:
 *    12 = BUFFER_LOAD_DWORD
 *    13 = BUFFER_LOAD_DWORDX2
 *    20 = BUFFER_LOAD_DWORD (variante com ADDR64/LDS)
 *    28 = BUFFER_STORE_DWORD
 *    29 = BUFFER_STORE_DWORDX2
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
/*  Kernel OpenCL (simples, sem volatile)                                    */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
"__kernel void matmul(\n"
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
"}\n";

/* ------------------------------------------------------------------------- */
/*  Utilitarios                                                              */
/* ------------------------------------------------------------------------- */

#define CHECK_CL(err, msg) do {                                         \
    if ((err) != CL_SUCCESS) {                                          \
        fprintf(stderr, "[ERRO] %s (codigo: %d)\n", (msg), (err));      \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static double get_time_ms(cl_event ev) {
    cl_ulong t0, t1;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &t1, NULL);
    return (double)(t1 - t0) / 1e6;
}

/* ------------------------------------------------------------------------- */
/*  Binary ISA Patcher                                                       */
/* ------------------------------------------------------------------------- */

typedef struct {
    unsigned char* data;
    size_t size;
} BinaryBlob;

/*
 * Extrai o binario compilado de um cl_program
 */
static BinaryBlob extract_binary(cl_program program) {
    BinaryBlob blob = {NULL, 0};

    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t), &blob.size, NULL);

    if (blob.size == 0) {
        fprintf(stderr, "[ERRO] Binario vazio\n");
        exit(1);
    }

    blob.data = (unsigned char*)malloc(blob.size);
    unsigned char* ptrs[] = { blob.data };
    clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                     sizeof(unsigned char*), ptrs, NULL);

    return blob;
}

/*
 * Localiza a secao .hsatext dentro do ELF aninhado.
 * Retorna o offset e tamanho do codigo ISA.
 */
typedef struct {
    size_t offset;   /* offset absoluto no blob.data */
    size_t size;
} CodeSection;

static CodeSection find_hsatext(const BinaryBlob* blob) {
    CodeSection cs = {0, 0};

    /* Procura o ELF interno (64-bit, dentro de .text do ELF externo) */
    for (size_t i = 0; i < blob->size - 4; i++) {
        if (blob->data[i]   == 0x7f &&
            blob->data[i+1] == 'E'  &&
            blob->data[i+2] == 'L'  &&
            blob->data[i+3] == 'F'  &&
            i > 0)  /* Pula o primeiro ELF (externo) */
        {
            /* Encontrou ELF interno. Parseia para achar .hsatext */
            unsigned char* elf = blob->data + i;
            int is_64 = (elf[4] == 2);

            if (!is_64) continue;

            /* ELF64: e_shoff at offset 40 */
            unsigned long long e_shoff;
            unsigned short e_shentsize, e_shnum, e_shstrndx;
            memcpy(&e_shoff,     elf + 40, 8);
            memcpy(&e_shentsize, elf + 58, 2);
            memcpy(&e_shnum,     elf + 60, 2);
            memcpy(&e_shstrndx,  elf + 62, 2);

            /* String table */
            unsigned long long shstr_off, shstr_sz;
            memcpy(&shstr_off, elf + e_shoff + e_shstrndx * e_shentsize + 24, 8);
            memcpy(&shstr_sz,  elf + e_shoff + e_shstrndx * e_shentsize + 32, 8);

            for (int s = 0; s < e_shnum; s++) {
                unsigned char* sh = elf + e_shoff + s * e_shentsize;
                unsigned int sh_name_idx;
                unsigned long long sh_offset, sh_size;
                memcpy(&sh_name_idx, sh, 4);
                memcpy(&sh_offset,   sh + 24, 8);
                memcpy(&sh_size,     sh + 32, 8);

                char* name = (char*)(elf + shstr_off + sh_name_idx);
                if (strcmp(name, ".hsatext") == 0) {
                    cs.offset = i + (size_t)sh_offset;
                    cs.size   = (size_t)sh_size;
                    return cs;
                }
            }
        }
    }

    fprintf(stderr, "[ERRO] .hsatext nao encontrada no binario\n");
    exit(1);
    return cs;
}

/*
 * Patcha as instrucoes MUBUF na secao .hsatext.
 *
 * mode:
 *   0 = nao alterar (original, GLC=0 SLC=0)
 *   1 = setar GLC=1, SLC=0  (bypass L1)
 *   2 = setar GLC=1, SLC=1  (bypass L1+L2)
 *
 * GFX8 MUBUF word0:
 *   bit 13 = GLC
 *   bit 17 = SLC
 *
 * Retorna quantas instrucoes foram patchadas.
 */
static int patch_mubuf(BinaryBlob* blob, const CodeSection* cs, int mode) {
    int patched = 0;

    /*
     * O .hsatext contem: [256 bytes header amd_kernel_code_t] [codigo ISA]
     * Precisamos PULAR o header de 256 bytes (0x100) para nao corromper
     * metadados do kernel.
     */
    size_t code_start = cs->offset + 0x100;  /* pula header */
    size_t code_end   = cs->offset + cs->size;

    if (code_start >= code_end) return 0;

    for (size_t pos = code_start; pos < code_end - 7; pos += 4) {
        unsigned int word0;
        memcpy(&word0, blob->data + pos, 4);

        unsigned int enc = (word0 >> 26) & 0x3F;
        if (enc != 0x38) continue;  /* Nao e MUBUF */

        unsigned int opcode = (word0 >> 18) & 0xFF;

        /* So patcha loads e stores de dados */
        int is_load  = (opcode >= 12 && opcode <= 15) || (opcode == 20);
        int is_store = (opcode >= 28 && opcode <= 31);

        if (!is_load && !is_store) continue;

        unsigned int orig = word0;

        /* Aplica patch */
        if (mode >= 1) {
            word0 |= (1 << 13);   /* GLC = 1 */
        }
        if (mode >= 2) {
            word0 |= (1 << 17);   /* SLC = 1 */
        }

        memcpy(blob->data + pos, &word0, 4);

        printf("      [patch] offset=0x%lx op=%u: 0x%08x -> 0x%08x\n",
               (unsigned long)(pos - cs->offset), opcode, orig, word0);

        patched++;

        pos += 4;  /* Pula word1 da instrucao 64-bit */
    }

    return patched;
}

/*
 * Carrega um binario patchado como cl_program
 */
static cl_program load_patched(cl_context ctx, cl_device_id device,
                                const BinaryBlob* blob) {
    cl_int err, bin_status;
    const unsigned char* ptrs[] = { blob->data };

    cl_program prog = clCreateProgramWithBinary(
        ctx, 1, &device, &blob->size, ptrs, &bin_status, &err);

    if (err != CL_SUCCESS || bin_status != CL_SUCCESS) {
        fprintf(stderr, "[ERRO] clCreateProgramWithBinary falhou (err=%d, status=%d)\n",
                err, bin_status);
        exit(1);
    }

    err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Build log: %s\n", log);
        exit(1);
    }

    return prog;
}


/* ------------------------------------------------------------------------- */
/*  Main                                                                     */
/* ------------------------------------------------------------------------- */

int main(void) {
    cl_int err;
    const int N = 1024;
    const size_t mat_size = (size_t)N * N * sizeof(float);
    const double gflops_op = 2.0 * (double)N * N * N / 1e9;

    printf("=== AMD Ellesmere Cache Bypass via Binary Patching ===\n");
    printf("Matriz: %dx%d (%.1f MB por matriz)\n\n", N, N,
           (double)mat_size / (1024.0*1024.0));

    /* -- Dados host -- */
    float* h_A = (float*)malloc(mat_size);
    float* h_B = (float*)malloc(mat_size);
    float* h_C = (float*)malloc(mat_size);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i+37) % 100) / 100.0f;
    }

    /* -- OpenCL setup -- */
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL(err, "clGetDeviceIDs");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("GPU: %s\n\n", dev_name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(
        ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 mat_size, h_A, &err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 mat_size, h_B, &err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, mat_size, NULL, &err);

    /* ================================================================== */
    /* PASSO 1: Compilar o kernel normalmente                             */
    /* ================================================================== */
    printf("[1] Compilando kernel OpenCL...\n");

    const char* src[] = { kernel_source };
    cl_program prog_original = clCreateProgramWithSource(ctx, 1, src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog_original, 1, &device,
                         "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog_original, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Build log: %s\n", log);
        exit(1);
    }

    /* ================================================================== */
    /* PASSO 2: Extrair binario e localizar .hsatext                      */
    /* ================================================================== */
    printf("[2] Extraindo binario ISA...\n");

    BinaryBlob original = extract_binary(prog_original);
    CodeSection hsatext = find_hsatext(&original);

    printf("    Binario: %lu bytes\n", (unsigned long)original.size);
    printf("    .hsatext: offset=0x%lx size=0x%lx\n\n",
           (unsigned long)hsatext.offset, (unsigned long)hsatext.size);

    /* ================================================================== */
    /* PASSO 3: Criar 3 versoes com diferentes flags de cache             */
    /* ================================================================== */

    /* Versao 0: Original (GLC=0, SLC=0) — ja temos */
    printf("[3] Patchando binarios...\n");

    /* Versao 1: GLC=1 (bypass L1) */
    BinaryBlob patched_glc;
    patched_glc.size = original.size;
    patched_glc.data = (unsigned char*)malloc(original.size);
    memcpy(patched_glc.data, original.data, original.size);
    int n1 = patch_mubuf(&patched_glc, &hsatext, 1);
    printf("    Versao GLC=1: %d instrucoes patchadas\n", n1);

    /* Versao 2: GLC=1 + SLC=1 (bypass L1+L2) */
    BinaryBlob patched_slc;
    patched_slc.size = original.size;
    patched_slc.data = (unsigned char*)malloc(original.size);
    memcpy(patched_slc.data, original.data, original.size);
    int n2 = patch_mubuf(&patched_slc, &hsatext, 2);
    printf("    Versao GLC=1+SLC=1: %d instrucoes patchadas\n\n", n2);

    /* ================================================================== */
    /* PASSO 4: Carregar binarios patchados                               */
    /* ================================================================== */
    printf("[4] Carregando binarios patchados...\n");

    cl_program prog_glc = load_patched(ctx, device, &patched_glc);
    cl_program prog_slc = load_patched(ctx, device, &patched_slc);

    printf("    Todos carregados com sucesso!\n\n");

    /* ================================================================== */
    /* PASSO 5: Criar kernels e benchmarkar                               */
    /* ================================================================== */

    cl_kernel k_orig = clCreateKernel(prog_original, "matmul", &err);
    CHECK_CL(err, "kernel original");
    cl_kernel k_glc  = clCreateKernel(prog_glc, "matmul", &err);
    CHECK_CL(err, "kernel glc");
    cl_kernel k_slc  = clCreateKernel(prog_slc, "matmul", &err);
    CHECK_CL(err, "kernel slc");

    cl_kernel kernels[] = { k_orig, k_glc, k_slc };
    const char* nomes[] = {
        "Original (GLC=0 SLC=0) — L1 -> L2 -> VRAM",
        "Patched  (GLC=1 SLC=0) — bypass L1, usa L2",
        "Patched  (GLC=1 SLC=1) — bypass L1+L2, direto VRAM"
    };

    for (int k = 0; k < 3; k++) {
        clSetKernelArg(kernels[k], 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(kernels[k], 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(kernels[k], 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(kernels[k], 3, sizeof(int),    &N);
    }

    size_t global[2] = { (size_t)N, (size_t)N };
    size_t local[2]  = { 16, 16 };

    const int WARMUP = 2;
    const int RUNS   = 5;

    printf("[5] Benchmark: %dx%d matmul, %d runs\n", N, N, RUNS);
    printf("================================================================\n\n");

    for (int k = 0; k < 3; k++) {
        printf("--- %s ---\n", nomes[k]);
        fflush(stdout);

        for (int w = 0; w < WARMUP; w++) {
            cl_event ev;
            cl_int enq_err = clEnqueueNDRangeKernel(queue, kernels[k], 2, NULL,
                                   global, local, 0, NULL, &ev);
            if (enq_err != CL_SUCCESS) {
                printf("  [ERRO] clEnqueueNDRangeKernel warmup falhou: %d\n", enq_err);
                fflush(stdout);
                goto next_kernel;
            }
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }
        printf("  Warmup OK\n");
        fflush(stdout);

        {
        double total = 0, mn = 1e9, mx = 0;
        for (int r = 0; r < RUNS; r++) {
            cl_event ev;
            cl_int enq_err = clEnqueueNDRangeKernel(queue, kernels[k], 2, NULL,
                                   global, local, 0, NULL, &ev);
            if (enq_err != CL_SUCCESS) {
                printf("  [ERRO] clEnqueueNDRangeKernel run falhou: %d\n", enq_err);
                fflush(stdout);
                goto next_kernel;
            }
            clWaitForEvents(1, &ev);

            double ms = get_time_ms(ev);
            total += ms;
            if (ms < mn) mn = ms;
            if (ms > mx) mx = ms;
            printf("  Run %d: %.3f ms\n", r+1, ms);
            fflush(stdout);
            clReleaseEvent(ev);
        }

        double avg = total / RUNS;
        printf("  Media: %.3f ms | Min: %.3f | Max: %.3f\n", avg, mn, mx);
        printf("  GFLOPS: %.2f\n\n", gflops_op / (avg / 1000.0));
        fflush(stdout);
        } /* fecha bloco do benchmark */
        next_kernel:;
    }

    /* -- Verificacao -- */
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_size, h_C, 0, NULL, NULL);

    /* Verificacao rapida com CPU */
    float test_sum = 0;
    int test_row = 0, test_col = 0;
    for (int k = 0; k < N; k++)
        test_sum += h_A[test_row * N + k] * h_B[k * N + test_col];

    printf("Verificacao C[0][0]: GPU=%.6f CPU=%.6f diff=%.8f\n",
           h_C[0], test_sum, fabsf(h_C[0] - test_sum));

    /* -- Salvar binarios para inspecao -- */
    FILE* fp;
    fp = fopen("patched_glc.bin", "wb");
    if (fp) { fwrite(patched_glc.data, 1, patched_glc.size, fp); fclose(fp); }
    fp = fopen("patched_slc.bin", "wb");
    if (fp) { fwrite(patched_slc.data, 1, patched_slc.size, fp); fclose(fp); }
    printf("\nBinarios salvos: patched_glc.bin, patched_slc.bin\n");

    /* -- Cleanup -- */
    clReleaseKernel(k_orig);
    clReleaseKernel(k_glc);
    clReleaseKernel(k_slc);
    clReleaseProgram(prog_original);
    clReleaseProgram(prog_glc);
    clReleaseProgram(prog_slc);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_A); free(h_B); free(h_C);
    free(original.data);
    free(patched_glc.data);
    free(patched_slc.data);

    printf("\n=== Fim ===\n");
    return 0;
}