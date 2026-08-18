/*
 * Debug: investiga crash do binary patching
 * Testa passo a passo com output em stderr (sem buffer)
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

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

#define LOG(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while(0)

int main(void) {
    cl_int err;
    const int N = 256;  /* Pequeno para teste rapido */
    const size_t mat_size = (size_t)N * N * sizeof(float);

    LOG("[0] Inicio\n");

    float* h_A = (float*)malloc(mat_size);
    float* h_B = (float*)malloc(mat_size);
    float* h_C = (float*)malloc(mat_size);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i+37) % 100) / 100.0f;
    }

    LOG("[1] Dados alocados\n");

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err) { LOG("ERRO GPU: %d\n", err); return 1; }

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    LOG("[2] GPU: %s\n", name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_size, h_A, &err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_size, h_B, &err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, mat_size, NULL, &err);

    LOG("[3] Buffers criados\n");

    /* Compilar */
    const char* src[] = { kernel_source };
    cl_program prog = clCreateProgramWithSource(ctx, 1, src, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    if (err) { LOG("ERRO build: %d\n", err); return 1; }

    LOG("[4] Kernel compilado\n");

    /* ========== TESTE 1: Kernel original ========== */
    {
        cl_kernel k = clCreateKernel(prog, "matmul", &err);
        clSetKernelArg(k, 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(k, 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(k, 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(k, 3, sizeof(int), &N);

        size_t global[2] = {(size_t)N, (size_t)N};
        size_t local[2] = {16, 16};

        LOG("[5] Executando kernel ORIGINAL...\n");
        cl_event ev;
        err = clEnqueueNDRangeKernel(queue, k, 2, NULL, global, local, 0, NULL, &ev);
        LOG("    enqueue err=%d\n", err);
        err = clWaitForEvents(1, &ev);
        LOG("    wait err=%d\n", err);

        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
        LOG("    tempo: %.3f ms\n", (double)(t1-t0)/1e6);

        clReleaseEvent(ev);
        clReleaseKernel(k);
    }

    LOG("[6] Original OK\n");

    /* ========== Extrair binario ========== */
    size_t bin_size = 0;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
    LOG("[7] Binario: %lu bytes\n", (unsigned long)bin_size);

    unsigned char* original = (unsigned char*)malloc(bin_size);
    unsigned char* ptrs[] = { original };
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), ptrs, NULL);

    /* Salvar original para referencia */
    FILE* fp = fopen("debug_original.bin", "wb");
    if (fp) { fwrite(original, 1, bin_size, fp); fclose(fp); }

    /* ========== Encontrar inner ELF e .hsatext ========== */
    size_t inner_elf_off = 0;
    for (size_t i = 1; i < bin_size - 4; i++) {
        if (original[i] == 0x7f && original[i+1] == 'E' &&
            original[i+2] == 'L' && original[i+3] == 'F') {
            inner_elf_off = i;
            break;
        }
    }
    LOG("[8] Inner ELF offset: 0x%lx\n", (unsigned long)inner_elf_off);

    if (inner_elf_off == 0) {
        LOG("ERRO: inner ELF nao encontrado\n");
        return 1;
    }

    /* Parse inner ELF64 para achar .hsatext */
    unsigned char* elf = original + inner_elf_off;
    unsigned long long e_shoff;
    unsigned short e_shentsize, e_shnum, e_shstrndx;
    memcpy(&e_shoff, elf + 40, 8);
    memcpy(&e_shentsize, elf + 58, 2);
    memcpy(&e_shnum, elf + 60, 2);
    memcpy(&e_shstrndx, elf + 62, 2);

    unsigned long long shstr_off;
    memcpy(&shstr_off, elf + e_shoff + e_shstrndx * e_shentsize + 24, 8);

    size_t hsatext_off = 0, hsatext_size = 0;
    for (int s = 0; s < e_shnum; s++) {
        unsigned char* sh = elf + e_shoff + s * e_shentsize;
        unsigned int sh_name_idx;
        unsigned long long sh_offset, sh_size;
        memcpy(&sh_name_idx, sh, 4);
        memcpy(&sh_offset, sh + 24, 8);
        memcpy(&sh_size, sh + 32, 8);

        char* sname = (char*)(elf + shstr_off + sh_name_idx);
        if (strcmp(sname, ".hsatext") == 0) {
            hsatext_off = inner_elf_off + (size_t)sh_offset;
            hsatext_size = (size_t)sh_size;
            break;
        }
    }

    LOG("[9] .hsatext: offset=0x%lx size=0x%lx\n",
        (unsigned long)hsatext_off, (unsigned long)hsatext_size);

    if (hsatext_size == 0) {
        LOG("ERRO: .hsatext nao encontrada\n");
        return 1;
    }

    /* ========== Patch apenas GLC (bit 13) nos MUBUF ========== */
    unsigned char* patched = (unsigned char*)malloc(bin_size);
    memcpy(patched, original, bin_size);

    /* Pula header de 256 bytes */
    size_t code_start = hsatext_off + 0x100;
    size_t code_end = hsatext_off + hsatext_size;
    int patch_count = 0;

    LOG("[10] Scanning MUBUF de 0x%lx a 0x%lx\n",
        (unsigned long)code_start, (unsigned long)code_end);

    for (size_t pos = code_start; pos < code_end - 7; pos += 4) {
        unsigned int word0;
        memcpy(&word0, patched + pos, 4);

        unsigned int enc = (word0 >> 26) & 0x3F;
        if (enc != 0x38) continue;

        unsigned int opcode = (word0 >> 18) & 0xFF;
        int is_mem = (opcode >= 12 && opcode <= 15) ||
                     (opcode >= 20 && opcode <= 21) ||
                     (opcode >= 28 && opcode <= 31);
        if (!is_mem) continue;

        unsigned int old = word0;
        word0 |= (1 << 13);  /* GLC = 1 */
        memcpy(patched + pos, &word0, 4);

        LOG("    MUBUF @0x%lx op=%u: 0x%08x -> 0x%08x (GLC=1)\n",
            (unsigned long)pos, opcode, old, word0);
        patch_count++;

        pos += 4; /* pula word1 */
    }

    LOG("[11] %d instrucoes patchadas\n", patch_count);

    /* Salvar patchado */
    fp = fopen("debug_patched.bin", "wb");
    if (fp) { fwrite(patched, 1, bin_size, fp); fclose(fp); }

    /* ========== TESTE 2: Carregar binario patchado ========== */
    LOG("[12] Carregando binario patchado...\n");

    cl_int bin_status;
    const unsigned char* cptrs[] = { patched };
    cl_program prog2 = clCreateProgramWithBinary(
        ctx, 1, &device, &bin_size, cptrs, &bin_status, &err);
    LOG("    createProgramWithBinary: err=%d status=%d\n", err, bin_status);

    if (err != CL_SUCCESS) {
        LOG("ERRO: driver rejeitou o binario!\n");
        free(original); free(patched);
        return 1;
    }

    LOG("[13] Building programa patchado...\n");
    err = clBuildProgram(prog2, 1, &device, NULL, NULL, NULL);
    LOG("    build err=%d\n", err);

    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog2, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        LOG("    build log: %s\n", log);
        free(original); free(patched);
        return 1;
    }

    LOG("[14] Criando kernel patchado...\n");
    cl_kernel k2 = clCreateKernel(prog2, "matmul", &err);
    LOG("    createKernel err=%d\n", err);

    if (err != CL_SUCCESS) {
        LOG("ERRO: kernel nao encontrado no binario patchado\n");
        free(original); free(patched);
        return 1;
    }

    clSetKernelArg(k2, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(k2, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(k2, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(k2, 3, sizeof(int), &N);

    size_t global2[2] = {(size_t)N, (size_t)N};
    size_t local2[2] = {16, 16};

    LOG("[15] Executando kernel PATCHADO (GLC=1)...\n");
    cl_event ev2;
    err = clEnqueueNDRangeKernel(queue, k2, 2, NULL, global2, local2, 0, NULL, &ev2);
    LOG("    enqueue err=%d\n", err);

    if (err != CL_SUCCESS) {
        LOG("ERRO: enqueue falhou!\n");
    } else {
        LOG("[16] Aguardando resultado...\n");
        err = clWaitForEvents(1, &ev2);
        LOG("    wait err=%d\n", err);

        if (err == CL_SUCCESS) {
            cl_ulong t0, t1;
            clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
            clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
            LOG("    tempo: %.3f ms\n", (double)(t1-t0)/1e6);

            /* Verificar resultado */
            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_size, h_C, 0, NULL, NULL);
            float ref = 0;
            for (int k = 0; k < N; k++)
                ref += h_A[k] * h_B[k * N];
            LOG("    Verificacao C[0][0]: GPU=%.4f CPU=%.4f diff=%.6f\n",
                h_C[0], ref, fabsf(h_C[0] - ref));
        }
        clReleaseEvent(ev2);
    }

    LOG("[17] Fim!\n");

    clReleaseKernel(k2);
    clReleaseProgram(prog2);
    clReleaseProgram(prog);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_A); free(h_B); free(h_C);
    free(original); free(patched);

    return 0;
}
