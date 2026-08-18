/*
 * =============================================================================
 *  AMD Ellesmere — Demonstracao Real de Hierarquia de Memoria
 * =============================================================================
 *
 *  Como nao conseguimos forcar GLC/SLC na Ellesmere via OpenCL,
 *  esta abordagem demonstra o impacto do cache usando PADROES DE ACESSO
 *  que naturalmente causam diferentes comportamentos de cache:
 *
 *  Cenario 1: Acesso SEQUENCIAL (cache-friendly)
 *     - Cada work-item le posicoes contíguas
 *     - L1 e L2 funcionam perfeitamente (prefetch, cache lines)
 *
 *  Cenario 2: Acesso STRIDE (parcialmente cache-hostile)
 *     - Cada work-item le com stride grande
 *     - L1 miss frequente, L2 ainda ajuda
 *
 *  Cenario 3: Acesso ALEATORIO (cache-hostile)
 *     - Cada work-item le posicoes pseudo-aleatorias
 *     - L1 e L2 quase inuteis, maioria dos acessos vai a VRAM
 *
 *  Cenario 4: Acesso via __local memory (LDS)
 *     - Dados carregados uma vez para LDS, reutilizados localmente
 *     - Demonstra controle explicito da hierarquia
 *
 *  Cenario 5: Stencil 2D (9-point)
 *     - Cada pixel le 9 vizinhos — alto reuso entre work-items adjacentes
 *     - L1 cache faz grande diferenca aqui
 *
 *  Cenario 6: Stencil 2D com __local memory
 *     - Mesma operacao, mas dados carregados em LDS primeiro
 *     - Elimina reloads redundantes do L1/L2
 *
 *  Compilar:
 *    g++ -Wall -O2 -D__USE_MINGW_ANSI_STDIO=1
 *        "-Wl,--enable-stdcall-fixup"
 *        -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *        mem_hierarchy.cpp -o output\mem_hierarchy.exe
 *        "C:\Windows\System32\OpenCL.dll"
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

/* -------------------------------------------------------------------------- */

static const char* kernels_source =
"/* ===== Cenario 1: Reducao com acesso SEQUENCIAL (cache-friendly) ===== */\n"
"__kernel void reduce_sequential(\n"
"    __global const float* data,\n"
"    __global float* out,\n"
"    const int N,\n"
"    const int iters)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < iters; i++) {\n"
"        int idx = (gid + i) % N;\n"
"        sum += data[idx];\n"
"    }\n"
"    out[gid] = sum;\n"
"}\n"
"\n"
"/* ===== Cenario 2: Reducao com acesso STRIDE (parcial cache miss) ===== */\n"
"__kernel void reduce_stride(\n"
"    __global const float* data,\n"
"    __global float* out,\n"
"    const int N,\n"
"    const int iters)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    float sum = 0.0f;\n"
"    int stride = 67;  /* stride primo, evita alinhamento com cache lines */\n"
"    for (int i = 0; i < iters; i++) {\n"
"        int idx = (gid * stride + i * stride) % N;\n"
"        sum += data[idx];\n"
"    }\n"
"    out[gid] = sum;\n"
"}\n"
"\n"
"/* ===== Cenario 3: Reducao com acesso ALEATORIO (cache-hostile) ===== */\n"
"__kernel void reduce_random(\n"
"    __global const float* data,\n"
"    __global float* out,\n"
"    const int N,\n"
"    const int iters)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    float sum = 0.0f;\n"
"    uint seed = (uint)gid * 2654435761u;  /* hash de Knuth */\n"
"    for (int i = 0; i < iters; i++) {\n"
"        seed ^= seed << 13;\n"
"        seed ^= seed >> 17;\n"
"        seed ^= seed << 5;\n"
"        int idx = seed % N;\n"
"        sum += data[idx];\n"
"    }\n"
"    out[gid] = sum;\n"
"}\n"
"\n"
"/* ===== Cenario 4: Reducao com __local memory (LDS) ===== */\n"
"__kernel void reduce_local(\n"
"    __global const float* data,\n"
"    __global float* out,\n"
"    const int N,\n"
"    const int iters)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int lid = get_local_id(0);\n"
"    int group_size = get_local_size(0);\n"
"    int group_id = get_group_id(0);\n"
"\n"
"    /* Carrega um bloco para LDS */\n"
"    __local float tile[256];\n"
"    int base = (group_id * group_size) % N;\n"
"    tile[lid] = data[(base + lid) % N];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    /* Reutiliza dados do LDS muitas vezes */\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < iters; i++) {\n"
"        int idx = (lid + i) % group_size;\n"
"        sum += tile[idx];\n"
"    }\n"
"    out[gid] = sum;\n"
"}\n"
"\n"
"/* ===== Cenario 5: Stencil 2D 9-point (sem __local) ===== */\n"
"__kernel void stencil_global(\n"
"    __global const float* in,\n"
"    __global float* out,\n"
"    const int W,\n"
"    const int H)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) {\n"
"        if (x < W && y < H) out[y*W+x] = in[y*W+x];\n"
"        return;\n"
"    }\n"
"    /* Media dos 9 vizinhos (blur box 3x3) */\n"
"    float sum = 0.0f;\n"
"    for (int dy = -1; dy <= 1; dy++) {\n"
"        for (int dx = -1; dx <= 1; dx++) {\n"
"            sum += in[(y+dy)*W + (x+dx)];\n"
"        }\n"
"    }\n"
"    out[y*W+x] = sum / 9.0f;\n"
"}\n"
"\n"
"/* ===== Cenario 6: Stencil 2D 9-point com __local memory ===== */\n"
"#define TILE_W 18  /* 16 + 2 (bordas) */\n"
"#define TILE_H 18\n"
"__kernel void stencil_local(\n"
"    __global const float* in,\n"
"    __global float* out,\n"
"    const int W,\n"
"    const int H)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int lx = get_local_id(0);\n"
"    int ly = get_local_id(1);\n"
"\n"
"    __local float tile[TILE_H][TILE_W];\n"
"\n"
"    /* Carrega tile + bordas para LDS */\n"
"    int src_x = x - 1;\n"
"    int src_y = y - 1;\n"
"\n"
"    /* Cada thread carrega seu pixel + possivelmente bordas */\n"
"    for (int j = ly; j < TILE_H; j += get_local_size(1)) {\n"
"        for (int i = lx; i < TILE_W; i += get_local_size(0)) {\n"
"            int gx = (int)get_group_id(0) * (int)get_local_size(0) - 1 + i;\n"
"            int gy = (int)get_group_id(1) * (int)get_local_size(1) - 1 + j;\n"
"            gx = clamp(gx, 0, W-1);\n"
"            gy = clamp(gy, 0, H-1);\n"
"            tile[j][i] = in[gy * W + gx];\n"
"        }\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    if (x >= W || y >= H) return;\n"
"    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) {\n"
"        out[y*W+x] = in[y*W+x];\n"
"        return;\n"
"    }\n"
"\n"
"    /* Leitura do LDS (rapido!) */\n"
"    float sum = 0.0f;\n"
"    for (int dy = 0; dy <= 2; dy++) {\n"
"        for (int dx = 0; dx <= 2; dx++) {\n"
"            sum += tile[ly+dy][lx+dx];\n"
"        }\n"
"    }\n"
"    out[y*W+x] = sum / 9.0f;\n"
"}\n"
"\n";

/* -------------------------------------------------------------------------- */

#define CHECK_CL(e, m) do { if((e)!=CL_SUCCESS){fprintf(stderr,"[ERRO] %s: %d\n",(m),(e));exit(1);}} while(0)

static double get_ms(cl_event ev) {
    cl_ulong t0, t1;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (double)(t1-t0)/1e6;
}

/* -------------------------------------------------------------------------- */

int main(void) {
    cl_int err;

    printf("=== AMD Ellesmere — Hierarquia de Memoria Demo ===\n\n");
    fflush(stdout);

    /* Setup */
    cl_platform_id plat; clGetPlatformIDs(1, &plat, NULL);
    cl_device_id dev;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    CHECK_CL(err, "GPU");

    char name[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("GPU: %s\n\n", name);

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);

    /* Compilar */
    const char* src[] = { kernels_source };
    cl_program prog = clCreateProgramWithSource(ctx, 1, src, NULL, &err);
    CHECK_CL(err, "source");
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Build:\n%s\n", log);
        exit(1);
    }

    /* ================================================================ */
    /* PARTE A: Reducao com diferentes padroes de acesso                */
    /* ================================================================ */
    {
        const int N = 4 * 1024 * 1024;  /* 16 MB */
        const int WORK = 256 * 1024;
        const int ITERS = 128;
        const size_t data_sz = (size_t)N * sizeof(float);
        const size_t out_sz  = (size_t)WORK * sizeof(float);

        float* h_data = (float*)malloc(data_sz);
        for (int i = 0; i < N; i++) h_data[i] = (float)(i % 1000) / 1000.0f;

        cl_mem d_data = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        data_sz, h_data, &err);
        cl_mem d_out  = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_sz, NULL, &err);

        const char* knames[] = {
            "reduce_sequential", "reduce_stride", "reduce_random", "reduce_local"
        };
        const char* descs[] = {
            "Sequencial   (cache-friendly, L1+L2 efetivos)",
            "Stride       (L1 miss, L2 parcial)",
            "Aleatorio    (L1+L2 miss, vai a VRAM)",
            "__local mem  (LDS, sem depender de cache)"
        };

        printf("========================================\n");
        printf("  PARTE A: Padroes de Acesso a Memoria\n");
        printf("  Dados: %d MB | Work-items: %d | Iteracoes: %d\n",
               (int)(data_sz/(1024*1024)), WORK, ITERS);
        printf("========================================\n\n");
        fflush(stdout);

        size_t global = (size_t)WORK;
        size_t local  = 256;

        for (int k = 0; k < 4; k++) {
            cl_kernel kern = clCreateKernel(prog, knames[k], &err);
            CHECK_CL(err, knames[k]);

            clSetKernelArg(kern, 0, sizeof(cl_mem), &d_data);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &d_out);
            clSetKernelArg(kern, 2, sizeof(int), &N);
            clSetKernelArg(kern, 3, sizeof(int), &ITERS);

            /* Warmup */
            for (int w = 0; w < 2; w++) {
                cl_event ev;
                clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
                clWaitForEvents(1, &ev);
                clReleaseEvent(ev);
            }

            /* Benchmark */
            double total = 0, mn = 1e9, mx = 0;
            const int RUNS = 5;
            for (int r = 0; r < RUNS; r++) {
                cl_event ev;
                clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, &ev);
                clWaitForEvents(1, &ev);
                double ms = get_ms(ev);
                total += ms;
                if (ms < mn) mn = ms;
                if (ms > mx) mx = ms;
                clReleaseEvent(ev);
            }

            double avg = total / RUNS;
            double gb = ((double)WORK * ITERS * sizeof(float)) / (avg/1000.0) / 1e9;

            printf("  %s\n", descs[k]);
            printf("    Media: %7.3f ms | Min: %7.3f | Max: %7.3f\n", avg, mn, mx);
            printf("    Bandwidth efetiva: %.2f GB/s\n\n", gb);
            fflush(stdout);

            clReleaseKernel(kern);
        }

        clReleaseMemObject(d_data);
        clReleaseMemObject(d_out);
        free(h_data);
    }

    /* ================================================================ */
    /* PARTE B: Stencil 2D (demonstra reuso de cache entre vizinhos)    */
    /* ================================================================ */
    {
        const int W = 4096, H = 4096;
        const size_t img_sz = (size_t)W * H * sizeof(float);

        float* h_img = (float*)malloc(img_sz);
        for (int i = 0; i < W*H; i++) h_img[i] = (float)(i % 256) / 256.0f;

        cl_mem d_in  = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       img_sz, h_img, &err);
        cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, img_sz, NULL, &err);

        printf("========================================\n");
        printf("  PARTE B: Stencil 2D (blur 3x3)\n");
        printf("  Imagem: %dx%d (%.1f MB)\n", W, H, (double)img_sz/(1024*1024));
        printf("========================================\n\n");
        fflush(stdout);

        const char* knames[] = { "stencil_global", "stencil_local" };
        const char* descs[] = {
            "Global memory  (9 reads/pixel, depende do L1 cache)",
            "__local memory (carrega tile uma vez, 9 reads do LDS)"
        };

        size_t global2[2] = { (size_t)W, (size_t)H };
        size_t local2[2]  = { 16, 16 };

        for (int k = 0; k < 2; k++) {
            cl_kernel kern = clCreateKernel(prog, knames[k], &err);
            CHECK_CL(err, knames[k]);

            clSetKernelArg(kern, 0, sizeof(cl_mem), &d_in);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &d_out);
            clSetKernelArg(kern, 2, sizeof(int), &W);
            clSetKernelArg(kern, 3, sizeof(int), &H);

            /* Warmup */
            for (int w = 0; w < 2; w++) {
                cl_event ev;
                clEnqueueNDRangeKernel(q, kern, 2, NULL, global2, local2, 0, NULL, &ev);
                clWaitForEvents(1, &ev);
                clReleaseEvent(ev);
            }

            /* Benchmark */
            double total = 0, mn = 1e9, mx = 0;
            const int RUNS = 5;
            for (int r = 0; r < RUNS; r++) {
                cl_event ev;
                clEnqueueNDRangeKernel(q, kern, 2, NULL, global2, local2, 0, NULL, &ev);
                clWaitForEvents(1, &ev);
                double ms = get_ms(ev);
                total += ms;
                if (ms < mn) mn = ms;
                if (ms > mx) mx = ms;
                clReleaseEvent(ev);
            }

            double avg = total / RUNS;
            /* 9 reads + 1 write per pixel (stencil_global) */
            double gb = ((double)W * H * 10 * sizeof(float)) / (avg/1000.0) / 1e9;

            printf("  %s\n", descs[k]);
            printf("    Media: %7.3f ms | Min: %7.3f | Max: %7.3f\n", avg, mn, mx);
            printf("    Bandwidth efetiva: ~%.2f GB/s\n\n", gb);
            fflush(stdout);

            clReleaseKernel(kern);
        }

        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        free(h_img);
    }

    /* Cleanup */
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    printf("=== Fim ===\n");
    return 0;
}
