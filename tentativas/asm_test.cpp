/*
 * Teste do kernel com inline assembly GCN
 * Compilar:
 *   g++ -Wall -O2 -D__USE_MINGW_ANSI_STDIO=1 "-Wl,--enable-stdcall-fixup"
 *       -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *       asm_test.cpp -o output\asm_test.exe "C:\Windows\System32\OpenCL.dll"
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

static const char* kernel_source =
"__kernel void cache_policy_test(__global float* data_in,\n"
"                                __global float* data_out,\n"
"                                int policy)\n"
"{\n"
"    uint id = get_global_id(0);\n"
"    float val;\n"
"    ulong addr = (ulong)&data_in[id];\n"
"\n"
"    if (policy == 0) {\n"
"        val = data_in[id];\n"
"    }\n"
"    else if (policy == 1) {\n"
"        __asm__ volatile (\n"
"            \"global_load_dword %0, %1, off glc \\n\\t\"\n"
"            \"s_waitcnt vmcnt(0)\"\n"
"            : \"=v\"(val)\n"
"            : \"v\"(addr)\n"
"        );\n"
"    }\n"
"    else if (policy == 2) {\n"
"        __asm__ volatile (\n"
"            \"global_load_dword %0, %1, off slc \\n\\t\"\n"
"            \"s_waitcnt vmcnt(0)\"\n"
"            : \"=v\"(val)\n"
"            : \"v\"(addr)\n"
"        );\n"
"    }\n"
"\n"
"    data_out[id] = val + 1.0f;\n"
"}\n";

int main(void) {
    cl_int err;
    const int N = 1024 * 1024;
    const size_t sz = N * sizeof(float);

    /* Setup */
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Sem GPU: %d\n", err); return 1; }

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("GPU: %s\n\n", name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    /* Compilar kernel */
    printf("Compilando kernel com inline asm...\n");
    fflush(stdout);

    const char* src[] = { kernel_source };
    cl_program program = clCreateProgramWithSource(ctx, 1, src, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource falhou: %d\n", err);
        return 1;
    }

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);

    /* Mostra o build log independente do resultado */
    char log[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(log), log, NULL);
    printf("=== Build Log ===\n%s\n=================\n\n", log);
    fflush(stdout);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram falhou: %d\n", err);
        fprintf(stderr, "O inline asm provavelmente nao e suportado nesta GPU/driver.\n");
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    printf("Compilacao OK! Testando...\n\n");
    fflush(stdout);

    /* Dados */
    float* h_in  = (float*)malloc(sz);
    float* h_out = (float*)malloc(sz);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    cl_mem d_in  = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz, h_in, &err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

    cl_kernel kernel = clCreateKernel(program, "cache_policy_test", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel falhou: %d\n", err);
        return 1;
    }

    size_t global = (size_t)N;
    size_t local  = 256;

    /* Testar cada policy */
    const char* policy_names[] = {
        "Policy 0: Normal (L1->L2->VRAM)",
        "Policy 1: Bypass L1 (GLC=1)",
        "Policy 2: Bypass L2 (SLC=1)"
    };

    for (int policy = 0; policy < 3; policy++) {
        printf("--- %s ---\n", policy_names[policy]);
        fflush(stdout);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(kernel, 2, sizeof(int), &policy);

        /* Warmup */
        cl_event ev;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        if (err != CL_SUCCESS) {
            printf("  clEnqueueNDRangeKernel falhou: %d\n\n", err);
            fflush(stdout);
            continue;
        }
        clWaitForEvents(1, &ev);
        clReleaseEvent(ev);

        /* Benchmark */
        double total = 0;
        const int RUNS = 5;
        for (int r = 0; r < RUNS; r++) {
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
            if (err != CL_SUCCESS) {
                printf("  Run %d falhou: %d\n", r+1, err);
                fflush(stdout);
                break;
            }
            clWaitForEvents(1, &ev);

            cl_ulong t0, t1;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
            double ms = (double)(t1-t0) / 1e6;
            total += ms;
            printf("  Run %d: %.3f ms\n", r+1, ms);
            fflush(stdout);
            clReleaseEvent(ev);
        }

        printf("  Media: %.3f ms\n\n", total / RUNS);
        fflush(stdout);
    }

    /* Verificacao */
    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sz, h_out, 0, NULL, NULL);
    printf("Verificacao: out[0]=%.1f (esperado: %.1f)\n", h_out[0], h_in[0] + 1.0f);

    /* Cleanup */
    clReleaseKernel(kernel);
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
