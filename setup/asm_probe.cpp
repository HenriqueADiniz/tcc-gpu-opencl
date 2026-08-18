/*
 * asm_probe.cpp — Sonda de suporte a inline assembly GCN em runtime OpenCL.
 *
 * Percorre TODAS as plataformas/dispositivos OpenCL visiveis e tenta compilar,
 * em cada um, dois kernels equivalentes:
 *   (1) load normal          -> baseline, deve sempre compilar
 *   (2) load via __asm__ glc -> emite flat_load_dword ... glc (GLC=1)
 *
 * Objetivo: identificar qual runtime OpenCL do Linux aceita __asm__ em kernel,
 * questao central do Experimento 5 (verificacao multiplataforma).
 *
 * Build: g++ -O2 setup/asm_probe.cpp -o output/asm_probe -lOpenCL
 */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

static const char* SRC_BASELINE =
"__kernel void probe(__global const float* in, __global float* out) {\n"
"    int i = get_global_id(0);\n"
"    out[i] = in[i] * 2.0f;\n"
"}\n";

static const char* SRC_ASM_GLC =
"__kernel void probe(__global const float* in, __global float* out) {\n"
"    int i = get_global_id(0);\n"
"    float v;\n"
"    __global const float* p = &in[i];\n"
"    __asm__ volatile(\n"
"        \"flat_load_dword %0, %1 glc\\n\"\n"
"        \"s_waitcnt vmcnt(0)\\n\"\n"
"        : \"=v\"(v) : \"v\"(p));\n"
"    out[i] = v * 2.0f;\n"
"}\n";

/* Tenta compilar `src` no dispositivo. Devolve CL_SUCCESS ou o erro de build,
   preenchendo `log` com o build log do compilador. */
static cl_int try_build(cl_context ctx, cl_device_id dev, const char* src,
                        char* log, size_t log_cap)
{
    log[0] = '\0';
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (err != CL_SUCCESS) {
        snprintf(log, log_cap, "clCreateProgramWithSource falhou: %d", err);
        return err;
    }

    cl_int berr = clBuildProgram(prog, 1, &dev, "", NULL, NULL);

    size_t log_size = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        char* tmp = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, tmp, NULL);
        tmp[log_size] = '\0';
        snprintf(log, log_cap, "%s", tmp);
        free(tmp);
    }

    clReleaseProgram(prog);
    return berr;
}

static void print_log(const char* label, cl_int err, const char* log)
{
    if (err == CL_SUCCESS) {
        printf("      %-22s COMPILOU\n", label);
    } else {
        printf("      %-22s FALHOU (erro %d)\n", label, err);
    }
    if (log[0]) {
        /* Indenta o build log para manter a saida legivel. */
        printf("        --- build log ---\n");
        const char* p = log;
        while (*p) {
            const char* nl = strchr(p, '\n');
            int len = nl ? (int)(nl - p) : (int)strlen(p);
            if (len > 0) printf("        | %.*s\n", len, p);
            if (!nl) break;
            p = nl + 1;
        }
    }
}

int main(void)
{
    cl_uint num_plat = 0;
    if (clGetPlatformIDs(0, NULL, &num_plat) != CL_SUCCESS || num_plat == 0) {
        printf("Nenhuma plataforma OpenCL encontrada.\n");
        return 1;
    }

    cl_platform_id* plats = (cl_platform_id*)malloc(num_plat * sizeof(cl_platform_id));
    clGetPlatformIDs(num_plat, plats, NULL);
    printf("Plataformas OpenCL encontradas: %u\n\n", num_plat);

    char buf[2048];
    char log[16384];

    for (cl_uint p = 0; p < num_plat; p++) {
        clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        printf("Plataforma #%u: %s\n", p, buf);
        clGetPlatformInfo(plats[p], CL_PLATFORM_VERSION, sizeof(buf), buf, NULL);
        printf("  versao: %s\n", buf);

        cl_uint num_dev = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_dev) != CL_SUCCESS
            || num_dev == 0) {
            printf("  (sem dispositivos)\n\n");
            continue;
        }

        cl_device_id* devs = (cl_device_id*)malloc(num_dev * sizeof(cl_device_id));
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, num_dev, devs, NULL);

        for (cl_uint d = 0; d < num_dev; d++) {
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
            printf("  Dispositivo #%u: %s\n", d, buf);

            cl_device_type dt = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dt), &dt, NULL);
            printf("    tipo: %s\n", (dt & CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU/outro");

            cl_uint cus = 0, freq = 0;
            cl_ulong gmem = 0, l2 = 0, lds = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(l2), &l2, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lds), &lds, NULL);
            printf("    CUs: %u | clock: %u MHz | global: %llu MB | cache: %llu KB | local: %llu KB\n",
                   cus, freq,
                   (unsigned long long)(gmem >> 20),
                   (unsigned long long)(l2 >> 10),
                   (unsigned long long)(lds >> 10));

            cl_int err = CL_SUCCESS;
            cl_context ctx = clCreateContext(NULL, 1, &devs[d], NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                printf("    clCreateContext falhou: %d\n", err);
                continue;
            }

            err = try_build(ctx, devs[d], SRC_BASELINE, log, sizeof(log));
            print_log("baseline (sem asm)", err, log);

            err = try_build(ctx, devs[d], SRC_ASM_GLC, log, sizeof(log));
            print_log("__asm__ glc", err, log);

            clReleaseContext(ctx);
            printf("\n");
        }
        free(devs);
        printf("\n");
    }
    free(plats);
    return 0;
}
