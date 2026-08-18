/*
 * build_cl_dump.cpp — Compila um arquivo .cl no dispositivo OpenCL escolhido e
 * grava o binario ISA resultante (clGetProgramInfo(CL_PROGRAM_BINARIES)).
 *
 * Ferramenta de apoio: permite iterar sobre formas de kernel e inspecionar o ISA
 * gerado sem precisar montar um benchmark completo.
 *
 * Uso:
 *   ./output/build_cl_dump <arquivo.cl> <saida.bin> [opcoes de build]
 *
 * Escolhe automaticamente a primeira plataforma que tenha um dispositivo GPU.
 * Build:
 *   g++ -Wall -O2 isa/build_cl_dump.cpp -o output/build_cl_dump -lOpenCL
 */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

static char* read_file(const char* path, size_t* len)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nao abriu %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(n + 1);
    size_t rd = fread(buf, 1, n, f);
    buf[rd] = '\0';
    fclose(f);
    if (len) *len = rd;
    return buf;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "uso: %s <arquivo.cl> <saida.bin> [opcoes de build]\n", argv[0]);
        return 2;
    }
    const char* cl_path  = argv[1];
    const char* out_path = argv[2];

    char opts[1024] = {0};
    for (int i = 3; i < argc; i++) {
        strncat(opts, argv[i], sizeof(opts) - strlen(opts) - 2);
        strncat(opts, " ",     sizeof(opts) - strlen(opts) - 2);
    }

    char* src = read_file(cl_path, NULL);
    if (!src) return 1;

    cl_uint np = 0;
    clGetPlatformIDs(0, NULL, &np);
    cl_platform_id* plats = (cl_platform_id*)malloc(np * sizeof(cl_platform_id));
    clGetPlatformIDs(np, plats, NULL);

    cl_platform_id plat = NULL;
    cl_device_id dev = NULL;
    for (cl_uint i = 0; i < np; i++) {
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) {
            plat = plats[i];
            break;
        }
    }
    if (!dev) { fprintf(stderr, "nenhum dispositivo GPU encontrado\n"); return 1; }

    char buf[512];
    clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
    printf("plataforma: %s\n", buf);
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
    printf("dispositivo: %s\n", buf);
    printf("opcoes de build: \"%s\"\n", opts);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateContext: %d\n", err); return 1; }

    const char* srcs[] = { src };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource: %d\n", err); return 1; }

    cl_int berr = clBuildProgram(prog, 1, &dev, opts, NULL, NULL);
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "--- build log ---\n%s\n", log);
        free(log);
    }
    if (berr != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram FALHOU: %d\n", berr);
        return 1;
    }

    size_t bin_size = 0;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
    unsigned char* bin = (unsigned char*)malloc(bin_size);
    unsigned char* bins[] = { bin };
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), bins, NULL);

    FILE* out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "nao gravou %s\n", out_path); return 1; }
    fwrite(bin, 1, bin_size, out);
    fclose(out);
    printf("binario ISA gravado: %s (%lu bytes)\n", out_path, (unsigned long)bin_size);

    clReleaseProgram(prog);
    clReleaseContext(ctx);
    return 0;
}
