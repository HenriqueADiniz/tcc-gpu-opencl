/*
 * sonda_wavefront.cpp — como o gfx803 forma as wavefronts de um work-group 16x16?
 *
 * Cada work-item grava o HW_ID da sua wave (s_getreg, uniforme na wave) e o seu
 * lane real (mbcnt). O programa confere, em 16 work-groups, que cada wave reúne
 * os ids locais achatados 64w..64w+63 (x mais rápido) e que lane == id % 64.
 * Consequência usada no Exp. 4: uma wavefront cobre 4 valores de row e 16 de col.
 *
 *   source setup/env_rocm.sh
 *   g++ -std=c++17 -Wall -Wextra -O2 experimentos/sonda_wavefront.cpp -o output/sonda_wavefront -lOpenCL
 *   ./output/sonda_wavefront
 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <map>
#include <set>
#include <vector>
static const char* src = R"CLC(
__kernel void probe(__global uint* hw, __global uint* lane) {
    uint g = get_global_id(1) * get_global_size(0) + get_global_id(0);
    uint id;
    __asm__ volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(id));
    hw[g] = id;
    lane[g] = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
)CLC";
int main() {
    cl_platform_id p; clGetPlatformIDs(1, &p, NULL);
    cl_device_id d; clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    char nome[128]; clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof nome, nome, NULL);
    cl_int e; cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueue(c, d, 0, &e);
    cl_program pr = clCreateProgramWithSource(c, 1, &src, NULL, &e);
    if (clBuildProgram(pr, 1, &d, "", NULL, NULL) != CL_SUCCESS) {
        char log[8192]; clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        printf("build falhou:\n%s\n", log); return 1;
    }
    cl_kernel k = clCreateKernel(pr, "probe", &e);
    const size_t N = 64, tot = N * N;          // 16 work-groups de 16x16
    cl_mem bh = clCreateBuffer(c, CL_MEM_WRITE_ONLY, tot * 4, NULL, &e);
    cl_mem bl = clCreateBuffer(c, CL_MEM_WRITE_ONLY, tot * 4, NULL, &e);
    clSetKernelArg(k, 0, sizeof bh, &bh); clSetKernelArg(k, 1, sizeof bl, &bl);
    size_t gs[2] = {N, N}, ls[2] = {16, 16};
    clEnqueueNDRangeKernel(q, k, 2, NULL, gs, ls, 0, NULL, NULL);
    std::vector<cl_uint> hw(tot), lane(tot);
    clEnqueueReadBuffer(q, bh, CL_TRUE, 0, tot * 4, hw.data(), 0, NULL, NULL);
    clEnqueueReadBuffer(q, bl, CL_TRUE, 0, tot * 4, lane.data(), 0, NULL, NULL);
    printf("dispositivo: %s\n", nome);
    int ok_lane = 0, ok_wave = 0, grupos = 0;
    for (size_t gy = 0; gy < N / 16; gy++) for (size_t gx = 0; gx < N / 16; gx++) {
        grupos++;
        std::map<cl_uint, std::set<int>> por_wave;       // HW_ID -> ids locais achatados
        bool lane_bate = true;
        for (int ly = 0; ly < 16; ly++) for (int lx = 0; lx < 16; lx++) {
            size_t g = (gy * 16 + ly) * N + gx * 16 + lx;
            int flat = ly * 16 + lx;
            por_wave[hw[g]].insert(flat);
            if (lane[g] != (cl_uint)(flat % 64)) lane_bate = false;
        }
        bool wave_bate = por_wave.size() == 4;
        for (auto& [h, s] : por_wave) {
            int w = *s.begin() / 64;
            if (s.size() != 64 || *s.begin() != 64 * w || *s.rbegin() != 64 * w + 63) wave_bate = false;
        }
        ok_lane += lane_bate; ok_wave += wave_bate;
        if (gy == 0 && gx == 0)
            for (auto& [h, s] : por_wave)
                printf("  WG(0,0) wave HW_ID=0x%08x: ids locais %d..%d -> linhas ly %d..%d, colunas lx 0..15\n",
                       h, *s.begin(), *s.rbegin(), *s.begin() / 16, *s.rbegin() / 16);
    }
    printf("work-groups com lane == id_achatado %% 64: %d/%d\n", ok_lane, grupos);
    printf("work-groups com 4 waves = blocos {64w..64w+63}: %d/%d\n", ok_wave, grupos);
}
