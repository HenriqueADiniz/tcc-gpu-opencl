/* Varre um buffer repetidas vezes. Se SLC=1 realmente ignorasse a L2, o caso
   "cabe na L2" desabaria para tempo de VRAM. Se SLC for so dica de eviccao,
   os tempos ficam colados no caso que cabe. Todos com asm: mesma codegen. */

__kernel void sweep_noflag(__global const float* A, __global float* out,
                           const int mask, const int R, const int gsize)
{
    int i = get_global_id(0); float acc = 0.0f; int idx = i;
    for (int it = 0; it < R; it++) {
        float v; __global const float* p = &A[idx];
        __asm__ volatile("flat_load_dword %0, %1\n s_waitcnt vmcnt(0)\n"
                         : "=v"(v) : "v"(p) : "memory");
        acc += v; idx = (idx + gsize) & mask;
    }
    out[i] = acc;
}

__kernel void sweep_glc(__global const float* A, __global float* out,
                        const int mask, const int R, const int gsize)
{
    int i = get_global_id(0); float acc = 0.0f; int idx = i;
    for (int it = 0; it < R; it++) {
        float v; __global const float* p = &A[idx];
        __asm__ volatile("flat_load_dword %0, %1 glc\n s_waitcnt vmcnt(0)\n"
                         : "=v"(v) : "v"(p) : "memory");
        acc += v; idx = (idx + gsize) & mask;
    }
    out[i] = acc;
}

__kernel void sweep_glc_slc(__global const float* A, __global float* out,
                            const int mask, const int R, const int gsize)
{
    int i = get_global_id(0); float acc = 0.0f; int idx = i;
    for (int it = 0; it < R; it++) {
        float v; __global const float* p = &A[idx];
        __asm__ volatile("flat_load_dword %0, %1 glc slc\n s_waitcnt vmcnt(0)\n"
                         : "=v"(v) : "v"(p) : "memory");
        acc += v; idx = (idx + gsize) & mask;
    }
    out[i] = acc;
}
