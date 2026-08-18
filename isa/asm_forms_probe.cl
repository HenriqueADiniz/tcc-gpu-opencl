/*
 * Formas de inline asm a validar no ISA antes de virarem benchmark.
 * Objetivo: obter as 4 celulas do desenho fatorial 2x2
 *           {GLC=0, GLC=1} x {loads sobrepostos, loads serializados}
 */

/* (a) GLC=0, serializado — waitcnt apos cada load */
__kernel void probe_glc0_serial(__global const float* A,
                                __global const float* B,
                                __global float* C, const int N)
{
    int i = get_global_id(0);
    if (i >= N) return;
    float a, b;
    __global const float* pa = &A[i];
    __global const float* pb = &B[i];
    __asm__ volatile(
        "flat_load_dword %0, %2\n"
        "s_waitcnt vmcnt(0)\n"
        "flat_load_dword %1, %3\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(a), "=v"(b)
        : "v"(pa), "v"(pb)
        : "memory");
    C[i] = a * b;
}

/* (b) GLC=1, sobreposto — os dois loads emitidos antes de um unico waitcnt */
__kernel void probe_glc1_overlap(__global const float* A,
                                 __global const float* B,
                                 __global float* C, const int N)
{
    int i = get_global_id(0);
    if (i >= N) return;
    float a, b;
    __global const float* pa = &A[i];
    __global const float* pb = &B[i];
    __asm__ volatile(
        "flat_load_dword %0, %2 glc\n"
        "flat_load_dword %1, %3 glc\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(a), "=v"(b)
        : "v"(pa), "v"(pb)
        : "memory");
    C[i] = a * b;
}

/* (c) GLC=1, serializado — deve reproduzir o codegen do volatile */
__kernel void probe_glc1_serial(__global const float* A,
                                __global const float* B,
                                __global float* C, const int N)
{
    int i = get_global_id(0);
    if (i >= N) return;
    float a, b;
    __global const float* pa = &A[i];
    __global const float* pb = &B[i];
    __asm__ volatile(
        "flat_load_dword %0, %2 glc\n"
        "s_waitcnt vmcnt(0)\n"
        "flat_load_dword %1, %3 glc\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(a), "=v"(b)
        : "v"(pa), "v"(pb)
        : "memory");
    C[i] = a * b;
}

/* (d) GLC=0, sobreposto — baseline em C puro, sem asm */
__kernel void probe_glc0_overlap(__global const float* A,
                                 __global const float* B,
                                 __global float* C, const int N)
{
    int i = get_global_id(0);
    if (i >= N) return;
    C[i] = A[i] * B[i];
}
