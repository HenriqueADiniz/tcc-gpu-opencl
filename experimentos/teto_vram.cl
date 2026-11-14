/* Kernels do Exp. 2 estendido (teto_vram.cpp). Os tres kernels originais vem
   de slc_l2.cl, compilado junto e sem alteracao; aqui ficam so as leituras com
   mais bytes em voo por wavefront.

   O kernel do Exp. 2 espera cada flat_load_dword (s_waitcnt vmcnt(0) dentro do
   asm): cada wavefront tem 64 x 4 B = 256 B em voo. Abaixo, sem asm, o
   compilador emite as cargas de uma iteracao e so espera ao usa-las (bytes em
   voo por wavefront conferidos na ISA, isa/teto_vram_disasm.txt):
     le_f1_u4     4 x dword    = 1 KB
     le_f4_u1     1 x dwordx4  = 1 KB
     le_f4_u4     4 x dwordx4, mas o escalonador deixa no maximo 2 em voo = 2 KB
     le_f4_u4_nt  idem, com __builtin_nontemporal_load (emite glc slc)
   e, com asm, para fixar 4 x dwordx4 = 4 KB em voo:
     asm_f4_u4, asm_f4_u4_glc_slc
   Mesma assinatura e mesmo percurso de slc_l2.cl: o indice avanca gsize
   elementos por carga, com mascara de potencia de 2. */

#define P(k) ((idx + (k) * gsize) & mask)

__kernel void le_f1_u4(__global const float* A, __global float* out,
                       const int mask, const int R, const int gsize)
{
    int i = get_global_id(0), idx = i;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int it = 0; it < R; it++) {
        a0 += A[P(0)]; a1 += A[P(1)]; a2 += A[P(2)]; a3 += A[P(3)];
        idx = P(4);
    }
    out[i] = a0 + a1 + a2 + a3;
}

__kernel void le_f4_u1(__global const float4* A, __global float* out,
                       const int mask, const int R, const int gsize)
{
    int i = get_global_id(0), idx = i;
    float4 a = 0.0f;
    #pragma unroll 1
    for (int it = 0; it < R; it++) {
        a += A[idx];
        idx = P(1);
    }
    out[i] = a.x + a.y + a.z + a.w;
}

__kernel void le_f4_u4(__global const float4* A, __global float* out,
                       const int mask, const int R, const int gsize)
{
    int i = get_global_id(0), idx = i;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int it = 0; it < R; it++) {
        a0 += A[P(0)]; a1 += A[P(1)]; a2 += A[P(2)]; a3 += A[P(3)];
        idx = P(4);
    }
    float4 s = a0 + a1 + a2 + a3;
    out[i] = s.x + s.y + s.z + s.w;
}

__kernel void le_f4_u4_nt(__global const float4* A, __global float* out,
                          const int mask, const int R, const int gsize)
{
    int i = get_global_id(0), idx = i;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int it = 0; it < R; it++) {
        a0 += __builtin_nontemporal_load(&A[P(0)]);
        a1 += __builtin_nontemporal_load(&A[P(1)]);
        a2 += __builtin_nontemporal_load(&A[P(2)]);
        a3 += __builtin_nontemporal_load(&A[P(3)]);
        idx = P(4);
    }
    float4 s = a0 + a1 + a2 + a3;
    out[i] = s.x + s.y + s.z + s.w;
}

__kernel void asm_f4_u4(__global const float4* A, __global float* out,
                        const int mask, const int R, const int gsize)
{
    int i = get_global_id(0), idx = i;
    float4 a = 0.0f;
    for (int it = 0; it < R; it++) {
        float4 v0, v1, v2, v3;
        __global const float4 *p0 = &A[P(0)], *p1 = &A[P(1)], *p2 = &A[P(2)], *p3 = &A[P(3)];
        __asm__ volatile("flat_load_dwordx4 %0, %4\n flat_load_dwordx4 %1, %5\n"
                         " flat_load_dwordx4 %2, %6\n flat_load_dwordx4 %3, %7\n"
                         " s_waitcnt vmcnt(0)\n"
                         : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3)
                         : "v"(p0), "v"(p1), "v"(p2), "v"(p3) : "memory");
        a += v0 + v1 + v2 + v3;
        idx = P(4);
    }
    out[i] = a.x + a.y + a.z + a.w;
}

__kernel void asm_f4_u4_glc_slc(__global const float4* A, __global float* out,
                                const int mask, const int R, const int gsize)
{
    int i = get_global_id(0), idx = i;
    float4 a = 0.0f;
    for (int it = 0; it < R; it++) {
        float4 v0, v1, v2, v3;
        __global const float4 *p0 = &A[P(0)], *p1 = &A[P(1)], *p2 = &A[P(2)], *p3 = &A[P(3)];
        __asm__ volatile("flat_load_dwordx4 %0, %4 glc slc\n flat_load_dwordx4 %1, %5 glc slc\n"
                         " flat_load_dwordx4 %2, %6 glc slc\n flat_load_dwordx4 %3, %7 glc slc\n"
                         " s_waitcnt vmcnt(0)\n"
                         : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3)
                         : "v"(p0), "v"(p1), "v"(p2), "v"(p3) : "memory");
        a += v0 + v1 + v2 + v3;
        idx = P(4);
    }
    out[i] = a.x + a.y + a.z + a.w;
}
