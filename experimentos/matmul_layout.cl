/* ==========================================================================
 * Multiplicação de matrizes: o efeito do LAYOUT de B sobre a localidade.
 *
 * Mesmo produto C = A*B, mesmos FLOPs, mesmos dados. Muda só como B está
 * disposta na memória e, com ela, o padrão de acesso do segundo operando:
 *
 *   PADRAO      A[row*N+k] * B[k*N+col]
 *               (ls = 16x16: cada wavefront cobre 4 valores de row e 16 de col)
 *               A: k contíguo, broadcast --- 4 endereços por instrução, cada um
 *                  pedido por 16 lanes
 *               B: k com passo de N*4 bytes; lanes em 16 endereços consecutivos
 *                  --- coalescido, 1 linha de cache por instrução
 *
 *   TRANSPOSTA  A[row*N+k] * Bt[col*N+k]      (Bt = B transposta no anfitrião)
 *               A: idêntico ao caso acima
 *               Bt: k contíguo dentro da linha de cada lane, mas cada col numa
 *                   linha distinta --- espalhado, 16 linhas por instrução
 *
 * Transpor B é a otimização clássica de matmul em CPU: troca um percurso por
 * coluna por um percurso por linha. Na GPU o efeito não é óbvio, porque o que
 * se ganha em contiguidade por lane se perde em coalescência entre lanes.
 *
 * As três configurações de flag usam a mesma construção em inline assembly,
 * com os dois loads emitidos antes de um único s_waitcnt, de modo que a razão
 * entre elas não carregue custo de construção nem perda de sobreposição.
 * ========================================================================== */

#define CORPO(EXPR_B, FLAGS)                                                  \
    int row = get_global_id(1);                                               \
    int col = get_global_id(0);                                               \
    if (row >= N || col >= N) return;                                         \
    float soma = 0.0f;                                                        \
    for (int k = 0; k < N; k++) {                                             \
        float a, b;                                                           \
        __global const float* pa = &A[row * N + k];                           \
        __global const float* pb = &B[EXPR_B];                                \
        __asm__ volatile("flat_load_dword %0, %2" FLAGS "\n"                  \
                         "flat_load_dword %1, %3" FLAGS "\n"                  \
                         "s_waitcnt vmcnt(0)\n"                               \
                         : "=v"(a), "=v"(b)                                   \
                         : "v"(pa), "v"(pb) : "memory");                      \
        soma += a * b;                                                        \
    }                                                                         \
    C[row * N + col] = soma;

__kernel void padrao_noflag(__global const float* A, __global const float* B,
                            __global float* C, const int N)
{ CORPO(k * N + col, "") }
__kernel void padrao_glc(__global const float* A, __global const float* B,
                         __global float* C, const int N)
{ CORPO(k * N + col, " glc") }
__kernel void padrao_glc_slc(__global const float* A, __global const float* B,
                             __global float* C, const int N)
{ CORPO(k * N + col, " glc slc") }

__kernel void transp_noflag(__global const float* A, __global const float* B,
                            __global float* C, const int N)
{ CORPO(col * N + k, "") }
__kernel void transp_glc(__global const float* A, __global const float* B,
                         __global float* C, const int N)
{ CORPO(col * N + k, " glc") }
__kernel void transp_glc_slc(__global const float* A, __global const float* B,
                             __global float* C, const int N)
{ CORPO(col * N + k, " glc slc") }
