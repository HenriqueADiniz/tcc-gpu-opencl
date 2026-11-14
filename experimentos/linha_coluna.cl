/* ==========================================================================
 * Localidade de acesso isolada dentro de um único kernel.
 *
 * Os dois padrões vêm da própria multiplicação de matrizes, onde convivem.
 *
 * Geometria (ls = 16x16): o hardware forma a wavefront com 64 work-items
 * consecutivos do id local achatado (x = col mais rápido), então cada wavefront
 * cobre 4 valores de row e 16 de col. Conferido no gfx803 lendo HW_ID e o lane
 * de cada work-item.
 *
 *   LINHA   A[row*N + k]  — k avança de 4 em 4 bytes (contíguo). Na wavefront
 *                           há 4 endereços, cada um pedido por 16 lanes:
 *                           broadcast, 4 linhas de cache por instrução.
 *                           Conjunto ativo por work-item: N*4 bytes contíguos.
 *
 *   COLUNA  A[k*N + col]  — k avança de N*4 bytes (passo largo). Os lanes pedem
 *                           16 endereços consecutivos: coalescido, 1 linha por
 *                           instrução. Mas k toca uma linha de cache NOVA a cada
 *                           iteração: conjunto ativo de N linhas por work-item.
 *
 *   DISPERSA A[((row+col)%N)*N + k]
 *                         — k avança de 4 em 4 bytes, como em LINHA, mas
 *                           row+col assume 19 valores na wavefront, cada um
 *                           pedido por até 4 lanes: broadcast reduzido e 19
 *                           linhas de cache vivas por instrução.
 *
 * As três variantes fecham um 2x2 incompleto que separa os dois fatores que
 * LINHA e COLUNA variavam juntos:
 *
 *                      broadcast entre lanes   |  espalhado entre lanes
 *   k contíguo                LINHA            |       DISPERSA
 *   k com passo largo           --             |       COLUNA
 *
 * LINHA vs DISPERSA isola o broadcast, com o passo mantido curto.
 * DISPERSA vs COLUNA isola o passo, com os lanes mantidos não-broadcast.
 *
 * Mesma matriz, mesmo número de loads (N por work-item, N^3 no total), mesma
 * saída em tamanho.
 *
 * As três configurações de flag são construídas do MESMO modo (inline asm), de
 * forma que a razão entre elas não carregue custo de construção --- o controle
 * que o experimento do matmul mostrou ser indispensável.
 * ========================================================================== */

#define CORPO(EXPR, FLAGS)                                                   \
    int row = get_global_id(1);                                              \
    int col = get_global_id(0);                                              \
    if (row >= N || col >= N) return;                                        \
    float soma = 0.0f;                                                       \
    for (int k = 0; k < N; k++) {                                            \
        float v;                                                             \
        __global const float* p = &A[EXPR];                                  \
        __asm__ volatile("flat_load_dword %0, %1" FLAGS "\n"                 \
                         "s_waitcnt vmcnt(0)\n"                              \
                         : "=v"(v) : "v"(p) : "memory");                     \
        soma += v;                                                           \
    }                                                                        \
    C[row * N + col] = soma;

__kernel void linha_noflag(__global const float* A, __global float* C, const int N)
{ CORPO(row * N + k, "") }
__kernel void linha_glc(__global const float* A, __global float* C, const int N)
{ CORPO(row * N + k, " glc") }
__kernel void linha_glc_slc(__global const float* A, __global float* C, const int N)
{ CORPO(row * N + k, " glc slc") }

__kernel void coluna_noflag(__global const float* A, __global float* C, const int N)
{ CORPO(k * N + col, "") }
__kernel void coluna_glc(__global const float* A, __global float* C, const int N)
{ CORPO(k * N + col, " glc") }
__kernel void coluna_glc_slc(__global const float* A, __global float* C, const int N)
{ CORPO(k * N + col, " glc slc") }

__kernel void dispersa_noflag(__global const float* A, __global float* C, const int N)
{ CORPO(((row + col) % N) * N + k, "") }
__kernel void dispersa_glc(__global const float* A, __global float* C, const int N)
{ CORPO(((row + col) % N) * N + k, " glc") }
__kernel void dispersa_glc_slc(__global const float* A, __global float* C, const int N)
{ CORPO(((row + col) % N) * N + k, " glc slc") }
