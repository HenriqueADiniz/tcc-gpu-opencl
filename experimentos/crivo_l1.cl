/*
 * =============================================================================
 *  crivo_l1.cl — Recuperando a sensibilidade ao L1 num algoritmo de primos
 * =============================================================================
 *
 *  O Experimento 8 mediu 1,00x para o crivo segmentado em todos os niveis de
 *  cache. O diagnostico foi duplo:
 *
 *    (i)  o load da tabela de primos e UNIFORME — os 64 lanes de uma wavefront
 *         leem o mesmo endereco, e o coalescedor resolve em 1 requisicao;
 *    (ii) o algoritmo faz seg/p ESCRITAS por load, de modo que o tempo e
 *         dominado por store (~19,6 M stores contra ~1,7 M loads), e as flags
 *         de coerencia so existem em instrucoes de load.
 *
 *  Este experimento troca a formulacao do teste de primalidade por uma que
 *  inverte essa razao, mantendo o algoritmo real e verificavel:
 *
 *    DIVISAO SUCESSIVA TABELADA (trial division por tabela de primos)
 *      cada work-item testa um candidato contra TODOS os primos ate sqrt(vmax);
 *      o teste de divisibilidade usa a forma exata de Granlund-Montgomery,
 *          p | v   <=>   v * inv_p (mod 2^32)  <=  floor((2^32-1)/p),
 *      com inv_p = p^-1 mod 2^32 pre-computado no anfitriao.
 *
 *      Custo por primo: 2 LOADS e ~5 instrucoes de ALU. Nao ha divisao inteira
 *      (que no GCN custa ~25 instrucoes) e nao ha escrita no laco. O trip count
 *      e FIXO em n_primos para todos os lanes, logo nao ha divergencia.
 *
 *  Duas variantes isolam exatamente o fator (i), com trabalho identico:
 *      tdiv_unif — todos os lanes leem tabela[j]        (acesso uniforme)
 *      tdiv_rot  — o lane l le tabela[(j + 37l) mod n]  (acesso disperso)
 *
 *  Working set reusado = 2 * n_primos * 4 B ~= 2,4 KB, folgadamente dentro dos
 *  16 KB do L1 por CU. Reuso de leitura = numero de candidatos (~1 M).
 *
 *  O crivo segmentado original e mantido no mesmo programa como linha de base.
 * =============================================================================
 */

/* Um load de 32 bits com modificador de cache parametrizado.
   "=&v" (earlyclobber) e obrigatorio: sem ele o compilador pode alocar o
   registrador de saida sobre um registrador de endereco ainda nao consumido. */
#define LOAD1(dst, ptr, F)                          \
    __asm__ volatile(                               \
        "flat_load_dword %0, %1" F "\n"             \
        "s_waitcnt vmcnt(0)\n"                      \
        : "=&v"(dst) : "v"(ptr) : "memory")


/* =====================================================================
 * A — DIVISAO SUCESSIVA TABELADA
 *
 * ROT define o indice inicial de cada lane na tabela:
 *   ROT_UNIF -> 0             (todos os lanes no mesmo endereco)
 *   ROT_DISP -> 37*lane mod n (lanes espalhados por toda a tabela)
 *
 * O passo 37 e primo com os tamanhos de tabela usados, o que garante que os
 * 64 lanes caiam em 64 posicoes distintas enquanto n_primos >= 64.
 * ===================================================================== */

#define ROT_UNIF (lane * 0u)
#define ROT_DISP ((lane * 37u) % n_primos)

/* Terceira variante: alem de espalhar os lanes, espalha os WORKGROUPS.
 *
 * ROT_DISP nao basta para variar o conjunto de trabalho ATIVO. Como todos os
 * work-items avancam com i++ em lock-step, as wavefronts de um mesmo CU
 * percorrem a tabela em fase, e a janela viva a qualquer instante e de ~64
 * linhas (4 KB) qualquer que seja o tamanho da tabela — que assim nunca
 * pressiona os 16 KB do L1.
 *
 * Multiplicar o group_id pela constante de Knuth (2654435761 = 2^32/phi)
 * distribui os grupos por toda a tabela, de modo que as wavefronts residentes
 * num mesmo CU leem regioes disjuntas. O conjunto ativo passa a escalar com o
 * tamanho da tabela, que e o que a varredura precisa variar.
 *
 * O trabalho e o trip count sao identicos aos das outras variantes: o laco
 * percorre a tabela inteira, so muda o ponto de partida. */
#define ROT_DSYN (((get_group_id(0) * 2654435761u) + lane * 37u) % n_primos)

#define K_TDIV(NOME, ROT, F)                                                  \
__kernel void NOME(__global const uint* restrict tab_inv,                     \
                   __global const uint* restrict tab_lim,                     \
                   const uint n_primos,                                       \
                   __global uchar* restrict marcas,                           \
                   const uint base,                                           \
                   const uint n_cand)                                         \
{                                                                             \
    uint gid = get_global_id(0);                                              \
    if (gid >= n_cand) return;                                                \
                                                                              \
    uint v = base + gid;                                                      \
    uint lane = get_local_id(0) & 63u;                                        \
                                                                              \
    /* A tabela contem apenas primos impares; a paridade sai do laco. */      \
    uint composto = ((v & 1u) == 0u) ? 1u : 0u;                               \
                                                                              \
    uint i = ROT;                                                             \
    for (uint j = 0; j < n_primos; j++) {                                     \
        uint qi, li;                                                          \
        __global const uint* pq = &tab_inv[i];                                \
        __global const uint* pl = &tab_lim[i];                                \
        LOAD1(qi, pq, F);                                                     \
        LOAD1(li, pl, F);                                                     \
        if (v * qi <= li) composto = 1u;                                      \
        i++;                                                                  \
        if (i >= n_primos) i = 0u;                                            \
    }                                                                         \
    marcas[gid] = (uchar)composto;                                            \
}

K_TDIV(tdiv_unif_L1,   ROT_UNIF, "")
K_TDIV(tdiv_unif_L2,   ROT_UNIF, " glc")
K_TDIV(tdiv_unif_VRAM, ROT_UNIF, " glc slc")

K_TDIV(tdiv_rot_L1,    ROT_DISP, "")
K_TDIV(tdiv_rot_L2,    ROT_DISP, " glc")
K_TDIV(tdiv_rot_VRAM,  ROT_DISP, " glc slc")

K_TDIV(tdiv_dsyn_L1,   ROT_DSYN, "")
K_TDIV(tdiv_dsyn_L2,   ROT_DSYN, " glc")
K_TDIV(tdiv_dsyn_VRAM, ROT_DSYN, " glc slc")


/* =====================================================================
 * B — CRIVO SEGMENTADO (linha de base, identico ao Experimento 8)
 *
 * Mantido para que a comparacao entre as duas formulacoes seja feita na
 * mesma execucao, no mesmo estado termico da GPU.
 * ===================================================================== */

#define K_CRIVO(NOME, F)                                                      \
__kernel void NOME(__global const uint* restrict primos_pequenos,             \
                   const uint n_primos,                                       \
                   __global uchar* restrict marcas,                           \
                   const uint base,                                           \
                   const uint seg_tam)                                        \
{                                                                             \
    uint gid = get_global_id(0);                                              \
    uint ini = base + gid * seg_tam;                                          \
    uint fim = ini + seg_tam;                                                 \
                                                                              \
    for (uint i = 0; i < n_primos; i++) {                                     \
        uint p;                                                               \
        __global const uint* pp = &primos_pequenos[i];                        \
        LOAD1(p, pp, F);                                                      \
        if (p > 65535u || p * p > fim) break;                                 \
                                                                              \
        uint m = ((ini + p - 1u) / p) * p;                                    \
        if (m < p * p) m = p * p;                                             \
        for (; m < fim; m += p) marcas[m - base] = 1;                         \
    }                                                                         \
}

K_CRIVO(crivo_L1,   "")
K_CRIVO(crivo_L2,   " glc")
K_CRIVO(crivo_VRAM, " glc slc")
