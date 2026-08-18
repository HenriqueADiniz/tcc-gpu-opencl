/*
 * =============================================================================
 *  algoritmos_cache.cl — Scan, primos e crivo sob os niveis de cache do GFX8
 * =============================================================================
 *
 *  Cada algoritmo e gerado tres vezes por macro, mudando apenas o modificador
 *  da instrucao de load:
 *      ""          -> DC500000  GLC=0 SLC=0  (L1 -> L2 -> VRAM)
 *      " glc"      -> DC510000  GLC=1 SLC=0  (contorna o L1)
 *      " glc slc"  -> DC530000  GLC=1 SLC=1  (contorna L1; nao-alocacao no L2)
 *
 *  As saidas do bloco asm usam "=&v" (earlyclobber). Sem isso o compilador pode
 *  alocar um registrador de saida sobre um registrador de endereco ainda nao
 *  consumido, corrompendo o endereco e causando falha de pagina na GPU.
 * =============================================================================
 */

/* Um load de 32 bits com modificador de cache parametrizado. */
#define LOAD1(dst, ptr, F)                          \
    __asm__ volatile(                               \
        "flat_load_dword %0, %1" F "\n"             \
        "s_waitcnt vmcnt(0)\n"                      \
        : "=&v"(dst) : "v"(ptr) : "memory")

#define TAM_BLOCO 256

/* =====================================================================
 * ALGORITMO A — SCAN (soma de prefixos inclusiva)
 *
 * Estrutura classica em tres fases (Blelloch; Harris et al.):
 *   1. scan_bloco  — scan inclusivo dentro de cada workgroup, via LDS,
 *                    gravando o total de cada bloco em `somas`
 *   2. recursao    — o mesmo kernel aplicado ao vetor `somas`
 *   3. add_offset  — soma a cada elemento o prefixo exclusivo do seu bloco
 *
 * O reuso de dados globais e praticamente unitario: cada elemento e lido
 * uma vez por fase. O algoritmo depende do LDS, nao do cache, para o reuso
 * dentro do bloco --- o que faz dele um contraponto util a multiplicacao
 * de matrizes.
 * ===================================================================== */

#define K_SCAN(NOME_SCAN, NOME_ADD, F)                                        \
__kernel void NOME_SCAN(__global const uint* restrict entrada,                \
                        __global uint* restrict saida,                        \
                        __global uint* restrict somas,                        \
                        __local uint* tmp,                                    \
                        const uint n)                                         \
{                                                                             \
    uint gid = get_global_id(0);                                              \
    uint lid = get_local_id(0);                                               \
    uint lsz = get_local_size(0);                                             \
                                                                              \
    uint v = 0;                                                               \
    if (gid < n) {                                                            \
        __global const uint* p = &entrada[gid];                               \
        LOAD1(v, p, F);                                                       \
    }                                                                         \
    tmp[lid] = v;                                                             \
    barrier(CLK_LOCAL_MEM_FENCE);                                             \
                                                                              \
    /* Hillis-Steele: todos leem, barreira, todos escrevem, barreira. */      \
    for (uint d = 1; d < lsz; d <<= 1) {                                      \
        uint x = (lid >= d) ? tmp[lid - d] : 0u;                              \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
        tmp[lid] += x;                                                        \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
    }                                                                         \
                                                                              \
    if (gid < n) saida[gid] = tmp[lid];                                       \
    if (lid == lsz - 1) somas[get_group_id(0)] = tmp[lid];                    \
}                                                                             \
                                                                              \
__kernel void NOME_ADD(__global uint* restrict dados,                         \
                       __global const uint* restrict prefixos,                \
                       const uint n)                                          \
{                                                                             \
    uint gid = get_global_id(0);                                              \
    uint bid = get_group_id(0);                                               \
    if (gid >= n || bid == 0) return;                                         \
    /* prefixo exclusivo do bloco = scan inclusivo das somas em bid-1 */       \
    uint off, val;                                                            \
    __global const uint* po = &prefixos[bid - 1];                             \
    __global const uint* pd = &dados[gid];                                    \
    LOAD1(off, po, F);                                                        \
    LOAD1(val, pd, F);                                                        \
    dados[gid] = val + off;                                                   \
}

K_SCAN(scan_bloco_L1,   add_offset_L1,   "")
K_SCAN(scan_bloco_L2,   add_offset_L2,   " glc")
K_SCAN(scan_bloco_VRAM, add_offset_VRAM, " glc slc")


/* =====================================================================
 * ALGORITMO B — PRIMOS POR DIVISAO SUCESSIVA
 *
 * Um work-item testa a primalidade de um candidato por divisoes sucessivas
 * ate a raiz quadrada. E deliberadamente COMPUTE-BOUND: ha exatamente um
 * load global por work-item e centenas de operacoes de ALU.
 *
 * Serve como CONTROLE NEGATIVO do estudo: se o nivel de cache nao alterar o
 * tempo desta carga, confirma-se que o efeito medido nas demais vem do
 * trafego de memoria, e nao de algum artefato das flags ou do inline asm.
 * ===================================================================== */

#define K_PRIMOS(NOME, F)                                                     \
__kernel void NOME(__global const uint* restrict candidatos,                  \
                   __global uint* restrict resultado,                         \
                   const uint n)                                              \
{                                                                             \
    uint gid = get_global_id(0);                                              \
    if (gid >= n) return;                                                     \
    uint v;                                                                   \
    __global const uint* p = &candidatos[gid];                                \
    LOAD1(v, p, F);                                                           \
                                                                              \
    uint primo = (v >= 2u) ? 1u : 0u;                                         \
    if (v >= 2u) {                                                            \
        for (uint d = 2u; d * d <= v; d++) {                                  \
            if (v % d == 0u) { primo = 0u; break; }                           \
        }                                                                     \
    }                                                                         \
    resultado[gid] = primo;                                                   \
}

K_PRIMOS(primos_L1,   "")
K_PRIMOS(primos_L2,   " glc")
K_PRIMOS(primos_VRAM, " glc slc")


/* =====================================================================
 * ALGORITMO C — CRIVO DE ERATOSTENES SEGMENTADO
 *
 * Cada work-item recebe um segmento disjunto do intervalo e marca nele os
 * multiplos de cada primo pequeno. Como os segmentos nao se sobrepoem, nao
 * ha condicao de corrida nas escritas.
 *
 * O padrao de acesso e o oposto do dos primos por divisao: a tabela de
 * primos pequenos (poucos KB) e lida por TODOS os work-items, o que gera
 * reuso temporal altissimo sobre um conjunto de trabalho que cabe no L1.
 * E, portanto, o candidato natural a sofrer mais com o bypass do L1.
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
        /* primeiro multiplo de p dentro do segmento, nunca abaixo de p*p */  \
        uint m = ((ini + p - 1u) / p) * p;                                    \
        if (m < p * p) m = p * p;                                             \
        for (; m < fim; m += p) marcas[m - base] = 1;                         \
    }                                                                         \
}

K_CRIVO(crivo_L1,   "")
K_CRIVO(crivo_L2,   " glc")
K_CRIVO(crivo_VRAM, " glc slc")
