/*
 * =============================================================================
 *  Hierarquia de memoria em GPU AMD — Multiplicacao de matrizes por caminho
 *  de cache (GCN3 / GFX8 / gfx803 — Ellesmere, RX 580)
 * =============================================================================
 *
 *  Controle de cache no GFX8: a instrucao flat_load_dword carrega dois bits
 *  que selecionam o caminho de memoria (GCN3 ISA, secao 9.1):
 *
 *    GLC=0            le pelo vL1 (16 KB por CU) — padrao
 *    GLC=1            ignora o vL1, atende a partir do L2 (2 MB)
 *    GLC=1 + SLC=1    ignora o vL1 e marca a linha do L2 como evict-first
 *                     (streaming). NAO ignora a L2 — ver abaixo.
 *
 *  NAO EXISTE bypass de L2 pelo lado do shader no GFX8. Medido em
 *  experimentos/slc_l2.cpp: varrendo um buffer de 1 MB, que cabe na L2 de
 *  2 MB, os loads com GLC=1 e SLC=1 sustentam 548 GB/s. O mesmo codigo sobre
 *  um buffer de 64 MB, que nao cabe na L2, satura em 222 GB/s — a banda real
 *  da GDDR5 desta placa. Se SLC=1 desligasse a L2, o caso de 1 MB teria caido
 *  para esses mesmos 222 GB/s. Ficou 2.5x acima, logo a L2 continua servindo.
 *  SLC=1 e politica de eviccao (evict-first), nao bypass. O cenario "direto a
 *  VRAM" da formulacao original do trabalho nao e alcancavel nesta
 *  arquitetura, e o kernel 6 nao deve ser descrito assim.
 *
 *  IMPORTANTE — o que NAO controla essas flags neste alvo:
 *
 *    volatile                    e uma propriedade do modelo de memoria da
 *                                linguagem, nao um seletor de cache. O
 *                                mapeamento volatile -> GLC e decisao de
 *                                codegen do backend e varia por compilador:
 *                                nao gera flag no AMDOCL (Windows) nem no RGA
 *                                usado para gfx803, mas gera GLC=1 no
 *                                LLVM do ROCm 5.2. Nao e um mecanismo estavel.
 *                                Os kernels 2 e 3 preservam essa hipotese
 *                                original: nao sao redundantes, sao o objeto
 *                                da comparacao multiplataforma. No ROCm o
 *                                volatile ainda serializa os loads (um
 *                                s_waitcnt por load), entao o tempo deles
 *                                mistura bypass com perda de sobreposicao —
 *                                o controle correto e o kernel 4.
 *
 *    __attribute__((address_space(1)))
 *                                nao faz bypass. No mapeamento do AMDGPU,
 *                                address space 1 E o espaco global — e o
 *                                mesmo que __global. Em modo OpenCL o Clang
 *                                ainda trata os dois como espacos distintos e
 *                                rejeita a conversao ("changes address space
 *                                of pointer"); quando declarado direto no
 *                                parametro do kernel, compila e gera ISA sem
 *                                nenhuma flag de cache.
 *
 *    STOREs                      nao saem com GLC=1 por padrao (as 50 formas
 *                                de store nos binarios deste projeto tem
 *                                GLC=0). As escritas alcancam o L2 porque o
 *                                vL1 do GCN e write-through por construcao,
 *                                nao por causa do bit GLC.
 *
 *  O mecanismo usado aqui e inline asm com as flags explicitas, mais
 *  __builtin_nontemporal_load como validacao cruzada sem asm. Ambos exigem um
 *  compilador OpenCL baseado em Clang/LLVM: no ROCm/Linux funcionam; no
 *  AMDOCL do Windows o inline asm nao compila ("use of undeclared identifier
 *  '__asm__'"), o que e em si um resultado da comparacao multiplataforma.
 *
 *  Compilar (Linux / ROCm):
 *    source setup/env_rocm.sh
 *    g++ -Wall -Wextra -O2 experimentos/matmul.cpp -o output/matmul -lOpenCL
 *
 *  Compilar (Windows / MinGW — kernels 2, 3 e 4 NAO compilam la):
 *    g++ -Wall -Wextra -O2 -D__USE_MINGW_ANSI_STDIO=1
 *        -Wl,--enable-stdcall-fixup
 *        -I "C:\Program Files (x86)\OCL_SDK_Light\include"
 *        matmul.cpp -o output\matmul.exe "C:\Windows\System32\OpenCL.dll"
 *
 *  Inspecionar a ISA gerada:
 *    llvm-objdump -d --mcpu=gfx803 isa/matmul_gfx803.bin | grep flat_load
 *
 * =============================================================================
 */

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <CL/cl.h>

/* ------------------------------------------------------------------------- */
/*  3 Kernels de multiplicacao de matrizes C = A * B                         */
/*  Cada um com nivel diferente de cache                                     */
/*  Matrizes quadradas NxN, row-major                                        */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
"/* ====================================================================== */\n"
"/* DESENHO EXPERIMENTAL                                                   */\n"
"/*                                                                        */\n"
"/* Os quatro primeiros kernels calculam C = A*B com o MESMO algoritmo     */\n"
"/* (naive, um work-item por elemento). Variam apenas no caminho de cache  */\n"
"/* dos LOADs de A e B, controlado pelos bits GLC/SLC da instrucao         */\n"
"/* flat_load_dword (GCN3/GFX8, secao 9.1 do ISA da AMD):                  */\n"
"/*                                                                        */\n"
"/*   GLC=0            -> le do vL1 (comportamento padrao)                 */\n"
"/*   GLC=1            -> ignora o vL1, le do L2                           */\n"
"/*   GLC=1 e SLC=1    -> ignora vL1 e marca a linha do L2 como            */\n"
"/*                       \"streaming\" (evict-first), aproximando um        */\n"
"/*                       acesso direto a VRAM                             */\n"
"/*                                                                        */\n"
"/* Os kernels 2 e 3 sao a HIPOTESE ORIGINAL do trabalho, preservada: a    */\n"
"/* tentativa de obter bypass com o qualificador volatile. Eles nao sao    */\n"
"/* redundantes — sao o objeto da comparacao multiplataforma. O mesmo      */\n"
"/* fonte produz ISA diferente conforme o compilador OpenCL:               */\n"
"/*                                                                        */\n"
"/*   AMDOCL (driver Windows)  -> sem flag   (GLC=0) — hipotese refutada   */\n"
"/*   RGA usado para gfx803    -> sem flag   (GLC=0)                       */\n"
"/*   Clang/LLVM do ROCm 5.2   -> flag glc   (GLC=1) — hipotese confirmada */\n"
"/*                                                                        */\n"
"/* Ou seja: volatile nao e um seletor de cache, e sim uma propriedade do  */\n"
"/* modelo de memoria cuja traducao para GLC e decisao de codegen do       */\n"
"/* backend. Por isso os kernels 5 a 7, que pedem as flags explicitamente, */\n"
"/* sao necessarios para uma medida reproduzivel — mas os kernels 2 e 3    */\n"
"/* documentam o percurso e sustentam o resultado multiplataforma.         */\n"
"/*                                                                        */\n"
"/* O kernel matmul_asm_baseline existe como CONTROLE: usa inline asm com  */\n"
"/* GLC=0, ou seja, tem a mesma perda de otimizacao dos kernels 5 e 6      */\n"
"/* (o bloco asm bloqueia unroll/vetorizacao e forca um s_waitcnt por      */\n"
"/* iteracao), mas nao faz bypass. Sem esse controle, comparar o kernel 1  */\n"
"/* com o kernel 5 confunde dois efeitos: o custo da barreira de           */\n"
"/* otimizacao e o efeito do bit GLC. As comparacoes validas sao:          */\n"
"/*                                                                        */\n"
"/*   kernel 1 vs 7 -> MEDIDA PRINCIPAL: ISA identica, so os bits glc/slc  */\n"
"/*   kernel 1 vs 2 -> efeito do volatile neste compilador                 */\n"
"/*   kernel 2 vs 3 -> repetibilidade (fontes identicos)                   */\n"
"/*   kernel 1 vs 4 -> custo do inline asm (barreira de otimizacao)        */\n"
"/*   kernel 4 vs 5 -> efeito isolado do bypass de L1  (GLC)               */\n"
"/*   kernel 5 vs 6 -> efeito isolado do streaming em L2 (SLC)             */\n"
"/* ====================================================================== */\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 1 — NORMAL: L1 -> L2 -> VRAM                                    */\n"
"/* ISA esperada: flat_load_dword v, v[..]           (GLC=0, SLC=0)        */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_via_L1(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 2 — HIPOTESE ORIGINAL: volatile                                 */\n"
"/* Preservado como registro do percurso cientifico e como objeto da       */\n"
"/* comparacao multiplataforma. O MESMO fonte gera ISA diferente conforme  */\n"
"/* o compilador:                                                          */\n"
"/*   AMDOCL / Windows : flat_load_dword v, v[..]       (DC500000, GLC=0)  */\n"
"/*   ROCm 5.2 / Linux : flat_load_dword v, v[..] glc   (DC510000, GLC=1)  */\n"
"/* O ponteiro de saida tambem e volatile; no GFX8 o store sai sem glc de  */\n"
"/* qualquer modo, porque o vL1 e write-through por construcao.            */\n"
"/*                                                                        */\n"
"/* ATENCAO ao interpretar o tempo deste kernel: sob o ROCm o volatile     */\n"
"/* faz DUAS coisas de uma vez. Alem do glc, ele serializa os acessos —    */\n"
"/* emite um s_waitcnt vmcnt(0) apos CADA load, enquanto o kernel 1 emite  */\n"
"/* um unico waitcnt para os dois loads sobrepostos. O custo medido        */\n"
"/* portanto mistura bypass de L1 com perda de sobreposicao de latencia.   */\n"
"/* O controle correto para este kernel e o kernel 4 (asm serializado com  */\n"
"/* GLC=0), nao o kernel 1. Confirmacao experimental: k2 cai praticamente  */\n"
"/* em cima do k5 (asm com glc, tambem serializado).                       */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_via_L2(\n"
"    __global volatile const float* A,\n"
"    __global volatile const float* B,\n"
"    __global volatile float*       C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 3 — HIPOTESE ORIGINAL: volatile \"agressivo\"                     */\n"
"/* Fonte deliberadamente IDENTICO ao kernel 2. Na formulacao original     */\n"
"/* supunha-se que um uso mais agressivo de volatile escalaria de L2 para  */\n"
"/* VRAM; a analise da ISA mostrou os dois kernels byte a byte iguais, o   */\n"
"/* que refutou a hipotese. Mantido por dois motivos: e a evidencia dessa  */\n"
"/* refutacao e, na bancada, da a repetibilidade da medida (dois kernels   */\n"
"/* identicos devem produzir o mesmo tempo dentro do ruido).               */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_direto_VRAM(\n"
"    __global volatile const float* A,\n"
"    __global volatile const float* B,\n"
"    __global volatile float*       C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 4 — CONTROLE: inline asm SEM flags                              */\n"
"/* Mesma estrutura de codigo dos kernels 3 e 4, mesma barreira de         */\n"
"/* otimizacao, mas caminho de memoria identico ao kernel 1.               */\n"
"/* ISA esperada: flat_load_dword v, v[..]           (GLC=0, SLC=0)        */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_asm_baseline(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            float a, b;\n"
"            __global const float* pa = &A[row * N + k];\n"
"            __global const float* pb = &B[k * N + col];\n"
"            __asm__ volatile(\n"
"                \"flat_load_dword %0, %2\\n\"\n"
"                \"flat_load_dword %1, %3\\n\"\n"
"                \"s_waitcnt vmcnt(0)\\n\"\n"
"                : \"=v\"(a), \"=v\"(b)\n"
"                : \"v\"(pa), \"v\"(pb)\n"
"                : \"memory\");\n"
"            sum += a * b;\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 5 — BYPASS DE L1: GLC=1 nos LOADs de A e B                      */\n"
"/* Difere do kernel 2 por exatamente um bit da instrucao (bit 16 do       */\n"
"/* encoding FLAT no GFX8). Encoding: DC510000 em vez de DC500000.         */\n"
"/* ISA esperada: flat_load_dword v, v[..] glc       (GLC=1, SLC=0)        */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_bypass_L1(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            float a, b;\n"
"            __global const float* pa = &A[row * N + k];\n"
"            __global const float* pb = &B[k * N + col];\n"
"            __asm__ volatile(\n"
"                \"flat_load_dword %0, %2 glc\\n\"\n"
"                \"flat_load_dword %1, %3 glc\\n\"\n"
"                \"s_waitcnt vmcnt(0)\\n\"\n"
"                : \"=v\"(a), \"=v\"(b)\n"
"                : \"v\"(pa), \"v\"(pb)\n"
"                : \"memory\");\n"
"            sum += a * b;\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 6 — BYPASS DE L1 + STREAMING EM L2: GLC=1 e SLC=1               */\n"
"/* SLC=1 nao \"desliga\" o L2: marca a linha como evict-first, de modo que  */\n"
"/* ela nao desloca dados reutilizaveis. A L2 CONTINUA servindo o acesso:  */\n"
"/* medido em slc_l2.cpp, 548 GB/s sobre um buffer que cabe na L2 contra   */\n"
"/* 222 GB/s quando nao cabe. Nao chamar este kernel de \"acesso direto a   */\n"
"/* VRAM\" — esse cenario nao existe no GFX8 pelo lado do shader.           */\n"
"/* ISA esperada: flat_load_dword v, v[..] glc slc   (GLC=1, SLC=1)        */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_bypass_L1_L2(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            float a, b;\n"
"            __global const float* pa = &A[row * N + k];\n"
"            __global const float* pb = &B[k * N + col];\n"
"            __asm__ volatile(\n"
"                \"flat_load_dword %0, %2 glc slc\\n\"\n"
"                \"flat_load_dword %1, %3 glc slc\\n\"\n"
"                \"s_waitcnt vmcnt(0)\\n\"\n"
"                : \"=v\"(a), \"=v\"(b)\n"
"                : \"v\"(pa), \"v\"(pb)\n"
"                : \"memory\");\n"
"            sum += a * b;\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 7 — VALIDACAO CRUZADA SEM INLINE ASM                            */\n"
"/* __builtin_nontemporal_load e um builtin do Clang: no backend AMDGPU    */\n"
"/* para GFX8 ele emite GLC=1 e SLC=1, o mesmo encoding do kernel 4, sem   */\n"
"/* bloco asm. Serve para mostrar que as flags do kernel 4 nao sao         */\n"
"/* artefato do inline asm. Como o Clang mantem as otimizacoes, a ISA    */\n"
"/* deste kernel e identica a do kernel 1 instrucao a instrucao, exceto  */\n"
"/* pelos bits glc/slc nos dois loads — e a comparacao mais limpa das    */\n"
"/* flags de cache que este experimento produz.                          */\n"
"/* ISA esperada: flat_load_dword v, v[..] glc slc   (GLC=1, SLC=1)        */\n"
"/* ====================================================================== */\n"
"__kernel void matmul_nontemporal(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += __builtin_nontemporal_load(&A[row * N + k])\n"
"                 * __builtin_nontemporal_load(&B[k * N + col]);\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 8 — REFERENCIA: tiled com LDS (__local)                         */\n"
"/* Nao mexe em flags de cache: reduz o trafego reaproveitando os dados    */\n"
"/* em memoria local. E o limite superior de desempenho da comparacao.     */\n"
"/* ====================================================================== */\n"
"#define TILE_SIZE 16\n"
"__kernel void matmul_tiled(\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float*       restrict C,\n"
"    const int N)\n"
"{\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    int localRow = get_local_id(1);\n"
"    int localCol = get_local_id(0);\n"
"\n"
"    __local float tileA[TILE_SIZE][TILE_SIZE];\n"
"    __local float tileB[TILE_SIZE][TILE_SIZE];\n"
"\n"
"    float sum = 0.0f;\n"
"    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;\n"
"\n"
"    for (int t = 0; t < numTiles; t++) {\n"
"        int tiledCol = t * TILE_SIZE + localCol;\n"
"        int tiledRow = t * TILE_SIZE + localRow;\n"
"\n"
"        if (row < N && tiledCol < N)\n"
"            tileA[localRow][localCol] = A[row * N + tiledCol];\n"
"        else\n"
"            tileA[localRow][localCol] = 0.0f;\n"
"\n"
"        if (tiledRow < N && col < N)\n"
"            tileB[localRow][localCol] = B[tiledRow * N + col];\n"
"        else\n"
"            tileB[localRow][localCol] = 0.0f;\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"        for (int k = 0; k < TILE_SIZE; k++) {\n"
"            sum += tileA[localRow][k] * tileB[k][localCol];\n"
"        }\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    if (row < N && col < N) {\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernels 9 e 10 — INTRINSIC DO LLVM, sem bloco de assembly              */\n"
"/*                                                                        */\n"
"/* llvm.amdgcn.raw.buffer.load.f32 e um intrinsic do backend AMDGPU, nao  */\n"
"/* do HIP: HIP e OpenCL sao front-ends que desembocam nele. A declaracao  */\n"
"/* abaixo o amarra por asm label, idioma que a Composable Kernel usa.     */\n"
"/*                                                                        */\n"
"/* O ultimo parametro NAO sao dois booleanos glc e slc: e UM inteiro      */\n"
"/* cachepolicy, com bit0 = glc e bit1 = slc.                              */\n"
"/*                                                                        */\n"
"/* A hipotese sob teste: diferentemente do bloco __asm__, que e opaco ao  */\n"
"/* otimizador e custa 1,5x por si so (kernel 4), o intrinsic tem semantica*/\n"
"/* conhecida pelo compilador, que pode escalonar em volta dele e manter a */\n"
"/* reducao de forca no calculo de endereco. Se a hipotese valer, o par    */\n"
"/* 9/10 mede a flag glc SOZINHA sobre codigo otimizado — a celula que     */\n"
"/* falta no desenho fatorial, inalcancavel pelas outras vias: o           */\n"
"/* nontemporal so emite glc+slc, e o asm cobra a barreira.                */\n"
"/*                                                                        */\n"
"/* Custo do metodo: o formato MUBUF exige um descritor de recurso de 128  */\n"
"/* bits. num_records (palavra 2) faz bounds check em hardware — se ficar  */\n"
"/* curto, o acesso fora do limite devolve ZERO em silencio, sem erro.     */\n"
"/* Dai a validacao contra CPU ser obrigatoria nestes dois kernels.        */\n"
"/* ====================================================================== */\n"
"float llvm_amdgcn_raw_buffer_load_fp32(uint4 srsrc, uint voffset,\n"
"                                       uint soffset, uint cachepolicy)\n"
"    __asm(\"llvm.amdgcn.raw.buffer.load.f32\");\n"
"\n"
"static uint4 descritor(__global const float* p, uint n_bytes)\n"
"{\n"
"    ulong base = (ulong)p;\n"
"    uint4 d;\n"
"    d.s0 = (uint)(base & 0xFFFFFFFFul);       /* base_lo                    */\n"
"    d.s1 = (uint)((base >> 32) & 0xFFFFul);   /* base_hi | stride = 0       */\n"
"    d.s2 = n_bytes;                           /* num_records, em bytes      */\n"
"    d.s3 = 0x0000F000u;                       /* formato padrao do LLVM     */\n"
"    return d;\n"
"}\n"
"\n"
"#define MATMUL_INTRIN(NOME, POLITICA)                                       \\\n"
"__kernel void NOME(                                                          \\\n"
"    __global const float* restrict A,                                        \\\n"
"    __global const float* restrict B,                                        \\\n"
"    __global float*       restrict C,                                        \\\n"
"    const int N)                                                             \\\n"
"{                                                                            \\\n"
"    int row = get_global_id(1);                                              \\\n"
"    int col = get_global_id(0);                                              \\\n"
"    if (row < N && col < N) {                                                \\\n"
"        uint nb = (uint)N * (uint)N * 4u;                                    \\\n"
"        uint4 ra = descritor(A, nb);                                         \\\n"
"        uint4 rb = descritor(B, nb);                                         \\\n"
"        float sum = 0.0f;                                                    \\\n"
"        for (int k = 0; k < N; k++) {                                        \\\n"
"            sum += llvm_amdgcn_raw_buffer_load_fp32(ra,                      \\\n"
"                        (uint)(row * N + k) * 4u, 0u, POLITICA)              \\\n"
"                 * llvm_amdgcn_raw_buffer_load_fp32(rb,                      \\\n"
"                        (uint)(k * N + col) * 4u, 0u, POLITICA);             \\\n"
"        }                                                                    \\\n"
"        C[row * N + col] = sum;                                              \\\n"
"    }                                                                        \\\n"
"}\n"
"\n"
"/* Kernel 9 — CONTROLE do intrinsic: mesma construcao, cachepolicy = 0     */\n"
"MATMUL_INTRIN(matmul_intrin_baseline, 0u)\n"
"\n"
"/* Kernel 10 — intrinsic com glc isolado: cachepolicy = 1 (bit0)           */\n"
"MATMUL_INTRIN(matmul_intrin_glc, 1u)\n";


/* ------------------------------------------------------------------------- */
/*  Utilitarios                                                              */
/* ------------------------------------------------------------------------- */

#define CHECK_CL(err, msg) do {                                         \
    if ((err) != CL_SUCCESS) {                                          \
        fprintf(stderr, "[ERRO OpenCL] %s (codigo: %d)\n", (msg), (err)); \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static double get_time_ms(cl_event ev) {
    cl_ulong t_start, t_end;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                           sizeof(cl_ulong), &t_start, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                           sizeof(cl_ulong), &t_end, NULL);
    return (double)(t_end - t_start) / 1e6;
}

/* Converte o nome do dispositivo num componente de nome de arquivo seguro:
   "Radeon RX 580 Series (POLARIS10...)" -> "Radeon_RX_580_Series_POLARIS10" */
static void slug_dispositivo(const char* nome, char* out, size_t out_sz) {
    size_t j = 0;
    int ultimo_foi_sep = 1;   /* evita '_' no inicio e repetidos */
    for (size_t i = 0; nome[i] && j + 1 < out_sz; i++) {
        unsigned char c = (unsigned char)nome[i];
        int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                 (c >= '0' && c <= '9') || c == '.' || c == '-';
        if (ok) {
            out[j++] = (char)c;
            ultimo_foi_sep = 0;
        } else if (!ultimo_foi_sep) {
            out[j++] = '_';
            ultimo_foi_sep = 1;
        }
    }
    while (j > 0 && out[j - 1] == '_') j--;   /* tira '_' final */
    if (j == 0) { const char* f = "desconhecido"; for (; *f && j + 1 < out_sz; f++) out[j++] = *f; }
    out[j] = '\0';
}

/*
 * Grava o binario ISA compilado pelo driver.
 *
 * O nome do arquivo inclui o dispositivo, e o destino e a pasta isa/. Sem isso,
 * rodar este programa em plataformas diferentes sobrescreve o dump anterior --
 * o binario do driver Windows (AMDOCL) e o do ROCm/Linux sao evidencias
 * distintas e precisam coexistir para a comparacao multiplataforma.
 */
static void dump_isa_info(cl_program program) {
    size_t bin_size = 0;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t), &bin_size, NULL);
    if (bin_size == 0) return;

    /* Nome do dispositivo para o qual este binario foi compilado */
    cl_device_id dev = NULL;
    char dev_nome[256] = "desconhecido";
    if (clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                         sizeof(cl_device_id), &dev, NULL) == CL_SUCCESS && dev) {
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dev_nome), dev_nome, NULL);
    }
    char slug[128];
    slug_dispositivo(dev_nome, slug, sizeof(slug));

    char caminho[320];
    snprintf(caminho, sizeof(caminho), "isa/matmul_%s.bin", slug);

    unsigned char* binary = (unsigned char*)malloc(bin_size);
    unsigned char* binaries[] = { binary };
    clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                     sizeof(unsigned char*), binaries, NULL);

    FILE* fp = fopen(caminho, "wb");
    if (!fp) {
        /* isa/ pode nao existir se o programa for executado de outra pasta */
        snprintf(caminho, sizeof(caminho), "matmul_%s.bin", slug);
        fp = fopen(caminho, "wb");
    }
    if (fp) {
        fwrite(binary, 1, bin_size, fp);
        fclose(fp);
        printf("[INFO] Binario ISA salvo em '%s' (%lu bytes)\n",
               caminho, (unsigned long)bin_size);
        printf("       Dispositivo: %s\n", dev_nome);
        printf("       Analise: llvm-objdump -d --triple=amdgcn-amd-amdhsa \\\n"
               "                  --mcpu=gfx803 %s   (Linux/ROCm)\n", caminho);
        printf("                clrxdisasm %s                       (binario do driver Windows)\n\n",
               caminho);
    } else {
        fprintf(stderr, "[AVISO] nao consegui gravar o binario ISA em '%s'\n", caminho);
    }
    free(binary);
}

/*
 * Razao entre duas medias com intervalo de confianca de 95% (metodo delta) e
 * estatistica t de Welch. So faz sentido com RUNS grande: com poucas amostras
 * o erro padrao e grande demais para separar efeito de ruido.
 *
 * O parametro "piso" e o tamanho de efeito abaixo do qual nao se deve afirmar
 * nada, mesmo com t alto — ele vem do par de kernels de fonte IDENTICO
 * (k2/k3). Com n na casa do milhar, o teste passa a detectar como
 * "significativa" ate a diferenca entre dois kernels iguais, que so pode vir
 * da ordem de execucao e do estado termico da GPU. Sem esse piso, a
 * significancia estatistica viraria um argumento vazio.
 */
static void razao_ic(const char* rotulo, double ma, double sea,
                     double mb, double seb, double piso)
{
    double r   = mb / ma;
    double rel = sqrt((sea/ma)*(sea/ma) + (seb/mb)*(seb/mb));
    double t   = (mb - ma) / sqrt(sea*sea + seb*seb);
    double efeito = fabs(r - 1.0) * 100.0;
    const char* nota;
    if (fabs(t) < 1.96)     nota = "nao significativo";
    else if (efeito < piso) nota = "abaixo do piso de ruido";
    else                    nota = "significativo";
    printf("  %-32s %6.4fx  IC95 [%.4f, %.4f]  t=%8.2f  %s\n",
           rotulo, r, r * (1.0 - 1.96 * rel), r * (1.0 + 1.96 * rel), t, nota);
}

/*
 * Executa uma medicao completa, opcionalmente dividida em lotes de linhas, e
 * devolve a soma dos tempos de execucao dos lotes.
 *
 * Motivo: o driver amdgpu declara travamento e RESETA a GPU quando um job de
 * compute passa de 60 s (padrao de bare metal, ver modinfo amdgpu). Em
 * N=16384 o kernel mais lento e estimado em ~58 s, dentro da margem de erro
 * desse limite --- e como a Ellesmere e a unica GPU da maquina, um reset
 * derruba a sessao grafica junto. Dividir a mesma NDRange em faixas de linhas
 * por global_work_offset mantem o trabalho total identico e cada lancamento
 * bem abaixo do limite. O custo e o overhead de lancamento, medido em 0,02 ms
 * por lote, desprezivel diante de segundos de execucao.
 *
 * get_global_id(1) devolve o indice global, ja incluindo o offset, entao os
 * kernels nao precisam de nenhuma alteracao.
 */
static double executa_run(cl_command_queue queue, cl_kernel kern, int N,
                          size_t* ls, int lote)
{
    double total = 0.0;
    for (int r0 = 0; r0 < N; r0 += lote) {
        size_t linhas    = (size_t)((r0 + lote <= N) ? lote : (N - r0));
        size_t offset[2] = { 0, (size_t)r0 };
        size_t gs[2]     = { (size_t)N, linhas };
        cl_event ev;
        cl_int e = clEnqueueNDRangeKernel(queue, kern, 2, offset, gs, ls,
                                          0, NULL, &ev);
        if (e != CL_SUCCESS) {
            fprintf(stderr, "[ERRO OpenCL] NDRange (lote em r0=%d, codigo %d)\n",
                    r0, e);
            exit(1);
        }
        clWaitForEvents(1, &ev);
        total += get_time_ms(ev);
        clReleaseEvent(ev);
    }
    return total;
}

/* Comparador para qsort — usado na mediana e no percentil 95 */
static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/*
 * Verificacao amostrada.
 *
 * A referencia completa em CPU e O(N^3) e domina o tempo total a partir de
 * N=8192 (estimativa de ~2,9 h em N=16384, contra ~7 min de GPU). Como uma
 * falha de bypass nao corrompe um elemento isolado --- um descritor mal
 * formado devolve zero em TODAS as posicoes, e uma flag mal emitida muda o
 * caminho de memoria de todos os acessos ---, verificar uma amostra aleatoria
 * de saidas detecta o mesmo conjunto de falhas a custo O(A*N).
 *
 * A verificacao e dupla: (i) cada kernel e comparado elemento a elemento com a
 * saida do kernel 1, em O(N^2), o que pega qualquer divergencia entre eles; e
 * (ii) o kernel 1 e conferido contra a CPU numa amostra de posicoes. Assim a
 * cadeia inteira fica ancorada num calculo independente.
 */
static void matmul_cpu_amostra(const float* A, const float* B, int N,
                               const int* idx, float* out, int n_amostra) {
    for (int t = 0; t < n_amostra; t++) {
        int i = idx[t] / N, j = idx[t] % N;
        float sum = 0.0f;
        for (int k = 0; k < N; k++) sum += A[i * N + k] * B[k * N + j];
        out[t] = sum;
    }
}



/* ------------------------------------------------------------------------- */
/*  Main                                                                     */
/* ------------------------------------------------------------------------- */

int main(int argc, char** argv) {
    cl_int err;

    /*
     * Uso:  ./output/matmul [RUNS] [N] [WARMUP]
     *
     * RUNS   execucoes medidas por cenario (padrao 1000). Com 8 cenarios e
     *        N=1024 isso da ~90 s de GPU; RUNS=50 serve para uma passada
     *        rapida, mas so RUNS na casa do milhar sustenta afirmacao sobre
     *        as diferencas de poucos por cento (efeito do SLC, por exemplo).
     * N      lado da matriz (padrao 1024, ~12 MB nos tres buffers).
     * WARMUP execucoes descartadas antes de medir (padrao 10).
     */
    /*
     * RUNS = 0 pede o modo adaptativo: em vez de um numero fixo de execucoes,
     * fixa-se um ORCAMENTO DE TEMPO por cenario e deriva-se o numero de
     * execucoes da duracao medida no aquecimento. Sem isso a varredura de
     * tamanhos e inviavel --- 1000 execucoes custam 1,8 min em N=1024 e 5,2
     * dias em N=16384, porque o kernel e O(N^3).
     *
     * O orcamento mantem a incerteza sob controle onde ela importa: em N
     * pequeno o kernel e curto, cabem muitas amostras e os efeitos de poucos
     * por cento ficam resolviveis; em N grande cada amostra ja e longa e
     * estavel, e poucas bastam. O n efetivo entra no CSV, de modo que a
     * precisao de cada ponto fica explicita.
     */
    const int RUNS_PEDIDO = (argc > 1) ? atoi(argv[1]) : 1000;
    const int N      = (argc > 2) ? atoi(argv[2]) : 1024;
    const int WARMUP = (argc > 3) ? atoi(argv[3]) : 10;
    /*
     * LOTE = linhas por lancamento. 0 (padrao) usa N, ou seja, um lancamento
     * unico. Valores menores dividem a NDRange para nao esbarrar no watchdog
     * de 60 s do amdgpu; precisa ser multiplo de 16, o lado do work-group.
     */
    int LOTE = (argc > 4) ? atoi(argv[4]) : 0;
    if (LOTE <= 0 || LOTE > N) LOTE = N;
    LOTE = (LOTE / 16) * 16;
    if (LOTE < 16) LOTE = 16;
    if (RUNS_PEDIDO < 0 || N < 16 || WARMUP < 0) {
        fprintf(stderr, "uso: %s [RUNS>=1, ou 0 = adaptativo] [N>=16] "
                        "[WARMUP>=0] [LINHAS_POR_LOTE]\n", argv[0]);
        return 2;
    }
    /* Orcamento de tempo por cenario no modo adaptativo, em milissegundos. */
    const double ORCAMENTO_MS = 60000.0;
    const int RUNS_MIN = 20, RUNS_MAX = 1000;
    const size_t mat_size = (size_t)N * N * sizeof(float);
    const double gflops_per_run = 2.0 * (double)N * N * N / 1e9;

    printf("=== AMD GPU Cache Bypass — Multiplicacao de Matrizes ===\n");
    printf("Matriz: %d x %d (%.1f MB por matriz, %.1f MB total)\n",
           N, N, (double)mat_size / (1024.0*1024.0),
           3.0 * (double)mat_size / (1024.0*1024.0));
    printf("FLOPs por execucao: %.2f GFLOP\n\n", gflops_per_run);

    /* -- Alocar e inicializar matrizes -- */
    float* h_A = (float*)malloc(mat_size);
    float* h_B = (float*)malloc(mat_size);
    float* h_C = (float*)malloc(mat_size);
    float* h_C_k1 = (float*)malloc(mat_size);   /* saida do kernel 1, ancora */
    if (!h_A || !h_B || !h_C || !h_C_k1) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        return 1;
    }

    /* Valores pequenos para evitar overflow */
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i + 37) % 100) / 100.0f;
    }

    /* -- Setup OpenCL -- */
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    CHECK_CL(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL(err, "clGetDeviceIDs");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("GPU: %s\n", dev_name);

    char vendor[256];
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    printf("Vendor: %s\n\n", vendor);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(
        ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    /* -- Buffers GPU -- */
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 mat_size, h_A, &err);
    CHECK_CL(err, "clCreateBuffer A");

    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 mat_size, h_B, &err);
    CHECK_CL(err, "clCreateBuffer B");

    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 mat_size, NULL, &err);
    CHECK_CL(err, "clCreateBuffer C");

    /* -- Compilar -- */
    const char* sources[] = { kernel_source };
    cl_program program = clCreateProgramWithSource(ctx, 1, sources, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        exit(1);
    }

    dump_isa_info(program);

    /* -- Criar kernels -- */
    /* -- Criar kernels -- */
    /*
     * A ordem importa para a leitura da tabela final: os quatro primeiros sao
     * o mesmo algoritmo variando so o caminho de cache, o quinto valida as
     * flags sem inline asm e o sexto e a referencia com LDS.
     */
    struct Cenario {
        const char* fn;       /* nome do kernel no fonte OpenCL   */
        const char* rotulo;   /* descricao para a saida           */
        const char* isa;      /* flags esperadas na ISA           */
        int         tiled;    /* usa __local (workgroup fixo)     */
        cl_kernel   k;
        double      avg_ms;   /* media das RUNS execucoes         */
        int         erros;
        double      med_ms;   /* mediana                          */
        double      min_ms;
        double      max_ms;
        double      sd_ms;    /* desvio padrao amostral           */
        double      sem_ms;   /* erro padrao da media (sd/sqrt n) */
        double      p95_ms;   /* percentil 95                     */
        double      deriva;   /* media do ultimo decimo / 1o (%)  */
        int         runs;     /* n efetivo deste cenario          */
    };

    struct Cenario cen[] = {
        { "matmul_via_L1",       "Kernel 1  normal (C puro)",              "GLC=0 SLC=0",  0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_via_L2",       "Kernel 2  volatile (hipotese original)", "GLC=1 SLC=0 *",0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_direto_VRAM",  "Kernel 3  volatile agressivo (idem k2)", "GLC=1 SLC=0 *",0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_asm_baseline", "Kernel 4  controle asm (sem flags)",     "GLC=0 SLC=0",  0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_bypass_L1",    "Kernel 5  bypass de L1",                 "GLC=1 SLC=0",  0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_bypass_L1_L2", "Kernel 6  bypass de L1 + streaming L2",  "GLC=1 SLC=1",  0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_nontemporal",  "Kernel 7  nontemporal (sem asm)",        "GLC=1 SLC=1",  0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_tiled",        "Kernel 8  tiled com LDS (referencia)",   "GLC=0 SLC=0",  1, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_intrin_baseline","Kernel 9  controle intrinsic (sem flag)","GLC=0 SLC=0",  0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 },
        { "matmul_intrin_glc",   "Kernel 10 intrinsic com glc isolado",    "GLC=1 SLC=0",  0, NULL, 0.0, 0, 0,0,0,0,0,0,0,0 }
    };
    const int NUM_KERNELS = (int)(sizeof(cen) / sizeof(cen[0]));

    for (int k = 0; k < NUM_KERNELS; k++) {
        cen[k].k = clCreateKernel(program, cen[k].fn, &err);
        CHECK_CL(err, cen[k].fn);
        clSetKernelArg(cen[k].k, 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(cen[k].k, 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(cen[k].k, 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(cen[k].k, 3, sizeof(int),    &N);
    }

    /* -- Referencia independente da CPU, sobre uma amostra de posicoes -- */
    const int N_AMOSTRA = 4096;
    int*   amostra_idx = (int*)  malloc(N_AMOSTRA * sizeof(int));
    float* amostra_ref = (float*)malloc(N_AMOSTRA * sizeof(float));
    if (!amostra_idx || !amostra_ref) { fprintf(stderr, "sem memoria\n"); return 1; }
    srand(12345);                       /* amostra reproduzivel entre execucoes */
    for (int t = 0; t < N_AMOSTRA; t++) {
        long r1 = ((long)rand() << 15) ^ (long)rand();
        amostra_idx[t] = (int)(((unsigned long)r1) % ((unsigned long)N * N));
    }
    printf("Referencia da CPU sobre %d posicoes amostradas...\n", N_AMOSTRA);
    matmul_cpu_amostra(h_A, h_B, N, amostra_idx, amostra_ref, N_AMOSTRA);
    printf("Referencia pronta.\n\n");

    /* -- Executar cada cenario -- */
    size_t local_size_normal[2] = { 16, 16 };   /* 256 threads */
    size_t local_size_tiled[2]  = { 16, 16 };   /* TILE_SIZE = 16 */

    /*
     * RUNS alto (padrao 1000) e o que permite tratar as diferencas pequenas
     * como resultado e nao como ruido: o efeito do bit SLC fica em ~1.03x, na
     * mesma ordem da dispersao entre execucoes. Com 1000 amostras por cenario
     * reporta-se mediana, desvio padrao e coeficiente de variacao, e a
     * comparacao passa a ter uma barra de incerteza associada.
     *
     * A carga sustentada tambem expoe variacao de clock: a coluna "deriva"
     * compara a media do ultimo decimo das execucoes com a do primeiro. Valor
     * proximo de 0% indica que a GPU manteve o clock; valor positivo alto
     * sugere throttling termico, e nesse caso a media do cenario nao e
     * comparavel com a dos demais sem ressalva.
     *
     * Ambos configuraveis pela linha de comando (ver uso no inicio de main).
     */
    printf("Matriz: %dx%d | Workgroup: 16x16 (256 threads)", N, N);
    if (LOTE < N) printf(" | Lotes: %d linhas (%d lancamentos por execucao)",
                         LOTE, (N + LOTE - 1) / LOTE);
    printf("\n");
    if (RUNS_PEDIDO == 0)
        printf("Warmup: %d | Runs: adaptativo (orcamento de %.0f s por cenario,"
               " entre %d e %d execucoes)\n",
               WARMUP, ORCAMENTO_MS / 1000.0, RUNS_MIN, RUNS_MAX);
    else
        printf("Warmup: %d | Runs: %d por cenario (%d execucoes no total)\n",
               WARMUP, RUNS_PEDIDO, RUNS_PEDIDO * NUM_KERNELS);
    printf("================================================================\n\n");

    double* amostras = (double*)malloc((size_t)RUNS_MAX * sizeof(double));
    if (!amostras) { fprintf(stderr, "sem memoria para as amostras\n"); exit(1); }

    /* CSV com todas as amostras deste N, para histograma/boxplot */
    char csv_nome[256];
    snprintf(csv_nome, sizeof(csv_nome), "resultados/matmul_amostras_N%d.csv", N);
    FILE* csv = fopen(csv_nome, "w");
    if (!csv) csv = fopen("matmul_amostras.csv", "w");
    if (csv) fprintf(csv, "kernel,flags_isa,execucao,tempo_ms\n");

    /*
     * Resumo acumulado da varredura: uma linha por (N, cenario), em modo
     * append. Cada N e um processo separado, entao o arquivo cresce ao longo
     * da varredura e sobrevive a uma interrupcao no meio dela.
     */
    int resumo_novo = (access("resultados/matmul_sweep.csv", F_OK) != 0);
    FILE* resumo = fopen("resultados/matmul_sweep.csv", "a");
    if (resumo && resumo_novo)
        fprintf(resumo, "N,kernel,flags_isa,runs,media_ms,mediana_ms,dp_ms,cv_pct,"
                        "min_ms,max_ms,p95_ms,deriva_pct,gflops,validacao\n");

    for (int k = 0; k < NUM_KERNELS; k++) {
        printf("--- %s [%s] ---\n", cen[k].rotulo, cen[k].isa);
        fflush(stdout);

        size_t* ls = cen[k].tiled ? local_size_tiled : local_size_normal;

        /* Warmup: nao entra na estatistica, serve para subir o clock e
           preencher os caches. A duracao da ultima execucao de aquecimento
           tambem alimenta o dimensionamento adaptativo de RUNS. */
        double ms_aquecimento = 0.0;
        for (int w = 0; w < WARMUP; w++)
            ms_aquecimento = executa_run(queue, cen[k].k, N, ls, LOTE);

        /* Quantas execucoes cabem no orcamento deste cenario */
        int RUNS = RUNS_PEDIDO;
        if (RUNS_PEDIDO == 0) {
            RUNS = (ms_aquecimento > 0.0)
                 ? (int)(ORCAMENTO_MS / ms_aquecimento) : RUNS_MAX;
            if (RUNS < RUNS_MIN) RUNS = RUNS_MIN;
            if (RUNS > RUNS_MAX) RUNS = RUNS_MAX;
            printf("  (adaptativo: %.2f ms por execucao -> %d execucoes)\n",
                   ms_aquecimento, RUNS);
        }

        /* Medicoes */
        double total_ms = 0.0;
        double min_ms = 1e30, max_ms = 0.0;
        int marco = RUNS / 10 > 0 ? RUNS / 10 : 1;

        for (int r = 0; r < RUNS; r++) {
            double ms = executa_run(queue, cen[k].k, N, ls, LOTE);
            amostras[r] = ms;
            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;

            if (csv) fprintf(csv, "%s,%s,%d,%.6f\n", cen[k].fn, cen[k].isa, r + 1, ms);

            /* Com RUNS grande, listar cada execucao inunda a saida: ate 20
               execucoes imprime tudo, acima disso so o progresso. */
            if (RUNS <= 20) {
                printf("  Run %d: %.3f ms\n", r + 1, ms);
            } else if ((r + 1) % marco == 0) {
                printf("  %3d%% (%d/%d)\r", (r + 1) * 100 / RUNS, r + 1, RUNS);
                fflush(stdout);
            }
        }
        if (RUNS > 20) printf("            \r");

        /* --- estatisticas --- */
        cen[k].runs   = RUNS;
        cen[k].avg_ms = total_ms / RUNS;
        cen[k].min_ms = min_ms;
        cen[k].max_ms = max_ms;

        double soma_q = 0.0;
        for (int r = 0; r < RUNS; r++) {
            double d = amostras[r] - cen[k].avg_ms;
            soma_q += d * d;
        }
        cen[k].sd_ms  = RUNS > 1 ? sqrt(soma_q / (RUNS - 1)) : 0.0;
        cen[k].sem_ms = cen[k].sd_ms / sqrt((double)RUNS);

        /* deriva: media do ultimo decimo contra a do primeiro decimo */
        int dec = RUNS / 10;
        if (dec >= 1) {
            double ini = 0.0, fim = 0.0;
            for (int r = 0; r < dec; r++)            ini += amostras[r];
            for (int r = RUNS - dec; r < RUNS; r++)  fim += amostras[r];
            ini /= dec; fim /= dec;
            cen[k].deriva = (fim - ini) / ini * 100.0;
        }

        /* mediana e p95 exigem ordenacao; ordena uma copia para nao perder
           a ordem temporal usada na deriva (ja calculada acima) */
        qsort(amostras, (size_t)RUNS, sizeof(double), cmp_double);
        cen[k].med_ms = (RUNS % 2)
            ? amostras[RUNS / 2]
            : 0.5 * (amostras[RUNS / 2 - 1] + amostras[RUNS / 2]);
        int idx95 = (int)(0.95 * (RUNS - 1));
        cen[k].p95_ms = amostras[idx95];

        printf("  Media: %.3f ms | Mediana: %.3f ms | Min: %.3f | Max: %.3f\n",
               cen[k].avg_ms, cen[k].med_ms, cen[k].min_ms, cen[k].max_ms);
        printf("  Desvio padrao: %.4f ms (cv %.2f%%) | p95: %.3f ms | deriva: %+.2f%%\n",
               cen[k].sd_ms, 100.0 * cen[k].sd_ms / cen[k].avg_ms,
               cen[k].p95_ms, cen[k].deriva);
        printf("  Performance: %.2f GFLOPS (pela mediana)\n",
               gflops_per_run / (cen[k].med_ms / 1000.0));

        /*
         * Validacao por kernel. Os kernels com inline asm montam os enderecos
         * manualmente: se o resultado bater com a CPU em todos os N*N
         * elementos, os LOADs com GLC/SLC leram os dados corretos e a
         * comparacao de tempo tem sentido. Verificar so o ultimo kernel, como
         * antes, deixaria os cenarios de bypass sem validacao nenhuma.
         */
        clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_size, h_C, 0, NULL, NULL);

        /* Tolerancia proporcional a N: a soma acumula erro de ponto flutuante
           ao longo de N termos, e um limite fixo daria falso negativo em
           matrizes grandes. */
        const float tol = 0.01f * ((float)N / 1024.0f);
        int errors = 0;
        float max_diff = 0.0f;
        int primeiro = -1;

        if (k == 0) {
            /* Kernel 1 e a ancora: guarda a saida e confere contra a amostra
               calculada na CPU, que e o unico calculo independente da GPU. */
            memcpy(h_C_k1, h_C, mat_size);
            for (int t = 0; t < N_AMOSTRA; t++) {
                float diff = fabsf(h_C[amostra_idx[t]] - amostra_ref[t]);
                if (diff > max_diff) max_diff = diff;
                if (diff > tol) {
                    if (primeiro < 0) primeiro = amostra_idx[t];
                    errors++;
                }
            }
        } else {
            /* Demais cenarios: comparacao completa contra a ancora. */
            for (int i = 0; i < N * N; i++) {
                float diff = fabsf(h_C[i] - h_C_k1[i]);
                if (diff > max_diff) max_diff = diff;
                if (diff > tol) {
                    if (primeiro < 0) primeiro = i;
                    errors++;
                }
            }
        }
        cen[k].erros = errors;
        if (errors == 0) {
            printf("  Validacao: OK (%s, max diferenca %.8f)\n",
                   k == 0 ? "amostra da CPU" : "contra kernel 1", max_diff);
        } else {
            printf("  Validacao: FALHOU — %d elementos errados, 1o em C[%d]\n",
                   errors, primeiro);
        }

        if (resumo) {
            fprintf(resumo, "%d,%s,%s,%d,%.6f,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,"
                            "%.4f,%.3f,%s\n",
                    N, cen[k].fn, cen[k].isa, RUNS, cen[k].avg_ms, cen[k].med_ms,
                    cen[k].sd_ms, 100.0 * cen[k].sd_ms / cen[k].avg_ms,
                    cen[k].min_ms, cen[k].max_ms, cen[k].p95_ms, cen[k].deriva,
                    gflops_per_run / (cen[k].med_ms / 1000.0),
                    errors == 0 ? "OK" : "FALHOU");
            fflush(resumo);
        }
        printf("\n");

        /* Zera C para que o proximo kernel nao herde o resultado deste */
        memset(h_C, 0, mat_size);
        clEnqueueWriteBuffer(queue, d_C, CL_TRUE, 0, mat_size, h_C, 0, NULL, NULL);
    }

    /* -- Tabela final -- */
    printf("================================================================\n");
    printf("%-40s %-14s %9s %9s %8s %6s %8s %9s %s\n",
           "cenario", "flags ISA", "media", "mediana", "dp(ms)", "cv%",
           "deriva%", "GFLOPS", "val");
    printf("--------------------------------------------------------------------"
           "-------------------------------------\n");
    for (int k = 0; k < NUM_KERNELS; k++) {
        printf("%-40s %-14s %9.3f %9.3f %8.4f %6.2f %+8.2f %9.2f %s\n",
               cen[k].rotulo, cen[k].isa, cen[k].avg_ms, cen[k].med_ms,
               cen[k].sd_ms, 100.0 * cen[k].sd_ms / cen[k].avg_ms, cen[k].deriva,
               gflops_per_run / (cen[k].med_ms / 1000.0),
               cen[k].erros == 0 ? "OK" : "FALHOU");
    }
    printf("--------------------------------------------------------------------"
           "-------------------------------------\n");
    printf("dp = desvio padrao amostral | cv = dp/media | deriva = ultimo decimo"
           " das execucoes contra o primeiro\n");
    printf("GFLOPS calculado pela mediana, menos sensivel a outliers. As razoes"
           " abaixo usam a media, porque\n");
    printf("o intervalo de confianca do metodo delta se apoia no erro padrao da"
           " media (dp/raiz(n)).\n");
    printf("----------------------------------------------------------------\n");
    printf("(*) k2 e k3 dependem do compilador: o mesmo fonte gera GLC=1 no\n");
    printf("    Clang/LLVM do ROCm e GLC=0 no AMDOCL do Windows. Confira sempre\n");
    printf("    a ISA realmente gerada antes de citar a flag.\n\n");
    printf("Comparacoes validas:\n");
    printf("  k1 vs k7 : MEDIDA PRINCIPAL — ISA identica instrucao a instrucao,\n");
    printf("             diferindo so pelos bits 'glc slc' em dois flat_load_dword\n");
    printf("  k1 -> k2 : efeito do volatile NESTE compilador (nao portavel).\n");
    printf("             CUIDADO: sob o ROCm o volatile emite glc E serializa os\n");
    printf("             loads (s_waitcnt apos cada um). O controle dele e o k4,\n");
    printf("             nao o k1 — senao bypass e serializacao ficam confundidos\n");
    printf("  k2 vs k3 : repetibilidade — fontes identicos, tempos devem coincidir\n");
    printf("  k1 -> k4 : custo do inline asm (barreira de otimizacao), sem bypass\n");
    printf("  k4 -> k5 : efeito do bypass de L1  (bit GLC), ja descontado o asm\n");
    printf("  k5 -> k6 : efeito do streaming em L2 (bit SLC)\n");
    printf("  k6 vs k7 : mesmas flags por caminhos diferentes (asm x builtin)\n\n");
    /*
     * Piso de ruido: k2 e k3 tem fonte identico e ISA identica, logo qualquer
     * diferenca entre eles e artefato da bancada (ordem de execucao, estado
     * termico). O tamanho dessa diferenca e o menor efeito que este
     * experimento consegue afirmar de fato.
     */
    double piso = fabs(cen[2].avg_ms / cen[1].avg_ms - 1.0) * 100.0;

    printf("Razoes (media, IC95 pelo metodo delta, t de Welch):\n");
    razao_ic("k3/k2 controle negativo",   cen[1].avg_ms, cen[1].sem_ms,
                                          cen[2].avg_ms, cen[2].sem_ms, 0.0);
    razao_ic("k7/k1 GLC+SLC, sem asm",    cen[0].avg_ms, cen[0].sem_ms,
                                          cen[6].avg_ms, cen[6].sem_ms, piso);
    razao_ic("k2/k1 volatile no ROCm",    cen[0].avg_ms, cen[0].sem_ms,
                                          cen[1].avg_ms, cen[1].sem_ms, piso);
    razao_ic("k4/k1 so o custo do asm",   cen[0].avg_ms, cen[0].sem_ms,
                                          cen[3].avg_ms, cen[3].sem_ms, piso);
    razao_ic("k5/k4 efeito do GLC",       cen[3].avg_ms, cen[3].sem_ms,
                                          cen[4].avg_ms, cen[4].sem_ms, piso);
    razao_ic("k6/k5 efeito do SLC",       cen[4].avg_ms, cen[4].sem_ms,
                                          cen[5].avg_ms, cen[5].sem_ms, piso);
    razao_ic("k2/k5 volatile x asm glc",  cen[4].avg_ms, cen[4].sem_ms,
                                          cen[1].avg_ms, cen[1].sem_ms, piso);
    razao_ic("k1/k8 LDS vs global",       cen[7].avg_ms, cen[7].sem_ms,
                                          cen[0].avg_ms, cen[0].sem_ms, piso);
    razao_ic("k9/k1  custo do intrinsic",  cen[0].avg_ms, cen[0].sem_ms,
                                          cen[8].avg_ms, cen[8].sem_ms, piso);
    razao_ic("k10/k9 GLC isolado (intrin)",cen[8].avg_ms, cen[8].sem_ms,
                                          cen[9].avg_ms, cen[9].sem_ms, piso);

    printf("\n  Piso de ruido da bancada (k2 vs k3, kernels identicos): %.2f%%\n", piso);
    printf("  Efeitos menores que isso nao devem ser afirmados, ainda que o\n");
    printf("  teste os aponte como significativos — com n=%d o t detecta ate a\n", cen[0].runs);
    printf("  diferenca entre dois kernels de ISA identica.\n\n");

    if (csv) {
        fclose(csv);
        printf("Amostras individuais gravadas em 'resultados/matmul_cache_flags.csv'\n");
        printf("(%d cenarios; n por cenario na coluna 'runs' de matmul_sweep.csv)\n\n",
               NUM_KERNELS);
    }
    free(amostras);

    /* -- Cleanup -- */
    for (int k = 0; k < NUM_KERNELS; k++) clReleaseKernel(cen[k].k);
    clReleaseProgram(program);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_k1);
    free(amostra_idx);
    free(amostra_ref);
    if (resumo) fclose(resumo);

    printf("\n=== Fim ===\n");
    return 0;
}