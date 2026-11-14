/*
 * =============================================================================
 *  Hierarquia de memoria em GPU AMD — Multiplicacao de matrizes por caminho
 *  de cache (GCN3 / GFX8 / gfx803 — Ellesmere, RX 580)
 * =============================================================================
 *
 *  Dez kernels calculam a MESMA C = A*B. Nove usam o algoritmo naive (um
 *  work-item por elemento) e diferem apenas no caminho de cache dos LOADs de
 *  A e B; o decimo (tiled, com LDS) e a referencia de desempenho.
 *
 *  Todo numero citado abaixo foi conferido nas evidencias deste repositorio:
 *  a ISA em isa/matmul_gfx803_disasm.txt (ROCm 5.2 / LLVM 12, gfx803) e os
 *  tempos em resultados/matmul_sweep.csv (media de 100 execucoes por cenario,
 *  N de 1024 a 16384).
 *
 *  ---------------------------------------------------------------------------
 *  1. COMO SE SELECIONA O CAMINHO DE MEMORIA NO GFX8
 *  ---------------------------------------------------------------------------
 *  A instrucao flat_load_dword carrega dois bits que escolhem o caminho
 *  (GCN3 ISA, secao 9.1). Encodings efetivamente presentes no binario:
 *
 *    DC500000   GLC=0 SLC=0   le pelo vL1 (16 KB por CU) — padrao
 *    DC510000   GLC=1 SLC=0   ignora o vL1; o acesso e atendido pela L2 (2 MB)
 *    DC530000   GLC=1 SLC=1   ignora o vL1 e marca a linha da L2 como
 *                             evict-first (streaming) — NAO ignora a L2
 *
 *  GLC e o bit 16 e SLC o bit 17 do encoding FLAT. Os kernels 9 e 10 usam
 *  MUBUF em vez de FLAT, onde GLC e o bit 14: E0501000 -> E0505000.
 *
 *  ---------------------------------------------------------------------------
 *  2. "LAST USE" AQUI E O SLC — E ELE NAO SAI DE GRACA
 *  ---------------------------------------------------------------------------
 *  Nao existe no GFX8 uma instrucao que descarte a linha depois de usada (o
 *  ld.global.lu do PTX, por exemplo, nao tem equivalente). O que existe e
 *  SLC=1: a linha entra na L2 ja marcada como primeira candidata a eviccao,
 *  que e a forma que a arquitetura tem de dizer "este dado nao sera reusado".
 *  A dica esta nos kernels 6 (asm com glc slc) e 7 (__builtin_nontemporal_-
 *  load), e seu efeito ISOLADO e a razao k6/k5 — mesmo codigo, mesmo GLC,
 *  so o bit 17 mudando:
 *
 *    k6/k5   1.018x (N=1024)   1.013x   1.027x   1.052x   1.118x (N=16384)
 *
 *  Marcar o load como de ultimo uso custa de 2% a 12%, e o custo CRESCE com o
 *  tamanho. Faz sentido: no matmul naive o dado NAO e de uso unico — cada
 *  linha de A e cada coluna de B sao relidas por varios work-items, entao
 *  antecipar a eviccao joga fora reuso que de fato existia, e quanto maior a
 *  matriz mais reuso se perde. A dica so seria neutra num acesso realmente
 *  de passagem unica.
 *
 *  ---------------------------------------------------------------------------
 *  3. SLC NAO E BYPASS DE L2 — NAO EXISTE "DIRETO A VRAM" PELO SHADER
 *  ---------------------------------------------------------------------------
 *  Medido em experimentos/slc_l2.cpp (resultados/slc_l2.txt): varrendo um
 *  buffer de 1 MB, que cabe na L2 de 2 MB, os loads com GLC=1 SLC=1 sustentam
 *  555.9 GB/s — contra 587.8 GB/s sem flag nenhuma, ou seja, 5% de perda, nao
 *  um desligamento. O mesmo codigo sobre 64 MB, que nao cabe na L2, satura em
 *  226.3 GB/s, a banda real da GDDR5 desta placa. Se SLC=1 desligasse a L2, o
 *  caso de 1 MB teria caido para esses mesmos ~226 GB/s; ficou 2.5x acima,
 *  logo a L2 continua servindo. O cenario "direto a VRAM" da formulacao
 *  original do trabalho nao e alcancavel nesta arquitetura, e o kernel 6 nao
 *  deve ser descrito assim.
 *
 *  ---------------------------------------------------------------------------
 *  4. O QUE NAO CONTROLA ESSAS FLAGS NESTE ALVO
 *  ---------------------------------------------------------------------------
 *    volatile     e uma propriedade do modelo de memoria da linguagem, nao um
 *                 seletor de cache. O mapeamento volatile -> GLC e decisao de
 *                 codegen do backend e varia por compilador: nao gera flag no
 *                 AMDOCL (Windows) nem no RGA usado para gfx803, mas gera
 *                 GLC=1 no LLVM do ROCm 5.2. Nao e mecanismo estavel. Os
 *                 kernels 2 e 3 preservam essa hipotese original: nao sao
 *                 redundantes, sao o objeto da comparacao multiplataforma. E
 *                 no ROCm o volatile faz DUAS coisas — alem do glc, serializa
 *                 os acessos: o laco do k2 tem DOIS s_waitcnt vmcnt(0), um
 *                 por load, contra UM no k1. O tempo dele mistura bypass com
 *                 perda de sobreposicao de latencia, e por isso seu controle
 *                 e o kernel 4, nao o kernel 1.
 *
 *    __attribute__((address_space(1)))
 *                 nao faz bypass. No mapeamento do AMDGPU, address space 1 E
 *                 o espaco global — o mesmo que __global. Em modo OpenCL o
 *                 Clang ainda trata os dois como espacos distintos e rejeita a
 *                 conversao ("changes address space of pointer"); quando
 *                 declarado direto no parametro do kernel, compila e gera ISA
 *                 sem nenhuma flag de cache.
 *
 *    STOREs       nao saem com GLC=1: os 10 flat_store_dword deste binario
 *                 (um por kernel) tem GLC=0. As escritas alcancam a L2 porque
 *                 o vL1 do GCN e write-through por construcao, e nao por
 *                 causa do bit GLC.
 *
 *  ---------------------------------------------------------------------------
 *  5. AS TRES VIAS USADAS PARA PEDIR AS FLAGS
 *  ---------------------------------------------------------------------------
 *    inline asm                    kernels 4, 5 e 6 — controle explicito de
 *                                  glc e slc, mas opaco ao otimizador; o
 *                                  preco esta medido no kernel 4
 *    __builtin_nontemporal_load    kernel 7 — o otimizador continua agindo,
 *                                  mas so consegue glc e slc JUNTOS
 *    llvm.amdgcn.raw.buffer.load   kernels 9 e 10 — cachepolicy e um inteiro,
 *                                  o que permite glc SOZINHO sobre codigo
 *                                  otimizado; e a celula que faltava
 *
 *  As tres exigem compilador OpenCL baseado em Clang/LLVM. No ROCm/Linux
 *  funcionam; no AMDOCL do Windows o inline asm nao compila ("use of
 *  undeclared identifier '__asm__'"), o que e em si um resultado da
 *  comparacao multiplataforma. Os kernels 7, 9 e 10 dependem de builtins e de
 *  intrinsics do Clang e NAO foram verificados no AMDOCL — nao afirmar nada
 *  sobre eles no Windows sem compilar la.
 *
 *  ---------------------------------------------------------------------------
 *  6. O QUE O CRONOMETRO MEDE — O OFFLOADING JA ESTA FORA
 *  ---------------------------------------------------------------------------
 *  O tempo de cada cenario vem de CL_PROFILING_COMMAND_START/END do evento do
 *  kernel: e tempo de EXECUCAO NA GPU. A e B sao enviados uma unica vez,
 *  antes de qualquer medicao, e a leitura de C acontece depois do laco
 *  cronometrado, so para validar. O offloading portanto ja esta excluido por
 *  construcao — nao ha o que descontar depois.
 *
 *  Para que a exclusao fique documentada e nao apenas afirmada, a rotina
 *  mede_offloading() cronometra H2D, D2H e a latencia de lancamento com os
 *  mesmos eventos de profiling, grava em resultados/matmul_offload.csv e a
 *  tabela final mostra a estimativa fim-a-fim de cada cenario.
 *
 *  Sobre "medir a porcentagem no cenario 1 e aplicar nos demais": as
 *  transferencias sao IDENTICAS em todos os cenarios — mesmos buffers, mesmos
 *  tamanhos, mesmo sentido, nenhuma delas depende das flags de cache. Logo o
 *  que se propaga entre cenarios e o valor ABSOLUTO em ms, e nao a
 *  porcentagem: como o denominador (o tempo de kernel) muda de cenario para
 *  cenario, a mesma transferencia pesa mais no k1, que e o mais rapido, e
 *  menos no k2. Aplicar a porcentagem do k1 aos outros superestimaria o
 *  offloading deles. Por isso o programa reporta os ms medidos e deriva a
 *  porcentagem de cada cenario, em vez do contrario.
 *
 *  Compilar (Linux / ROCm):
 *    source setup/env_rocm.sh
 *    g++ -Wall -Wextra -O2 experimentos/matmul.cpp -o output/matmul -lOpenCL
 *
 *  Compilar (Windows / MinGW — os kernels 4, 5 e 6 NAO compilam la):
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
#include <time.h>
#include <CL/cl.h>

/* ------------------------------------------------------------------------- */
/*  10 kernels de multiplicacao de matrizes C = A * B                        */
/*  Matrizes quadradas NxN, row-major, um work-item por elemento             */
/*  (o kernel 8 e a excecao: tiled 16x16 com reuso em LDS)                   */
/* ------------------------------------------------------------------------- */

static const char* kernel_source =
"/* ====================================================================== */\n"
"/* DESENHO EXPERIMENTAL                                                   */\n"
"/*                                                                        */\n"
"/* Nove dos dez kernels calculam C = A*B com o MESMO algoritmo naive (um  */\n"
"/* work-item por elemento). Variam apenas no caminho de cache dos LOADs   */\n"
"/* de A e B, dado pelos bits GLC/SLC (GCN3/GFX8, secao 9.1 do ISA):       */\n"
"/*                                                                        */\n"
"/*   GLC=0            -> le do vL1 (comportamento padrao)                 */\n"
"/*   GLC=1            -> ignora o vL1; o acesso e atendido pela L2        */\n"
"/*   GLC=1 e SLC=1    -> ignora o vL1 e marca a linha da L2 como          */\n"
"/*                       evict-first. NAO e acesso direto a VRAM.         */\n"
"/*                                                                        */\n"
"/* As flags sao pedidas por tres vias de proposito, porque cada via cobra */\n"
"/* um preco diferente em codigo gerado. Contagem sobre o laco interno em  */\n"
"/* isa/matmul_gfx803_disasm.txt (o k8 e tiled e nao entra na comparacao): */\n"
"/*                                                                        */\n"
"/*   via              kernels   instr/iter   s_waitcnt vmcnt   VGPRs      */\n"
"/*   C puro                 1        15             1            11       */\n"
"/*   volatile            2, 3        16             2            11       */\n"
"/*   inline asm       4, 5, 6        20             1            11       */\n"
"/*   builtin               7        15             1            11        */\n"
"/*   intrinsic         9, 10         9             1             7        */\n"
"/*                                                                        */\n"
"/* Os kernels 2 e 3 sao a HIPOTESE ORIGINAL do trabalho, preservada: a    */\n"
"/* tentativa de obter bypass com o qualificador volatile. Nao sao         */\n"
"/* redundantes — sao o objeto da comparacao multiplataforma. O mesmo      */\n"
"/* fonte produz ISA diferente conforme o compilador OpenCL:               */\n"
"/*                                                                        */\n"
"/*   AMDOCL (driver Windows)  -> sem flag   (GLC=0) — hipotese refutada   */\n"
"/*   RGA usado para gfx803    -> sem flag   (GLC=0)                       */\n"
"/*   Clang/LLVM do ROCm 5.2   -> flag glc   (GLC=1) — hipotese confirmada */\n"
"/*                                                                        */\n"
"/* COMPARACOES VALIDAS — cada efeito com o seu controle:                  */\n"
"/*                                                                        */\n"
"/*   k1 vs k7   MEDIDA PRINCIPAL. Conferido no disassembly: as duas       */\n"
"/*              listagens sao identicas instrucao a instrucao, diferindo  */\n"
"/*              so pelos bits glc slc em dois flat_load_dword.            */\n"
"/*   k9 vs k10  SEGUNDA MEDIDA LIMPA, e a unica que isola o GLC SEM o SLC */\n"
"/*              junto: identicas exceto o glc em dois buffer_load_dword.  */\n"
"/*   k2 vs k3   repetibilidade — fontes identicos, ISA byte a byte igual; */\n"
"/*              a diferenca medida entre eles e o piso de ruido da        */\n"
"/*              bancada (0.1% a 1.2% conforme o tamanho).                 */\n"
"/*   k1 vs k4   custo do inline asm, sem nenhuma flag envolvida.          */\n"
"/*   k4 vs k5   efeito do GLC, ja descontado o custo do asm.              */\n"
"/*   k5 vs k6   efeito do SLC (a dica de ultimo uso) isolado.             */\n"
"/*   k1 vs k8   quanto o reuso em LDS ganha sem tocar em flag nenhuma.    */\n"
"/*                                                                        */\n"
"/* NAO comparar k1 com k5 diretamente: isso soma o custo do asm ao efeito */\n"
"/* do GLC e produz um numero que nao mede nem um nem outro. O controle do */\n"
"/* k5 e o k4. Pela mesma razao o controle do k2 e o k4, e nao o k1.       */\n"
"/* ====================================================================== */\n"
"\n"
"/* ====================================================================== */\n"
"/* Kernel 1 — LINHA DE BASE: caminho padrao vL1 -> L2 -> VRAM             */\n"
"/* ISA gerada: flat_load_dword v9, v[1:2] / v4, v[7:8]  (DC500000)        */\n"
"/* Laco de 15 instrucoes com UM s_waitcnt vmcnt(0), servindo aos dois     */\n"
"/* loads sobrepostos. O otimizador aplica reducao de forca no endereco de */\n"
"/* A: o ponteiro avanca com v_add_u32 +4 por iteracao, sem refazer a      */\n"
"/* aritmetica de 64 bits. Guardar isso: e exatamente o que os kernels com */\n"
"/* asm perdem, e a razao de eles serem 1.5x mais lentos sem nenhuma flag. */\n"
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
"/* ATENCAO ao interpretar o tempo deste kernel: sob o ROCm o volatile faz */\n"
"/* DUAS coisas de uma vez. Alem do glc, ele serializa os acessos — o laco */\n"
"/* deste kernel tem DOIS s_waitcnt vmcnt(0), um apos cada load, enquanto  */\n"
"/* o k1 tem UM so para os dois loads sobrepostos (conferido na ISA). O    */\n"
"/* tempo medido portanto mistura bypass de L1 com perda de sobreposicao   */\n"
"/* de latencia, e o controle correto e o kernel 4, nao o kernel 1.        */\n"
"/*                                                                        */\n"
"/* Confirmacao experimental: o k2 fica proximo do k5 (asm com glc, um     */\n"
"/* waitcnt so) — razao k2/k5 entre 0.995x e 1.053x nos cinco tamanhos,    */\n"
"/* enquanto k2/k1 vai de 1.59x a 1.92x. Proximo nao e igual: as duas vias */\n"
"/* pagam custos diferentes (serializacao no k2, aritmetica de endereco no */\n"
"/* k5) que por acaso dao quase no mesmo.                                  */\n"
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
"/* VRAM; a analise da ISA mostrou os dois kernels byte a byte iguais (o   */\n"
"/* diff das duas listagens no disassembly e vazio), o que refutou a       */\n"
"/* hipotese. Mantido por dois motivos: e a evidencia dessa refutacao e,   */\n"
"/* na bancada, mede a repetibilidade — dois kernels de ISA identica devem */\n"
"/* dar o mesmo tempo dentro do ruido. Medido: k3/k2 = 0.988x, 1.005x,     */\n"
"/* 1.001x, 1.002x, 1.001x de N=1024 a N=16384. E esse o piso de ruido     */\n"
"/* usado la embaixo para julgar os demais efeitos.                        */\n"
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
"/* Mesma estrutura dos kernels 5 e 6, mesma barreira de otimizacao, mas   */\n"
"/* caminho de memoria identico ao kernel 1. Sem esse controle, comparar   */\n"
"/* k1 com k5 confundiria o custo do asm com o efeito do bit GLC.          */\n"
"/* ISA gerada: flat_load_dword v1, v[5:6] / v5, v[7:8]  (DC500000)        */\n"
"/*                                                                        */\n"
"/* ONDE O ASM CUSTA — e NAO e onde se supunha. Conferido na ISA:          */\n"
"/*   - o laco NAO deixa de ser desenrolado por causa do asm: nem o k1 nem */\n"
"/*     este kernel sao desenrolados (um v_mac_f32 por iteracao nos dois); */\n"
"/*   - o asm NAO acrescenta espera: os dois tem UM s_waitcnt vmcnt(0) por */\n"
"/*     iteracao (o kernel que serializa e o k2, com dois).                */\n"
"/* O custo real e o calculo de endereco. A restricao \"v\" obriga o         */\n"
"/* compilador a materializar um ponteiro de 64 bits completo para A e     */\n"
"/* para B a cada iteracao (v_ashrrev + v_lshlrev_b64 + v_add/v_addc para  */\n"
"/* cada um), enquanto o k1 mantem o endereco de A num par de VGPRs que    */\n"
"/* apenas avanca 4 bytes. Resultado: laco de 20 instrucoes contra 15, e   */\n"
"/* ainda um s_waitcnt lgkmcnt(0) que o k1 nao tem dentro do laco.         */\n"
"/* Medido: k4/k1 = 1.52x, 1.51x, 1.50x, 1.47x, 1.46x de N=1024 a 16384 —  */\n"
"/* estavel, como se espera de um custo de codigo e nao de memoria.        */\n"
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
"/* Difere do KERNEL 4 por exatamente um bit da instrucao (bit 16 do       */\n"
"/* encoding FLAT no GFX8): DC510000 em vez de DC500000. Conferido: o diff */\n"
"/* das duas listagens tem so as duas linhas dos loads.                    */\n"
"/* ISA gerada: flat_load_dword v, v[..] glc         (GLC=1, SLC=0)        */\n"
"/* Efeito medido, ja descontado o asm (k5/k4): 1.20x em N=1024 e ~1.06x   */\n"
"/* de N=4096 em diante. Ver tambem o par k9/k10, que mede o mesmo bit     */\n"
"/* sobre codigo otimizado e da um efeito bem maior.                       */\n"
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
"/* Kernel 6 — BYPASS DE L1 + DICA DE ULTIMO USO NA L2: GLC=1 e SLC=1      */\n"
"/* SLC=1 nao \"desliga\" a L2: marca a linha como evict-first, isto e,      */\n"
"/* declara que o dado nao sera reusado. E o que existe no GFX8 de mais    */\n"
"/* proximo de um load de ultimo uso — nao ha instrucao que descarte a     */\n"
"/* linha. A L2 CONTINUA servindo o acesso: medido em slc_l2.cpp, 555.9    */\n"
"/* GB/s sobre um buffer de 1 MB que cabe na L2, contra 226.3 GB/s sobre   */\n"
"/* 64 MB que nao cabe. Nao chamar este kernel de \"acesso direto a VRAM\" — */\n"
"/* esse cenario nao existe no GFX8 pelo lado do shader.                   */\n"
"/* ISA gerada: flat_load_dword v, v[..] glc slc     (DC530000)            */\n"
"/* Efeito isolado do SLC (k6/k5): 1.018x, 1.013x, 1.027x, 1.052x, 1.118x  */\n"
"/* de N=1024 a N=16384 — a dica de ultimo uso e sempre um custo aqui, e   */\n"
"/* um custo crescente, porque o matmul naive de fato reusa os dados.      */\n"
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
"/* Kernel 7 — MEDIDA PRINCIPAL: as flags sem inline asm                   */\n"
"/* __builtin_nontemporal_load e um builtin do Clang: no backend AMDGPU    */\n"
"/* para GFX8 ele emite GLC=1 e SLC=1, o mesmo encoding do KERNEL 6        */\n"
"/* (DC530000), sem bloco asm. Serve para mostrar que as flags dos         */\n"
"/* kernels 5 e 6 nao sao artefato do inline asm, e semanticamente e o     */\n"
"/* jeito portavel de dizer \"este dado nao sera reusado\".                  */\n"
"/*                                                                        */\n"
"/* Como o Clang mantem as otimizacoes, a ISA deste kernel e identica a do */\n"
"/* kernel 1 instrucao a instrucao — conferido: o diff das duas listagens  */\n"
"/* tem exatamente duas linhas, os dois flat_load_dword, e o laco continua */\n"
"/* com 15 instrucoes e um s_waitcnt. E a comparacao mais limpa de flags   */\n"
"/* de cache que este experimento produz.                                  */\n"
"/* ISA gerada: flat_load_dword v, v[..] glc slc     (DC530000)            */\n"
"/* Medido (k7/k1): 1.78x, 1.55x, 1.48x, 1.46x, 1.59x de N=1024 a 16384.   */\n"
"/* Como aqui glc e slc vem juntos, este numero e o efeito COMBINADO; para */\n"
"/* separar os dois bits sao necessarios os pares k5/k4 e k6/k5.           */\n"
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
"/* Nao mexe em flags de cache: reduz o trafego reaproveitando os dados em */\n"
"/* memoria local. E o limite superior de desempenho da comparacao — e o   */\n"
"/* unico kernel cujo laco e desenrolado (76 instrucoes, 16 v_mac_f32 por  */\n"
"/* iteracao de tile, contra 1 nos demais). Medido (k1/k8): 3.34x, 3.42x,  */\n"
"/* 3.45x, 3.42x, 3.20x — reorganizar o acesso vale muito mais do que      */\n"
"/* qualquer coisa que se peca ao cache por flag.                          */\n"
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
"/* ====================================================================== */\n"
"/* Kernels 9 e 10 — INTRINSIC DO LLVM, sem bloco de assembly              */\n"
"/*                                                                        */\n"
"/* llvm.amdgcn.raw.buffer.load.f32 e um intrinsic do backend AMDGPU, nao  */\n"
"/* do HIP: HIP e OpenCL sao front-ends que desembocam nele. A declaracao  */\n"
"/* abaixo o amarra por asm label, idioma que a Composable Kernel usa.     */\n"
"/*                                                                        */\n"
"/* O ultimo parametro NAO sao dois booleanos glc e slc: e UM inteiro      */\n"
"/* cachepolicy, com bit0 = glc e bit1 = slc. E por isso que este par e o  */\n"
"/* unico caminho para pedir glc SOZINHO — o builtin nontemporal so emite  */\n"
"/* glc+slc juntos, e o asm cobra a barreira de otimizacao.                */\n"
"/*                                                                        */\n"
"/* HIPOTESE CONFIRMADA. Diferentemente do bloco __asm__, opaco ao         */\n"
"/* otimizador, o intrinsic tem semantica conhecida pelo compilador, que   */\n"
"/* escalona em volta dele. Conferido na ISA: o laco destes dois kernels   */\n"
"/* tem 9 instrucoes, contra 15 do k1 e 20 dos kernels com asm, e usa 7    */\n"
"/* VGPRs contra 11. O ganho vem do formato MUBUF: o descritor fica em     */\n"
"/* SGPRs (s[0:3] e s[4:7]) e o endereco vira um voffset de 32 bits, de    */\n"
"/* modo que some a aritmetica de 64 bits que o k1 ainda precisa fazer.    */\n"
"/* Consequencia: o k9 e MAIS RAPIDO que o k1 em C puro — k9/k1 = 0.78x,   */\n"
"/* 0.74x, 0.78x, 0.78x, 0.79x (396 contra 307 GFLOPS em N=1024).          */\n"
"/*                                                                        */\n"
"/* Com isso o par 9/10 mede o bit glc SOZINHO sobre codigo otimizado — a  */\n"
"/* celula que faltava no desenho. As duas listagens diferem apenas pelo   */\n"
"/* glc em dois buffer_load_dword (E0501000 -> E0505000, bit 14 do MUBUF), */\n"
"/* e o efeito medido e o maior de todos: k10/k9 = 2.46x, 2.07x, 1.78x,    */\n"
"/* 1.69x, 1.65x de N=1024 a N=16384. Quanto melhor o codigo, mais caro    */\n"
"/* fica contornar a L1 — o que explica por que o mesmo bit parece custar  */\n"
"/* so 1.06x quando medido sobre os kernels com asm (k5/k4).               */\n"
"/*                                                                        */\n"
"/* Custo do metodo: o formato MUBUF exige um descritor de recurso de 128  */\n"
"/* bits. num_records (palavra 2) faz bounds check em hardware — se ficar  */\n"
"/* curto, o acesso fora do limite devolve ZERO em silencio, sem erro.     */\n"
"/* Dai a validacao contra a CPU ser obrigatoria nestes dois kernels.      */\n"
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



/*
 * ---------------------------------------------------------------------------
 *  OFFLOADING: o que o cronometro dos cenarios NAO mede
 * ---------------------------------------------------------------------------
 *
 * Os tempos reportados por cenario vem de CL_PROFILING_COMMAND_START/END do
 * evento do kernel, ou seja, sao tempo de execucao na GPU. As transferencias
 * ficam de fora por construcao: A e B sobem uma unica vez, na criacao dos
 * buffers, e a leitura de C acontece depois do laco cronometrado.
 *
 * Esta rotina existe para que a exclusao fique MEDIDA, e nao apenas afirmada.
 * Ela cronometra, com os mesmos eventos de profiling:
 *
 *   H2D          envio de A e B  (2 x mat_size)
 *   D2H          leitura de C    (1 x mat_size)
 *   lancamento   CL_PROFILING_COMMAND_QUEUED -> COMMAND_START de uma NDRange
 *                minima (um unico work-group), que e a latencia de enfileirar
 *                e despachar sem contar trabalho util
 *
 * Sobre propagar isso para os demais cenarios: as transferencias sao as
 * MESMAS em todos eles — mesmos buffers, mesmos tamanhos, mesmo sentido, e
 * nenhuma depende das flags de cache. Logo o que vale entre cenarios e o
 * valor ABSOLUTO em ms, nao a porcentagem. A porcentagem e derivada: como o
 * denominador (o tempo de kernel) muda de cenario para cenario, a mesma
 * transferencia pesa mais no cenario rapido e menos no lento. Medir a
 * porcentagem no kernel 1 e aplica-la aos outros superestimaria o offloading
 * deles, porque o kernel 1 e justamente o mais rapido da lista.
 *
 * O peso tambem cai depressa com N: a transferencia cresce com N^2 e o kernel
 * com N^3, de modo que o offloading so e relevante nos tamanhos pequenos.
 */
struct Offload { double h2d_ms; double d2h_ms; double lanc_ms; int n; };

static double mediana_de(double* v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static struct Offload mede_offloading(cl_command_queue queue,
                                      cl_mem d_A, cl_mem d_B, cl_mem d_C,
                                      const float* h_A, const float* h_B,
                                      float* h_C, size_t mat_size,
                                      cl_kernel k_ref, int n)
{
    struct Offload o; o.h2d_ms = 0.0; o.d2h_ms = 0.0; o.lanc_ms = 0.0; o.n = n;
    double* v_h2d  = (double*)malloc((size_t)n * sizeof(double));
    double* v_d2h  = (double*)malloc((size_t)n * sizeof(double));
    double* v_lanc = (double*)malloc((size_t)n * sizeof(double));
    if (!v_h2d || !v_d2h || !v_lanc) { free(v_h2d); free(v_d2h); free(v_lanc);
                                       o.n = 0; return o; }

    for (int i = 0; i < n; i++) {
        cl_event ea, eb, ec;
        clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, mat_size, h_A, 0, NULL, &ea);
        clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, mat_size, h_B, 0, NULL, &eb);
        clEnqueueReadBuffer (queue, d_C, CL_TRUE, 0, mat_size, h_C, 0, NULL, &ec);
        v_h2d[i] = get_time_ms(ea) + get_time_ms(eb);
        v_d2h[i] = get_time_ms(ec);
        clReleaseEvent(ea); clReleaseEvent(eb); clReleaseEvent(ec);

        /* NDRange minima: um work-group 16x16. Nao interessa o resultado, so
           a distancia entre enfileirar e comecar a executar. */
        size_t gs[2] = { 16, 16 }, ls[2] = { 16, 16 };
        cl_event ek;
        if (clEnqueueNDRangeKernel(queue, k_ref, 2, NULL, gs, ls, 0, NULL, &ek)
            == CL_SUCCESS) {
            clWaitForEvents(1, &ek);
            cl_ulong tq = 0, ts = 0;
            clGetEventProfilingInfo(ek, CL_PROFILING_COMMAND_QUEUED,
                                    sizeof(cl_ulong), &tq, NULL);
            clGetEventProfilingInfo(ek, CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong), &ts, NULL);
            v_lanc[i] = (ts > tq) ? (double)(ts - tq) / 1e6 : 0.0;
            clReleaseEvent(ek);
        } else {
            v_lanc[i] = 0.0;
        }
    }

    o.h2d_ms  = mediana_de(v_h2d,  n);
    o.d2h_ms  = mediana_de(v_d2h,  n);
    o.lanc_ms = mediana_de(v_lanc, n);
    free(v_h2d); free(v_d2h); free(v_lanc);
    return o;
}



/* ------------------------------------------------------------------------- */
/*  Main                                                                     */
/* ------------------------------------------------------------------------- */

int main(int argc, char** argv) {
    cl_int err;

    /*
     * Uso:  ./output/matmul [RUNS] [N] [WARMUP] [LINHAS_POR_LOTE]
     *
     * RUNS   execucoes medidas por cenario (padrao 1000). Sao 10 cenarios;
     *        em N=1024 uma rodada completa dos dez custa ~104 ms, entao
     *        RUNS=1000 da ~1,7 min de GPU e RUNS=100 da ~10 s. As campanhas
     *        ja publicadas usaram RUNS=1000 em N=1024
     *        (resultados/matmul_cache_flags.csv) e RUNS=100 na varredura de
     *        tamanhos (resultados/matmul_sweep.csv). RUNS baixo serve para
     *        uma passada rapida, mas afirmacao sobre diferenca de poucos por
     *        cento — o efeito do SLC, por exemplo — precisa de RUNS alto o
     *        bastante para que o erro padrao fique abaixo do efeito.
     * N      lado da matriz (padrao 1024, ~12 MB nos tres buffers).
     * WARMUP execucoes descartadas antes de medir (padrao 10).
     */
    /*
     * RUNS = 0 pede o modo adaptativo: em vez de um numero fixo de execucoes,
     * fixa-se um ORCAMENTO DE TEMPO por cenario e deriva-se o numero de
     * execucoes da duracao medida no aquecimento. Sem isso a varredura de
     * tamanhos e inviavel: uma rodada dos dez cenarios custa 0,10 s em
     * N=1024 e 387 s em N=16384 (medido, resultados/matmul_sweep.csv), de
     * modo que 1000 execucoes dariam 1,7 min no primeiro caso e 4,5 dias no
     * segundo, porque o kernel e O(N^3). Foi por isso que a varredura rodou
     * com RUNS=100: 10 s em N=1024 e ~11 h em N=16384.
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

    /*
     * -- A memoria da placa NAO e unificada com a do host --
     *
     * Pergunta recorrente ao ler os tempos: a L2 da GPU e a RAM do sistema
     * sao o mesmo espaco? Nao, e o proprio runtime diz isso. Numa Polaris
     * discreta a hierarquia e vL1 (16 KB/CU) -> L2 (2 MB) -> GDDR5 da placa,
     * tudo dentro da GPU; a RAM do host so e alcancada pelo PCIe, por
     * transferencia explicita. Por isso existe custo de offloading, e por
     * isso ele e uma etapa separada e nao um acesso de memoria.
     *
     * Os tres campos abaixo sustentam a afirmacao:
     *   CL_DEVICE_HOST_UNIFIED_MEMORY  0  nesta placa (num APU seria 1)
     *   CL_DEVICE_GLOBAL_MEM_SIZE      8 GB, que e a VRAM, nao os 14 GB do host
     *   CL_DEVICE_GLOBAL_MEM_CACHE_SIZE  16 KB
     *
     * ATENCAO ao terceiro: 16 KB e o tamanho do vL1, nao o da L2. O driver
     * reporta o primeiro nivel nesse campo e nao expoe a L2 de 2 MB por
     * nenhuma consulta de OpenCL — o valor de 2 MB vem da documentacao da
     * arquitetura e da curva medida em experimentos/pchase_v2.cpp. Nao citar
     * esse campo como se fosse a L2.
     */
    cl_bool unificada = CL_FALSE;
    cl_ulong mem_global = 0, cache_global = 0, lds = 0;
    cl_uint  linha_cache = 0;
    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY,
                    sizeof(unificada), &unificada, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(mem_global), &mem_global, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                    sizeof(cache_global), &cache_global, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                    sizeof(linha_cache), &linha_cache, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lds), &lds, NULL);
    printf("Memoria: %.1f GB de VRAM | cache reportado pelo driver %.0f KB "
           "(e o vL1, nao a L2) | linha %u B | LDS %.0f KB\n",
           (double)mem_global / (1024.0*1024.0*1024.0),
           (double)cache_global / 1024.0, linha_cache, (double)lds / 1024.0);
    printf("Memoria unificada com o host: %s — a L2 e a VRAM ficam na placa, a\n",
           unificada ? "SIM" : "NAO");
    printf("RAM do sistema so e alcancada por transferencia explicita no PCIe.\n\n");


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
    /*
     * A ordem importa para a leitura da tabela final: k1 a k7 sao o mesmo
     * algoritmo naive variando so o caminho de cache (k1 em C puro, k2/k3
     * com volatile, k4/k5/k6 com inline asm, k7 com builtin), k8 e a
     * referencia com LDS e k9/k10 sao o par do intrinsic, que isola o glc
     * sobre codigo otimizado.
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
    /*
     * A CPU entra aqui como REFERENCIA DE CORRECAO, nao como concorrente: o
     * kernel 1 e conferido contra ela e os demais cenarios contra o kernel 1.
     * O tempo abaixo e cronometrado so para que o custo dessa etapa fique
     * explicito e para dar uma ordem de grandeza — e um matmul ingenuo de uma
     * unica thread, sem blocagem e sem SIMD explicita, entao NAO serve de
     * baseline de CPU para o artigo. A extrapolacao para a matriz inteira e
     * exata no numero de operacoes (o custo por posicao e o mesmo O(N)), mas
     * otimista quanto a cache: calcular N*N posicoes seguidas tem localidade
     * melhor do que 4096 posicoes sorteadas.
     */
    printf("Referencia da CPU sobre %d posicoes amostradas...\n", N_AMOSTRA);
    clock_t t0_cpu = clock();
    matmul_cpu_amostra(h_A, h_B, N, amostra_idx, amostra_ref, N_AMOSTRA);
    double cpu_amostra_s = (double)(clock() - t0_cpu) / CLOCKS_PER_SEC;
    double cpu_total_est_s = cpu_amostra_s *
                             ((double)N * (double)N / (double)N_AMOSTRA);
    printf("Referencia pronta: %.3f s para %d posicoes (1 thread).\n",
           cpu_amostra_s, N_AMOSTRA);
    printf("Extrapolando para as %.0f posicoes da matriz: ~%.1f s na CPU.\n\n",
           (double)N * N, cpu_total_est_s);

    /*
     * -- Offloading: medir o que fica de fora do cronometro dos cenarios --
     *
     * 100 medicoes por padrao, como nos cenarios. Acima de 64 MB por matriz
     * (N > 4096) cada iteracao move 3 x mat_size pelo PCIe e a etapa passaria
     * de um minuto sem acrescentar precisao — nesses tamanhos o offloading ja
     * e desprezivel diante de um kernel de dezenas de segundos —, entao o
     * numero de medicoes cai para 20.
     */
    const int n_off = (mat_size <= (size_t)64 * 1024 * 1024) ? 100 : 20;
    printf("Medindo offloading (%d medicoes de H2D, D2H e lancamento)...\n", n_off);
    fflush(stdout);
    struct Offload off = mede_offloading(queue, d_A, d_B, d_C, h_A, h_B, h_C,
                                         mat_size, cen[0].k, n_off);
    const double off_ms = off.h2d_ms + off.d2h_ms + off.lanc_ms;
    printf("  H2D (A+B, %.1f MB): %.3f ms | D2H (C, %.1f MB): %.3f ms | "
           "lancamento: %.3f ms\n",
           2.0 * (double)mat_size / (1024.0*1024.0), off.h2d_ms,
           (double)mat_size / (1024.0*1024.0), off.d2h_ms, off.lanc_ms);
    printf("  Total de transferencia por execucao fim-a-fim: %.3f ms "
           "(NAO entra nos tempos abaixo)\n\n", off_ms);

    /* Uma linha por N, em append: o offloading nao depende do cenario, entao
       nao faz sentido repeti-lo dez vezes no CSV do sweep — que, alem disso,
       ja esta publicado com o cabecalho atual e nao pode ganhar colunas sem
       invalidar as linhas antigas. */
    {
        int off_novo = (access("resultados/matmul_offload.csv", F_OK) != 0);
        FILE* fo = fopen("resultados/matmul_offload.csv", "a");
        if (fo) {
            if (off_novo) fprintf(fo, "N,mat_mb,n,h2d_ms,d2h_ms,lancamento_ms,"
                                      "total_ms,cpu_amostra_s,cpu_total_est_s\n");
            fprintf(fo, "%d,%.3f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    N, (double)mat_size / (1024.0*1024.0), off.n,
                    off.h2d_ms, off.d2h_ms, off.lanc_ms, off_ms,
                    cpu_amostra_s, cpu_total_est_s);
            fclose(fo);
        }
    }


    /* -- Executar cada cenario -- */
    size_t local_size_normal[2] = { 16, 16 };   /* 256 threads */
    size_t local_size_tiled[2]  = { 16, 16 };   /* TILE_SIZE = 16 */

    /*
     * O tamanho de RUNS e o que decide quais efeitos este experimento pode
     * afirmar. Os efeitos pequenos aqui sao o do bit SLC (k6/k5, medido entre
     * 1.018x e 1.118x) e o piso de ruido da bancada (k3/k2, entre 0.988x e
     * 1.005x): para separar um do outro o erro padrao da media precisa cair
     * bem abaixo de 1%, o que exige RUNS na casa das centenas ou do milhar.
     * Com RUNS baixo o programa ainda roda, mas as linhas de razao no fim
     * sairao como "nao significativo" — e essa e a leitura honesta.
     *
     * A carga sustentada tambem expoe variacao de clock: a coluna "deriva"
     * compara a media do ultimo decimo das execucoes com a do primeiro. Valor
     * proximo de 0% indica que a GPU manteve o clock; valor positivo alto
     * sugere throttling termico, e nesse caso a media do cenario nao e
     * comparavel com a dos demais sem ressalva. Na campanha de N=1024 com
     * RUNS=1000 a deriva ficou em +9.2% no primeiro cenario medido e perto de
     * zero nos demais — sinal de que o aquecimento de 10 execucoes nao basta
     * para o primeiro da fila.
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
    printf("Offloading medido (mediana de %d medicoes) — NAO incluido nas colunas acima:\n", off.n);
    printf("  H2D (A+B) %.3f ms + D2H (C) %.3f ms + lancamento %.3f ms = %.3f ms\n",
           off.h2d_ms, off.d2h_ms, off.lanc_ms, off_ms);
    printf("Estimativa fim-a-fim por cenario (kernel + transferencias):\n");
    for (int k = 0; k < NUM_KERNELS; k++) {
        double e2e = cen[k].med_ms + off_ms;
        printf("  %-40s kernel %10.3f ms | fim-a-fim %10.3f ms | offloading %6.2f%%\n",
               cen[k].rotulo, cen[k].med_ms, e2e, 100.0 * off_ms / e2e);
    }
    printf("O offloading e IDENTICO em todos os cenarios em valor absoluto — mesmos buffers,\n");
    printf("mesmos tamanhos, e nenhuma transferencia depende das flags de cache. O que muda e\n");
    printf("o peso relativo, porque o denominador muda: por isso a porcentagem e derivada de\n");
    printf("cada cenario, e nao medida no k1 e transportada para os demais (o k1 e o mais\n");
    printf("rapido da lista, entao a porcentagem dele e a MAIOR de todas).\n");
    printf("Referencia da CPU (1 thread, ingenua, extrapolada da amostra): ~%.1f s para a\n",
           cpu_total_est_s);
    printf("matriz inteira, ~%.0fx o tempo do kernel 1. Numero informativo: nao e um baseline\n",
           cpu_total_est_s * 1000.0 / cen[0].med_ms);
    printf("de CPU otimizado e nao deve ser citado como comparacao CPU x GPU.\n");

    printf("----------------------------------------------------------------\n");
    printf("(*) k2 e k3 dependem do compilador: o mesmo fonte gera GLC=1 no\n");
    printf("    Clang/LLVM do ROCm e GLC=0 no AMDOCL do Windows. Confira sempre\n");
    printf("    a ISA realmente gerada antes de citar a flag.\n\n");
    printf("Comparacoes validas:\n");
    printf("  k1 vs k7  : MEDIDA PRINCIPAL — ISA identica instrucao a instrucao,\n");
    printf("              diferindo so pelos bits 'glc slc' em dois flat_load_dword\n");
    printf("  k9 vs k10 : SEGUNDA MEDIDA LIMPA, e a unica que isola o GLC sem o SLC:\n");
    printf("              identicas exceto o 'glc' em dois buffer_load_dword\n");
    printf("  k1 -> k2  : efeito do volatile NESTE compilador (nao portavel).\n");
    printf("              CUIDADO: sob o ROCm o volatile emite glc E serializa os\n");
    printf("              loads (s_waitcnt apos cada um). O controle dele e o k4,\n");
    printf("              nao o k1 — senao bypass e serializacao ficam confundidos\n");
    printf("  k2 vs k3  : repetibilidade — fontes identicos, tempos devem coincidir\n");
    printf("  k1 -> k4  : custo do inline asm, que NAO e perda de desenrolamento nem\n");
    printf("              espera extra (ambos tem 1 waitcnt por iteracao): e a\n");
    printf("              aritmetica de endereco de 64 bits refeita a cada volta\n");
    printf("  k4 -> k5  : efeito do bypass de L1 (bit GLC), ja descontado o asm\n");
    printf("  k5 -> k6  : efeito da dica de ultimo uso na L2 (bit SLC)\n");
    printf("  k6 vs k7  : mesmas flags por caminhos diferentes (asm x builtin)\n");
    printf("  k1 -> k9  : o intrinsic e MAIS RAPIDO que o C puro (laco de 9\n");
    printf("              instrucoes contra 15, 7 VGPRs contra 11)\n\n");
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
        printf("Amostras individuais gravadas em '%s'\n", csv_nome);
        printf("(%d cenarios x %d execucoes; o n de cada cenario tambem vai na\n",
               NUM_KERNELS, cen[0].runs);
        printf(" coluna 'runs' de resultados/matmul_sweep.csv)\n\n");
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