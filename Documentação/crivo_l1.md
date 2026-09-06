# Recuperando a Sensibilidade ao L1 num Algoritmo de Primos

**Experimento 9** — Ambiente: Ubuntu 20.04, ROCm 5.2, AMD RX 580 (Polaris / gfx803).

---

## 1. Resultado em uma frase

Trocando a **formulação** do teste de primalidade — sem trocar o problema — o
custo de contornar o L1 saltou de **1,00× para 7,01×**, o maior efeito medido em
todo o trabalho, quase **4× acima do stencil 3×3** (1,80×), que até aqui era o
caso mais sensível.

| Kernel | Perfil de memória | Mediana L1 | L2/L1 | VRAM/L1 | dp rel. |
|---|---|---|---|---|---|
| `crivo` (segmentado, Exp. 8) | tabela 2,2 KB, uniforme, dominado por escrita | 10,114 ms | 0,999× | 1,000× | 3,2% |
| `tdiv_unif` | tabela 2,4 KB, **muitos loads**, uniforme | 1,481 ms | **2,272×** | 2,210× | 6,4% |
| `tdiv_rot` | tabela 2,4 KB, **muitos loads**, disperso | 4,287 ms | **7,014×** | 6,984× | 9,5% |
| *(referência)* matmul *naive* | reúso 1024, disperso em 8 MB | — | 1,273× | 1,311× | — |
| *(referência)* stencil 3×3 | reúso 9, janela de 3 linhas | — | 1,796× | 1,943× | — |

As três variantes de cada kernel produzem o resultado correto: **69.581 primos**
em [3.000.000, 4.048.576) e **513.708** em [2²³, 2²⁴), idênticos à referência em
CPU.

---

## 2. O que mudou em relação ao Experimento 8

O crivo segmentado media 1,00× por dois motivos independentes, e a correção teve
de atacar o segundo:

| Fator | Crivo segmentado | Divisão tabelada |
|---|---|---|
| (i) Lanes por instrução de load | 1 endereço (uniforme) | 1 ou 64, conforme a variante |
| (ii) **Trabalho por load** | `seg/p` **escritas** | 1 multiplicação, **nenhuma escrita** |
| Loads globais no kernel | 9,2 M | **638 M** |
| Stores globais no kernel | ~19,6 M | 1,0 M |

A primeira tentativa considerada — rotacionar o índice da tabela por lane dentro
do crivo segmentado — **foi descartada por análise antes de ser implementada**.
Ela corrige (i) mas não (ii), e introduz um problema novo: o laço interno tem
*trip count* `seg/p`, de modo que lanes em primos diferentes divergem. No passo
em que `p = 2` cai na janela de 64 lanes, a wavefront inteira espera 256
iterações. O custo estimado é de ordem 60×, e o número de requisições de cache
permaneceria o mesmo. Corrigiria o padrão e pioraria o resultado.

### A formulação que funciona

**Divisão sucessiva tabelada**: cada work-item testa um candidato contra todos os
primos ímpares até √(vmax). O teste de divisibilidade usa a forma exata de
Granlund & Montgomery (1994):

$$p \mid v \iff v \cdot (p^{-1} \bmod 2^{32}) \le \left\lfloor \frac{2^{32}-1}{p} \right\rfloor$$

com $p^{-1} \bmod 2^{32}$ pré-computado no anfitrião por iteração de Newton. O
custo por primo é de **2 loads e ~5 instruções de ALU**, sem divisão inteira —
que no GCN custa cerca de 25 instruções e tornaria o kernel *compute-bound*,
reproduzindo o algoritmo B do Experimento 8. O *trip count* é fixo em
`n_primos` para todos os lanes, logo **não há divergência**.

Isso inverte a razão do algoritmo: de 2,1 escritas por load passa a 0,0016 --- um
fator de mais de mil.

---

## 3. O experimento controlado sobre a uniformidade

`tdiv_unif` e `tdiv_rot` executam **o mesmo trabalho, o mesmo número de loads e
o mesmo número de iterações**. Diferem apenas no índice inicial de cada lane na
tabela:

```c
#define ROT_UNIF (lane * 0u)              /* todos os lanes no mesmo endereço  */
#define ROT_DISP ((lane * 37u) % n_primos) /* 64 lanes em 64 posições distintas */
```

O passo 37 é primo com o tamanho da tabela, o que garante que os 64 lanes caiam
em 64 posições distintas. Como a tabela ocupa 1.216 B por array — **19 linhas de
cache de 64 B** — uma wavefront no modo disperso toca praticamente todas elas a
cada instrução de load.

Contagem de requisições de linha de cache por instrução de load:

| Variante | Linhas tocadas por wavefront | Requisições no kernel | Taxa medida (nível L1) |
|---|---|---|---|
| `tdiv_unif` | 1 | 9,96 M | 6,7 G req/s |
| `tdiv_rot` | ~19 | 189 M | **44,2 G req/s** |

A taxa de `tdiv_rot` no L1 equivale a **0,91 requisição por ciclo por CU**
(36 CUs × 1,35 GHz), praticamente o teto arquitetural de 1 requisição/ciclo/CU
do L1 vetorial do GCN. O kernel está limitado pela taxa de serviço do L1 — e é
exatamente por isso que retirar o L1 do caminho custa 7×: essa mesma taxa passa
a ter de ser sustentada pelo L2, que é compartilhado pelas 36 CUs.

---

## 4. O efeito é de vazão, não de latência

O Experimento 7 mediu, por P-chase com um único work-item, a **latência** de cada
nível:

| | sem flags | `glc` | razão |
|---|---|---|---|
| Latência (1–16 KB) | ~171 ns | ~244 ns | **1,42×** |

A razão de latência é 1,42×, mas o tempo de kernel do `tdiv_rot` piora 7,01×. A
diferença mede **contenção**: com um único work-item o L2 atende sem disputa;
com 36 CUs saturando requisições, ele se torna o gargalo. Este é o argumento
mais direto do trabalho para a tese de que a hierarquia de memória importa por
vazão agregada, e não apenas por latência de acesso individual.

Vale registrar a ressalva metodológica que atravessa todo o trabalho: o bloco de
*inline assembly* emite `s_waitcnt vmcnt(0)` após cada load, o que impede o
compilador de sobrepor loads dentro de um mesmo work-item. O paralelismo entre
wavefronts permanece, mas o *memory-level parallelism* intra-thread é suprimido —
o mesmo fator já quantificado no Experimento 5 (custo do asm = 1,54×). As
comparações entre níveis são válidas porque as três variantes compartilham
exatamente essa construção; o número absoluto não deve ser lido como o
desempenho de uma implementação otimizada.

---

## 5. Verificação de ISA

As contagens de instrução são idênticas entre os três níveis de cada kernel —
apenas as flags mudam:

```
kernel              inst  laço  loads  vmcnt  flags de cache
tdiv_unif_L1          52    23      2      2  GLC=0 SLC=0
tdiv_unif_L2          52    23      2      2  GLC=1 SLC=0
tdiv_unif_VRAM        52    23      2      2  GLC=1 SLC=1
tdiv_rot_L1           68    21      2      2  GLC=0 SLC=0
tdiv_rot_L2           68    21      2      2  GLC=1 SLC=0
tdiv_rot_VRAM         68    21      2      2  GLC=1 SLC=1
crivo_L1              92    10      1      0  GLC=0 SLC=0
```

`tdiv_rot` tem 16 instruções a mais que `tdiv_unif` — o `% n_primos` do índice
inicial, executado **uma vez por work-item, fora do laço**. Não afeta a
comparação entre níveis, que é feita dentro de cada kernel.

O store continua sendo `DC600000` (GLC=0 SLC=0) em todas as variantes: as flags
de coerência só existem em instruções de load. No `tdiv` isso é irrelevante — há
1 store por work-item contra 608 loads.

---

## 6. Quadro consolidado

Custo de contornar o L1, por perfil de acesso, somando todos os experimentos:

| Carga | Loads/trabalho | Working set reusado | Acesso | Custo do *bypass* |
|---|---|---|---|---|
| *Streaming* puro | 1 load, 0 reúso | — | coalescido | ≈1,00× (n.m.) |
| Scan | 1 load/elemento/fase | — | coalescido | 1,00× |
| Primos (divisão simples) | 1 load / ~1000 ALU | — | — | 1,00× |
| Crivo segmentado | 1 load / 2,1 stores | 2,2 KB | uniforme | 1,00× |
| Matmul *naive* | 2 loads / 2 ALU | 8 MB | disperso | 1,27× |
| *Stencil* 2D 3×3 | 9 loads / ~10 ALU | 3 linhas | vizinhança | 1,80× |
| **Divisão tabelada, uniforme** | **2 loads / 5 ALU** | **2,4 KB** | uniforme | **2,27×** |
| **Divisão tabelada, dispersa** | **2 loads / 5 ALU** | **2,4 KB** | **disperso** | **7,01×** |

Duas conclusões, em ordem de importância:

1. **A densidade de loads é o fator dominante.** `tdiv_unif` tem exatamente o
   mesmo padrão de acesso do crivo segmentado — tabela pequena, lida
   uniformemente por todos os lanes — e mede 2,27× contra 1,00×. O que mudou foi
   apenas quantos loads o algoritmo emite por unidade de trabalho.
2. **A dispersão amplifica.** Fixada a densidade de loads, quebrar a uniformidade
   multiplica o efeito por mais 3,1× (2,27 → 7,01), porque multiplica o número de
   requisições de linha por instrução.

Isso corrige a formulação do Experimento 8, que atribuía o efeito ao "reúso de
leitura que cabe em 16 KB". O reúso e o tamanho do conjunto são **necessários,
não suficientes**: o crivo segmentado satisfaz ambos e não reage. A condição
adicional é que o algoritmo passe **tempo suficiente emitindo loads**.

---

## 6b. Reprodutibilidade

Cinco execuções independentes do programa completo:

| Kernel | mín | máx | mediana | dp entre execuções |
|---|---|---|---|---|
| `tdiv_rot` | 7,102× | 7,218× | **7,147×** | **0,7%** |
| `tdiv_unif` | 2,114× | 2,279× | **2,178×** | 2,7% |
| `crivo` | 0,972× | 1,004× | **0,982×** | 1,5% |

O efeito de 7,15× é o resultado mais estável de todo o trabalho — dispersão de
0,7% entre execuções, contra 26% do *streaming* do Experimento 6. Os 297 pontos
das cinco execuções foram verificados contra referência em CPU sem uma única
divergência.

Os valores da tabela da §1 são de uma execução isolada e diferem em até 2% das
medianas acima; o artigo usa as medianas.

---

## 7. Limitações

1. ~~**Um só tamanho de tabela.**~~ **Resolvido pelo Experimento 10**
   (`tdiv_ws.md`): a varredura de 2 KB a 51 KB confirma o pico exatamente em
   16 KB (9,39×) e a queda para 2,78× em 32 KB. O experimento também mostrou que
   `tdiv_rot` **não** basta para variar o conjunto ativo — as wavefronts avançam
   em fase e a janela viva fica em ~4 KB independentemente do tamanho da tabela.
   Foi preciso dessincronizar os workgroups (`tdiv_dsyn`).
2. **`s_waitcnt vmcnt(0)` por load** suprime o MLP intra-thread (ver §4).
3. **dp relativo de 9,5% no `tdiv_rot`**, o maior dos três — mas o efeito medido
   (7,01×) está duas ordens de grandeza acima do ruído.

---

## 8. Reprodução

```bash
source setup/env_rocm.sh
g++ -Wall -O2 experimentos/crivo_l1.cpp -o output/crivo_l1 -lOpenCL
./output/crivo_l1          # executar a partir da raiz do projeto

llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 \
    isa/rocm_gfx803_crivo_l1.bin > isa/crivo_l1_disasm.txt
python3 isa/analyze_disasm.py isa/crivo_l1_disasm.txt
```

| Arquivo | Conteúdo |
|---|---|
| `experimentos/crivo_l1.cl` | Kernels das três formulações, gerados por macro |
| `experimentos/crivo_l1.cpp` | Anfitrião, tabelas GM, verificação em CPU e medição |
| `resultados/crivo_l1.csv` | Dados brutos |
| `isa/rocm_gfx803_crivo_l1.bin` | ISA, para conferir as flags |

## Referência adicional

Granlund, T. e Montgomery, P. L. (1994). *Division by invariant integers using
multiplication*. ACM SIGPLAN PLDI. — origem do teste de divisibilidade por
multiplicação usado nos kernels `tdiv`.
