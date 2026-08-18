# Os Três Níveis da Hierarquia, Medidos Diretamente

**Experimento 6** — Ambiente: Ubuntu 20.04, ROCm 5.2 (OpenCL 2.1), AMD RX 580 (Polaris / gfx803).

> ## ⚠ CORRIGIDO PELO EXPERIMENTO 7
>
> O Experimento 7 (`pchase_niveis.md`) mediu **latência** com P-chase e mostrou
> que **`SLC=1` não contorna o L2**: as curvas de `glc` e `glc slc` são
> indistinguíveis (diferença média de 0,30%). Portanto:
>
> - os kernels chamados aqui de `*_VRAM` **não acessam a VRAM** — são servidos
>   pelo L2; leia-os como "`glc slc` / não-alocação no L2";
> - o incremento de 3–8% atribuído abaixo ao "terceiro nível" **não é** custo de
>   ir à VRAM;
> - existem **dois** níveis controláveis por flag, não três: com L1 e sem L1.
>
> **Permanece válido**: os encodings e o mapeamento dos bits; toda a análise de
> `glc` (que o Exp. 7 confirma diretamente); o achado de que em streaming puro
> contornar o L1 é mais rápido; e a conclusão sobre o reúso ter de caber em 16 KB.
>
> A seção 4.3 abaixo levantava as hipóteses H1 e H2 — **H2 é a correta**.

---

## 1. O problema que este experimento resolve

Desde a Etapa 1 o trabalho persegue três cenários de acesso:

- **(a)** via L1 → L2 → VRAM (caminho normal)
- **(b)** contornando o L1 (servido pelo L2)
- **(c)** contornando L1 e L2 (direto da VRAM)

**Nenhuma tentativa anterior alcançou (c).** Pior: o kernel batizado de
`matmul_direto_VRAM` **nunca chegou à VRAM**. A verificação de ISA mostra que
ele gera código byte-idêntico ao `matmul_via_L2`:

| Kernel | O nome sugere | ISA real |
|---|---|---|
| `matmul_via_L1` | usa o L1 | `DC500000` — GLC=0 SLC=0 |
| `matmul_via_L2` | contorna o L1 | `DC510000` — GLC=1 SLC=0 *(só no ROCm; no Windows é `DC500000`)* |
| `matmul_direto_VRAM` | contorna L1+L2 | **`DC510000` — idêntico ao anterior** |

Em todas as medições feitas até aqui, nos dois sistemas operacionais,
**SLC = 0 sem exceção**.

### Por que trocar de algoritmo não resolveria

O nível de cache **não depende do algoritmo — depende das flags da instrução de
load**. Implementar scan, crivo de Eratóstenes ou FFT daria exatamente o mesmo
resultado, porque `volatile` não emite SLC em nenhum deles. O que faltava era o
modificador `slc`, que só o inline assembly produz — e que só compila em runtime
no ROCm.

---

## 2. As quatro combinações de flags

Confirmadas no ISA do gfx803 (`isa/rocm_gfx803_slc_probe.bin`):

| Inline asm | Encoding | GLC | SLC | Caminho |
|---|---|---|---|---|
| `flat_load_dword %0, %1` | `DC500000` | 0 | 0 | L1 → L2 → VRAM |
| `flat_load_dword %0, %1 glc` | `DC510000` | 1 | 0 | L2 → VRAM |
| `flat_load_dword %0, %1 slc` | `DC520000` | 0 | 1 | L1, sem alocar no L2 |
| `flat_load_dword %0, %1 glc slc` | **`DC530000`** | **1** | **1** | **VRAM** |

Formato FLAT no GFX8: bit 16 = GLC, bit 17 = SLC. As diferenças de encoding
(`0x10000` e `0x20000`) correspondem exatamente a esses dois bits.

A combinação `slc` sozinha (GLC=0, SLC=1) é registrada por completude do
fatorial de flags, mas não corresponde a um cenário de interesse: um miss no L1
vai ao L2 de qualquer forma.

---

## 3. Desenho experimental

Fatorial **nível de cache × carga**:

- **3 níveis**: L1 (normal), L2 (`glc`), VRAM (`glc slc`)
- **3 cargas**, escolhidas por terem fatores de reúso muito diferentes:

| Carga | Reúso por elemento | Descrição |
|---|---|---|
| `stream` | **1** | grid-stride coalescido; cada elemento lido exatamente uma vez |
| `stencil3x3` | **9** | blur 3×3; cada entrada participa de 9 saídas |
| `matmul` | **N = 1024** | naive; cada elemento participa de N saídas |

As três variantes de cada carga são geradas por macro a partir do **mesmo
código-fonte**, diferindo apenas na string do modificador. A verificação de ISA
confirma que a construção é controlada — as contagens de instruções são
idênticas entre os níveis:

```
kernel              inst  laco  loads  vmcnt  flags de cache
stream_L1             60    34      4      1  GLC=0 SLC=0 (usa L1)
stream_L2             60    34      4      1  GLC=1 SLC=0 (contorna L1)
stream_VRAM           60    34      4      1  GLC=1 SLC=1 (contorna L1 e L2)
stencil_L1           148   115     10      3  GLC=0 SLC=0
stencil_L2           148   115     10      3  9 loads GLC=1 + 1 de borda
stencil_VRAM         148   115     10      3  9 loads GLC=1 SLC=1 + 1 de borda
mm_L1                 57    20      2      1  GLC=0 SLC=0
mm_L2                 57    20      2      1  GLC=1 SLC=0
mm_VRAM               57    20      2      1  GLC=1 SLC=1
```

O décimo load do stencil é o caminho de borda (`out[y*W+x] = in[y*W+x]`, em C
puro, executado apenas pelos work-items da moldura) e não carrega as flags.

### Cuidado de implementação: earlyclobber

As saídas dos blocos asm usam **`"=&v"`**, não `"=v"`. Sem o `&`, o compilador
pode alocar um registrador de saída sobre um registrador de endereço que ainda
será lido por um load posterior do mesmo bloco, corrompendo o endereço. Com 4
loads por bloco isso produz **falha de página na GPU de forma reproduzível**
(`Memory access fault ... Reason: Page not present`). Com 2 loads o erro não se
manifesta, mas a construção continua incorreta.

---

## 4. Resultados

3 warmups + 10 execuções cronometradas por evento OpenCL.

| Carga | Reúso | L1 (ms) | L2 (ms) | VRAM (ms) | L2/L1 | VRAM/L1 |
|---|---|---|---|---|---|---|
| `stream` | 1 | 0,366 | 0,348 | 0,321 | **0,95×** | **0,88×** |
| `stencil3x3` | 9 | 1,073 | 1,927 | 2,084 | **1,80×** | **1,94×** |
| `matmul` | 1024 | 10,617 | 13,518 | 13,918 | **1,27×** | **1,31×** |

> Os tempos **absolutos não são comparáveis entre cargas** — a largura de
> sobreposição de loads difere (4, 3 e 2, seguindo a unidade natural de cada
> algoritmo). Apenas as razões dentro de cada carga são comparáveis.

### 4.1 Sem reúso, o cache atrapalha

No `stream`, contornar a hierarquia é **mais rápido**: 0,95× ao pular o L1 e
0,88× ao pular também o L2. Com reúso igual a 1, nenhum dado é tocado duas
vezes, então o cache não tem o que oferecer — mas continua pagando o custo de
alocar linhas e poluir o conjunto. O `glc slc` evita essa alocação e ganha 12%.

Este é o resultado mais limpo do experimento: **um cache não é gratuito, e em
carga de streaming puro ele é overhead líquido.**

### 4.2 A hipótese inicial está errada

A hipótese era que o custo de contornar o cache cresceria monotonicamente com o
reúso. **Os dados não sustentam isso**: o stencil, com reúso 9, sofre muito mais
(1,80×) que o matmul, com reúso 1024 (1,27×).

A explicação que ajusta os três pontos não é o fator de reúso bruto, e sim **se
o conjunto de trabalho onde o reúso ocorre cabe nos 16 KB do L1**:

- **stencil** — o reúso é *espacialmente local*: as 9 leituras de cada saída
  ficam numa janela de 3 linhas. Uma linha de 4096 floats ocupa 16 KB, e a
  janela relevante para um workgroup de 16×16 é bem menor. Cabe no L1 →
  o L1 é altamente eficaz → contorná-lo custa caro.
- **matmul** — o reúso está espalhado pela dimensão N inteira. O conjunto de
  trabalho de A e B (8 MB) excede tanto o L1 (16 KB) quanto o L2 (2 MB), então
  boa parte do reúso já não era capturada pelo L1 → contorná-lo custa menos.
- **stream** — não há reúso, em escala nenhuma.

Enunciado corrigido:

> O valor do L1 não é função do fator de reúso, e sim de **quanto do reúso
> acontece dentro de um conjunto de trabalho que cabe em 16 KB por CU**. Um
> algoritmo com reúso modesto mas localizado (stencil) se beneficia mais do L1
> do que um com reúso alto porém disperso (matmul).

Isso é consistente com o achado do Experimento 5 (`analise_volatile_rocm.md`,
seção 8.3), onde o ganho do L1 no matmul saltava justamente nos tamanhos em que
o stride provoca conflito de conjunto — ou seja, também ali o que governava era
o mapeamento no L1, não a quantidade de reúso.

### 4.3 O incremento do L2 é pequeno

Pular o L2 além do L1 acrescenta pouco: 8% no stream, 8% no stencil, 3% no
matmul. Duas leituras possíveis, que este experimento **não distingue**:

1. Nessas cargas o L2 realmente contribui pouco além do que o L1 já captura.
2. `SLC=1` não é um bypass rígido do L2.

A segunda é plausível: a documentação da ISA GCN3 descreve SLC como *system
level coherent*, e em várias gerações AMD a flag funciona como dica de
**não-alocação / streaming** no L2, e não como desvio completo. Pelo tempo
apenas não é possível decidir entre as duas hipóteses.

---

## 5. Limitações

1. **A semântica exata de SLC=1 não foi verificada.** Chamar o nível de "VRAM
   direto" é uma simplificação: o encoding `DC530000` está confirmado, mas o
   comportamento microarquitetural (bypass rígido vs. não-alocação) não. Para
   decidir seria preciso um P-chase com `glc slc` medindo se a latência bate com
   a da VRAM ou com a do L2.
2. **Larguras de sobreposição diferentes** entre cargas (4/3/2) impedem comparar
   tempos absolutos entre elas.
3. **Um tamanho por carga.** O Experimento 5 mostrou que o efeito varia com N de
   forma não trivial (conflitos de stride); estes números são de um ponto só.
4. **Só leituras.** O L1 do GFX8 é write-through; o comportamento de escrita não
   foi investigado.
5. **Exclusivo do Linux/ROCm.** Nada disto é reproduzível no driver Windows, que
   não compila inline assembly.

---

## 6. Reprodução

```bash
source setup/env_rocm.sh
g++ -Wall -O2 experimentos/niveis_cache.cpp -o output/niveis_cache -lOpenCL
./output/niveis_cache

llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 \
    isa/rocm_gfx803_niveis_cache.bin > isa/niveis_cache_disasm.txt
python3 isa/analyze_disasm.py isa/niveis_cache_disasm.txt
```

| Arquivo | Conteúdo |
|---|---|
| `experimentos/niveis_cache.cpp` | Experimento 3 níveis × 3 cargas |
| `isa/rocm_gfx803_slc_probe.bin` | Sonda das 4 combinações de flags |
| `isa/rocm_gfx803_niveis_cache.bin` | ISA do experimento |
| `resultados/niveis_cache.csv` | Dados brutos |

---

## 7. Efeito sobre o restante do trabalho

1. **Renomear `matmul_direto_VRAM`.** O nome afirma algo que o kernel não faz.
   Sugestão: `matmul_volatile_2` ou `matmul_via_L2_b`, deixando claro que é uma
   segunda variante com `volatile`, não um terceiro nível.
2. **A Etapa 1 pode ser reescrita com o resultado que ela buscava.** A hipótese
   original — "controlar o nível de cache e medir o impacto" — agora é
   executável. O que mudou não foi o algoritmo, foi a plataforma.
3. **O `volatile` deixa de ser necessário** como mecanismo: no ROCm ele dá
   GLC=1, mas o inline asm dá os três níveis com controle explícito.
