# O `volatile` no ROCm/Linux — Isolamento do Efeito do L1 no gfx803

**Experimento 5 — verificação multiplataforma do controle de cache**
Ambiente: Ubuntu 20.04, ROCm 5.2 (OpenCL 2.1, AMD-APP 3452.0), AMD RX 580 (Ellesmere / Polaris / gfx803).

---

## 1. Resumo

Executar os mesmos kernels do Experimento 1 sob o runtime OpenCL do ROCm, na **mesma
GPU física** usada no Windows, produziu um resultado que **contradiz e corrige** a
conclusão anterior sobre o `volatile` no GFX8.

| | Windows (AMDOCL, Adrenalin) | Linux (ROCm 5.2, Clang/LLVM) |
|---|---|---|
| `volatile` → flags de cache | `DC500000` (GLC=0) | **`DC510000` (GLC=1)** |
| `__asm__` em kernel OpenCL | não compila | **compila e executa** |

Como o alvo (`gfx803`) e o hardware são idênticos nos dois casos, a diferença **só pode
ser atribuída ao compilador**. Isso elimina a limitação metodológica registrada na
`analise_isa.md`, onde a comparação entre gerações usava versões diferentes do RGA e
não permitia separar arquitetura de codegen.

Medindo com um desenho fatorial 2×2 que separa os fatores confundidos, o custo do
bypass do L1 nesta carga é de **1,36×** — e não os 2,1× que a comparação ingênua
entre o kernel normal e o kernel `volatile` sugeria.

---

## 2. Evidência de ISA

Binário extraído com `clGetProgramInfo(CL_PROGRAM_BINARIES)` sob o ROCm. Diferente do
Windows (ELF de 32 bits contendo um ELF interno), o ROCm devolve **um code object HSA
direto** (ELF64, `e_machine = 0xE0 = EM_AMDGPU`), o que dispensa o parse aninhado.

```
llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 isa/matmul_gfx803.bin
```

| Kernel | Encoding | Flags |
|---|---|---|
| `matmul_via_L1` (sem volatile) | `DC500000` | GLC=0 SLC=0 |
| `matmul_via_L2` (volatile) | **`DC510000`** | **GLC=1** SLC=0 |
| `matmul_direto_VRAM` (volatile) | **`DC510000`** | **GLC=1** SLC=0 |
| `matmul_tiled` (LDS) | `DC500000` | GLC=0 SLC=0 |

Decodificação do campo (formato FLAT no GFX8): bit 16 = GLC, bit 17 = SLC.
`0xDC510000 − 0xDC500000 = 0x10000 = 2¹⁶`, logo a diferença está exatamente no bit GLC.

> Atenção: no formato **MUBUF** (`buffer_load_dword`, usado pelo driver Windows) o GLC
> está em outra posição. Comparações entre as duas plataformas devem levar em conta que
> o ROCm emite **FLAT** e o AMDOCL emite **MUBUF** para o mesmo código-fonte.

`matmul_via_L2` e `matmul_direto_VRAM` geram sequências de instruções **idênticas**,
confirmando o achado anterior de que o OpenCL não permite endereçar L1 e L2
separadamente — nenhum dos dois produz SLC=1.

---

## 3. O confundidor

A comparação direta de tempos sob o ROCm dava:

| Kernel | Tempo |
|---|---|
| `matmul_via_L1` | 6,84 ms |
| `matmul_via_L2` (volatile) | 14,09 ms |

Atribuir esses 2,06× ao bypass do L1 seria **incorreto**. A inspeção do laço interno
mostra que o codegen do `volatile` altera **duas coisas ao mesmo tempo**:

```asm
;--- sem volatile: os dois loads são emitidos antes de um único waitcnt
flat_load_dword v9, v[1:2]
flat_load_dword v4, v[7:8]
...
s_waitcnt vmcnt(0)          ; uma única espera — as latências se sobrepõem
v_mac_f32_e32 v6, v9, v4

;--- com volatile: cada load tem sua própria espera
flat_load_dword v9, v[1:2] glc
s_waitcnt vmcnt(0)          ; <-- serializa
flat_load_dword v4, v[7:8] glc
s_waitcnt vmcnt(0)          ; <-- serializa
```

Ou seja: além de `glc`, o `volatile` **elimina a sobreposição de latência entre os dois
loads** (memory-level parallelism). Os dois efeitos estão confundidos e não podem ser
separados por essa comparação.

Este é exatamente o mesmo confundidor que a `analise_isa.md` já havia identificado para
o kernel `matmul_glc_asm` — só que aqui ele aparece no codegen do próprio `volatile`.

---

## 4. Desenho fatorial 2×2

Como o inline assembly **funciona em runtime no ROCm**, é possível construir as quatro
combinações e medir cada fator isoladamente:

| | loads sobrepostos | loads serializados |
|---|---|---|
| **GLC=0** (usa L1) | (1) baseline | (2) só serialização |
| **GLC=1** (contorna L1) | (3) só bypass do L1 | (4) ambos |

**Ponto crítico do desenho**: as quatro células são construídas com inline assembly.
Uma versão anterior deste experimento usava C puro na célula (1) e asm nas demais, o
que inflava o efeito atribuído ao GLC — porque o bloco `__asm__` é opaco ao otimizador
e **impede a redução de força no cálculo de endereço**, adicionando ALU por iteração:

| Kernel | Instruções no laço interno |
|---|---|
| `mm_plain_c` (C puro) | 15 |
| `mm_glc0_overlap` (asm) | 20 |
| `mm_volatile` | 16 |

Comparar C puro com asm mistura o efeito da flag de cache com 5 instruções de
aritmética de endereço por iteração. Só células asm são comparáveis entre si.

Verificação do ISA de todas as células (`python3 isa/analyze_disasm.py`):

```
kernel                    inst  laco  loads  vmcnt  flags de cache
mm_glc0_overlap             57    20      2      1  GLC=0 SLC=0 (usa L1)
mm_glc0_serial              58    21      2      2  GLC=0 SLC=0 (usa L1)
mm_glc1_overlap             57    20      2      1  GLC=1 SLC=0 (contorna L1)
mm_glc1_serial              58    21      2      2  GLC=1 SLC=0 (contorna L1)
mm_plain_c                  58    15      2      1  GLC=0 SLC=0 (usa L1)
mm_volatile                 58    16      2      2  GLC=1 SLC=0 (contorna L1)
```

Cada célula gera exatamente o que foi projetado, e as quatro células do fatorial
compartilham a mesma estrutura de endereçamento (20–21 instruções).

---

## 5. Resultados

Matmul 1024×1024, workgroup 16×16, 3 warmups + 10 execuções cronometradas por evento
OpenCL. Todas as células validadas numericamente contra a CPU (`maxdiff = 0.00000`).

| Célula | Média (ms) | dp | GFLOPS |
|---|---|---|---|
| (1) asm GLC=0, sobreposto — *baseline* | 10,204 | 0,215 | 210,5 |
| (2) asm GLC=0, serializado | 11,285 | 0,398 | 190,3 |
| (3) asm GLC=1, sobreposto | 13,850 | 0,801 | 155,1 |
| (4) asm GLC=1, serializado | 15,696 | 0,696 | 136,8 |
| (0) C puro — *referência externa* | 7,469 | 0,350 | 287,5 |
| (5) `volatile` puro — *controle* | 14,275 | 0,527 | 150,4 |

### Decomposição

| Efeito | Razão |
|---|---|
| Serialização isolada — (2)/(1) | 1,11× |
| **Bypass do L1 isolado — (3)/(1)** | **1,36×** |
| Combinado — (4)/(1) | 1,54× |
| Produto dos isolados | 1,50× |
| Interação (combinado ÷ produto) | 1,02× — aproximadamente multiplicativos |

**O L1 responde por uma redução de 26% no tempo desta carga** (equivalente a 3,65 ms
absolutos), com todos os demais fatores mantidos fixos.

### Consistência interna

- **Controle**: `volatile` (14,275 ms) vs célula (4) (15,696 ms) → razão 0,91. As duas
  têm ISA equivalente em flags e waitcnt; a diferença de 9% é explicada pelas 5
  instruções de ALU a mais por iteração da versão asm (21 vs 16 no laço). O `volatile`
  atinge a mesma semântica de cache preservando a redução de força do endereçamento.
- **Custo do asm**: célula (1) vs (0) = 1,37×, quantificando o preço de construir os
  loads manualmente. É por isso que (0) fica fora do fatorial.
- A interação de 1,02× indica que os dois fatores agem de forma quase independente,
  consistente com serem gargalos distintos (largura de banda vs. latência exposta).

---

## 6. Correções à `analise_isa.md`

Três conclusões do documento anterior precisam ser revistas:

| Conclusão anterior | Estado |
|---|---|
| *"No GFX8 o backend não emite flags de coerência para `volatile`"* | **Restringir a AMDOCL/Windows.** No ROCm/LLVM, o mesmo gfx803 emite `glc`. É propriedade do compilador, não da arquitetura. |
| *"O único mecanismo funcional para GLC=1 no GFX8 é inline assembly"* | **Falso no ROCm.** Ali o `volatile` basta — e é preferível, por preservar o endereçamento otimizado. |
| *"Inline asm é exclusivo de compilação offline (RGA/HIP)"* | **Falso.** Compila e executa no runtime OpenCL do ROCm. |

Em compensação, a **limitação metodológica** que o documento registrava — a de que
gfx803 e gfx900/gfx950 foram compilados com versões diferentes do RGA, impedindo separar
arquitetura de versão de compilador — **fica resolvida para o caso do gfx803**. Agora há
dois compiladores distintos gerando ISA diferente para o **mesmo alvo e o mesmo
hardware**, o que é evidência direta de que o mapeamento `volatile` → ISA é decisão de
codegen.

Enunciado corrigido:

> O mapeamento de `volatile` para flags de coerência de cache é uma decisão de codegen
> do compilador, não uma propriedade da arquitetura. Para o mesmo alvo gfx803 e o mesmo
> hardware, o compilador proprietário AMDOCL (Windows) não emite GLC, enquanto o
> Clang/LLVM do ROCm (Linux) emite GLC=1.

---

## 7. Limitações

1. **`glc` contorna o L1, não a hierarquia inteira.** SLC permanece 0 em todos os casos:
   os acessos continuam servidos pelo L2. O experimento mede o efeito do L1, não o
   custo de ir até a VRAM.
2. **Um único algoritmo e um único tamanho.** 1,36× vale para matmul 1024×1024 com
   workgroup 16×16. Cargas com mais ou menos reúso de dados dariam fatores diferentes;
   o número não deve ser generalizado como "o valor do L1 na Polaris".
3. **Duas variáveis mudam entre Windows e Linux** — compilador e sistema operacional.
   A atribuição ao compilador se apoia na evidência de ISA (que é gerada em tempo de
   compilação e independe do SO), não apenas na diferença de tempo.
4. **O `volatile` não foi projetado para isso.** Ele especifica semântica de memória da
   linguagem; que o LLVM a implemente com `glc` no gfx803 é um detalhe de
   implementação, sujeito a mudar entre versões. Não é interface documentada de
   controle de cache.
5. **Sobreposição entre iterações não foi isolada.** Todas as células asm forçam um
   `s_waitcnt` por iteração, impedindo o pipeline de software que o compilador aplicaria.
   Por isso o baseline asm (10,2 ms) é mais lento que o C puro (7,5 ms).

---

## 8. Escalabilidade — o efeito do L1 em função de N

O fatorial da seção 5 mediu um único tamanho. Repetindo-o para 16 valores de N
entre 256 e 3072 (`experimentos/glc_scaling.cpp`), o quadro muda de forma
importante — **e corrige o número da seção 5**.

### 8.1 Modelo

Para cada célula ajustou-se, por mínimos quadrados,

$$T(N) = a \cdot N^3 + b \cdot N^2$$

O termo N³ acompanha tanto o número de operações quanto o de loads emitidos
(ambos 2N³); o termo N² absorve efeitos de borda — lançamento do kernel, prólogo
por work-item, escrita de C. A razão entre os coeficientes `a` de duas células é
o **custo assintótico** do fator que as separa, livre desses efeitos de borda.

| Célula | a (ns/N³) | R² |
|---|---|---|
| (1) asm GLC=0, sobreposto | 0,010180 | 0,99999 |
| (2) asm GLC=0, serializado | 0,011173 | 0,99999 |
| (3) asm GLC=1, sobreposto | 0,011620 | 0,99988 |
| (4) asm GLC=1, serializado | 0,011984 | 0,99991 |
| (0) C puro | 0,006625 | 0,99998 |
| (5) `volatile` | 0,010769 | 0,99986 |

O ajuste é praticamente exato (R² ≥ 0,9998 em todas as células), confirmando que
o regime é N³ em toda a faixa medida.

| Efeito | Razão assintótica |
|---|---|
| **Bypass do L1 — (3)/(1)** | **1,141×** |
| Serialização — (2)/(1) | 1,098× |
| Ambos — (4)/(1) | 1,177× |
| Custo do inline asm — (1)/(0) | 1,537× |

### 8.2 Correção ao valor da seção 5

O fator 1,36× medido em N=1024 **não é o valor assintótico**. O custo assintótico
do bypass do L1 é **1,141×** — ou seja, o L1 responde por cerca de **12%** de
redução de tempo nesta carga, não os 26% que o ponto único sugeria.

O motivo é a seção seguinte: N=1024 é um dos tamanhos em que o padrão de acesso
sofre conflito de cache, o que infla localmente o valor do L1.

### 8.3 Conflito de cache dependente do stride

A razão (3)/(1) não decai suavemente: ela oscila, com picos em tamanhos
específicos.

| N | (3)/(1) | | N | (3)/(1) |
|---|---|---|---|---|
| 256 | **1,346** | | 768 | **1,301** |
| 320 | 1,003 | | 896 | 1,092 |
| 384 | 1,134 | | 1024 | **1,264** |
| 448 | 1,031 | | 1280 | 1,089 |
| **512** | **1,780** | | 1536 | 1,177 |
| 576 | 1,032 | | 2048 | 1,114 |
| 640 | 1,068 | | 2560 | 1,130 |
| 704 | 1,038 | | 3072 | 1,137 |

Os picos caem exatamente nos **N múltiplos de 256**. Como a matriz é row-major e
`float` ocupa 4 bytes, o stride entre `B[k][col]` e `B[k+1][col]` é de 4N bytes;
N múltiplo de 256 significa stride múltiplo de 1 KB. Strides que são múltiplos
de potências de dois grandes mapeiam os acessos em poucos conjuntos do L1,
produzindo **conflict misses**.

Que o efeito é real, e não ruído de medição, fica claro no próprio baseline:

| N | Baseline (1) | GFLOPS |
|---|---|---|
| 448 | 0,858 ms | 209,5 |
| **512** | **1,850 ms** | **145,1** |
| 576 | 1,797 ms | 212,7 |

**N=576 é maior que N=512 e mesmo assim mais rápido em tempo absoluto.** O desvio
padrão nesses pontos é de 4–8%, muito abaixo da anomalia. O resultado se
reproduziu em duas execuções independentes (1,78× e 1,76×).

A leitura é que, quando o stride provoca conflito, **o L1 é justamente o que
atenua o dano**: sem ele (GLC=1) o tempo em N=512 salta para 3,26 ms. O valor do
L1 não é constante — depende de o padrão de acesso conseguir ou não explorá-lo.

### 8.4 Largura de banda efetiva

Contando os loads emitidos (2N³ × 4 bytes), a taxa efetiva assintótica é:

| Célula | BW efetiva |
|---|---|
| (0) C puro | ~1190 GB/s |
| (1) asm GLC=0 | ~790 GB/s |
| (3) asm GLC=1 | ~690 GB/s |
| VRAM GDDR5 (pico teórico) | ~256 GB/s |

**Todas as células ficam muito acima do pico teórico da VRAM**, inclusive as com
GLC=1. Isso é consistente com SLC=0: contornar o L1 não leva os acessos à VRAM —
eles passam a ser servidos pelo **L2**, que é quem sustenta a maior parte do
tráfego nesta carga. O L1 acrescenta ~14% sobre isso.

### 8.5 Figuras

PNG em 300 dpi, prontos para `\includegraphics` no LaTeX:

| Arquivo | Conteúdo |
|---|---|
| `analise/fig_glc_desempenho.png` | GFLOPS × N, com e sem o L1 |
| `analise/fig_glc_razao.png` | Razões sobre o baseline; picos nos N múltiplos de 256 |
| `analise/fig_glc_bandwidth.png` | Taxa efetiva de load contra o pico da VRAM |
| `analise/figuras_glc.png` | Painel único, para conferência |
| `analise/modelo_glc.txt` | Coeficientes do ajuste |

```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{analise/fig_glc_razao.png}
\caption{Efeito isolado de cada fator sobre o baseline...}
\label{fig:glc-razao}
\end{figure}
```

As séries usam cor **e** marcador **e** traço distintos — a identidade não
depende só da cor, o que mantém as figuras legíveis impressas em tons de cinza.

**Atenção a uma redundância**: largura de banda efetiva e GFLOPS são a mesma
curva, já que $BW = 2N^3 \cdot 4/T$ e $GFLOPS = 2N^3/T$, ou seja $BW = 4 \cdot
GFLOPS$. As duas figuras sustentam argumentos distintos (desempenho alcançado
vs. comparação contra o pico da VRAM), mas **não são evidências independentes**.
Usar as duas no artigo apenas se ambos os argumentos forem feitos; caso
contrário, uma basta.

Por esse motivo a figura de desempenho usa GFLOPS em vez de tempo: num gráfico
de tempo em log-log todas as curvas N³ se sobrepõem e a diferença entre células
fica ilegível. Normalizar pelo trabalho (2N³) achata as curvas e torna a
diferença visível.

### 8.6 Limitação adicional

A varredura usa `workgroup` fixo de 16×16 e apenas o algoritmo naive. O fator
1,141× é específico dessa combinação. Um kernel com blocagem explícita em LDS
(como o `matmul_tiled`) reduz drasticamente o tráfego que chega ao L1 e teria um
fator diferente — provavelmente menor, já que o LDS já cumpre o papel de captura
de reúso.

---

## 9. Reprodução

```bash
source setup/env_rocm.sh                      # exige ROC_ENABLE_PRE_VEGA=1

# fatorial em um único tamanho (N=1024)
g++ -Wall -O2 experimentos/glc_factorial.cpp -o output/glc_factorial -lOpenCL
./output/glc_factorial | tee resultados/glc_factorial_rocm.txt

# varredura de N + gráficos e ajuste do modelo
g++ -Wall -O2 experimentos/glc_scaling.cpp -o output/glc_scaling -lOpenCL
./output/glc_scaling                          # ou: ./output/glc_scaling 512 576 1024
python3 analise/plot_glc_scaling.py

# verificação do ISA
llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 \
    isa/rocm_gfx803_glc_factorial.bin > isa/glc_factorial_disasm.txt
python3 isa/analyze_disasm.py isa/glc_factorial_disasm.txt           # tabela-resumo
python3 isa/analyze_disasm.py isa/glc_factorial_disasm.txt --loops   # laços internos
```

### Arquivos

| Arquivo | Conteúdo |
|---|---|
| `experimentos/glc_factorial.cpp` | Experimento fatorial 2×2 (N único) |
| `experimentos/glc_scaling.cpp` | Varredura de N sobre o fatorial → CSV |
| `analise/plot_glc_scaling.py` | Figuras e ajuste do modelo T(N) = aN³ + bN² |
| `resultados/glc_scaling.csv` | Dados brutos da varredura (16 tamanhos) |
| `isa/analyze_disasm.py` | Analisador de disassembly: flags GLC/SLC, laço interno, diff |
| `isa/build_cl_dump.cpp` | Compila um `.cl` e grava o ISA gerado |
| `isa/asm_forms_probe.cl` | Formas de inline asm validadas antes de virarem benchmark |
| `isa/matmul_gfx803.bin` | ISA do `matmul.cpp` sob o ROCm |
| `isa/rocm_gfx803_glc_factorial.bin` | ISA do experimento fatorial |
| `resultados/glc_factorial_rocm.txt` | Saída completa da execução |
| `setup/asm_probe.cpp` | Sonda de suporte a `__asm__` em todas as plataformas |
