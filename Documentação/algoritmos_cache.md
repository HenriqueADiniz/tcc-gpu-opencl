# Scan, Primos e Crivo sob os Níveis de Cache

**Experimento 8** — Ambiente: Ubuntu 20.04, ROCm 5.2, AMD RX 580 (Polaris / gfx803).

---

## 1. Resultado em uma frase

**Nos três algoritmos, o nível de cache não produziu efeito mensurável** — e cada
um deles não produziu pelo motivo oposto ao dos outros. O conjunto reforça a
conclusão do Experimento 6: o L1 só importa quando o algoritmo tem **reúso de
leitura que cabe nele**.

| Algoritmo | Perfil de memória | Mediana L1 | L2/L1 | VRAM/L1 | dp rel. |
|---|---|---|---|---|---|
| Scan | reúso global ≈ 1, reúso interno no LDS | 1,423 ms | 1,000× | 1,003× | 8,7% |
| Primos (divisão) | *compute-bound*, 1 load por work-item | 1,715 ms | **1,000×** | **1,000×** | 0,4% |
| Crivo segmentado | dominado por escrita | 10,224 ms | 0,994× | 0,999× | 1,4% |

Todos os resultados foram verificados contra referência em CPU antes de qualquer
tempo ser reportado: o scan por comparação exata dos 16 M elementos, os primos
contra teste de primalidade em CPU (9.459 primos no intervalo), e o crivo pela
contagem de primos em $[2^{23}, 2^{24})$ — **513.708**, idêntico à referência.

---

## 2. Por que cada um não reagiu

### 2.1 Scan — o reúso mora no LDS, não no cache

O scan de soma de prefixos em três fases (Blelloch; Harris et al.) lê cada
elemento uma vez por fase: o reúso global é aproximadamente unitário. O reúso
que o algoritmo de fato explora — a árvore de somas dentro do bloco — acontece
na **memória local (LDS)**, que é gerenciada explicitamente e não passa pelo L1.

Isso reproduz, num algoritmo real e citável, o mesmo comportamento medido com a
carga sintética de *streaming* no Experimento 6 (0,95×).

### 2.2 Primos por divisão sucessiva — o controle negativo

Este algoritmo foi incluído **deliberadamente como controle negativo**. Cada
work-item executa um load global e centenas de operações de ALU (divisões
sucessivas até a raiz quadrada do candidato). A razão medida é de exatamente
**1,000× em ambos os níveis**, com desvio padrão de 0,4% — a menor dispersão de
todo o trabalho.

O valor deste resultado é metodológico. Ele demonstra que **as diferenças
medidas nos demais experimentos vêm do tráfego de memória**, e não de algum
artefato das flags de coerência ou da construção por *inline assembly*. Se o
mero uso de `glc` ou `glc slc` impusesse custo por si só, ele apareceria aqui.
Não aparece.

### 2.3 Crivo segmentado — o tráfego é de escrita

Este foi o caso em que a previsão inicial **falhou**, e o motivo é instrutivo.

A expectativa era de alta sensibilidade ao L1: a tabela de primos pequenos ocupa
2,2 KB (564 primos até 4096), cabe folgadamente no L1 e é percorrida por todos
os 16.384 work-items. Reúso temporal altíssimo sobre conjunto pequeno — o perfil
que, pelo Experimento 6, deveria sofrer mais com o *bypass*.

A inspeção do ISA mostra por que não sofre:

```
crivo_L2:  DC510000  GLC=1 SLC=0   flat_load_dword v3, v[4:5] glc   <- tabela de primos
           DC600000  GLC=0 SLC=0   flat_store_byte v[5:6], v2       <- marcação
```

Há **um único load** por iteração do laço externo e um **store por marcação** no
laço interno. Duas consequências:

1. Todos os *lanes* de uma *wavefront* leem `primos_pequenos[i]` no **mesmo
   endereço**. O acesso é uniforme e o coalescedor o resolve em uma requisição
   por *wavefront*, e não 64 — o tráfego de leitura é desprezível.
2. O volume de memória do algoritmo está nas **escritas** em `marcas`, que são
   geradas em C puro e **não carregam as flags de coerência**, que só se aplicam
   a instruções de *load*.

Ou seja: o crivo é dominado por escrita, e este experimento controla apenas
leitura. Isso conecta diretamente com a limitação já registrada no Experimento 7
— o L1 do GFX8 é *write-through*, e o comportamento das flags em escrita
permanece em aberto. O crivo é o candidato natural para essa investigação
futura.

---

## 3. Cuidados de medição

Duas correções foram necessárias antes de os números serem confiáveis, e ambas
valem registro porque afetam qualquer medição de kernel longo nesta GPU.

**Desbalanceamento de carga.** Peneirando $[0, N)$, os segmentos iniciais
descartam quase todos os primos pela condição $p^2 > \text{fim}$, enquanto os
finais aplicam a tabela inteira. O desvio padrão chegava a **30%**. Peneirar
apenas $[N/2, N)$ torna a carga uniforme entre todos os work-items, porque para
todo segmento vale $\text{fim} > 4096^2$.

**Ociosidade entre execuções.** Um `clFinish` entre o preenchimento do buffer e
o lançamento do kernel deixava a GPU ociosa a cada iteração, permitindo que o
gerenciamento dinâmico de frequência reduzisse o *clock*. O resultado era uma
distribuição **bimodal** (mediana ≈ 10 ms com execuções isoladas de ≈ 5 ms).
Como a fila é *in-order*, o `clFinish` era desnecessário.

**Estatística.** As razões são calculadas sobre a **mediana**, não sobre a média
nem o mínimo. Restaram execuções rápidas esporádicas cuja variante muda entre
execuções do programa — em uma rodada o desvio caiu no `glc slc`, na seguinte no
`glc` —, o que demonstra que são artefato e não efeito das flags. A mediana é
robusta a esses casos nas duas direções.

---

## 4. Onde isto posiciona os três algoritmos

Somando aos experimentos anteriores, o quadro completo do custo de contornar o
L1 por perfil de acesso:

| Carga | Reúso de leitura | Cabe no L1? | Custo do *bypass* |
|---|---|---|---|
| *Streaming* puro | 1 | — | 0,95× (contornar é **mais rápido**) |
| Scan | ≈ 1 (reúso no LDS) | — | 1,00× |
| Primos (divisão) | 1 load/work-item | — | 1,00× |
| Crivo segmentado | alto, mas leitura uniforme | sim, mas irrelevante | 1,00× |
| Matmul *naive* | $N$, disperso em 8 MB | não | 1,27× |
| *Stencil* 2D 3×3 | 9, janela de 3 linhas | **sim** | **1,80×** |

O padrão é consistente: o L1 só cobra quando existe **reúso de leitura,
volumoso, sobre um conjunto de trabalho que cabe em 16 KB por CU**. Reúso alto
mas uniforme (crivo) não conta, porque o coalescedor já o resolve; reúso alto
mas disperso (matmul) conta pouco; reúso modesto e localizado (stencil) conta
muito.

---

## 5. Limitações

1. **Só leituras.** As flags aplicam-se a instruções de *load*. O crivo, sendo
   dominado por escrita, é medido apenas parcialmente por este desenho.
2. **Um tamanho por algoritmo.** O Experimento 5 mostrou que o efeito pode variar
   de forma não trivial com o tamanho do problema.
3. **O scan tem dispersão de 8,7%**, maior que a dos demais, por ser uma
   sequência de cinco lançamentos de kernel cronometrada por eventos de
   fronteira; parte da variação vem do escalonamento entre lançamentos.

---

## 6. Reprodução

```bash
source setup/env_rocm.sh
g++ -Wall -O2 experimentos/algoritmos_cache.cpp -o output/algoritmos_cache -lOpenCL
./output/algoritmos_cache          # executar a partir da raiz do projeto

llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 \
    isa/rocm_gfx803_algoritmos_cache.bin > isa/algoritmos_cache_disasm.txt
python3 isa/analyze_disasm.py isa/algoritmos_cache_disasm.txt
```

| Arquivo | Conteúdo |
|---|---|
| `experimentos/algoritmos_cache.cl` | Kernels dos três algoritmos, gerados por macro |
| `experimentos/algoritmos_cache.cpp` | Anfitrião, verificação em CPU e medição |
| `resultados/algoritmos_cache.csv` | Dados brutos |
| `isa/rocm_gfx803_algoritmos_cache.bin` | ISA, para conferir as flags |
