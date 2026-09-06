# Bateria de Reprodução — Todos os Experimentos

**Ambiente**: Ubuntu 20.04, ROCm 5.2, AMD RX 580 (Polaris / gfx803), `ROC_ENABLE_PRE_VEGA=1`.
Todos os 11 programas de `experimentos/` compilam sem *warnings* com
`g++ -Wall -O2 ... -lOpenCL` e executam com código de saída 0.

---

## 1. Quadro geral

| Exp. | Programa | Métrica-chave | Documentado | Reproduzido | |
|---|---|---|---|---|---|
| 1 | `matmul` | *tiled* (LDS) | 2,07 ms / 1038 GFLOPS | 2,067 ms / 1039 GFLOPS | ✅ |
| 3 | `mem_hierarchy` | acesso aleatório | 18,03 ms / 7,4 GB/s | 18,27 ms / 7,35 GB/s | ✅ |
| 4 | `pchase` | patamar do L1 | ~39 ns | 38,6--38,8 ns | ✅ |
| 4 | `pchase` | salto em 20 KB | +52% | +52% (38,8 → 59,0) | ✅ |
| 4 | `pchase` | faixa 1--8 MB | ~137 ns | **152--240 ns** | ❌ |
| 5 | `glc_scaling` | serialização | 1,098× | 1,099× | ✅ |
| 5 | `glc_scaling` | custo do *asm* | 1,537× | 1,522× | ✅ |
| 5 | `glc_scaling` | *bypass* do L1 | 1,141× | **1,114×** / 1,070× | ⚠️ |
| 5 | `glc_factorial` | *bypass* isolado | — | 1,31× | ✅ |
| 6 | `niveis_cache` | *stencil* L2/L1 | 1,796× | 1,70--1,82× (7 exec.) | ✅ |
| 6 | `niveis_cache` | matmul L2/L1 | 1,273× | 1,25--1,31× (7 exec.) | ✅ |
| 6 | `niveis_cache` | *streaming* L2/L1 | 0,950× | **0,93--1,19× (7 exec.)** | ❌ |
| 7 | `pchase_flags` | `glc` em 1 KB | 1,4226× | 1,424--1,438× (4 exec.) | ✅ |
| 7 | `pchase_flags` | faixa ≥ 256 KB | — | **±10--15%, com inversões** | ❌ |
| 8 | `algoritmos_cache` | scan / primos / crivo | 1,000 / 1,000 / 0,994× | 1,002 / 1,000 / 0,989× | ✅ |
| 9 | `crivo_l1` | `tdiv_rot` | 7,014× | idem (desta sessão) | ✅ |
| 10 | `tdiv_ws` | pico em 16 KB | 9,390× | idem (desta sessão) | ✅ |

Onze das dezessete métricas reproduzem dentro da variação esperada. As quatro
marcadas em vermelho estão detalhadas abaixo; todas resultaram em correção do
`main.tex`.

---

## 2. O resultado que não se sustenta: *streaming* puro (Exp. 6)

O artigo afirmava que, em *streaming* puro, contornar o L1 é **mais rápido**
(0,95×) — resultado que constava do resumo, da Tabela do Experimento 6 e da
conclusão. Sete execuções independentes:

| Carga | mín | máx | amplitude | Reprodutível? |
|---|---|---|---|---|
| *streaming* | 0,929× | 1,191× | **26%** | **não** |
| *stencil* 3×3 | 1,702× | 1,815× | 6,6% | sim |
| matmul | 1,254× | 1,314× | 4,8% | sim |

A razão do *streaming* cruza 1,0 nos dois sentidos e não tem sinal consistente,
enquanto as outras duas cargas do **mesmo experimento, nas mesmas execuções**,
reproduzem-se de forma estreita. O problema é local ao *streaming*, não ao
experimento.

**Causa identificada.** Em cada execução, uma das três variantes apresenta desvio
padrão inferior a 0,5% e as outras duas entre 8% e 32% — e qual variante fica
"limpa" muda de execução para execução:

| Execução | dp(L1) | dp(L2) | dp(VRAM) | variante limpa |
|---|---|---|---|---|
| documentada | 23,9% | 17,0% | **0,3%** | VRAM |
| reprodução 1 | **0,3%** | 31,9% | 20,9% | L1 |

O valor "limpo" é ~0,33 ms nas duas, em todas as variantes. As razões publicadas
são inteiramente artefato de execuções esporádicas lentas.

É **o mesmo artefato já documentado no Experimento 8** (`algoritmos_cache.md`,
§3), onde a mediana foi adotada como estatística robusta justamente por isso. O
Experimento 6 usa a média e nunca foi corrigido. A carga de *streaming* é a mais
exposta porque tem o menor tempo absoluto (0,33 ms), de modo que um único
*outlier* domina a média de 10 execuções.

**Correção aplicada**: o resumo (PT e EN), a tabela do Experimento 6, o parágrafo
de discussão e a conclusão passaram a registrar *efeito não mensurável*. A tese
do trabalho não depende desse ponto — ela se apoia no contraste entre *stencil*
(1,80×) e matmul (1,27×), ambos reprodutíveis.

---

## 3. Tabela do P-chase com uma linha impossível (Exp. 4)

A Tabela 4 do artigo registrava:

```
32--512 KB    86--140 ns    L2
1--8 MB       ~137 ns       VRAM     <- menor que o topo da faixa do L2
```

A VRAM não pode ser mais rápida que o L2. Medição real (`pchase`, 26 pontos):

| Working set | Latência | Nível |
|---|---|---|
| 256 B -- 16 KB | 38,63 -- 38,88 ns | L1 |
| 20 KB | 58,96 ns (+52%) | transição L1→L2 |
| 24 -- 256 KB | 69,9 -- 129,6 ns | L2 |
| 512 KB -- 32 MB | 139,0 -- 256,2 ns | transição L2→VRAM, **gradual** |

O valor de ~137 ns corresponde a 512 KB, e foi transcrito para a linha errada.
Não há patamar de VRAM nem salto nítido em 2 MB — com um único *work-item* a
transição é suave, como o próprio Experimento 7 já reconhecia. **Correção
aplicada** na tabela.

---

## 4. Coeficiente assintótico depende de tamanhos não documentados (Exp. 5)

`glc_scaling` aceita os tamanhos por linha de comando e, sem argumentos, usa
`DEFAULT_SIZES = {256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072}` — nove
valores, todos múltiplos de 256. Mas o `resultados/glc_scaling.csv` e o
`analise/modelo_glc.txt` que produziram o número publicado contêm **dezesseis**
tamanhos, incluindo 320, 384, 448, 576, 640, 704 e 896.

| Conjunto de tamanhos | *bypass* do L1 | serialização | custo do *asm* |
|---|---|---|---|
| 16 tamanhos (os do baseline) | **1,114×** | 1,099× | 1,522× |
| 9 tamanhos (padrão do código) | 1,070× | 1,080× | 1,519× |
| publicado | 1,141× | 1,098× | 1,537× |

O ajuste de $a$ em $T(N) = aN^3 + bN^2$ é sensível ao conjunto de pontos: os
múltiplos de 256 sofrem conflito de mapeamento no L1 e puxam a razão para cima.
O comando documentado em `README`/`niveis_cache.md` **não reproduz o número
publicado**.

**Correções**: a tabela do Experimento 5 passou a registrar 1,11--1,14×, faixa
que cobre a variação entre execuções, em vez de 1,141× com três decimais. Fica
pendente decidir se `DEFAULT_SIZES` deve incorporar os dezesseis tamanhos — o
que tornaria o comando sem argumentos reprodutível.

---

## 5. Faixa não reprodutível no P-chase com flags (Exp. 7)

Quatro execuções independentes. O resultado central é sólido; a cauda não é.

| Ponto | execuções | amplitude |
|---|---|---|
| razão `glc` em 1 KB | 1,4226 / 1,4241 / 1,4242 / 1,4380 | **0,5%** |
| latência em 8 MB | 307,4 / 283,4 / 287,3 / 281,6 ns | 9% |
| latência em 256 KB | 268,6 / 257,9 / 254,1 / 237,3 ns | 12% |

Na execução 2, `glc` em 256 KB (244,18 ns) ficou **mais rápido** que o acesso sem
flags (257,90 ns) — fisicamente impossível, portanto ruído.

**Hipótese de DVFS testada e descartada**: amostrando
`/sys/class/drm/card0/device/pp_dpm_sclk` durante a execução, a frequência
permanece em 1350 MHz e `gpu_busy_percent` em 100% do início ao fim. A causa
provável é alocação de páginas / TLB entre execuções.

**Consequência**: a afirmação "diferença média de 0,30% e máxima de 0,84%" declara
precisão maior do que a reprodutibilidade permite. **Correção aplicada**: o texto
passou a reportar a faixa observada em quatro execuções e a basear a conclusão na
magnitude da penalidade (+71 ns, contra +70 ns de L1→L2 e +207 ns de L1→VRAM
medidos em C puro), que é robusta.

---

## 6. Itens de implementação pendentes

1. **`pchase_flags.cpp:177`** seleciona a primeira plataforma que exponha uma GPU,
   sem filtrar por nome. Com Clover e PoCL instalados, pode cair no *runtime*
   errado. `crivo_l1.cpp` já filtra por `"AMD Accelerated"`; convém uniformizar.
2. **`pchase.cpp`** rotula como "VRAM" tudo acima de 512 KB, embora o L2 tenha
   2 MB. Os rótulos da saída não correspondem aos níveis reais.
3. **Sem desvio padrão em `pchase_flags`**: 5 execuções, só a média. A conclusão
   central é uma afirmação de equivalência, que exige medida de dispersão.
4. **Ausência de mediana no Experimento 6**, causa raiz do item 2 desta bateria.

---

## 7. Reprodução desta bateria

```bash
source setup/env_rocm.sh
for e in matmul mem_hierarchy niveis_cache pchase algoritmos_cache \
         glc_factorial glc_scaling crivo_l1 tdiv_ws pchase_flags; do
    ./output/$e
done
# Exp. 5 com os tamanhos do baseline:
./output/glc_scaling 256 320 384 448 512 576 640 704 768 896 1024 1280 1536 2048 2560 3072
```
