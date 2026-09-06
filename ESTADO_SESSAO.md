# Estado da sessão — 20/08/2026

Documento de retomada. Registra o que mudou, o que está rodando e o que ficou
pendente, para que o trabalho continue sem depender do histórico da conversa.

## Análise de sensibilidade ao tamanho (concluída)

Análise de sensibilidade ao tamanho em terminal próprio (`tools/cadeia_sweep.sh`), aberta na
tela. Sequência: **N=4096 → 8192 → validação dos lotes → 16384**.

- 100 execuções por cenário, 10 cenários por tamanho
- N=1024 e N=2048 já concluídos e gravados
- Resultados acumulam em `resultados/matmul_sweep.csv` (append + `fflush`)
- Log em `resultados/cadeia_log.txt`
- Estimativa: ~1,5 h até o 8192, mais ~11 h no 16384

O **16384 só roda se a validação dos lotes passar** — a etapa compara o
lançamento fatiado contra o lançamento único em N=2048 e aborta se divergir.

### Por que existem lotes

O amdgpu reseta a GPU em job de compute acima de **60 s**, e em N=16384 um
lançamento único levaria ~58 s. Como o Core Ultra 5 **245KF** não tem gráfico
integrado, a RX 580 é a única saída de vídeo: um reset derrubaria o desktop e o
job de 11 h juntos. A NDRange é fatiada por `global_work_offset` em faixas de
linhas (5º argumento da linha de comando), o que mantém o trabalho total
idêntico e cada lançamento em ~7 s.

### Limites apurados

| N | VRAM (3 matrizes) | RAM host (4 vetores) | Situação |
|---|---|---|---|
| 8192 | 0,75 GB | 1 GB | ok |
| 16384 | 3 GB | 4 GB | teto viável |
| 32768 | 12 GB | 16 GB | **impossível** (placa tem 8 GB, máquina tem 15) |

## experimentos/matmul.cpp — o que mudou

De 4 para **10 kernels**, todos validados elemento a elemento:

| # | Kernel | Flags | Papel |
|---|---|---|---|
| 1 | `matmul_via_L1` | GLC=0 | linha de base em C puro |
| 2 | `matmul_via_L2` | GLC=1* | hipótese original com `volatile` |
| 3 | `matmul_direto_VRAM` | GLC=1* | fonte idêntico ao 2 — controle negativo |
| 4 | `matmul_asm_baseline` | GLC=0 | controle: mede o custo do bloco `asm` |
| 5 | `matmul_bypass_L1` | GLC=1 | `asm` com `glc` |
| 6 | `matmul_bypass_L1_L2` | GLC=1 SLC=1 | `asm` com `glc slc` |
| 7 | `matmul_nontemporal` | GLC=1 SLC=1 | `__builtin_nontemporal_load`, sem `asm` |
| 8 | `matmul_tiled` | GLC=0 | referência com LDS |
| 9 | `matmul_intrin_baseline` | GLC=0 | controle do intrinsic |
| 10 | `matmul_intrin_glc` | GLC=1 | `llvm.amdgcn.raw.buffer.load.f32` |

`*` depende do compilador: `glc` no ROCm, nenhuma flag no AMDOCL do Windows.

Instrumentação: RUNS configurável (0 = adaptativo por orçamento de tempo),
mediana, desvio, cv, p95, deriva, IC 95% pelo método delta, t de Welch e **piso
de ruído** derivado do par de kernels de fonte idêntico. Validação em duas
camadas: cada cenário contra a saída do kernel 1 em O(N²), e o kernel 1 contra
uma amostra de 4096 posições calculada na CPU.

Uso: `./output/matmul [RUNS] [N] [WARMUP] [LINHAS_POR_LOTE]`

## Achados desta sessão

1. **`volatile` sob ROCm emite `glc`**, sob AMDOCL não. Mesmo fonte, mesma GPU,
   mesmo target — é decisão de codegen. E ele faz duas coisas: além da flag,
   serializa os loads com `s_waitcnt` após cada um.
2. **`address_space(1)` não faz bypass** — é o mesmo espaço que `__global`.
3. **`slc` não contorna a L2.** Buffer que cabe na L2 sustenta 556 GB/s com
   `glc slc`, contra 222 GB/s medidos para a VRAM (`experimentos/slc_l2.cpp`).
4. **Nenhum store recebe `glc`** — a L1 é write-through por construção.
5. **O intrinsic é mais rápido que o C puro**: 396 contra 307 GFLOPS em N=1024,
   por usar 7 VGPRs contra 11. Dá a célula que faltava: `glc` isolado, sem a
   barreira do `asm`.
6. **Não existe um número único para "custo de contornar a L1"** — vai de 1,24×
   a 2,43× conforme a linha de base contra a qual se mede.
7. O tempo de offloading **não entra** na medição: o cronômetro é o evento de
   kernel. H2D de 4 MB custa 0,37 ms, D2H 1,89 ms, lançamento 0,02 ms.

## Pendências

- **Artigo com 18 páginas**, limite SBC é 16. Precisa decidir o que cortar:
  mover os Experimentos 3 e 4 para o TCC II (~2,5 pág), condensar Trabalhos
  Relacionados (~1 pág), ou cortar o Experimento 1 (~0,8 pág).
- **Metodologia não diz que a temporização exclui as transferências.** Falta a
  frase, e falta registrar que os GFLOPS são de kernel, não fim-a-fim.
- **Incorporar a análise ao artigo** quando terminar, substituindo as medidas
  de ponto único — N=1024 é múltiplo de 256 e cai em conflict miss, não serve
  como tamanho representativo.
- `Documentação/arquivo_experimentos.tex` guarda o P-chase com controle de
  flags, que é a prova direta de que `glc` contorna a L1. Vale trazer de volta.
