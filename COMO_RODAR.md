# Como Rodar os Experimentos

Guia completo de compilação e execução. Ambiente de referência: **Ubuntu 20.04,
ROCm 5.2, AMD RX 580 (Ellesmere / Polaris / gfx803)**.

---

## 0. Antes de tudo

### 0.1 Preparar o ambiente (uma vez só)

Os três scripts de `setup/` instalam o necessário. Precisam de root; como o `sudo`
pede senha, use `pkexec` com caminho absoluto (o `pkexec` reseta o diretório):

```bash
cp setup/01_base.sh /tmp/ && pkexec bash /tmp/01_base.sh
cp setup/02_fix_clover_libclc.sh /tmp/ && pkexec bash /tmp/02_fix_clover_libclc.sh
cp setup/03_rocm_opencl.sh /tmp/ && pkexec bash /tmp/03_rocm_opencl.sh
```

Depois é preciso **logout/login** para o grupo `render` valer na sessão. Antes
disso, prefixe os comandos com `sg render -c '...'`.

### 0.2 Carregar o ambiente (toda vez que abrir um terminal)

```bash
cd ~/"Área de Trabalho/tcc-gpu-opencl"
source setup/env_rocm.sh
```

> **`ROC_ENABLE_PRE_VEGA=1` é obrigatório.** Sem essa variável o ROCm 5.2 enumera
> a plataforma AMD com **zero dispositivos**, porque Polaris foi depreciado. É o
> que o `env_rocm.sh` define.

### 0.3 Verificar que a GPU aparece

```bash
./output/asm_probe        # se ainda não existir, veja a seção 1
```

Deve listar `AMD Accelerated Parallel Processing` → `gfx803`, com
`__asm__ glc  COMPILOU`.

### 0.4 Duas regras que evitam a maioria dos erros

1. **Rode sempre a partir da raiz do projeto.** Os programas `algoritmos_cache`,
   `crivo_l1` e `tdiv_ws` leem o arquivo `.cl` por caminho relativo e falham com
   `nao abriu 'experimentos/...'` se você estiver noutro diretório.
2. **Os resultados são sobrescritos.** Cada execução regrava o CSV
   correspondente em `resultados/`. Se quiser preservar uma execução, copie o CSV
   antes de rodar de novo.

---

## 1. Compilar tudo de uma vez

```bash
mkdir -p output
for f in experimentos/*.cpp setup/asm_probe.cpp; do
    b=$(basename "$f" .cpp)
    g++ -Wall -O2 "$f" -o "output/$b" -lOpenCL && echo "OK   $b" || echo "FALHOU $b"
done
```

Todos devem compilar sem nenhum *warning*. O comando individual, se precisar:

```bash
g++ -Wall -O2 experimentos/matmul.cpp -o output/matmul -lOpenCL
```

---

## 2. Os experimentos, um a um

Todos assumem `source setup/env_rocm.sh` já executado e o terminal na raiz do
projeto.

### Experimentos do artigo — comandos que geraram os dados

Atualizado em 14/10/2026. Numeração do artigo a partir dessa data (o antigo
Exp. 1 virou a §4.1 e os demais desceram um número); contexto em `CLAUDE.md`.
Rodar sempre da raiz, com `source setup/env_rocm.sh`.

| Exp. | Comando | Saída | Duração |
|---|---|---|---|
| §4.1 e 1 | `./output/matmul 1000 1024 10` | `resultados/matmul_cache_flags.csv` + ISA em `isa/` | ~2 min |
| 2 | `./output/slc_l2 > resultados/slc_l2.txt` | só texto | segundos |
| 2 (extensão, 04/11/2026) | `./output/teto_vram \| tee resultados/teto_vram.txt` | `resultados/teto_vram.csv` (o programa se recusa a sobrescrevê-lo) + ISA em `isa/rocm_gfx803_teto_vram.bin` | ~20 s |
| 3 (GPU) | `./output/matmul 100 N 5` para N = 4096, 8192; `./output/matmul 100 16384 3 2048` (ver `tools/cadeia_sweep.sh`) | `resultados/matmul_sweep.csv` | ~11 h em 16384 |
| 3 (CPU) | `./output/matmul_cpu "1024 2048 4096 8192 16384" 60` (`tools/cpu_sweep.sh`) | `resultados/matmul_cpu_sweep.csv` | horas |
| 4 (padrões) | `./output/linha_coluna "1024 2048 4096 8192 16384" 4000` | `resultados/linha_coluna.csv` | ~1,5 h (estimativa; quase todo em N=16384) |
| 4 (B transposta) | `./output/matmul_layout "1024 2048 4096 8192" 6000 2048` e `./output/matmul_layout 16384 6000 1024` | `resultados/matmul_layout.csv` | ~1,5 h |

Observações que afetam a reprodução:

- **Exp. 2, extensão (04/11/2026):** `teto_vram` compila `slc_l2.cl` (intocado)
  junto com `teto_vram.cl` e tem quatro partes: A, varredura de 256 KB a 2 GB
  com os kernels originais; B, kernels com 256 B a 4 KB em voo por wavefront
  (conferidos em `isa/teto_vram_disasm.txt`); C, L2 ocupado por outro buffer
  de 64 MB antes de cada execução; D, 1 MB em 10 buffers distintos. Aquece
  0,2 s antes de cada buffer: sem isso, o protocolo de `slc_l2` chegou a medir
  64 MB a 115,4 GB/s, com a memória no estado de 1000 MHz
  (`resultados/slc_l2_reexecucao.txt`, três execuções do binário de 19/08).
  Na parte C, o kernel de limpeza termina (`clFinish`) antes do medido: logo
  atrás de outro kernel, o START do ROCm sai antes do fim do anterior.
- **Exp. 3:** N=1024 e 2048 foram rodados com `RUNS=100` e aquecimento de 5
  antes do script de cadeia (registrado em `resultados/sweep_log.txt`); N=4096 e
  8192, pela cadeia, também com 5; N=16384 com 3 (`resultados/cadeia_log.txt`).
  O N=4096 do `sweep_log.txt` foi interrompido no kernel 6 e refeito pela cadeia;
  o CSV contém a execução completa.
- **Os programas gravam nos CSVs das campanhas** — antes de qualquer reexecução,
  copiar `resultados/` ou redirecionar a saída (regra 2 do `CLAUDE.md`):
  `matmul` acrescenta a `matmul_sweep.csv` e **sobrescreve**
  `matmul_amostras_N<N>.csv`; `matmul_layout` acrescenta a `matmul_layout.csv`
  (caminho fixo no fonte); `linha_coluna` acrescenta a `linha_coluna.csv` quando
  o 4º argumento é omitido. A amostra do Exp. 1 (`matmul_cache_flags.csv`) foi
  gravada por uma versão anterior do programa e não é mais regravada.
- **Exp. 4, `linha_coluna`:** cada medição é **um lançamento único**, sem lotes.
  Na campanha original o 3º argumento recebeu `2048`/`1024`, mas era ignorado; a
  partir de 14/10/2026 o 3º argumento é o **filtro de padrão**
  (`linha|dispersa|coluna|todos`) e o 4º é o CSV de saída. **Não repetir** o
  comando antigo com número no 3º argumento: nenhum kernel casaria com o filtro.
- **Remedição** (14/10/2026): `tools/remedicao_dispersa.sh` →
  `resultados/linha_coluna_remedicao.csv`. Remedição nunca vai para o CSV da
  campanha original.
- **2ª remedição** (20/10/2026, 20:38–21:38): `tools/remedicao_dispersa_2.sh` →
  `resultados/linha_coluna_remedicao2.csv` (log em `…_remedicao2.txt`). Mesmas
  condições (`dispersa`, N=8192 e 16384, orçamento de 4000 ms, 5 execuções). O
  script recusa rodar se o CSV já existir. Uma 3ª remedição precisa de script e
  arquivo próprios.
- **`matmul_layout`:** o 3º argumento é o tamanho do lote em linhas
  (fatiamento por `global_work_offset`), usado como precaução.

### Tabelas do artigo a partir dos CSVs

```bash
python3 analise/gera_tabelas.py              # regrava Documentação/tab_*.tex (Tabelas 4, 5, 6 e 7)
python3 analise/gera_tabelas.py --saida DIR  # grava em outro diretório, para comparar antes
python3 analise/gera_tabelas.py --conferir   # imprime as Tabelas 2 e 3 e os números do texto do Exp. 2
```

As Tabelas 2 e 3 estão digitadas no `main.tex`; o `--conferir` serve para
compará-las com o CSV. Desde 04/11/2026 a Tabela 4 tem dois blocos gerados:
`tab_slc_original.tex` (campanha original, de `slc_l2.txt`) e `tab_slc.tex`
(varredura, de `resultados/teto_vram.csv`, parte A), com tempo, vazão e a razão
`glc slc`/`glc`; os
números do texto do Exp. 2 (patamar, variantes, 17 medições de 1 MB, controle
do L2, modelo) saem do `--conferir`, que também lê a campanha original
(`slc_l2.txt`) e as reexecuções (`slc_l2_reexecucao.txt`).
Em 20/10/2026 o script reproduziu byte a byte os `tab_*.tex` existentes; em
04/11/2026, de novo, os cinco anteriores (os `tab_slc*.tex` são novos).

### Experimentos arquivados (fora do artigo)

Os itens abaixo são do percurso anterior e estão reservados para a banca
(texto em `Documentação/arquivo_experimentos.tex`). A descrição de `matmul` como
"4 kernels" está **desatualizada**: o programa hoje tem 10 cenários e os
argumentos `[RUNS] [N] [WARMUP] [LINHAS_POR_LOTE]`.

### Padrões de acesso à memória (`mem_hierarchy`)

Redução sob acesso sequencial, *stride*, aleatório e LDS; mais um *stencil* 2D.

```bash
./output/mem_hierarchy
```

*Duração: ~1 s.*

### P-chase — mapeamento dos níveis (`pchase` e `pchase_v2`)

Percorre uma lista encadeada aleatória variando o *working set*. É o que revela
o tamanho do L1 pelo salto de latência em 20 KB.

```bash
./output/pchase          # ~21 s — 26 pontos, single work-item
./output/pchase_v2       # duas fases: single e multi work-item (36 CUs)
```

### Números primos, scan e crivo (`algoritmos_cache`)

Três algoritmos clássicos sob os três níveis de cache. Verifica tudo contra
referência em CPU antes de reportar tempo.

```bash
./output/algoritmos_cache
```

*Duração: ~2 s.* Gera `resultados/algoritmos_cache.csv` e
`isa/rocm_gfx803_algoritmos_cache.bin`.

### Primos por divisão tabelada — o maior efeito medido (`crivo_l1`)

Compara o crivo segmentado com a divisão sucessiva tabelada, nas variantes de
acesso uniforme e disperso.

```bash
./output/crivo_l1
```

*Duração: ~4 s.* Gera `resultados/crivo_l1.csv` e `isa/rocm_gfx803_crivo_l1.bin`.

### Varredura do conjunto de trabalho (`tdiv_ws`)

Varre a tabela de 2 KB a 51 KB e localiza o pico do custo em 16 KB. Reaproveita
os kernels de `crivo_l1.cl`.

```bash
./output/tdiv_ws
```

*Duração: ~31 s.* Gera `resultados/tdiv_ws.csv`.

### Níveis de cache × padrão de acesso (`niveis_cache`)

Cruza os três níveis de cache com *streaming*, *stencil* e matmul.

```bash
./output/niveis_cache
```

*Duração: ~1 s.* Gera `resultados/niveis_cache.csv`.

### P-chase com controle de flags (`pchase_flags`)

Repete a curva de latência sob `sem flags`, `glc` e `glc slc`. É o experimento
que prova que `glc` contorna o L1 e que `slc` **não** contorna o L2.

```bash
./output/pchase_flags
```

*Duração: ~63 s.* Gera `resultados/pchase_flags.csv` e
`isa/rocm_gfx803_pchase_flags.bin`.

### Desenho fatorial 2×2 (`glc_factorial`)

Isola o custo do *bypass* do L1 do custo da serialização, num único tamanho.

```bash
./output/glc_factorial
```

*Duração: ~1 s.*

### Escalonamento e ajuste do modelo (`glc_scaling`)

Mede as seis células do fatorial em vários tamanhos de matriz e ajusta
T(N) = a·N³ + b·N².

```bash
# Padrão: 9 tamanhos, múltiplos de 256 — ~50 s
./output/glc_scaling

# Os 16 tamanhos que produziram os números do artigo — ~2 min
./output/glc_scaling 256 320 384 448 512 576 640 704 768 896 1024 1280 1536 2048 2560 3072
```

> **Atenção:** o coeficiente assintótico depende do conjunto de tamanhos. Com os
> 9 padrão o *bypass* do L1 dá 1,07×; com os 16 acima, 1,11×. **Os números do
> artigo usam os 16 tamanhos** — use a segunda forma para reproduzi-los.

Gera `resultados/glc_scaling.csv`.

---

## 3. Rodar todos em sequência

```bash
source setup/env_rocm.sh
for e in matmul mem_hierarchy niveis_cache pchase algoritmos_cache \
         glc_factorial crivo_l1 tdiv_ws pchase_flags; do
    echo "=== $e — $(date +%T) ==="
    ./output/$e
done
./output/glc_scaling 256 320 384 448 512 576 640 704 768 896 1024 1280 1536 2048 2560 3072
```

*Duração total: ~4 minutos.*

---

## 4. Análise do ISA

Confirma que as *flags* GLC/SLC foram realmente emitidas — passo obrigatório
antes de confiar em qualquer medição de nível de cache.

```bash
# 1. o experimento grava o binário em isa/ ao rodar
./output/crivo_l1

# 2. desmontar
llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 \
    isa/rocm_gfx803_crivo_l1.bin > isa/crivo_l1_disasm.txt

# 3. resumir flags por kernel
python3 isa/analyze_disasm.py isa/crivo_l1_disasm.txt
```

A saída lista, por kernel, o número de instruções, *loads* e as *flags*:

```
tdiv_rot_L1    68  21  2  2  GLC=0 SLC=0 (usa L1)
tdiv_rot_L2    68  21  2  2  GLC=1 SLC=0 (contorna L1)
tdiv_rot_VRAM  68  21  2  2  GLC=1 SLC=1 (contorna L1 e L2)
```

O mesmo vale para `rocm_gfx803_algoritmos_cache.bin` e
`rocm_gfx803_pchase_flags.bin`.

---

## 5. Gráficos

```bash
python3 analise/plot_pchase_flags.py     # -> analise/fig_pchase_flags.png
python3 analise/plot_tdiv_ws.py          # -> analise/fig_tdiv_ws.png
python3 analise/plot_glc_scaling.py      # -> figuras + analise/modelo_glc.txt
```

Requer apenas `matplotlib` (sem `pandas`). Para usar as figuras no artigo:

```bash
cp analise/fig_tdiv_ws.png analise/fig_pchase_flags.png Documentação/
```

---

## 6. Compilar o artigo

```bash
cd Documentação
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # a terceira passada resolve as referências
cd ..
```

São **quatro** comandos e a ordem importa: o `bibtex` precisa do `main.aux` que a
primeira passada gera, e só a terceira passada resolve todas as referências
cruzadas.

Para verificar que não sobrou erro nem referência indefinida, examine o log da
**última** passada:

```bash
cd Documentação
pdflatex -interaction=nonstopmode main.tex > /tmp/tex.log 2>&1
grep -E "^!|[Uu]ndefined (control|citation|reference)|There were undefined" /tmp/tex.log
echo "^ vazio = tudo certo"
cd ..
```

> **Não verifique com uma passada isolada depois de apagar os arquivos
> auxiliares.** Sem `main.aux` e `main.bbl`, o LaTeX reporta *todas* as citações
> e referências como indefinidas — um falso positivo. Rode a sequência completa
> antes de conferir.

Limpar os auxiliares (opcional, só quando terminar):

```bash
cd Documentação && rm -f main.aux main.bbl main.blg main.log main.out && cd ..
```

---

## 7. Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `Nenhuma GPU OpenCL encontrada` | falta `ROC_ENABLE_PRE_VEGA=1` | `source setup/env_rocm.sh` |
| `nao abriu 'experimentos/*.cl'` | rodou fora da raiz do projeto | `cd` para a raiz |
| `FALHA no build ... __asm__` | caiu num runtime sem Clang | confira se a plataforma é `AMD Accelerated Parallel Processing` |
| Permissão negada em `/dev/kfd` | sessão fora do grupo `render` | logout/login, ou `sg render -c './output/...'` |
| `Memory access fault ... Page not present` | `"=v"` em vez de `"=&v"` no asm | use *earlyclobber* (`"=&v"`) nas saídas |
| Resultados oscilando muito | GPU ociosa entre execuções, ou média contaminada por *outlier* | use a **mediana**, não a média |

---

## 8. Onde cada coisa vai parar

| Diretório | Conteúdo |
|---|---|
| `output/` | executáveis compilados |
| `resultados/` | CSVs (**sobrescritos** a cada execução) |
| `isa/` | binários ISA e *disassemblies* |
| `analise/` | scripts de gráfico e figuras |
| `Documentação/` | artigo LaTeX e os relatórios `.md` de cada experimento |
