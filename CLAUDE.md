# TCC — Impacto da Hierarquia de Memória em GPU AMD sob OpenCL

Fonte única de contexto do projeto. Atualizado em 20/10/2026.
`CONTEXTO_LINUX.md` e `ESTADO_SESSAO.md` são históricos; `COMO_RODAR.md` é o
manual operacional.

## O trabalho

- **Aluno:** Henrique de Almeida Diniz — Ciência da Computação, PUC Minas.
- **Orientador:** Pedro Henrique Ramos Costa.
- **Estágio:** **TCC II**. Próxima entrega: **intermediária** ao orientador, **21/10/2026**.
- **Artigo:** `Documentação/main.tex`, formato SBC, **10 a 16 páginas** (Art. 4º).
- **Título:** "Análise do Impacto da Hierarquia de Memória em GPU AMD Utilizando OpenCL".

### Pergunta central (decidida em 14/10/2026)

**Qual o impacto da hierarquia de memória no desempenho de algoritmos paralelos
em GPU AMD sob OpenCL?** As *flags* de cache (GLC/SLC) são o **instrumento** de
medição, não o objeto: servem para tirar ou pôr o L1 no caminho e medir quanto
ele vale. Núcleo do artigo: o que o L1 vale e do que isso depende (tamanho do
problema, localidade, algoritmo completo). Como as flags foram obtidas é
apoio metodológico.

## Experimentos (numeração do artigo, desde 14/10/2026)

O antigo "Experimento 1" (mecanismos) virou a **§4.1 Obtenção e Verificação do
Instrumento**, sem seção de resultados; os demais foram renumerados (2→1, 3→2,
4→3, 5→4). Arquivos e CSVs não mudaram de nome.

| # | Pergunta | Programa | Dados |
|---|---|---|---|
| §4.1 | Que mecanismo emite as flags? (volatile, asm, builtin, address_space) | `experimentos/matmul.cpp` | ISA em `isa/` |
| 1 | Quanto custa contornar o L1, isolado do custo de construção? | `matmul.cpp` (N=1024, 1000 exec.) | `resultados/matmul_cache_flags.csv` |
| 2 | SLC=1 contorna o L2? (não: é política de evicção) | `experimentos/slc_l2.cpp`; extensão `teto_vram.cpp` (04/11) | `resultados/slc_l2.txt` (só texto); `teto_vram.csv` (Tabela 4 desde 04/11) |
| 3 | Como o custo varia com N (1024–16384)? + referência em CPU | `matmul.cpp`, `matmul_cpu.cpp` | `matmul_sweep.csv`, `matmul_cpu_sweep.csv` |
| 4 | Quanto o valor do L1 depende da localidade? (linha/dispersa/coluna) + matmul com B transposta | `linha_coluna.cpp`, `matmul_layout.cpp` | `linha_coluna.csv`, `matmul_layout.csv` |

Arquivados (fora do artigo, reserva para banca): P-chase, padrões de acesso
(51×), `tdiv_ws` (pico de 9,79× em 16 KB), scan/primos/crivo — texto em
`Documentação/arquivo_experimentos.tex`. Comparativo Windows × Linux: reservado
para a apresentação.

### Resultados centrais (conferidos contra os CSVs)

- `glc` contorna o L1 (verificado na ISA: `DC510000`); o custo cai de 1,20× (N=1024) a 1,06× (N=16384).
- Em leituras FLAT sobre a VRAM, `slc` **não** contorna o L2: 555,9 GB/s com conjunto que cabe no L2, acima do pico teórico de 256 GB/s da VRAM (o patamar medido fora do L2 é 222,3–223,9 GB/s, teto prático). De 2 a 64 MB, `glc slc` é até 1,51× **mais rápido** que `glc`. O manual GCN3 descreve o SLC como "forced to miss in level 2" para MUBUF/MTBUF (Tab. 8.2) e como política com GLC e MTYPE para FLAT (Tab. 9.1); MUBUF e outros MTYPE não foram testados.
- Localidade: contornar o L1 custa até 2,19× em percurso contíguo e 1,01× em percurso por coluna. O que governa é o reúso dentro da linha de 64 B, não o *broadcast*.
- Transpor B (otimização clássica de CPU) deixa a matmul 2,31–2,46× **mais lenta**; a soma dos padrões isolados prevê a razão (erro < 3% a partir de N=2048).
- Comparação ingênua superestima o efeito do cache: o bloco de asm custa sozinho ~1,5×; o `volatile` serializa os loads.

## Ambiente

- **GPU:** RX 580 8 GB (Ellesmere, Polaris, GCN 4, **gfx803**), 36 CUs; L1 16 KB/CU (write-through), L2 2 MB, LDS 64 KB/CU. É a **única** saída de vídeo (CPU 245KF sem gráfico integrado).
- **Sistema:** Ubuntu, kernel 5.15, **ROCm 5.2** (Polaris depreciada: exige `ROC_ENABLE_PRE_VEGA=1`). Compilador de kernel: Clang/LLVM (`comgr`). Host: GCC. ISA: `llvm-objdump`.
- O percurso em **Windows** (AMDOCL, MinGW) foi a etapa inicial: lá o `volatile` não emite flags e o `__asm__` não compila. Não é mais o ambiente de trabalho.

```bash
source setup/env_rocm.sh
g++ -Wall -Wextra -O2 experimentos/<prog>.cpp -o output/<prog> -lOpenCL
./output/<prog> ...        # rodar SEMPRE da raiz do projeto
```

## Regulamento (Regulamento_TCC_CC (1).pdf) — o que o artigo precisa ter

- Introdução: contextualização, motivação, justificativa, problema, objetivos geral e específicos, **síntese do que foi feito e principais resultados**, organização do texto.
- Referencial teórico e revisão de literatura.
- Metodologia **com diagramas/fluxogramas/figuras** e citação das ferramentas; descrita de modo **reproduzível** (Art. 5º, §3).
- Resultados em gráficos, figuras ou tabelas, comentados e analisados.
- **Conclusões e Trabalhos Futuros**: ressaltar contribuições e comparar os resultados **com os da literatura**.
- ≥ 10 referências. Apêndice só se o orientador indicar (Art. 4º, §2).
- Modelo SBC (Art. 3º): **resumo e abstract com no máximo 10 linhas cada, ambos na página 1**; Times 12; numeração de seções opcional.
- Declaração de IA (Arts. 6º–8º): ferramenta, **versão**, finalidade e **seções** em que contribuiu.
- **Não** incluir cronograma (é exigência do TCC I).

## Feito em 14/10/2026

- Reestruturação do artigo em torno da pergunta central: Introdução com síntese
  dos resultados e objetivos realinhados; justificativa de SO (Chen, Gupta) fora
  (13 referências); diagrama TikZ da hierarquia na Metodologia
  (`Documentação/fig_hierarquia.tex`); Exp. 1 → §4.1; seção final renomeada
  "Conclusões e Trabalhos Futuros", com três conclusões e comparação com a
  literatura; resumo e abstract reescritos com paridade numérica.
- Corrigidos: afirmação de reset em 60 s (agora "por precaução"); §Exp. 4 dizia
  que houve lotes — não houve; 1,015× desatualizado (é 1,019×); parágrafo da CPU
  movido para o Exp. 3; §2.2 prometia tratar a edição do binário (não trata).

## Feito em 19/10/2026

- Incluído Jia, Shaw e Martonosi (ICS 2012), o trabalho mais próximo da pergunta
  central (L1 ligado/desligado no Fermi, taxonomia de localidade): §3.2, Tabela 1,
  §3.4 e comparação nas Conclusões. Não confundir com Jia et al. 2018 (Zhe Jia).
- Corrigida a geometria da *wavefront* no Exp. 4: com `ls = {16,16}` cada
  *wavefront* cobre 4 `row` × 16 `col` (verificado com
  `experimentos/sonda_wavefront.cpp`); `linha` toca 4 linhas por instrução,
  `coluna` 1, `dispersa` 19 — o texto dizia 1 e 64. Detalhes em `CORTES.md`.
- Escopo do SLC restrito a leituras FLAT sobre a VRAM (resumo, abstract,
  Introdução, §2.2, Metodologia, Exp. 2, Conclusões, Limitações). A §2.2 cita as
  duas descrições do manual (Tabelas 8.2 e 9.1) no lugar de "é comum encontrar".
- Auditoria: todos os CSVs são a última execução completa e o código atual mede
  igual (detalhes em `CORTES.md`). Corrigidos no texto: cv do Exp. 4 (até 19,4%
  em `dispersa`), cv da `coluna` e "indistinguível de zero" → "pequeno, mas
  mensurável"; variação entre campanhas em N=1024 (1,19–1,237×); orçamento de
  execuções do Exp. 4. **Reexecutar programas grava nos CSVs das campanhas** —
  ver `COMO_RODAR.md` antes.

## Feito em 20/10/2026

- Conferidas as Tabelas 2 a 7 contra os CSVs (todas batem). Corrigidos: CPU
  contra cenário 1 usava uma execução isolada (7,191 ms) — agora mediana 7,223 e
  34×; legendas das Tabelas 2 e 5 com o número real de execuções da CPU; **"fator
  de 256 em volume de dados"** (era 4.096, que é o fator do trabalho N³) no resumo,
  abstract, Exp. 3 e Conclusões.
- Novo `analise/gera_tabelas.py`, que reproduz byte a byte os `tab_*.tex`.
- Figura 2 refeita: um eixo, três curvas com numerador ÷ denominador por extenso
  na legenda; texto passou de "painel" a "curva"; corrigido "cresce
  monotonicamente" (o slc cai em N=2048).
- Disco liberado (421 MB → 2,3 GB livres: *journal* reduzido a 100 MB e 7 revisões
  desativadas de *snap* removidas). 2ª remedição da `dispersa`
  (`resultados/linha_coluna_remedicao2.csv`): `glc slc` 3,360× (N=8192) e 2,515×
  (N=16384). A queda 8192 → 16384 persiste nas três campanhas (−37%, −17%, −25%);
  medianas das três: 3,360× e 2,500×. `glc` reproduz a ≤ 2,3%. Texto do Exp. 4
  atualizado com as três campanhas (aprovado pelo aluno). A Tabela 6 fica
  só com a campanha original, sem nota (decisão do aluno).

## Feito em 04/11/2026

- Orientador pediu revisão da afirmação dos 222 GB/s do Exp. 2. Novo
  `experimentos/teto_vram.cpp` (+ `.cl`; `slc_l2.cl` intocado) →
  `resultados/teto_vram.csv`; binário original reexecutado 3× →
  `resultados/slc_l2_reexecucao.txt`. Texto do Exp. 2 reescrito em torno do
  contrafactual (um *bypass* limitaria 1 MB a 256 GB/s, na prática ~222); Tabela 4
  passou a ser a varredura (`tab_slc.tex`, gerada); resumo, abstract e Conclusões
  trocam "teto de 222,3" por "pico teórico de 256". Cortes e ajustes mínimos de
  redação para manter 16 páginas: `Documentação/CORTES.md`.
- Depois, a pedido do aluno (sem se preocupar com as 16 páginas por ora): Tabela 4
  no formato original (tempo e vazão por variante), em dois blocos, campanha
  original (1 e 64 MB) e varredura (14 tamanhos), mais a razão `glc slc`/`glc`;
  Exp. 2 na versão completa; os ajustes de redação em texto do aluno revertidos.
- Travessões usados como vírgula trocados (63 de 70) por vírgula, parênteses,
  dois-pontos ou ponto e vírgula; ficam as células "não se aplica" de tabela.
  Detalhes no `CORTES.md`.
- Declaração de IA encurtada a pedido do aluno (de 8 para 3 linhas): sem citar
  artigos do regulamento nem a frase de autoria; mantém o que os Arts. 7º e 8º
  pedem (ferramenta, fornecedor, versões, finalidade e seções). Versões tiradas
  dos registros das sessões do Claude Code nesta máquina: Opus 5 e Opus 5.5.
  A análise fica fora da finalidade declarada, como na versão anterior (decisão
  do aluno).
- Pico teórico desta placa: 256 GB/s (`pp_dpm_mclk` 2000 MHz × 4 × 256 bits / 8).
  "Teto de 222,3" é vazão medida, não limite físico; o argumento forte é
  555,9 > 256 (2,17×).
- Patamar independe do tamanho: de 8 MB a 2 GB, sem flag e `glc` dão 222,3–223,9
  GB/s. Até 16× mais bytes em voo por *wavefront* não passam de 223,3 na mediana
  (225,3 na melhor execução, 88%): ~222 é o teto prático de leitura.
- Terceira evidência contra o *bypass*: de 2 a 64 MB, `glc slc` é **mais rápido**
  que `glc` (até 1,51× em 4 MB), e o ganho segue a fração 2 MB/W que fica no L2
  (erro ≤ 1,7% de 8 a 128 MB). Explica o 226,3 da Tabela 4.
- Controles: com o L2 ocupado por outro buffer antes de cada execução, `glc slc`
  em 1 MB dá 555,9 (não depende da ordem dos kernels). O valor de 1 MB varia
  com a alocação: 17 medidas, mediana 548,6, faixa 431,5–564,1.
- Protocolo de `slc_l2` (3 aquecimentos < 1 ms) mediu uma vez 115,4 GB/s em
  64 MB (memória a 1000 MHz); a Tabela 4 não foi afetada.

## Feito em 05/11/2026

- Artigo levado ao Overleaf (TeX Live 2026): `Documentação/overleaf.zip` com os 13
  arquivos necessários (testado em pasta vazia). No LaTeX 2026 o `\input` dentro de
  `tabular` quebra o `\midrule`/`\bottomrule` seguinte ("Misplaced \noalign");
  as tabelas geradas entram por `\tabinput{tab_x}` (`\@@input` primitivo, definido
  no preâmbulo). `babel` passou de `brazil` (obsoleto) a `brazilian`. PDF local
  idêntico. Ao mudar arquivos aqui, reenviar ao Overleaf (ou refazer o zip).

## Feito em 12 e 13/11/2026

- Revisão rigorosa do artigo (fontes, interpretação, linguagem) e volta às 16
  páginas. Bibliografia corrigida: duas referências tinham metadados inexistentes
  (hoje Baghsorkhi et al. 2012 e Boras et al. 2020), além de DOI, autores e
  gerações de GPU errados em outras. Detalhes, cortes e correções de conteúdo no
  `Documentação/CORTES.md`.
- Números derivados acrescentados na revisão recalculados dos CSVs: `slc`
  isolado do Exp. 4 (`linha_coluna*.csv`) e erros do modelo do Exp. 2
  (`teto_vram.csv`).
- Parágrafo de N=1008 retirado: os números não têm CSV. Só volta se for remedido.
- Decisões do aluno: não remedir a transposição em OpenCL C (o texto declara a
  construção usada); menção breve à etapa em Windows (versão do driver nas
  limitações); sem intervalo para o 2,19× (o texto dá n e cv do ponto).
- Declaração de IA: acrescentado o Claude Fable 5.2, usado nesta revisão (o regulamento
  pede a versão). Redação de resto intacta.
- `overleaf.zip` refeito com os 13 arquivos atuais. **Reenviar ao Overleaf.**

## Feito em 18/11/2026

- Orientador pediu mais figuras: nova Figura 2, tempo da CPU e da GPU por N em
  escala logarítmica (`analise/plot_tempos.py` → `Documentação/fig_tempos.pdf`).
  A figura de razões virou Figura 3. Trabalho futuro com RDNA 4 (ideia do
  aluno). Cortes e ajustes tipográficos no `CORTES.md`.
- `overleaf.zip` com 14 arquivos (inclui `fig_tempos.pdf`). **Reenviar ao Overleaf.**
- Conferência com o regulamento: síntese com números, contribuições explícitas
  nas Conclusões, Python/matplotlib citados. Folga: ~3 linhas. Entrega à banca
  (4º PC do TCC II): **até 23/11/2026, já revisada pelo orientador**.
- Atenção: `analise/plot_sensibilidade.py` usa `set_xscale("log", basex=2)`,
  que não existe no matplotlib ≥ 3.5; roda no 3.1 do Ubuntu 20.04. O
  `plot_tempos.py` funciona nos dois.

## Feito em 23/11/2026

- Revisão de outro agente aplicada (detalhes no `CORTES.md`): Vidye et al.
  conferido no PDF e descrito corretamente (atribuem ao compilador e ao driver);
  comentários do LLVM 14 citados no Exp. 2; reprodutibilidade (opções de
  compilação, medição de tempo, validação); Boras et al. retirado.
- Segunda rodada: Vidye et al. retirado (o tema de SOs sai do artigo), citação do
  LLVM no contexto certo, "mesma construção" nos Exps. 3 e 4, cenários definidos
  na §4.2, contribuição metodológica de volta ao resumo. Motivo da RX 580 (placa
  disponível): só na defesa.
- Fechamento: Conclusões ligadas à pergunta e aos objetivos 1 a 4; itálico em
  *microbenchmarking*, separador de milhar e `\ref` no roteiro da Introdução.
- Terceira rodada: E1–E7, conferência das saídas do Exp. 2 na §4.3 e limitação
  da tolerância do Exp. 4 (só distingue padrões em N=1024; ver CORTES.md e
  `analise/poder_validacao_exp4.py`). Nenhum código pode ser executado de novo
  (aluno, 23/11/2026): a limitação é definitiva.
- Quarta rodada: R1–R4 e a validação exata dos Exps. 1 e 3 (logs com diferença
  máxima zero), ver CORTES.md.
- Quinta rodada (revisor deu o texto por pronto): S1, S2 e "a medição na CPU"
  na 4.1. A igualdade exata dos Exps. 1 e 3 vem de `v_mac_f32` na GPU e de float
  sem FMA na CPU.
- Negrito reduzido (tabelas, títulos em linha, rótulos da 4.5 e as três
  conclusões); padrões em itálico; "(diferença máxima na GPU: zero)" na 4.1.
  Pendente para a versão final: o \resizebox da Tabela 1.
- Réplica do cenário 3: na 4.2 entra à parte dos cinco kernels, como controle
  negativo; a numeração dos cenários segue o desenho, não a execução (no log,
  cenário 3 = Kernel 2, réplica = Kernel 3, controle = Kernel 4).

## Pendências

- [ ] Revisão do aluno da nova redação do Exp. 2 (04/11/2026).
- [x] Voltar a 16 páginas (Art. 4º): feito em 13/11/2026.
- [ ] Leitura do orientador sobre a versão revisada (13/11/2026).
- [x] Vidye et al. (2023) conferido no PDF (23/11/2026).
- [ ] Conferir no PDF de Jia, Shaw e Martonosi (2012) as afirmações da §3.2 (tráfego L1–L2; taxonomia).
- [x] Declaração de IA: versão curta do aluno (04/11/2026), que substitui a
  redação do orientador mantida desde 19/10/2026. Ao mexer, manter ferramenta,
  fornecedor, versão, finalidade e seções (Arts. 7º e 8º).
- [ ] Revisão do aluno sobre o texto reestruturado antes da entrega de 21/10/2026.

Estado em 23/11/2026: **16 páginas** (~7,5 linhas de folga), 0 erros, 0 Overfull,
21 referências, 3 figuras; resumo e abstract com 10 linhas cada.

## Regras de trabalho

1. **Números do artigo saem dos CSVs.** Antes de escrever ou revisar um número, recalcular a partir do CSV. Tabelas `Documentação/tab_*.tex` são geradas por `analise/gera_tabelas.py` (`--conferir` imprime as Tabelas 2 e 3, digitadas no `main.tex`).
2. **Nunca sobrescrever o CSV de uma campanha.** Remedição vai para arquivo próprio. (As Tabelas 3 e 4 já ficaram irreproduzíveis uma vez por isso.)
3. **Descrever a metodologia pelo que de fato rodou:** conferir o fonte e os argumentos usados (`tools/*.sh`), não a memória da sessão.
4. **Cortes seguem o regulamento e a coerência**, não só a contagem de páginas: objetivos ↔ metodologia ↔ resultados ↔ conclusões. Ao cortar, checar se alguma conclusão perde a evidência. Registro dos cortes em `Documentação/CORTES.md`.
5. Antes de propor cortes: `pdfinfo main.pdf`; procurar figuras e tabelas órfãs (`grep -c 'ref{<label>}'`).
6. Linguagem precisa: "contornar o L1" é afirmação verificada (ISA + desempenho); "bypass do L2" **não** é — o SLC é política de evicção.
7. Texto do artigo em pt-BR, vírgula decimal, formato SBC; resumo e abstract com **os mesmos números**.
8. Jobs longos: terminal visível na tela (`gnome-terminal`), log com `tee`, espera e kill **por PID** (nunca `pgrep -f`/`pkill -f`). Antes de deixar rodando à noite, checar `df -h /` (o snap do Discord inunda o journal).
9. Lançamentos longos: o limite de 60 s do amdgpu **não** se confirmou (218 s rodou sem reset). Lotes por `global_work_offset` são precaução opcional, não necessidade.
