# Cortes de 18 → 16 páginas

Registro das alterações em `main.tex` e `sbc-template.bib` para enquadrar o
artigo no limite do Art. 4º do Regulamento de TCC (10–16 páginas).

**Estado final:** 16 páginas · 0 `Overfull \hbox` · 0 referências cruzadas
quebradas · 15 referências (mínimo 10) · 9 linhas de folga na última página.

O critério de todo corte foi **remover repetição, não conteúdo**. Nenhum dado,
número ou conclusão foi perdido. Dois erros pré-existentes foram corrigidos de
passagem (ver seção 8).

---

## Resumo

| # | Corte | Efeito |
|---|---|---|
| 1 | Figura 2, que replicava a Tabela 7 | −24 linhas |
| 2 | Metodologia deixa de antecipar o raciocínio dos Resultados (Exp. 2, 3, 5) | −1 página |
| 3 | Conclusão enxuta (767 → ~600 palavras) | −20 linhas |
| 4 | Referencial Teórico: preview vago do resultado + detalhe órfão | −10 linhas |
| 5 | Filler de abertura de seção (4 frases) | −12 linhas |
| 6 | Objetivos específicos: 5 → 4, e passam a cobrir o Exp. 5 | −3 linhas |
| 7 | Compressões pontuais (Intro, §5.1, §5.4, §5.5, Declaração de IA) | −12 linhas |
| 8 | `booktitle` com prefixo de sigla redundante | −1 linha |

---

## 1. Figura 2 — replicava a Tabela 7

A legenda declarava a duplicação na primeira frase, e o texto cita valores
exatos (`1,233×`, `3,376×`, `2,111×`) que só existem na tabela. A figura não era
fonte de nenhuma afirmação.

**Removido:**

```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{fig_linha_coluna}
\caption{Os dados da Tabela~\ref{tab:localidade}. Os dois painéis compartilham a
escala vertical: a diferença entre eles --- o que a \textit{flag} \texttt{slc}
acrescenta sobre \texttt{glc} --- é ela própria um resultado, e escalas
independentes a esconderiam.}
\label{fig:localidade}
\end{figure}
```

**Ajuste de referência** (§5.5), porque o texto apontava para um painel que
deixou de existir:

```diff
-\textbf{O \texttt{slc} confirma-se como política de evicção.} O painel~(b) mostra
-que seu efeito cresce com a pressão sobre o L2:
+\textbf{O \texttt{slc} confirma-se como política de evicção.} A metade inferior da
+Tabela~\ref{tab:localidade} mostra que seu efeito cresce com a pressão sobre o L2:
```

> `fig_linha_coluna.pdf/.png` continuam no diretório — servem para a
> apresentação, onde o gráfico funciona melhor que a tabela.

---

## 2. Metodologia deixava de antecipar o raciocínio dos Resultados

A Metodologia enunciava a previsão condicional que os Resultados reencenam ao
lado do número medido. O desenho fica na Metodologia; o *se-então* passa a
aparecer uma vez só, onde há dado para confrontá-lo.

### §4.2 — Experimento 2

```diff
-Para separá-los, o experimento emprega seis \textit{kernels} com o mesmo algoritmo,
-variando apenas o caminho de memória, entre eles um \textbf{controle} construído em
-\textit{assembly} sem \textit{flag} alguma: comparar o \textit{kernel} normal com o
-controle mede o custo da construção em \textit{assembly}, e comparar o controle com
-a variante \texttt{glc} mede o efeito da \textit{flag} com esse custo descontado.
-Como as três variantes em \textit{assembly} compartilham a mesma construção, a
-diferença entre elas resume-se aos bits de \textit{cache} dos dois \textit{loads}.
+Para separá-los, o experimento emprega seis \textit{kernels} com o mesmo algoritmo,
+variando apenas o caminho de memória, entre eles um \textbf{controle} construído em
+\textit{assembly} sem \textit{flag} alguma, que reproduz a construção das variantes
+com \textit{flag} sem alterar o caminho de memória.
```

*Motivo:* §5.2 refaz a leitura inteira — *"o desenho se apoia no cenário 2… a
comparação 2→4 mede o bit `glc` com o custo da construção descontado dos dois
lados"*.

### §4.3 — Experimento 3

```diff
 Para decidir se \texttt{SLC=1} realiza o \textit{bypass} do L2, mede-se a vazão da
 leitura repetida de um mesmo \textit{buffer} em dois regimes: 1~MB, que
-\textbf{cabe} no L2 de 2~MB, e 64~MB, que \textbf{não cabe}. O segundo calibra a
-bancada, estabelecendo em GB/s o custo de um acesso atendido pela VRAM; o primeiro
-é o teste. Se \texttt{SLC=1} desligasse o L2, a vazão com \texttt{glc slc} sobre
-1~MB cairia ao patamar do regime de 64~MB; se a \textit{flag} for apenas política
-de \textit{eviction}, as três variantes permanecem próximas, pois com o conjunto
-residente não há o que substituir. As três variantes
+\textbf{cabe} no L2 de 2~MB, e 64~MB, que \textbf{não cabe} --- o segundo calibra a
+bancada em GB/s para um acesso atendido pela VRAM, e o primeiro é o teste.
+As três variantes
```

*Motivo:* §5.3 diz exatamente isso com o número na mão — *"se `SLC=1`
realizasse o bypass do L2, a vazão teria caído ao patamar de 222 GB/s"*.

### §4.5 — Experimento 5

O `itemize` dos três padrões perdeu o mecanismo que §5.5 explica em detalhe,
mantendo fórmula e caracterização:

```diff
-\item \textbf{linha}, $A[\,\mathrm{row}\,N + k\,]$: $k$ avança 4~bytes, de modo que
-16 acessos consecutivos caem na mesma linha de \textit{cache}. Numa
-\textit{wavefront} $\mathrm{row}$ é constante e os 64 \textit{lanes} pedem o mesmo
-endereço --- \textit{broadcast}, uma linha por instrução.
+\item \textbf{linha}, $A[\,\mathrm{row}\,N + k\,]$: $k$ avança 4~bytes, de modo que
+16 acessos consecutivos caem na mesma linha de \textit{cache}; numa
+\textit{wavefront} os 64 \textit{lanes} pedem o mesmo endereço (\textit{broadcast}).
```

(as entradas **coluna** e **dispersa** receberam o mesmo tratamento)

### §4 — abertura

```diff
 (AMDOCL) não reconhece \textit{inline assembly} e não emite \textit{flags} de
-coerência nem sob \texttt{volatile}, ao passo que o Clang/LLVM as emite para o mesmo
-\textit{target}, o que torna o mapeamento de \texttt{volatile} uma decisão de
-geração de código, e não uma propriedade da arquitetura.
+coerência nem sob \texttt{volatile}, ao passo que o Clang/LLVM as emite para o
+mesmo \textit{target} --- contraste de que a Seção~\ref{sec:multi} se vale.
```

*Motivo:* a frase "decisão de geração de código, não propriedade da
arquitetura" aparecia **três vezes** (§4, §5.1, Conclusão). Ficou nas duas em
que é conclusão, não premissa.

---

## 3. Conclusão — 767 → ~600 palavras

Estava em 1,4 página contra ~0,7 p nos TCCs de referência. Dos 14 números
citados, 11 já constavam dos Resultados.

Alterações, mantendo as quatro conclusões e todos os achados:

- **1ª conclusão:** os *encodings* `DC510000`/`DC500000` viraram `GLC=1`/`GLC=0`
  — a Tabela 2 já dá o *encoding*. Saiu a frase sobre `address_space(1)`, que
  §5.1 demonstra com ISA.
- **2ª:** "refutada por duas vias" ficou; saiu a repetição dos percentuais
  `1,9% → 11,8%`, já dados duas vezes antes.
- **3ª:** inalterada em substância — é onde entra o resultado de localidade.
- **4ª:** absorveu o parágrafo solto da referência em CPU, que virou uma frase.
- **Limitações:** saiu "e de uma única carga" (o Exp. 5 usa três padrões de
  acesso — a afirmação tinha ficado **falsa** com a adição do experimento) e a
  aposição que reexplicava o desenho fatorial.

```diff
-Como limitações, os resultados provêm de uma única geração de \textit{hardware} e
-de uma única carga; apenas leituras foram investigadas, ficando em aberto o
+Como limitações, os resultados provêm de uma única geração de \textit{hardware};
+apenas leituras foram investigadas, ficando em aberto o
```

---

## 4. Referencial Teórico — preview do resultado e detalhe órfão

### §2.2, 3º parágrafo

Antecipava, em termos vagos ("*pode resultar em diferenças*"), o que §5.1
demonstra com o binário na mão:

```diff
-A possibilidade de controlar esses bits a partir de código OpenCL depende da
-arquitetura, do compilador e da plataforma de execução. Mecanismos que teoricamente
-permitiriam esse controle --- o qualificador \texttt{volatile}, o \textit{inline
-assembly} GCN (\texttt{\_\_asm\_\_}) e a edição direta do binário ISA ---
-apresentam limitações que variam conforme a geração da GPU e o compilador
-utilizado. Em particular, o compilador proprietário do \textit{driver} AMD (AMDOCL,
-presente no Windows e no AMDGPU-PRO do Linux) difere do compilador baseado em
-Clang/LLVM utilizado pelo ROCm, o que pode resultar em diferenças no suporte a
-esses mecanismos, conforme investigado neste trabalho.
+Três mecanismos poderiam, em tese, controlar esses bits a partir de código OpenCL
+--- o qualificador \texttt{volatile}, o \textit{inline assembly} GCN
+(\texttt{\_\_asm\_\_}) e a edição direta do binário ISA ---, e o que cada um de
+fato produz no GFX8 é objeto da Seção~\ref{sec:multi}.
```

### Detalhe órfão

```diff
-bit~14 e SLC o bit~17~\cite{amdgcn3}. Em gerações posteriores (GFX9+) as
-\textit{flags} foram renomeadas para \texttt{sc0}/\texttt{sc1}. A nomenclatura
+bit~14 e SLC o bit~17~\cite{amdgcn3}. A nomenclatura
```

`sc0`/`sc1` aparecia **uma única vez** no artigo — resto da validação cruzada em
GFX9+ que saiu do texto numa revisão anterior.

### §2.1 — comparação CUDA × OpenCL

```diff
-Entre os \textit{frameworks} de programação, o CUDA é proprietário da NVIDIA e o
-OpenCL é um padrão aberto e portável entre fabricantes~\cite{memeti2017}, embora o
-desempenho ótimo costume exigir otimizações específicas de
-plataforma~\cite{seo2011, vidye2023}.
+O desempenho ótimo, porém, costuma exigir otimizações específicas de
+plataforma~\cite{memeti2017, seo2011, vidye2023}.
```

"OpenCL é padrão aberto e portável entre fabricantes" já está na Introdução, e
§3.1 é uma subseção inteira sobre comparação de *frameworks*. As três citações
foram preservadas.

### Fecho dos Trabalhos Relacionados

Removida a última frase, que repetia os objetivos específicos da Introdução:

```diff
-atendido e o que esse controle custa. Para isso combina análise de ISA entre
-\textit{targets} e compiladores, controle explícito das \textit{flags} por
-\textit{inline assembly} e \textit{intrinsics}, e um desenho que separa o efeito do
-\textit{cache} dos custos de geração de código que a comparação ingênua confunde
-com ele.
+atendido e o que esse controle custa.
```

E duas frases de fecho sem conteúdo:

```diff
-por técnicas probabilísticas. Embora validado em GPUs NVIDIA, seus princípios são
-aplicáveis a outras arquiteturas.
+por técnicas probabilísticas.
```
```diff
-desempenho de aplicações. Embora não aborde GPUs diretamente, o estudo reforça a
-importância de considerar o ambiente de execução completo.
+desempenho de aplicações.
```

---

## 5. Filler de abertura de seção

Quatro frases que anunciavam o que a seção já mostra — o padrão que os TCCs de
referência usam para encher espaço:

```diff
-Nesta seção são descritos os principais conceitos relacionados à computação
-paralela em GPU, ao \textit{framework} OpenCL e à hierarquia de memória em GPUs AMD.
```
```diff
-Diversos estudos investigaram o desempenho de \textit{frameworks} de programação
-paralela em GPUs e o impacto da hierarquia de memória no desempenho de algoritmos.
-Esta seção analisa os principais trabalhos que fundamentam a presente pesquisa,
-organizados por eixo temático.
```
```diff
-Esta seção apresenta os resultados dos cinco experimentos.
```
```diff
-A Tabela~\ref{tab:trabalhos} sintetiza as características dos trabalhos
-relacionados e posiciona a contribuição deste estudo.
```

A última era dispensável porque o parágrafo seguinte já abre com *"Conforme a
Tabela 1, a literatura concentra-se em GPUs NVIDIA sob Linux"*.

---

## 6. Objetivos específicos — 5 → 4, e agora cobrem o Experimento 5

Os objetivos 2 e 3 diziam a mesma coisa ("analisar a ISA" / "achar o mecanismo
que funciona") e foram fundidos. O último foi estendido porque **nenhum
objetivo mencionava padrão de acesso** — lacuna aberta quando o Experimento 5
entrou no trabalho.

```diff
-\item Analisar o binário ISA gerado pelo compilador OpenCL da AMD, verificando
-como qualificadores de memória são traduzidos em instruções no GFX8.
-\item Investigar os mecanismos disponíveis para controle direto das \textit{flags}
-de coerência e determinar quais produzem a instrução pretendida de forma previsível.
+\item Analisar o binário ISA gerado pelo compilador OpenCL da AMD e determinar
+quais mecanismos de controle das \textit{flags} de coerência produzem a instrução
+pretendida de forma previsível no GFX8.
 \item Isolar, por desenho controlado, o custo de contornar o L1, separando-o dos
 fatores que a comparação ingênua confunde com ele.
-\item Verificar como esse custo varia com o tamanho do problema, ao longo de várias
-ordens de grandeza em volume de dados.
+\item Verificar como esse custo varia com o tamanho do problema e com a
+localidade do padrão de acesso.
```

> A banca confere objetivos contra entrega. Sem isso, o Experimento 5 — o
> resultado mais forte do trabalho — não tinha objetivo que o sustentasse.

---

## 7. Compressões pontuais

### §5.1 — mecanismo já demonstrado na ISA

```diff
-porque dois outros fatores variam junto com a \textit{flag}: o \textit{codegen} do
-\texttt{volatile}, que insere \texttt{s\_waitcnt vmcnt(0)} após cada \textit{load}
-e elimina a sobreposição de latência entre acessos independentes; e o próprio
+porque dois outros fatores variam junto com a \textit{flag}: o \textit{codegen} do
+\texttt{volatile}, que serializa os \textit{loads} (Seção~\ref{sec:multi}); e o próprio
```

### §5.4 — degradação da CPU

É referência de escala, não objeto de estudo. Mesma explicação, mais curta; e os
dois parágrafos sobre qualidade da bancada viraram um. Todos os números
preservados (`4%`, `11,8×`, `35×`, `400×`, `15,4%`, `7,223 ms`, `6,99 ms`,
`3,3%`, `0,14%`).

### §5.5 — hipótese não testada

```diff
-2,111$\times$. O $cv$ de 0,24\% afasta ruído de medição; uma explicação plausível é
-que, com 1~GB de matriz contra 2~MB de L2, a pressão já seja tão alta que marcar
-linhas como candidatas preferenciais a evicção deixe de alterar o que seria
-evictado de qualquer modo. A hipótese não foi testada e fica registrada como tal.
+2,111$\times$. O $cv$ de 0,24\% afasta ruído de medição; uma explicação plausível ---
+não testada, e registrada como tal --- é que, com 1~GB de matriz contra 2~MB de L2,
+a pressão já seja tão alta que marcar linhas como candidatas preferenciais deixe de
+alterar o que seria evictado de qualquer modo.
```

A ressalva de que a hipótese não foi testada continua lá — só saiu da frase
própria.

### Outros

- **Introdução:** `influencia de forma mensurável` → `afeta de forma mensurável`;
  `o que justifica investigações que considerem a plataforma` → `o que justifica
  considerar a plataforma`.
- **§5.1:** `O formato MUBUF também foi verificado e é viável` → `O formato MUBUF
  é viável` (a verificação está implícita).
- **§5.2:** removido `Uma medida de ponto único não caracteriza, portanto, o
  custo de contornar o L1` — §5.4 demonstra isso com a varredura de cinco
  tamanhos; aqui era asserção sem dado.
- **Declaração de IA:** reescrita mais curta, com o mesmo conteúdo legal
  (Arts. 6º–8º e 9º preservados).

---

## 8. Correções — dois erros pré-existentes

Encontrados durante os cortes. **Ambos introduzidos quando o Experimento 5 foi
adicionado ao trabalho.**

### 8.1 Abstract e resumo diziam "quatro experimentos"

O corpo tem cinco (`§4` diz *"A metodologia compreende cinco experimentos"*; a
Conclusão diz *"Os cinco experimentos permitem quatro conclusões"*).

```diff
-Ellesmere GPU (Polaris/GCN, GFX8) under OpenCL. Four experiments progress from
+Ellesmere GPU (Polaris/GCN, GFX8) under OpenCL. Five experiments progress from
```
```diff
-GPU AMD Ellesmere (Polaris/GCN, GFX8) sob OpenCL. Quatro experimentos vão da
+GPU AMD Ellesmere (Polaris/GCN, GFX8) sob OpenCL. Cinco experimentos vão da
```

### 8.2 O resultado de localidade não estava no abstract

O Experimento 5 é a 3ª conclusão do trabalho e não aparecia no resumo. Foi
acrescentado nos dois idiomas:

```diff
-and shown to vary with problem size, falling from 1.20$\times$ to 1.06$\times$
-over a 4,096-fold range in data volume.
+and varies both with problem size --- from 1.20$\times$ to 1.06$\times$ over a
+4,096-fold range in data volume --- and with locality, from 2.19$\times$ on a
+contiguous traversal to 1.01$\times$ on a column traversal, indistinguishable
+from zero.
```

Verificado por script: abstract e resumo têm **paridade numérica exata**
(`1,20` `1,06` `4096` `2,19` `1,01` `555,9` `222,3`) e os seis valores constam
do corpo.

### 8.3 Limitação que virou falsa

`"os resultados provêm de uma única geração de hardware e de uma única carga"` —
o Experimento 5 usa três padrões de acesso. Corrigido no item 3.

---

## 9. Bibliografia — prefixo de sigla redundante

Foi o que fechou a última página. `sbc-template.bib`:

```diff
-booktitle = {PPoPP '12: Proceedings of the 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
+booktitle = {Proceedings of the 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP)},
```
```diff
-booktitle = {ARMS-CC '17: Proceedings of the 2017 Workshop on Adaptive Resource Management and Scheduling for Cloud Computing},
+booktitle = {Proceedings of the 2017 Workshop on Adaptive Resource Management and Scheduling for Cloud Computing (ARMS-CC)},
```
```diff
-booktitle = {Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis},
+booktitle = {Proceedings of the SC '25 Workshops},
```

As **15 referências foram mantidas** — nenhuma citação sacrificada por espaço.

---

## O que foi preservado

Registro do que **não** se tocou, para que nenhum corte futuro vá no lugar errado:

- Experimento 5 inteiro, incluindo o padrão **dispersa** construído para separar
  passo de *broadcast*.
- Piso de ruído e controle negativo (§5.2).
- Ponto não monotônico do `slc` em `N=16384` e a ressalva de hipótese não testada.
- Tabela 1 (posicionamento frente à literatura) e as 15 referências.
- Figura 1 (`fig_sensibilidade`), única figura restante: mostra as três *formas*
  — plana, decrescente, crescente — que nenhuma tabela mostra.
- Todos os números, intervalos de confiança e coeficientes de variação.

## Backups

| arquivo | estado |
|---|---|
| `v0_original.tex` | 18 páginas, antes dos cortes |
| `pre_float.tex` | antes do teste de posicionamento de floats (revertido) |
| `sbc-template.bib` | bibliografia original |

Em `/tmp/claude-1000/-home-henrique--rea-de-Trabalho-tcc-gpu-opencl/b07e8eb8-83bc-44d3-bf00-4077ff58bb02/scratchpad/`.

---
---

# Reestruturação de 14/10/2026

Feita depois que o artigo voltou a 17 páginas com a entrada da multiplicação com
$B$ transposta. Desta vez o critério não foi só espaço: o artigo foi
**realinhado à pergunta central** e conferido item a item contra o regulamento.

## Decisões do aluno que orientaram tudo

| Pergunta | Decisão |
|---|---|
| Pergunta central | **Impacto da hierarquia de memória.** As flags são o instrumento, não o objeto. |
| Resultados inteiros | Isolamento do L1, SLC, sensibilidade ao tamanho, localidade. |
| Antigo Experimento 1 (mecanismos) | Vira a **§4.1 Obtenção e Verificação do Instrumento**; os demais são renumerados. |
| §3.3 (Vidye, Balen) | **Fica**: é o único confronto direto com a literatura. |
| Justificativa de SO na Introdução (Chen, Gupta) | **Sai**: o trabalho não estuda SO. |
| Diagrama na Metodologia | **Entra** (Art. 4º, §1, III). |
| Declaração de IA | Fica como está até consulta ao orientador. |

## O que o regulamento exigia e faltava

| Exigência | Onde estava | O que foi feito |
|---|---|---|
| Introdução com síntese do que foi feito e dos principais resultados (TCC II, 4º PC) | ausente | parágrafo "Em síntese" antes do roteiro |
| Metodologia com diagramas/figuras (Art. 4º, §1, III) | nenhuma figura | `fig_hierarquia.tex` (TikZ): caminhos da leitura e o experimento que mede cada um |
| Seção "Conclusões e Trabalhos Futuros" (Art. 4º, §1, V) | "Conclusão" | renomeada |
| Resultados frente à literatura (Art. 5º, §3) | zero citações na Conclusão | parágrafo com Wong, Mei e Chu, Jia, Seo, Fraire, Vidye, Balen, Fang |

## Mudanças de estrutura

1. **Exp. 1 → §4.1.** A tabela de mecanismos saiu. O que sustenta os resultados
   ficou: encodings verificados na ISA, as duas razões que desqualificam o
   `volatile` (depende do compilador; serializa os loads), `address_space(1)`,
   escolha de FLAT sobre MUBUF, `__builtin_nontemporal_load` e a ausência de
   `glc` nas escritas. Saíram a menção ao Mesa Clover e a contagem das 50 formas
   de *store*.
2. **Renumeração:** Isolamento do L1 = Exp. 1; SLC = Exp. 2; tamanho = Exp. 3;
   localidade = Exp. 4.
3. **Objetivos específicos** reescritos na ordem dos experimentos: obter e
   verificar o instrumento; isolar o custo; testar quais limites são
   selecionáveis; variação com tamanho e localidade.
4. **Conclusão:** de quatro conclusões para três, reordenadas pela pergunta
   central. O valor do L1 vem primeiro; a dependência do compilador foi
   absorvida pela conclusão metodológica.
5. **Resumo e abstract** reescritos no novo eixo, com paridade numérica conferida
   por script.
6. **13 referências**: chen1993 e gupta1991 deixaram de ser citadas.

## Correções factuais

| Antes | Por que estava errado | Agora |
|---|---|---|
| Nota de rodapé: "o `amdgpu` reinicia a GPU quando um trabalho de compute ultrapassa 60 s" | Um lançamento único de 218 s (Exp. 4, N=16384) terminou sem reset | Lotes "adotados por precaução", sem afirmar o reset |
| Exp. 4 (localidade): "os lançamentos são fatiados em faixas de 1.024 linhas" | O programa **nunca teve** fatiamento; o argumento passado era ignorado | "Cada medição é um lançamento único" |
| §5.2: "Isso explica o 1,015× do Experimento 2" | Sobra da campanha perdida; o valor arquivado é 1,019× | 1,019× |
| §2.2 prometia examinar "a edição direta do binário ISA" | O artigo não trata disso (tentativa do percurso em Windows) | só `volatile` e *inline assembly* |
| Introdução: LDS "centenas de vezes mais rápido" que a VRAM | Sem citação e exagerado | formulação qualitativa |
| Parágrafo da CPU dentro do Exp. de um único tamanho, falando em "os mesmos cinco tamanhos" | Fora de lugar | movido para o Exp. 3 |
| Tabela da $B$ transposta sem referência no texto | Ficou órfã ao compactar o parágrafo | referenciada |

## Fechamento (mesmo dia)

**Resumo e abstract violavam o modelo SBC.** O modelo oficial (link do Art. 3º)
diz: *"abstracts should not have more than 10 lines and must be in the first page
of the paper"*. Tinham 18 e 17 linhas. Foram reescritos com **10 linhas cada**,
ambos na página 1, mantendo pergunta central, instrumento e os quatro resultados
com paridade numérica (1,20; 1,06; 4.096; 2,19; 1,01; 2,3–2,5; 555,9; 222,3).
Foi isso que fechou as 16 páginas, com ~12 linhas de folga.

**Repetição criada pela reestruturação:** o contraste AMDOCL × Clang aparecia na
abertura da Metodologia e na §4.1. A abertura passou a uma frase com remissão à
§4.1.

**Anomalia do `slc` (Exp. 4, `dispersa`) remedida** em N=8192 e 16384, mesmas
condições (`resultados/linha_coluna_remedicao.csv`):

| | original | remedição |
|---|---|---|
| `glc`, N=8192 / 16384 | 1,626× / 1,498× | 1,619× / 1,498× |
| `glc slc`, N=8192 / 16384 | 3,376× / 2,111× | 3,016× / 2,500× |

O `glc` reproduz a menos de 0,5%; o `glc slc` não (−11% e +18%, contra
dispersão interna de até 7,4%). A queda persiste em sentido, não em magnitude. O
parágrafo foi reescrito com isso, e "único ponto não monotônico" virou "única
queda expressiva" (a tabela tem inversões menores em N=2048).

## Backups

`main_pre_layout.tex` (antes da $B$ transposta) e `main_pre_reestruturacao.tex`
(antes desta reestruturação), no scratchpad da sessão.

# Revisão de 19/10/2026 — Jia, Shaw e Martonosi e geometria da *wavefront*

Origem: avaliação da pergunta central com a skill `idea-evaluator`. A busca de
literatura achou o trabalho mais próximo da pergunta do TCC, que não estava
citado: Jia, Shaw e Martonosi, ICS 2012 (DOI 10.1145/2304576.2304582), lido na
íntegra antes de citar.

## Correção factual: geometria da *wavefront* no Exp. 4

Os três programas da multiplicação usam `ls = {16, 16}`. A *wavefront* reúne 64
*work-items* consecutivos do id local achatado (x = `col` mais rápido), logo
cobre **4 valores de `row` e 16 de `col`**, e não 64 de `col`. Verificado na
RX 580 com `experimentos/sonda_wavefront.cpp` (HW_ID e *lane* por *work-item*,
16/16 *work-groups*).

| Padrão | Texto anterior | Correto (linhas de 64 B por instrução) |
|---|---|---|
| `linha` | 64 *lanes* no mesmo endereço | 4 endereços, 16 *lanes* cada |
| `coluna` | — | 1 linha (16 endereços, 4 *lanes* cada) |
| `dispersa` | 64 linhas vivas | 19 linhas, até 4 *lanes* por endereço |

Os dados favorecem a geometria correta: vazão `linha`/`dispersa` de 4,17 a
4,68× (`linha_coluna.csv`), contra 19/4 = 4,75. A conclusão "reúso dentro da
linha, não o *broadcast*" se mantém: `dispersa` *reduz* (não elimina) o
compartilhamento e depende mais do L1. Trechos corrigidos: itens do Exp. 4 na
Metodologia; "O fator determinante…"; "A penalidade do acesso disperso…" (19
linhas em vez de 4, no lugar de 64 transações em vez de uma); "…que mantém 19
linhas vivas"; Conclusões ("reduzi-lo", no lugar de "removê-lo"). O
*work-group* 16×16, que o artigo não informava, entrou no Exp. 1. Comentários de
`linha_coluna.cl` e `matmul_layout.cl` corrigidos.

Efeito colateral: "o L1 nunca tem um acerto para oferecer" (padrão `coluna`)
deixou de ser exato, porque as 4 *wavefronts* do grupo leem a mesma linha no
mesmo `k`. Passou a "sem reúso dentro do *work-item*".

## Inclusão de Jia, Shaw e Martonosi (2012)

§3.2 (parágrafo), Tabela 1 (SO "---": o artigo não informa), Introdução (lista de
trabalhos em NVIDIA), §3.4 (posicionamento: a pergunta já foi feita no Fermi; a
diferença está no controle, oferecido pelo CUDA e aqui construído e verificado
na ISA) e Conclusões (concorda: o L1 não aproveita o compartilhamento entre
*lanes*; diverge: lá o L1 não aproveitou o reúso entre instruções, aqui sua
principal fonte de valor; lá chegou a retardar programas, aqui contornar o L1
nunca acelerou — conferido nos CSVs, menor custo 1,006×, `coluna` N=4096).

## Compressões para voltar a 16 páginas (sem corte de conteúdo)

As inclusões levaram a 17 páginas (+7 linhas). Recuperado encurtando parágrafos
cuja última linha tinha uma ou duas palavras, sem retirar informação:
relevância ("medi-lo…, ausente da literatura…"); objetivos ("sob OpenCL",
"definem-se os objetivos específicos"); organização do texto ("O texto
organiza-se assim", "a Seção 5, os resultados"); Exp. 1 ("contra a CPU");
"diferem em dois fatores" (sem "ao mesmo tempo"), "troca só o segundo"; deriva
de *clock* ("explicação das diferenças"); piso de ruído (sem "critério" e "como
reais"); Exp. 3 ("ininterpretável (Seção 5.1)"); padrão `coluna` (sem "A razão
é direta").

## Escopo do SLC (falha 2 da avaliação)

O Exp. 2 usou só `flat_load_dword` sobre *buffers* comuns na VRAM
(`slc_l2.cl`, `slc_l2.cpp:57`). O manual citado (GCN3 rev. 1.1) descreve o SLC
de duas formas: Tabela 8.2 (MUBUF/MTBUF), *"accesses are forced to miss in level
2 texture cache and are coherent with system memory"*; Tabela 9.1 (FLAT),
*"used in conjunction with GLC and MTYPE to determine cache policies"*. O
resultado confirma a descrição do FLAT; a do MUBUF não foi testada.

Mudanças: §2.2 cita as duas descrições (`\cite[Tabelas~8.2 e~9.1]{amdgcn3}`) no
lugar de "é comum encontrar a leitura"; "em leituras FLAT" no resumo, abstract
(paridade mantida, 10 linhas cada), síntese da Introdução, abertura da
Metodologia, Exp. 2 (método: `flat_load_dword` sobre *buffers* na VRAM;
resultado: "o caminho direto à VRAM que a descrição do manual para o MUBUF sugere
não existe por esse meio") e segunda conclusão ("embora o manual descreva o SLC,
nos formatos MUBUF e MTBUF, como forçando a falta no L2", no lugar de "contra a
leitura difundida"). Limitações: MUBUF e outros MTYPE em aberto.

## Auditoria dos números e da proveniência dos dados

Antes de corrigir, verificou-se que cada CSV é a última execução completa do seu
programa e que o código atual mede da mesma forma: `matmul.cpp` mudou em 09/09
só por acréscimo (medição de transferências, em CSV próprio), com *kernels*
idênticos e binário de ISA de 26704 B nas duas campanhas de agosto; o
`matmul` não rodou depois de 21/08; `linha_coluna.cpp` ganhou em 14/10 apenas os
argumentos de filtro e saída, e a remedição reproduz o `glc` a ≤ 0,5%. Os
problemas abaixo são do texto, não dos dados.

| Trecho | Antes | Depois | Fonte |
|---|---|---|---|
| Exp. 4, abertura dos resultados | cv "entre 0,02% e 4,5%" | 0,02–4,1% em `linha`/`coluna`, até 19,4% em `dispersa` | `linha_coluna.csv` (19,36%: `dispersa` N=1024 `glc slc`, 47 execuções) |
| Exp. 4, padrão `coluna` | "cv de 0,08% — efeito que não se distingue de zero"; "removê-lo não custa nada" | "cv entre 0,06% e 2,5% — efeito pequeno, mas mensurável"; "quase não custa"; título "quase indiferente" | Com cv ≤ 0,34% a partir de N=2048, 0,6–1,4% é distinguível de zero |
| Conclusões | "1,01× … indistinguível de zero" | "praticamente nulo" | idem |
| Exp. 3 | 1,20× em N=1024 sem menção ao 1,237× do Exp. 1 | Frase: campanhas independentes deram 1,237× (Exp. 1) e 1,19× (Seção 5.4); a queda com o tamanho excede a variação | `matmul_cache_flags.csv`, `matmul_sweep.csv`, `matmul_layout.csv` |
| Exp. 4, Metodologia | sem número de execuções | `linha_coluna`: 1 aquecimento + orçamento de 4 s (5 a 200); `matmul_layout`: 6 s (5 a 100) | `COMO_RODAR.md`, fonte, CSVs |

`COMO_RODAR.md`: corrigida a nota "aquecimento não registrado" (são 5, e 3 em
N=16384, nos logs) e acrescentado o aviso de que os programas gravam nos CSVs das
campanhas.

Compressões para voltar a 16 páginas (+9 linhas), sem perda de informação:
legendas das Tabelas 2, 4 e 5; "memória nos chips de GDDR5 não entregaria";
"entre 0,6% e 1,4%" (sem "nos cinco tamanhos"); parágrafo de Jia, Shaw e
Martonosi; Bernal et al. (sem "por técnicas probabilísticas", redundante com
"estocástico"); Exp. 1 (sem "medidas", "exatamente", "precisamente").

Estado: 16 páginas, ~9 linhas de folga, 0 erros, 0 Overfull, 14 referências.
Sem backup em arquivo: o estado anterior é o atual menos as edições listadas.

# Revisão de 20/10/2026 — conferência das tabelas

Origem: explicação das Tabelas 2 a 7 ao aluno. Todos os valores foram
recalculados dos CSVs e conferem. Quatro problemas de texto corrigidos:

| Trecho | Antes | Depois | Fonte |
|---|---|---|---|
| Exp. 1, parágrafo da CPU | "247,3 ms contra 7,191 ms … 35 vezes" | "contra 7,223 ms … 34 vezes" | 7,191 era uma execução isolada (nº 114 de `matmul_cache_flags.csv`); a mediana é 7,223; 247,297/7,223 = 34,2. Os 35× do Exp. 3 vêm da Tabela 5 (6,99 ms, outra campanha) |
| Legenda da Tabela 2 | "1000 execuções cada" | "1000 execuções por cenário de GPU e 25 na CPU" | `matmul_cpu_sweep.csv` |
| Legenda da Tabela 5 | "sobre 100 execuções" | "100 execuções na GPU e de 3 a 25 na CPU" | CPU: 25, 21, 3, 3, 3 |
| Resumo, abstract, Exp. 3 (×3), Conclusões | "fator de 4.096 em volume de dados" / "4,096-fold range in data volume" | "256" / "256-fold" | De N=1024 a 16384 cada matriz vai de 4 MB a 1 GB (256×); 4.096× é o fator do trabalho (N³), não do volume de dados |

Novo `analise/gera_tabelas.py`: regenera `tab_cpu_ref`, `tab_medianas_artigo`,
`tab_loc_glc`, `tab_loc_glcslc` e `tab_layout`, cujos geradores não estavam no
projeto. Saída idêntica byte a byte aos arquivos existentes; `--conferir`
imprime os valores das Tabelas 2 e 3.

Estado: 16 páginas, ~6 linhas de folga, 0 erros, 0 Overfull, 14 referências;
resumo e abstract com 10 linhas cada e os mesmos números.

## Figura 2 refeita (mesmo dia)

Pedido do aluno: na versão anterior não se via o que era numerador e
denominador (três painéis com escalas próprias, rótulos "asm / normal", "base
asm" e nenhum no painel c). Nova versão (`analise/plot_sensibilidade.py`):

- um único eixo para as três razões (mesma unidade), curvas (a), (b) e (c);
- legenda com o nome de cada fator e, por extenso, numerador ÷ denominador e os
  cenários da Tabela 5 ([2 ÷ 1], [4 ÷ 2], [5 ÷ 4]); eixo y "tempo do numerador ÷
  tempo do denominador"; referência em 1,00× = sem diferença;
- rótulos só nas extremidades, calculados do CSV (antes eram digitados);
- grade sólida (antes tracejada); marcadores distintos para impressão sem cor;
  cores: slots 1–3 da paleta categórica de referência, sem alteração.

Texto: legenda da figura reescrita; "painel a/b/c" → "curva a/b/c" (5 trechos);
"cresce monotonicamente" → "fica entre 1,2% e 2,6% até N=4096, cresce a partir
daí" (a razão cai de 1,019 para 1,012 em N=2048; o antigo texto contradizia a
figura).

Estado: 16 páginas, ~9 linhas de folga, 0 erros, 0 Overfull.

## 2ª remedição da `dispersa` incorporada (mesmo dia)

`resultados/linha_coluna_remedicao2.csv` (20/10, 20:38–21:38, mesmas condições).
`glc slc`: 3,360× (N=8192) e 2,515× (N=16384); `glc`: 1,589× e 1,489×.

Texto do Exp. 4: "Uma remedição … (3,016× e 2,500×): a queda persiste em sentido,
não em magnitude, e a variação entre campanhas, de 11% a 18%…" → "Duas remedições
… reproduzem o custo do `glc` a menos de 2,5%, mas não o do `glc slc` (3,016× e
3,360× em N=8192; 2,500× e 2,515× em N=16384): a queda persiste nas três
campanhas, de 17% a 37%, e a amplitude entre elas, de até 12% em N=8192 e 19% em
N=16384, supera a dispersão interna, de até 7,4%. As medianas das três campanhas
são 3,360× e 2,500×." A Tabela 6 mantém a campanha original (regra 2).

Estado: 16 páginas, ~6 linhas de folga, 0 erros, 0 Overfull.

## Nota de remedição na Tabela 6 (mesmo dia)

Decisão do aluno (opção A entre "nota" e "trocar pelos valores típicos"): a
Tabela 6 continua com a campanha original (regra 2), e as duas células
remedidas da `dispersa` (`glc slc`, N=8192 e 16384) recebem "†". Nota gerada por
`analise/gera_tabelas.py` em `tab_loc_nota.tex`: "Remedido em mais duas
campanhas; mediana das três: 3,360 (N=8192) e 2,500 (N=16384)." O gerador lê
`linha_coluna_remedicao*.csv`; as demais tabelas saem idênticas.

Estado: 16 páginas, ~3 linhas de folga, 0 erros, 0 Overfull.

**Revertido no mesmo dia, a pedido do aluno:** a nota e as marcas "†" foram
retiradas (o pedido de remedição partira da leitura do 2,111 como tempo, e não
como razão). `tab_loc_nota.tex` apagado; gerador e `tab_loc_glcslc.tex` de volta
ao original (idênticos byte a byte). O parágrafo do Exp. 4 com as três campanhas
permanece.


# Revisão de 04/11/2026 — o contrafactual do Exp. 2 (pedido do orientador)

Origem: o orientador questionou se se pode afirmar que, com o L2 contornado, a
vazão em 1 MB "cairia a 222 GB/s" em vez dos ~550 medidos. Para responder,
`experimentos/teto_vram.cpp` (+ `teto_vram.cl`; `slc_l2.cl` intocado) →
`resultados/teto_vram.csv` / `.txt`, e o binário original reexecutado 3× →
`resultados/slc_l2_reexecucao.txt`. Detalhes de execução em `COMO_RODAR.md`.

**Resposta:** sim, com duas grandezas distintas. O limite físico é o pico teórico
da GDDR5 (256 GB/s: `pp_dpm_mclk` 2000 MHz × 4 × 256 bits / 8), que nenhuma
leitura da VRAM supera; o prático é o patamar medido fora do L2 (222,3–223,9 GB/s
de 8 MB a 2 GB, sem flag e `glc`), que nove variantes com 4 a 16× mais bytes em
voo não superam (melhor execução 225,3). "Teto de 222,3" chamava de limite uma
medida; o texto passa a usar 256 como limite e ~222 como valor esperado.

## Mudanças no texto

| Trecho | Antes | Depois |
|---|---|---|
| Resumo / abstract | "contra o teto de 222,3 GB/s da VRAM" | "acima do pico teórico de 256 GB/s da VRAM" (10 linhas cada, mantidas) |
| §4.3 (Metodologia) | dois regimes, 1 e 64 MB | varredura de 14 tamanhos, aquecimento de 0,2 s, nove variantes de vazão, dois controles |
| §5.2 (Resultados) | Tabela 4 com 1 e 64 MB (campanha original) | Tabela 4 = varredura (`tab_slc.tex`, gerada); três parágrafos: o que um *bypass* produziria (256 / ~222), o que se mede (539,8 e 555,9; 17 medições, 431,5–564,1; controle do L2, ≤ 3,0%), e `glc slc` mais rápido que `glc` de 2 a 64 MB (até 1,51×; modelo h = 2 MB/W, erro ≤ 1,7%) |
| Conclusões, 2ª | "duas vias … contra o teto de 222,3" | 555,9 acima do pico de 256; a *flag* acelera até 1,51× logo acima do L2 e custa mais na multiplicação — política de *eviction* |

Números conferidos com `python3 analise/gera_tabelas.py --conferir`. Paridade
numérica de resumo e abstract: (1,20; 1,06; 256; 2,19; 1,01; 2,3–2,5; 555,9;
256 GB/s). A Tabela 4 da campanha original sai do artigo; seus valores de 1 MB
(555,9) continuam citados no texto, com a fonte `resultados/slc_l2.txt`.

## Cortes para manter 16 páginas

O texto novo custava ~19 linhas e a folga era de ~6. Nenhum dado ou conclusão
perdido; verificado que nenhuma conclusão perde evidência (regra 4):

- §5.2: sai "A diferença entre os regimes confirma ainda, por vazão, que o L2
  reside no *die*…" (corolário; o §2.2 já diz que o L2 está no *die*) e a frase
  final "o caminho 'direto à VRAM' que o manual descreve para o MUBUF não existe
  por esse meio", que as Conclusões repetem com a ressalva de MUBUF e MTBUF. A
  frase em negrito passa a "apenas o L1 é selecionável pelas flags", como na
  Introdução.
- Ajustes mínimos de redação em fins de parágrafo com 1–3 palavras (sem mudar o
  sentido): Introdução, "estudos que investiguem o comportamento de cache" →
  "estudos sobre cache"; §2.1, "sobre centenas" → "em centenas", "adequadas a
  algoritmos de paralelismo" → "adequadas ao paralelismo", "O desempenho ótimo,
  porém," → "Mas o desempenho ótimo", "específicas de plataforma" → "por
  plataforma"; §3.2, "Mais próximos" → "Mais perto", "dentro do warp" → "no
  warp"; §5.3, "o conjunto de trabalho cabe" → "o conjunto cabe", "há mais dados
  concorrendo do que capacidade disponível" → "os dados excedem a capacidade",
  sai ", portanto,"; Conclusões (literatura), "Confirmam também" → "Confirmam",
  "E qualificam … entre plataformas" → "Qualificam … de plataforma", "em linha
  com o que Fang et al. observaram" → "como Fang et al. observaram".

Estado: 16 páginas, ~1 linha de folga, 0 erros, 0 Overfull, 14 referências;
resumo e abstract com 10 linhas cada. Backup do `main.tex` anterior no
scratchpad da sessão (`backup_pre_exp2/`).

## Tabela 4 no formato original, estendida (mesmo dia, pedido do aluno)

O aluno esperava a Tabela 4 original (tempo e vazão por variante) com mais
linhas, e não a versão transposta em inteiros, feita só para caber em 16
páginas; e pediu para não se preocupar, por ora, com o limite.

- **Tabela 4:** as colunas da original (tempo e vazão das três variantes) mais a
  razão de vazão `glc slc`/`glc`, em dois blocos: "Campanha original" (1 e
  64 MB, de `slc_l2.txt`, valores idênticos aos de antes, 555,9 em negrito) e
  "Varredura (segunda campanha)" (14 tamanhos, de `teto_vram.csv`). Gerada por
  `gera_tabelas.py`: `tab_slc_original.tex` e `tab_slc.tex`.
- **Exp. 2 na versão completa:** a metodologia explica por que há variantes e
  controles; os resultados têm quatro parágrafos (o que um *bypass* produziria; o
  que se mede; a evidência de 2 a 64 MB, com o modelo e sua aplicação à campanha
  original, 226,4 previsto contra 226,3 medido; a conclusão). Voltam as duas frases
  do aluno que tinham saído: a do MUBUF, idêntica, e a do L2 no *die*, com "mais
  que o dobro do pico da própria GDDR5" no lugar de "2,5× a banda da GDDR5" (2,5×
  vale contra o patamar medido, não contra o pico).
- **Conclusões:** "três vias" por extenso.
- **Revertidos os 10 ajustes de redação** em texto do aluno (Introdução, §2.1,
  §3.2, §5.3, Conclusões/literatura). O diff contra o backup confirma que só
  diferem resumo, abstract, Exp. 2 e o parágrafo da 2ª conclusão. Os ajustes
  continuam listados na seção anterior como candidatos se for preciso voltar a 16
  páginas (rendiam ~5 linhas).

Estado: **18 páginas** (acima do limite de 16 do Art. 4º; enquadramento adiado
por decisão do aluno), 0 erros, 0 Overfull; resumo e abstract com 10 linhas.

## Travessões no papel de vírgula (mesmo dia, pedido do aluno)

Em pt-BR o travessão usado como vírgula é incomum e deixa o texto com aspecto de
rascunho. Trocados 63 dos 70 travessões do `main.tex`, mais o rótulo da Figura 1:

- **vírgula** onde isolavam aposto ou oração explicativa (maioria dos casos);
- **parênteses** onde o trecho isolado era uma enumeração com vírgulas internas
  (com vírgulas a frase ficaria ambígua: "As três variantes, sem flags, glc e glc
  slc, …"): Referencial (quatro regiões), Trabalhos Relacionados (geometria),
  Metodologia (três caminhos, escritas, três variantes, nove variantes, CPU),
  Exp. 2 (17 medições), Exp. 3 (vazão da GPU), Exp. 4 (45 pontos, GFLOPS);
- **dois-pontos ou ponto e vírgula** onde introduziam explicação ou oração
  independente (duas camadas, perguntas distintas, "é pequeno neste tamanho",
  "descontá-los com segurança", "capacidade disponível", entre outros);
- **rótulos de tabela** com dois-pontos ("GPU: cenários de caminho de cache",
  "glc: desvio do L1", "glc slc: L1 e política do L2") e Figura 1 com vírgula
  ("Não, evicção no L2").

Ajustes mínimos para a frase seguir correta: "pois o volatile…", "a de que o acesso
à memória domina", "objeto deste estudo", "o piso é necessário porque",
"diferença de 3,3%", "cuja redução aumenta", "As duas últimas são…", o aposto
"otimização clássica em CPU" passa para junto de "transposta no anfitrião", e sai
a vírgula serial antes de "e memória global".

Mantidos: as células "não se aplica" das Tabelas 1, 2 e 5 (não são pontuação) e
os dois travessões da Declaração de IA (redação do orientador; `CLAUDE.md` manda
não alterar). Backup anterior no scratchpad da sessão (`backup_pre_travessoes/`).

Estado: 18 páginas, 0 erros, 0 Overfull; resumo e abstract com 10 linhas.

## Declaração de IA encurtada (mesmo dia, pedido do aluno)

O aluno achou a declaração extensa e pediu para não citar os artigos do
regulamento. Passou de 8 para 3 linhas:

> Utilizou-se o Claude Code (Anthropic), com os modelos Claude Opus 5 e Claude
> Opus 5.5, como apoio ao planejamento, à implementação, à execução e à análise
> dos experimentos, bem como à redação e à revisão de todas as seções do texto.

Saíram as remissões aos Arts. 6º a 9º e a frase de autoria ("A concepção, o
desenho experimental, a condução das medições ... são de autoria do aluno"); o
Art. 9º vale com ou sem a frase. Ficou o que os Arts. 7º e 8º exigem e a versão
anterior não trazia por inteiro: versão do modelo (era só "modelo Opus") e
seções (não constavam). A finalidade passou a cobrir o que de fato se fez com a
ferramenta nas sessões registradas, inclusive o planejamento, a execução e a
redação do Exp. 2 estendido. As versões saem dos registros das sessões do Claude
Code nesta máquina (`claude-opus-5` e `claude-opus-5-5`); uso em outro ambiente
não aparece neles. Com a troca, saem também os dois últimos travessões do texto.

Estado: 18 páginas, 0 erros, 0 Overfull.

Em seguida, por decisão do aluno, sai "à análise" da finalidade, como na versão
anterior, que não a declarava. Texto final:

> Utilizou-se o Claude Code (Anthropic), com os modelos Claude Opus 5 e Claude
> Opus 5.5, como apoio ao planejamento, à implementação e à execução dos
> experimentos, bem como à redação e à revisão de todas as seções do texto.

Estado: 18 páginas, 0 erros, 0 Overfull.
