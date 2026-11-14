# -*- coding: utf-8 -*-
"""Figura da Tabela de medianas por tamanho (Experimento 4).

Le resultados/matmul_sweep.csv e gera Documentacao/fig_medianas.pdf com
exatamente o conteudo da tabela: a mediana do tempo de execucao de cada cenario
em cada tamanho de matriz.

Forma: um painel com as medianas NORMALIZADAS pela do cenario 1 em cada tamanho.
A versao em tempo absoluto foi descartada: o eixo cobre quatro decadas (7 ms a
52 s) e a razao maxima entre cenarios e 1,9x, de modo que as cinco series se
sobrepunham e o grafico carregava menos informacao que a propria tabela.
Normalizar preserva a informacao --- os absolutos estao na tabela ao lado --- e
torna visivel a separacao entre cenarios, que e o objeto do experimento.

ATENCAO: "normal" aqui e o kernel escrito em OpenCL C puro, SEM inline assembly
e SEM intrinsics. E um kernel de GPU como todos os outros; nao tem relacao com
execucao em CPU. A CPU so aparece no programa como referencia de validacao, que
nao e cronometrada nem reportada em nenhuma tabela.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "matmul_sweep.csv")
SAIDA = os.path.join(RAIZ, "Documentação", "fig_medianas.pdf")

# Paleta categorica validada, slots 1-5 na ordem fixa documentada. Linhas usam
# a pairlist adjacente, que a paleta passa em modo claro.
AZUL, LARANJA, AGUA, AMARELO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d8d4"

med = {}
with open(CSV) as f:
    for lin in csv.DictReader(f):
        med.setdefault(int(lin["N"]), {})[lin["kernel"]] = float(lin["mediana_ms"])
Ns = sorted(med)

# (kernel, rotulo curto, cor, marcador) — mesma ordem da tabela
CEN = [
    ("matmul_via_L1",       "1. normal (OpenCL C)",   AZUL,    "o"),
    ("matmul_asm_baseline", "2. controle asm",      LARANJA, "s"),
    ("matmul_via_L2",       "3. volatile",          AGUA,    "^"),
    ("matmul_bypass_L1",    "4. asm glc",           AMARELO, "D"),
    ("matmul_bypass_L1_L2", "5. asm glc slc",       MAGENTA, "v"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": TINTA2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6.3, 3.0))

base = [med[n]["matmul_via_L1"] for n in Ns]

# O cenario 1 e a BASE da normalizacao: plotado, seria a constante 1,0 por
# construcao, gastando uma cor e um lugar na legenda sem dizer nada --- e ainda
# por cima sobreposto a linha de referencia. Entra como a propria linha
# tracejada, rotulada, e sai das series.
for chave, rotulo, cor, marca in CEN[1:]:
    y = [med[n][chave] / b0 for n, b0 in zip(Ns, base)]
    ax.plot(Ns, y, color=cor, lw=1.8, marker=marca, ms=4.6, mew=0.9,
            mec="white", zorder=3, label=rotulo, clip_on=False)
    # rotulo direto: identidade nunca depende so da cor
    ax.annotate(rotulo.split(". ")[1], xy=(Ns[-1], y[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=7, color=cor,
                va="center", ha="left", annotation_clip=False)

ax.axhline(1.0, color=TINTA2, lw=0.9, ls="--", zorder=1)
ax.annotate("cenário 1 (normal, OpenCL C) = referência", xy=(Ns[0], 1.0),
            xytext=(2, 5), textcoords="offset points", fontsize=7,
            color=TINTA2, ha="left")
ax.set_xscale("log", basex=2)
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel("$N$ (ordem da matriz)", color=TINTA2)
ax.set_ylabel("tempo relativo ao cenário 1", color=TINTA2)
ax.set_ylim(0.97, 2.0)
ax.grid(axis="y", color=GRADE, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.tick_params(colors=TINTA2, length=2.5, width=0.6)

# valores de extremo citados no texto
ax.annotate("1,92", xy=(Ns[0], med[Ns[0]]["matmul_via_L2"] / base[0]),
            xytext=(4, 4), textcoords="offset points", fontsize=7, color=TINTA)
ax.annotate("1,46", xy=(Ns[-1], med[Ns[-1]]["matmul_asm_baseline"] / base[-1]),
            xytext=(-2, -11), textcoords="offset points", fontsize=7,
            color=TINTA, ha="right")

leg = ax.legend(loc="upper right", frameon=False, fontsize=7,
                handlelength=1.8, borderaxespad=0.3, labelspacing=0.3,
                ncol=2, columnspacing=1.2)
for txt in leg.get_texts():
    txt.set_color(TINTA2)
fig.tight_layout(rect=[0, 0, 0.87, 1], pad=0.4)
fig.savefig(SAIDA)
fig.savefig(SAIDA.replace(".pdf", ".png"), dpi=170)
print("gravado:", SAIDA)
for chave, rotulo, _, _ in CEN:
    print("  %-20s %s" % (rotulo, "  ".join("%9.1f" % med[n][chave] for n in Ns)))
