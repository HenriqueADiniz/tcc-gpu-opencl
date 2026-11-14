# -*- coding: utf-8 -*-
"""Figura do experimento de localidade (linha x coluna).

Le resultados/linha_coluna.csv e gera Documentacao/fig_linha_coluna.pdf.

Mostra o custo de contornar o L1 --- a razao entre o tempo com flag e o tempo
sem flag, medidos com a MESMA construcao em assembly --- para os dois padroes de
percurso. Um unico painel e uma unica escala: as quatro series se separam
sozinhas, e forcar escalas diferentes distorceria a comparacao, que e
justamente de magnitude entre os padroes.
"""
import csv, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "linha_coluna.csv")
SAIDA = os.path.join(RAIZ, "Documentação", "fig_linha_coluna.pdf")

# paleta categorica validada, slots na ordem fixa documentada
AZUL, LARANJA, AGUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d8d4"

d = collections.defaultdict(dict)
for r in csv.DictReader(open(CSV)):
    d[int(r["N"])][(r["padrao"], r["flags"])] = float(r["mediana_ms"])
Ns = sorted(d)

PADROES = [("linha",    "linha",    LARANJA, "o"),
           ("dispersa", "dispersa", AMARELO, "s"),
           ("coluna",   "coluna",   AZUL,    "^")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": TINTA2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
# Dois paineis, MESMA escala vertical: a comparacao entre eles e de magnitude
# --- quanto o slc acrescenta sobre o glc ---, entao escalas proprias
# esconderiam justamente o que interessa.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
for ax, fl, titulo in [(ax1, "glc", "(a) \\texttt{glc}: desvio do L1"),
                       (ax2, "glc slc", "(b) \\texttt{glc slc}: L1 e política do L2")]:
    for chave, rot, cor, marca in PADROES:
        y = [d[n][(chave, fl)] / d[n][(chave, "GLC=0")] for n in Ns]
        ax.plot(Ns, y, color=cor, lw=1.7, marker=marca, ms=4.2, mew=0.8,
                mec="white", zorder=3, label=rot, clip_on=False)
        # rotulo direto: identidade nunca depende so da cor
        ax.annotate(rot, xy=(Ns[-1], y[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.8, color=cor,
                    va="center", ha="left", annotation_clip=False)
    ax.axhline(1.0, color=TINTA2, lw=0.9, ls="--", zorder=1)
    ax.set_title(titulo.replace("\\texttt{", "").replace("}", ""), pad=6, color=TINTA)
    ax.set_xscale("log", basex=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"])
    ax.grid(axis="y", color=GRADE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA2, length=2.5, width=0.6)
ax1.set_ylim(0.95, 3.5)
ax1.set_ylabel("custo de contornar o L1\n(razão sobre GLC=0)", color=TINTA2)
ax1.annotate("sem efeito", xy=(Ns[0], 1.0), xytext=(2, 5),
             textcoords="offset points", fontsize=7, color=TINTA2)
fig.text(0.47, 0.012, "$N$ (ordem da matriz)", ha="center", fontsize=8, color=TINTA2)
fig.tight_layout(rect=[0, 0.05, 0.93, 1], pad=0.4)
fig.savefig(SAIDA)
fig.savefig(SAIDA.replace(".pdf", ".png"), dpi=170)
print("gravado:", SAIDA)
for fl in ("glc", "glc slc"):
    for chave, rot, _, _ in PADROES:
        print("  %-18s %s" % (rot + " " + fl,
              "  ".join("%.3f" % (d[n][(chave, fl)] / d[n][(chave, "GLC=0")]) for n in Ns)))
