# -*- coding: utf-8 -*-
"""Figura da analise de sensibilidade ao tamanho do problema (Experimento 5).

Le resultados/matmul_sweep.csv e gera Documentacao/fig_sensibilidade.pdf, com
tres paineis que separam os fatores por comportamento: custos de construcao
(invariantes), efeito de glc (decrescente) e efeito de slc (crescente).
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "matmul_sweep.csv")
SAIDA = os.path.join(RAIZ, "Documentação", "fig_sensibilidade.pdf")

# Paleta categorica validada (slots 1-3), segura para CVD em todos os pares.
AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d8d4"

# ---------------------------------------------------------------- dados
med = {}
with open(CSV) as f:
    for lin in csv.DictReader(f):
        med.setdefault(int(lin["N"]), {})[lin["kernel"]] = float(lin["mediana_ms"])
Ns = sorted(med)


def razao(num, den):
    return [med[n][num] / med[n][den] for n in Ns]


series = {
    "asm":        razao("matmul_asm_baseline",    "matmul_via_L1"),
    "intrin":     razao("matmul_intrin_baseline", "matmul_via_L1"),
    "glc_intrin": razao("matmul_intrin_glc",      "matmul_intrin_baseline"),
    "glc_asm":    razao("matmul_bypass_L1",       "matmul_asm_baseline"),
    "slc":        razao("matmul_bypass_L1_L2",    "matmul_bypass_L1"),
}
for k, v in series.items():
    print("%-11s %s" % (k, "  ".join("%.3f" % x for x in v)))

# ---------------------------------------------------------------- estilo
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": TINTA2,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

virg = FuncFormatter(lambda v, _: ("%.2f" % v).replace(".", ","))
fig, eixos = plt.subplots(1, 3, figsize=(6.3, 2.15))

paineis = [
    ("(a) Custos de construção: invariantes",
     [("asm", LARANJA, "o", "-", "asm / normal", 0.10),
      ("intrin", AZUL, "s", "-", "intrinsic / normal", 0.07)],
     (0.70, 1.68)),
    ("(b) Efeito de \\texttt{glc}: decresce",
     [("glc_intrin", AZUL, "s", "-", "base intrinsic", 0.13),
      ("glc_asm", LARANJA, "o", "-", "base asm", 0.13)],
     (0.95, 2.70)),
    ("(c) Efeito de \\texttt{slc}: cresce",
     [("slc", AGUA, "^", "-", None, 0.0)],
     (0.99, 1.16)),
]

for ax, (titulo, linhas, ylim) in zip(eixos, paineis):
    ax.axhline(1.0, color=GRADE, lw=0.8, ls="--", zorder=1)
    for chave, cor, marca, traco, rotulo, desloc in linhas:
        y = series[chave]
        ax.plot(Ns, y, color=cor, lw=1.5, ls=traco, marker=marca,
                ms=4.0, mew=0.8, mec="white", zorder=3, clip_on=False)
        # rotulo direto: identidade nunca depende so da cor
        if rotulo is None:
            continue
        ax.annotate(rotulo, xy=(Ns[2], y[2] + desloc), color=cor,
                    ha="center", fontsize=7, zorder=4)
    ax.set_title(titulo.replace("\\texttt{", "").replace("}", ""), pad=6,
                 color=TINTA)
    ax.set_xscale("log", basex=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"])
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(virg)
    ax.grid(axis="y", color=GRADE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA2, length=2.5, width=0.6)

# valores de extremo citados no texto
eixos[1].annotate("2,47", xy=(Ns[0], series["glc_intrin"][0]), xytext=(0, 6),
                  textcoords="offset points", ha="left", fontsize=7, color=TINTA)
eixos[1].annotate("1,65", xy=(Ns[-1], series["glc_intrin"][-1]), xytext=(0, 7),
                  textcoords="offset points", ha="right", fontsize=7, color=TINTA)
eixos[2].annotate("+1,9%", xy=(Ns[0], series["slc"][0]), xytext=(2, -10),
                  textcoords="offset points", ha="left", fontsize=7, color=TINTA)
eixos[2].annotate("+11,8%", xy=(Ns[-1], series["slc"][-1]), xytext=(0, 5),
                  textcoords="offset points", ha="right", fontsize=7, color=TINTA)

eixos[0].set_ylabel("razão entre medianas", color=TINTA2)
fig.text(0.5, 0.005, "$N$ (ordem da matriz)", ha="center", fontsize=8,
         color=TINTA2)
fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.20, wspace=0.32)
fig.savefig(SAIDA, format="pdf")
print("\nescrito: " + SAIDA)
