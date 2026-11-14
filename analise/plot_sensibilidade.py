# -*- coding: utf-8 -*-
"""Figura 2 do artigo: sensibilidade ao tamanho do problema (Experimento 3).

Le resultados/matmul_sweep.csv e gera Documentacao/fig_sensibilidade.pdf.

Cada curva e a razao entre as medianas de DOIS cenarios da Tabela 5, no mesmo
N -- numerador / denominador, escritos por extenso na legenda:

  (a) construcao em asm   asm sem flag / normal           (cenario 2 / 1)
  (b) valor do L1         asm com glc / asm sem flag      (cenario 4 / 2)
  (c) bit slc             asm glc slc / asm com glc       (cenario 5 / 4)

As tres razoes tem a mesma unidade, entao dividem UM eixo: a altura das curvas
e comparavel entre elas (a versao anterior usava tres paineis com escalas
proprias, o que obrigava a legenda a pedir que a altura fosse ignorada).

Todos os rotulos de valor sao calculados do CSV -- nada digitado a mao.
Cores: slots 1-3 da paleta categorica de referencia (validados em todos os
pares, nos dois modos); marcadores distintos por curva para leitura em
impressao sem cor.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "matmul_sweep.csv")
SAIDA = os.path.join(RAIZ, "Documentação", "fig_sensibilidade.pdf")

AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3
TINTA, TINTA2, GRADE, BASE = "#0b0b0b", "#52514e", "#e4e4e0", "#9a9994"

# ---------------------------------------------------------------- dados
med = {}
with open(CSV) as f:
    for lin in csv.DictReader(f):
        med.setdefault(int(lin["N"]), {})[lin["kernel"]] = float(lin["mediana_ms"])
Ns = sorted(med)


def razao(num, den):
    return [med[n][num] / med[n][den] for n in Ns]


# (letra, nome, numerador / denominador por extenso, cenarios, serie, cor, marca)
CURVAS = [
    ("a", "Construção em asm", "asm sem flag ÷ normal", "2 ÷ 1",
     razao("matmul_asm_baseline", "matmul_via_L1"), AZUL, "o"),
    ("b", "Valor do L1 (bit glc)", "asm com glc ÷ asm sem flag", "4 ÷ 2",
     razao("matmul_bypass_L1", "matmul_asm_baseline"), LARANJA, "s"),
    ("c", "Bit slc", "asm glc slc ÷ asm com glc", "5 ÷ 4",
     razao("matmul_bypass_L1_L2", "matmul_bypass_L1"), AGUA, "^"),
]
for letra, _, rot, cen, y, _, _ in CURVAS:
    print("(%s) %-28s %s" % (letra, rot, "  ".join("%.3f" % v for v in y)))

vx = lambda v: ("%.2f×" % v).replace(".", ",")

# ---------------------------------------------------------------- estilo
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": TINTA2,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(6.3, 2.3))
ax = fig.add_axes([0.105, 0.19, 0.44, 0.76])

# referencia em 1,0x: sem diferenca entre numerador e denominador
ax.axhline(1.0, color=BASE, lw=0.8, zorder=1)
ax.text(Ns[0], 1.0 - 0.008, "1,00× = sem diferença", color=TINTA2,
        fontsize=6.5, ha="left", va="top")

for letra, nome, rot, cen, y, cor, marca in CURVAS:
    ax.plot(Ns, y, color=cor, lw=1.5, marker=marca, ms=5.5, mew=1.0,
            mec="white", solid_capstyle="round", solid_joinstyle="round",
            zorder=3, clip_on=False)
    # rotulos seletivos: so as extremidades, em tinta de texto
    ax.annotate(vx(y[0]), xy=(Ns[0], y[0]), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7, color=TINTA)
    ax.annotate("%s  (%s)" % (vx(y[-1]), letra), xy=(Ns[-1], y[-1]), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=7, color=TINTA)

ax.set_xscale("log", basex=2)
ax.xaxis.set_major_locator(FixedLocator(Ns))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlim(Ns[0] / 1.25, Ns[-1] * 1.25)
ax.set_ylim(0.95, 1.62)
ax.yaxis.set_major_locator(FixedLocator([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%.1f×" % v).replace(".", ",")))
ax.grid(axis="y", color=GRADE, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.tick_params(colors=TINTA2, length=2.5, width=0.6)
ax.set_xlabel("$N$ (ordem da matriz)", color=TINTA2, labelpad=2)
ax.set_ylabel("tempo do numerador\n÷ tempo do denominador", color=TINTA2)

# ---------------------------------------------------------------- legenda
# Cada entrada diz o que divide o que; a identidade vem do traco + marcador
# ao lado, nunca do texto colorido.
x0, y0, passo = 0.655, 0.80, 0.28
fig.text(x0, y0 + 0.10, "numerador ÷ denominador\n(medianas dos cenários da Tabela 5)",
         fontsize=7, color=TINTA2, va="bottom", linespacing=1.2)
for i, (letra, nome, rot, cen, y, cor, marca) in enumerate(CURVAS):
    yy = y0 - i * passo
    fig.lines.append(Line2D([x0, x0 + 0.04], [yy - 0.02] * 2, transform=fig.transFigure,
                            color=cor, lw=1.5, solid_capstyle="round"))
    fig.lines.append(Line2D([x0 + 0.02], [yy - 0.02], transform=fig.transFigure,
                            color=cor, marker=marca, ms=5.5, mew=1.0, mec="white", lw=0))
    fig.text(x0 + 0.055, yy, "(%s) %s" % (letra, nome), fontsize=8, color=TINTA,
             va="center", weight="bold")
    fig.text(x0 + 0.055, yy - 0.085, "%s   [%s]" % (rot, cen),
             fontsize=7.5, color=TINTA2, va="center")

fig.savefig(SAIDA, format="pdf")
print("\nescrito: " + SAIDA)
