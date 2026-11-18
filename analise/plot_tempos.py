# -*- coding: utf-8 -*-
"""Figura do Experimento 3: tempo da multiplicacao na CPU e na GPU.

Le resultados/matmul_sweep.csv (cenario 1, matmul_via_L1) e
resultados/matmul_cpu_sweep.csv (laco triplo, 14 threads) e gera
Documentacao/fig_tempos.pdf.

Os tempos vao de 7 ms a 3,3 h. Uma escala linear esmagaria a GPU contra o eixo,
e um segundo eixo so para a CPU tornaria arbitraria a distancia entre as curvas
(o grafico poderia ate cruza-las). A escala logaritmica mantem as duas series
num unico eixo, e a distancia vertical entre elas passa a ser a propria razao
CPU/GPU, anotada nas extremidades.

Todos os rotulos de valor sao calculados do CSV -- nada digitado a mao.
Cores: slots 1-2 da paleta categorica de referencia, as mesmas da Figura de
sensibilidade; marcadores distintos para impressao sem cor. Desenhada no
tamanho de impressao (10 cm de largura), para que o texto saia a 7,5 pt.

Compativel com matplotlib 3.1 (Ubuntu 20.04) e posteriores.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "resultados")
SAIDA = os.path.join(RAIZ, "Documentação", "fig_tempos.pdf")

AZUL, LARANJA = "#2a78d6", "#eb6834"                 # slots 1-2
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e4e0"


def ler(nome):
    with open(os.path.join(RES, nome), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- dados (s)
gpu = {int(r["N"]): float(r["mediana_ms"]) / 1000.0
       for r in ler("matmul_sweep.csv") if r["kernel"] == "matmul_via_L1"}
cpu = {int(r["N"]): float(r["mediana_ms"]) / 1000.0 for r in ler("matmul_cpu_sweep.csv")}
Ns = sorted(gpu)
assert Ns == sorted(cpu), "os dois CSVs devem cobrir os mesmos N"
yg = [gpu[n] for n in Ns]
yc = [cpu[n] for n in Ns]


def duracao(s):
    """Rotulo de duracao em pt-BR, com virgula decimal."""
    if s >= 3600:
        return ("%.1f h" % (s / 3600)).replace(".", ",")
    if s >= 60:
        return "%.0f min" % (s / 60)
    if s >= 10:
        return "%.0f s" % s
    if s >= 1:
        return ("%.1f s" % s).replace(".", ",")
    return "%.0f ms" % (s * 1000)


def razao(n):
    return "%.0f×" % (cpu[n] / gpu[n])


for n in Ns:
    print("N=%5d  GPU %-7s CPU %-7s CPU/GPU %s" % (n, duracao(gpu[n]), duracao(cpu[n]), razao(n)))

# ---------------------------------------------------------------- estilo
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",          # N em itálico serifado, como no texto
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "text.color": TINTA,
    "axes.edgecolor": TINTA2,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

CM = 1 / 2.54
fig = plt.figure(figsize=(10.0 * CM, 4.0 * CM))
ax = fig.add_axes([0.125, 0.245, 0.865, 0.735])

SERIES = [("CPU: laço triplo, 14 threads", yc, LARANJA, "s"),
          ("GPU: cenário 1 (OpenCL C)", yg, AZUL, "o")]
for _, y, cor, marca in SERIES:
    ax.plot(Ns, y, color=cor, lw=1.3, marker=marca, ms=4.8, mew=0.9, mec="white",
            solid_capstyle="round", solid_joinstyle="round", zorder=3, clip_on=False)
    # rotulo seletivo: so o valor final, em tinta de texto
    ax.annotate(duracao(y[-1]), xy=(Ns[-1], y[-1]), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center", fontsize=7.5)

# razao CPU/GPU nas extremidades: a distancia vertical entre as curvas
for i, lado in ((0, 1), (-1, -1)):
    n = Ns[i]
    x = n * 1.12 ** lado
    ax.annotate("", xy=(x, cpu[n]), xytext=(x, gpu[n]),
                arrowprops=dict(arrowstyle="<->", color=TINTA2, lw=0.6,
                                shrinkA=1.5, shrinkB=1.5, mutation_scale=6))
    ax.text(x * 1.05 ** lado, (cpu[n] * gpu[n]) ** 0.5 * 1.6, razao(n),
            ha="left" if lado > 0 else "right", va="center", fontsize=7.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_locator(FixedLocator(Ns))
ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in Ns]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlim(Ns[0] / 1.25, Ns[-1] * 1.6)
YT = [0.01, 1, 60, 3600]
ax.yaxis.set_major_locator(FixedLocator(YT))
ax.yaxis.set_major_formatter(FixedFormatter(["10 ms", "1 s", "1 min", "1 h"]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(0.004, 3.0e4)
ax.grid(axis="y", color=GRADE, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.tick_params(colors=TINTA2, length=2.5, width=0.6)
ax.set_xlabel("$N$ (ordem da matriz)", color=TINTA2, labelpad=2)
ax.set_ylabel("tempo (escala log.)", color=TINTA2, labelpad=2)

# legenda: identidade pelo traco + marcador, texto em tinta (nunca colorido)
ax.legend(handles=[Line2D([0], [0], color=c, lw=1.3, marker=m, ms=4.8, mew=0.9,
                          mec="white", label=rot) for rot, _, c, m in SERIES],
          loc="upper left", frameon=False, fontsize=7.5, handlelength=1.8,
          borderaxespad=0.2, labelspacing=0.3)

fig.savefig(SAIDA, format="pdf")
print("\nescrito: " + SAIDA)
