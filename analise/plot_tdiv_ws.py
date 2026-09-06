#!/usr/bin/env python3
"""
Figura do Experimento 10: custo de contornar o L1 em funcao do conjunto de
trabalho reusado, na divisao sucessiva tabelada.

Le resultados/tdiv_ws.csv e gera analise/fig_tdiv_ws.png.
Uso:  python3 analise/plot_tdiv_ws.py     (a partir da raiz do projeto)
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "resultados/tdiv_ws.csv"
OUT = "analise/fig_tdiv_ws.png"
L1_KB = 16.0

ESTILO = {
    "tdiv_unif": ("o-",  "#888888", "acesso uniforme (1 linha/wave)"),
    "tdiv_rot":  ("s-",  "#1f77b4", "disperso, workgroups em fase"),
    "tdiv_dsyn": ("^-",  "#d62728", "disperso, workgroups dessincronizados"),
}

dados = {}          # kernel -> {ws: {nivel: mediana}}
with open(CSV) as f:
    for r in csv.DictReader(f):
        k = r["kernel"]
        ws = float(r["working_set_KB"])
        dados.setdefault(k, {}).setdefault(ws, {})[r["nivel"]] = float(r["mediana_ms"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- Painel A: razao L2/L1 -------------------------------------------------
for k, (marca, cor, rot) in ESTILO.items():
    if k not in dados:
        continue
    xs = sorted(dados[k])
    ys = [dados[k][x]["L2 (glc)"] / dados[k][x]["L1 (normal)"] for x in xs]
    ax1.plot(xs, ys, marca, color=cor, label=rot, markersize=5)

ax1.axvline(L1_KB, color="black", ls="--", lw=1)
ax1.text(L1_KB * 1.06, ax1.get_ylim()[1] * 0.95, "L1 = 16 KB/CU",
         fontsize=8, va="top")
ax1.axhline(1.0, color="gray", ls=":", lw=0.8)
ax1.set_xscale("log", basex=2) if matplotlib.__version__ < "3.3" else ax1.set_xscale("log", base=2)
ax1.set_xlabel("conjunto de trabalho reusado (KB)")
ax1.set_ylabel("tempo com glc / tempo com L1")
ax1.set_title("(a) custo de contornar o L1")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# --- Painel B: tempos absolutos do kernel dessincronizado -------------------
k = "tdiv_dsyn"
if k in dados:
    xs = sorted(dados[k])
    ax2.plot(xs, [dados[k][x]["L1 (normal)"] for x in xs], "^-",
             color="#d62728", label="com L1 (sem flag)", markersize=5)
    ax2.plot(xs, [dados[k][x]["L2 (glc)"] for x in xs], "^--",
             color="#7f7f7f", label="sem L1 (glc)", markersize=5)
ax2.axvline(L1_KB, color="black", ls="--", lw=1)
ax2.set_xscale("log", basex=2) if matplotlib.__version__ < "3.3" else ax2.set_xscale("log", base=2)
ax2.set_xlabel("conjunto de trabalho reusado (KB)")
ax2.set_ylabel("tempo mediano do kernel (ms)")
ax2.set_title("(b) tdiv_dsyn: o L1 perde eficacia apos 16 KB")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle("Divisao sucessiva tabelada — gfx803 (RX 580), 640 M loads por ponto",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print("figura:", OUT)
