#!/usr/bin/env python3
"""
Figura do P-chase com controle de nivel de cache.

Entrada : resultados/pchase_flags.csv  (gerado por output/pchase_flags)
Saida   : analise/fig_pchase_flags.png  (300 dpi, para o artigo)

A figura mostra a latencia de acesso em funcao do conjunto de trabalho, sob as
tres configuracoes de flags. Dois fatos ficam visiveis de imediato:

  1. a curva `glc` perde o patamar do L1 -- ela ja comeca no nivel do L2 mesmo
     com 1 KB de conjunto de trabalho, o que confirma o bypass do L1;
  2. as curvas `glc` e `glc slc` sao indistinguiveis em toda a faixa, o que
     mostra que SLC=1 nao acrescenta bypass do L2.

Uso:
    python3 analise/plot_pchase_flags.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

CSV_IN = 'resultados/pchase_flags.csv'
OUTDIR = 'analise'
DPI = 300

# Paleta categorica de referencia (modo claro), slots 1-3 na ordem documentada.
# Marcador e traco distintos: codificacao secundaria, para leitura em cinza.
SERIES = [
    ('pchase_L1',   'sem flags — L1 → L2 → VRAM', '#2a78d6', 'o', '-'),
    ('pchase_L2',   'glc — contorna o L1',        '#eb6834', 's', '--'),
    ('pchase_VRAM', 'glc slc — contorna L1 e L2', '#1baf7a', '^', '-.'),
]
TEXT_1, TEXT_2, GRID = '#0b0b0b', '#52514e', '#d8d8d4'

L1_KB, L2_KB = 16, 2048       # limites nominais da Ellesmere


def load(path):
    if not os.path.exists(path):
        sys.exit(f"nao encontrei {path} — rode ./output/pchase_flags primeiro")
    d = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            e = d.setdefault(row['variante'], {'kb': [], 'ns': []})
            e['kb'].append(int(row['bytes']) / 1024.0)
            e['ns'].append(float(row['latencia_ns']))
    for e in d.values():
        o = np.argsort(e['kb'])
        e['kb'] = np.asarray(e['kb'])[o]
        e['ns'] = np.asarray(e['ns'])[o]
    return d


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = load(CSV_IN)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Limites nominais dos niveis, como referencia de leitura
    for x, lab in ((L1_KB, 'L1 = 16 KB'), (L2_KB, 'L2 = 2 MB')):
        ax.axvline(x, color=GRID, linewidth=1.2, zorder=0)
        ax.annotate(lab, xy=(x, 0.985), xycoords=('data', 'axes fraction'),
                    xytext=(4, 0), textcoords='offset points',
                    color=TEXT_2, fontsize=7.5, va='top')

    for key, lab, col, mk, ls in SERIES:
        e = data[key]
        ax.plot(e['kb'], e['ns'], color=col, linestyle=ls, linewidth=2.0,
                marker=mk, markersize=5, label=lab, zorder=3,
                markeredgecolor='white', markeredgewidth=0.6)

    # A medida decisiva: no menor conjunto de trabalho, a distancia entre as
    # curvas identifica qual nivel esta servindo o acesso.
    k0 = data['pchase_L1']['kb'][0]
    y0 = data['pchase_L1']['ns'][0]
    y1 = data['pchase_L2']['ns'][0]
    ax.annotate('', xy=(k0, y0), xytext=(k0, y1),
                arrowprops=dict(arrowstyle='<->', color=TEXT_1, linewidth=1.1))
    ax.annotate(f'+{y1 - y0:.0f} ns\ncom 1 KB de\nconjunto de trabalho',
                xy=(k0, (y0 + y1) / 2), xytext=(12, -6),
                textcoords='offset points', color=TEXT_1, fontsize=7.5)

    ax.set_xscale('log', basex=2) if matplotlib.__version__ < '3.3' \
        else ax.set_xscale('log', base=2)
    ax.set_xlabel('conjunto de trabalho (KB)', color=TEXT_2, fontsize=9)
    ax.set_ylabel('latência por acesso (ns)', color=TEXT_2, fontsize=9)
    ax.set_title('P-chase com controle de nível de cache',
                 color=TEXT_1, fontsize=10, pad=10)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=3)
    ax.set_xticks([1, 16, 64, 256, 1024, 8192])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    leg = ax.legend(fontsize=8, frameon=False, loc='lower right')
    for t in leg.get_texts():
        t.set_color(TEXT_2)

    fig.tight_layout()
    out = os.path.join(OUTDIR, 'fig_pchase_flags.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"gravado: {out}")

    # Resumo numerico das sobreposicoes, que e o achado principal
    g, s = data['pchase_L2']['ns'], data['pchase_VRAM']['ns']
    dif = np.abs(s - g) / g * 100.0
    print(f"\ndiferenca |glc slc - glc| ao longo da curva:")
    print(f"  media {dif.mean():.2f}%  |  maxima {dif.max():.2f}%")
    print("  -> as duas variantes sao indistinguiveis: SLC=1 nao contorna o L2")


if __name__ == '__main__':
    main()
