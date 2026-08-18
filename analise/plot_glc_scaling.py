#!/usr/bin/env python3
"""
Figuras e ajuste de modelo para o experimento de escalabilidade do GLC.

Entrada : resultados/glc_scaling.csv  (gerado por output/glc_scaling)
Saida   : analise/fig_glc_desempenho.png
          analise/fig_glc_razao.png
          analise/fig_glc_bandwidth.png
          analise/figuras_glc.png       (painel unico, para conferencia)
          analise/modelo_glc.txt        (coeficientes do ajuste)

PNG em 300 dpi, pronto para \\includegraphics no LaTeX.

NOTA SOBRE REDUNDANCIA
----------------------
Largura de banda efetiva e GFLOPS sao a mesma curva:
    BW = 2N^3*4/T   e   GFLOPS = 2N^3/T   =>   BW = 4 * GFLOPS
As duas figuras existem porque sustentam argumentos diferentes — desempenho
alcancado (fig. desempenho) e comparacao contra o pico da VRAM (fig. bandwidth)
— mas NAO sao evidencias independentes. Usar as duas no artigo so se ambos os
argumentos forem feitos; caso contrario, uma basta.

MODELO
------
Ajusta-se, por minimos quadrados,  T(N) = a*N^3 + b*N^2.
O termo N^3 acompanha operacoes e loads emitidos (ambos 2N^3); o termo N^2
absorve efeitos de borda. A razao entre os coeficientes `a` de duas celulas e o
custo assintotico do fator que as separa.

Uso:
    python3 analise/plot_glc_scaling.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

CSV_IN = 'resultados/glc_scaling.csv'
OUTDIR = 'analise'
DPI = 300

# Cor fixa POR SERIE, igual em todas as figuras — a cor identifica a entidade,
# nunca a posicao dela num grafico especifico. Paleta categorica de referencia
# (modo claro), slots 1-4 na ordem documentada.
# Marcador e traco distintos por serie: codificacao secundaria, para o artigo
# continuar legivel impresso em tons de cinza.
STYLE = {
    'mm_glc0_overlap': ('GLC=0, sobreposto',  '#2a78d6', 'o', '-'),
    'mm_glc0_serial':  ('GLC=0, serializado', '#eb6834', 's', '--'),
    'mm_glc1_overlap': ('GLC=1, sobreposto',  '#1baf7a', '^', '-.'),
    'mm_glc1_serial':  ('GLC=1, serializado', '#eda100', 'D', ':'),
    'mm_plain_c':      ('C puro (sem asm)',   '#8a8a86', 'v', (0, (1, 1))),
    'mm_volatile':     ('volatile',           '#b5b5b0', 'x', (0, (3, 1, 1, 1))),
}

TEXT_1, TEXT_2, GRID = '#0b0b0b', '#52514e', '#d8d8d4'
VRAM_GBS = 256.0   # pico teorico da GDDR5 da RX 580
XTICKS = [256, 512, 1024, 2048, 3072]


def load(path):
    if not os.path.exists(path):
        sys.exit(f"nao encontrei {path} — rode ./output/glc_scaling primeiro")
    data = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            d = data.setdefault(row['kernel'],
                                {'N': [], 't': [], 'sd': [], 'bw': [], 'gf': []})
            d['N'].append(int(row['N']))
            d['t'].append(float(row['media_ms']))
            d['sd'].append(float(row['dp_ms']))
            d['bw'].append(float(row['bw_efetiva_GBs']))
            d['gf'].append(float(row['gflops']))
    for d in data.values():
        order = np.argsort(d['N'])
        for k in ('N', 't', 'sd', 'bw', 'gf'):
            d[k] = np.asarray(d[k], dtype=float)[order]
    return data


def fit_cubic(N, T):
    M = np.column_stack([N ** 3, N ** 2])
    coef, *_ = np.linalg.lstsq(M, T, rcond=None)
    pred = M @ coef
    ss_res = float(np.sum((T - pred) ** 2))
    ss_tot = float(np.sum((T - np.mean(T)) ** 2))
    return coef[0], coef[1], (1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    if title:
        ax.set_title(title, color=TEXT_1, fontsize=10, pad=10)
    ax.grid(True, which='major', color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=3)
    if matplotlib.__version__ < '3.3':
        ax.set_xscale('log', basex=2)
    else:
        ax.set_xscale('log', base=2)
    ax.set_xticks(XTICKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


def draw(ax, data, key, field, emphasis=True, label=None):
    lab, col, mk, ls = STYLE[key]
    d = data[key]
    ax.plot(d['N'], d[field], color=col, linestyle=ls,
            linewidth=2.0 if emphasis else 1.3,
            marker=mk, markersize=5 if emphasis else 3.5,
            label=label or lab, zorder=3 if emphasis else 2,
            markeredgecolor='white', markeredgewidth=0.6 if emphasis else 0.0)


def legenda(ax, loc):
    leg = ax.legend(fontsize=8, frameon=False, loc=loc)
    for t in leg.get_texts():
        t.set_color(TEXT_2)


# --------------------------------------------------------------------------- #
#  Fig. 1 — desempenho alcancado                                              #
#  So as duas celulas que respondem a pergunta do L1, mais o C puro como       #
#  contexto em cinza. As outras duas celulas vivem na figura da razao.        #
#  GFLOPS em vez de tempo: normaliza pelo trabalho (2N^3) e achata as curvas, #
#  tornando visivel a diferenca entre celulas. Num grafico de tempo em log-log#
#  todas as curvas N^3 se sobrepoem e nada se le.                             #
# --------------------------------------------------------------------------- #
def fig_desempenho(data, ax):
    draw(ax, data, 'mm_plain_c',      'gf', emphasis=False)
    draw(ax, data, 'mm_glc0_overlap', 'gf', label='GLC=0 — L1 ativo')
    draw(ax, data, 'mm_glc1_overlap', 'gf', label='GLC=1 — L1 contornado')
    style(ax, 'N (matriz N×N)', 'desempenho (GFLOPS)',
          'Desempenho alcançado — com e sem o L1')
    ax.set_ylim(bottom=0)
    legenda(ax, 'lower right')


# --------------------------------------------------------------------------- #
#  Fig. 2 — resultado principal: efeito isolado de cada fator                 #
# --------------------------------------------------------------------------- #
def fig_razao(data, ax):
    base = data['mm_glc0_overlap']
    curvas = [
        ('mm_glc1_overlap', 'bypass do L1  (3)/(1)',  5),
        ('mm_glc0_serial',  'serialização  (2)/(1)', -6),
        ('mm_glc1_serial',  'ambos         (4)/(1)',  0),
    ]
    # N multiplo de 256 => stride de linha (4N bytes) multiplo de 1 KB, que
    # mapeia os acessos a B em poucos conjuntos do L1 e gera conflito.
    for n in base['N']:
        if int(n) % 256 == 0:
            ax.axvline(n, color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(1.0, color=TEXT_2, linewidth=1.0, alpha=0.5, zorder=1)
    ax.annotate('sem efeito', xy=(base['N'][0], 1.0), xytext=(0, -12),
                textcoords='offset points', color=TEXT_2, fontsize=7.5)

    for key, lab, dy in curvas:
        _l, col, mk, ls = STYLE[key]
        r = data[key]['t'] / base['t']
        ax.plot(base['N'], r, color=col, linestyle=ls, linewidth=2.0,
                marker=mk, markersize=5, zorder=3,
                markeredgecolor='white', markeredgewidth=0.6)
        # rotulo direto: identidade nunca depende so da cor
        ax.annotate(lab, xy=(base['N'][-1], r[-1]), xytext=(6, dy),
                    textcoords='offset points', color=col, fontsize=7.5,
                    va='center', fontweight='bold')

    i512 = int(np.argmin(np.abs(base['N'] - 512)))
    if abs(base['N'][i512] - 512) < 1:
        r512 = data['mm_glc1_overlap']['t'][i512] / base['t'][i512]
        ax.annotate('N=512: conflito de cache\npor stride potência de 2',
                    xy=(512, r512), xytext=(24, 6), textcoords='offset points',
                    color=TEXT_1, fontsize=7,
                    arrowprops=dict(arrowstyle='-', color=TEXT_2, linewidth=0.8))

    style(ax, 'N (matriz N×N) — linhas verticais: N múltiplo de 256',
          'razão sobre o baseline (×)',
          'Efeito isolado de cada fator — picos onde o stride conflita no L1')
    ax.set_xlim(right=base['N'][-1] * 2.6)


# --------------------------------------------------------------------------- #
#  Fig. 3 — taxa efetiva de load contra o pico da VRAM                        #
#  Unico argumento desta figura: mesmo contornando o L1, a taxa fica muito    #
#  acima da VRAM, logo o trafego e servido pelo L2 (SLC=0). Duas series bastam#
# --------------------------------------------------------------------------- #
def fig_bandwidth(data, ax):
    ax.axhline(VRAM_GBS, color=TEXT_1, linewidth=1.4, linestyle='--', zorder=1)
    ax.annotate(f'pico teórico da VRAM GDDR5 ≈ {VRAM_GBS:.0f} GB/s',
                xy=(XTICKS[-1], VRAM_GBS), xytext=(0, 8), ha='right',
                textcoords='offset points', color=TEXT_1, fontsize=8)
    draw(ax, data, 'mm_glc0_overlap', 'bw', label='GLC=0 — L1 ativo')
    draw(ax, data, 'mm_glc1_overlap', 'bw', label='GLC=1 — L1 contornado')
    style(ax, 'N (matriz N×N)', 'taxa efetiva de load (GB/s)',
          'Mesmo sem o L1, o tráfego é servido pelo L2')
    ax.set_ylim(0, max(data['mm_glc0_overlap']['bw'].max(),
                       data['mm_glc1_overlap']['bw'].max()) * 1.25)
    legenda(ax, 'lower right')


def modelo(data, path):
    L = ["Ajuste  T(N) = a*N^3 + b*N^2  por minimos quadrados",
         "=" * 72,
         f"{'celula':26s} {'a (ns/N^3)':>14s} {'b (us/N^2)':>14s} {'R^2':>8s}",
         "-" * 72]
    coefs = {}
    for key in STYLE:
        if key not in data:
            continue
        d = data[key]
        a, b, r2 = fit_cubic(d['N'], d['t'])
        coefs[key] = a
        L.append(f"{STYLE[key][0]:26s} {a * 1e6:14.6f} {b * 1e3:14.4f} {r2:8.5f}")

    base = coefs['mm_glc0_overlap']
    L += ["", "Custo assintotico (razao dos coeficientes a)", "=" * 72]
    for key, lab in (('mm_glc1_overlap', 'bypass do L1     (3)/(1)'),
                     ('mm_glc0_serial',  'serializacao     (2)/(1)'),
                     ('mm_glc1_serial',  'ambos            (4)/(1)')):
        L.append(f"  {lab:28s} = {coefs[key] / base:.3f}x")
    if 'mm_plain_c' in coefs:
        L.append(f"  {'custo do asm     (1)/(0)':28s} = {base / coefs['mm_plain_c']:.3f}x")

    L += ["", "Razao medida por tamanho — bypass do L1 (3)/(1)", "=" * 72]
    b0, b1 = data['mm_glc0_overlap'], data['mm_glc1_overlap']
    for n, t0, t1 in zip(b0['N'], b0['t'], b1['t']):
        marca = "  <- N multiplo de 256" if int(n) % 256 == 0 else ""
        L.append(f"  N = {int(n):5d}   {t1 / t0:.3f}x{marca}")

    txt = "\n".join(L)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(txt + "\n")
    print(txt)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = load(CSV_IN)

    for nome, fn in (('fig_glc_desempenho', fig_desempenho),
                     ('fig_glc_razao',      fig_razao),
                     ('fig_glc_bandwidth',  fig_bandwidth)):
        fig, ax = plt.subplots(figsize=(5.4, 3.7))
        fn(data, ax)
        fig.tight_layout()
        out = os.path.join(OUTDIR, nome + '.png')
        fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"gravado: {out}")

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.0))
    fig_desempenho(data, axes[0])
    fig_razao(data, axes[1])
    fig_bandwidth(data, axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'figuras_glc.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"gravado: {OUTDIR}/figuras_glc.png")

    print()
    modelo(data, os.path.join(OUTDIR, 'modelo_glc.txt'))


if __name__ == '__main__':
    main()
