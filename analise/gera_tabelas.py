# -*- coding: utf-8 -*-
"""Gera as tabelas do artigo que sao carregadas por \\tabinput (\\input primitivo) no main.tex.

Le os CSVs das campanhas e escreve, em Documentacao/ (ou no diretorio dado em
--saida):

  tab_slc_original.tex     Tabela 4, campanha original (slc_l2.txt, so texto)
  tab_slc.tex              Tabela 4, varredura      (teto_vram.csv, parte A)
  tab_cpu_ref.tex          Tabela 5, linha da CPU   (matmul_cpu_sweep.csv)
  tab_medianas_artigo.tex  Tabela 5, cenarios 1-5   (matmul_sweep.csv)
  tab_loc_glc.tex          Tabela 6, metade glc     (linha_coluna.csv)
  tab_loc_glcslc.tex       Tabela 6, metade glc slc (linha_coluna.csv)
  tab_layout.tex           Tabela 7                 (matmul_layout.csv + linha_coluna.csv)

Todas as razoes sao entre MEDIANAS da mesma campanha, recalculadas aqui a partir
de mediana_ms (e nao da coluna razao_sobre_noflag, arredondada a 4 casas).

As Tabelas 2 e 3 estao digitadas no main.tex. Para elas o script so imprime os
valores recalculados (--conferir), para comparacao manual:
  Tabelas 2 e 3 <- matmul_cache_flags.csv (1000 execucoes por cenario)
O --conferir imprime tambem os numeros do texto do Exp. 2 (Secao 5.2, resumo,
conclusoes), de teto_vram.csv e das execucoes do programa original
(slc_l2.txt, campanha original, e slc_l2_reexecucao.txt).

Uso (da raiz do projeto):
  python3 analise/gera_tabelas.py                  # regrava Documentacao/tab_*.tex
  python3 analise/gera_tabelas.py --saida /tmp/x   # grava em outro diretorio
  python3 analise/gera_tabelas.py --conferir       # imprime Tabelas 2 e 3 e o Exp. 2
"""
import argparse
import collections
import csv
import math
import os
import re
import statistics

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "resultados")
NS = [1024, 2048, 4096, 8192, 16384]


def ler(nome):
    with open(os.path.join(RES, nome), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def br(x, casas):
    """Decimal com virgula, como no texto do artigo."""
    return ("%.*f" % (casas, x)).replace(".", ",")


def tempo(ms):
    """Tempo em ms: 2 casas abaixo de 100, 1 abaixo de 1000, inteiro com
    separador de milhar a partir de 1000 (6,99 | 449,0 | 29.934)."""
    if ms < 100:
        return br(ms, 2)
    if ms < 1000:
        return br(ms, 1)
    return format(int(round(ms)), ",").replace(",", ".")


def linha(rotulo, valores):
    return rotulo + " & " + " & ".join(valores) + " \\\\\n"


def tab_cpu_ref():
    cpu = {int(r["N"]): float(r["mediana_ms"]) for r in ler("matmul_cpu_sweep.csv")}
    return linha("\\quad laço triplo & ---", [tempo(cpu[n]) for n in NS])


# Ordem e rotulos da Tabela 5 (cenarios 1-5 da Tabela 2)
CENARIOS = [
    ("matmul_via_L1", "1. normal (OpenCL C) & GLC=0 SLC=0"),
    ("matmul_asm_baseline", "2. controle \\textit{asm} & GLC=0 SLC=0"),
    ("matmul_via_L2", "3. \\texttt{volatile} & GLC=1$^\\ast$"),
    ("matmul_bypass_L1", "4. \\textit{asm} \\texttt{glc} & GLC=1"),
    ("matmul_bypass_L1_L2", "5. \\textit{asm} \\texttt{glc slc} & GLC=1 SLC=1"),
]


def tab_medianas_artigo():
    med = collections.defaultdict(dict)
    for r in ler("matmul_sweep.csv"):
        med[r["kernel"]][int(r["N"])] = float(r["mediana_ms"])
    return "".join(linha(rot, [tempo(med[k][n]) for n in NS]) for k, rot in CENARIOS)


PADROES = [
    ("linha", "\\quad linha (contíguo, \\textit{broadcast})"),
    ("dispersa", "\\quad dispersa (contíguo, espalhado)"),
    ("coluna", "\\quad coluna (passo largo, coalescido)"),
]


def medianas_padroes():
    return {(int(r["N"]), r["padrao"], r["flags"]): float(r["mediana_ms"])
            for r in ler("linha_coluna.csv")}


def tab_loc(flag):
    """Tabela 6: mediana(padrao, flag) / mediana(mesmo padrao, sem flag)."""
    P = medianas_padroes()
    return "".join(linha(rot, [br(P[(n, p, flag)] / P[(n, p, "GLC=0")], 3) for n in NS])
                   for p, rot in PADROES)


def tab_layout():
    """Tabela 7. A previsao soma os padroes isolados sem flag:
    (linha + dispersa) / (linha + coluna) -- o laco da matmul faz um load de A
    (padrao linha) e um de B (coluna se usual, ~dispersa se transposta)."""
    L = {(int(r["N"]), r["layout"], r["flags"]): float(r["mediana_ms"])
         for r in ler("matmul_layout.csv")}
    P = medianas_padroes()
    lin = lambda n: P[(n, "linha", "GLC=0")]
    out = linha("transposta / usual, medido",
                [br(L[(n, "transposta", "GLC=0")] / L[(n, "normal", "GLC=0")], 3) for n in NS])
    out += linha("transposta / usual, previsto",
                 [br((lin(n) + P[(n, "dispersa", "GLC=0")]) / (lin(n) + P[(n, "coluna", "GLC=0")]), 3)
                  for n in NS])
    out += "\\midrule\n"
    out += linha("\\texttt{glc}, $B$ usual",
                 [br(L[(n, "normal", "glc")] / L[(n, "normal", "GLC=0")], 3) for n in NS])
    out += linha("\\texttt{glc}, $B$ transposta",
                 [br(L[(n, "transposta", "glc")] / L[(n, "transposta", "GLC=0")], 3) for n in NS])
    # denominador e o glc, nao o sem flag: isola o slc
    out += linha("\\texttt{slc} sobre \\texttt{glc}, $B$ transposta",
                 [br(L[(n, "transposta", "glc slc")] / L[(n, "transposta", "glc")], 3) for n in NS])
    return out


MB = 1 << 20
PICO = 256.0   # GB/s: GDDR5 a 2000 MHz (pp_dpm_mclk) x 4 x 256 bits / 8
FLAGS_SLC = ["GLC=0 SLC=0", "GLC=1 SLC=0", "GLC=1 SLC=1"]


def varredura_slc():
    """{(bytes, flags): (ms, GB/s) medianos} da parte A de teto_vram.csv."""
    return {(int(r["ws_bytes"]), r["flags"]): (float(r["mediana_ms"]), float(r["gbs_mediana"]))
            for r in ler("teto_vram.csv") if r["parte"] == "A"}


def campanha_original_slc():
    """{(bytes, flags): (ms, GB/s)} da campanha original (slc_l2.txt, so texto)."""
    with open(os.path.join(RES, "slc_l2.txt"), encoding="utf-8") as f:
        blocos = f.read().split("=== ")[1:]
    out = {}
    for b in blocos:
        w = int(re.search(r"\(buffer (\d+) MB\)", b).group(1)) * MB
        for fl, ms, gb in re.findall(r"sweep_\w+\s+(GLC=\d SLC=\d)\s+([\d.]+)\s+([\d.]+)", b):
            out[(w, fl)] = (float(ms), float(gb))
    return out


def linhas_slc(V):
    """Tabela 4: uma linha por tamanho, com tempo e vazao das tres variantes e a
    razao de vazao glc slc / glc (= tempo glc / tempo glc slc). O valor de teste,
    glc slc em 1 MB, vai em negrito, como na tabela original."""
    out = ""
    for w in sorted({w for w, _ in V}):
        n, u = (w >> 10, "KB") if w < MB else ((w >> 20, "MB") if w < 1024 * MB else (w >> 30, "GB"))
        l2 = "cabe" if w < 2 * MB else ("no limite" if w == 2 * MB else "não cabe")
        t = [V[(w, fl)][0] for fl in FLAGS_SLC]
        g = [br(V[(w, fl)][1], 1) for fl in FLAGS_SLC]
        if w == MB:
            g[2] = "\\textbf{%s}" % g[2]
        out += linha("%d~%s & %s" % (n, u, l2), [br(x, 4) for x in t] + g + [br(t[1] / t[2], 3)])
    return out


def tab_slc_original():
    return linhas_slc(campanha_original_slc())


def tab_slc():
    return linhas_slc(varredura_slc())


def conferir_slc():
    """Numeros do texto do Exp. 2, recalculados dos resultados."""
    R = ler("teto_vram.csv")
    V = {k: v[1] for k, v in varredura_slc().items()}
    O = {k: v[1] for k, v in campanha_original_slc().items()}
    g = lambda r: float(r["gbs_mediana"])
    pct = lambda x: br(100 * x, 1) + "%"
    print("\nExp. 2 (pico teorico %s GB/s)" % br(PICO, 0))
    plat = [v for (w, fl), v in V.items() if w >= 8 * MB and fl != "GLC=1 SLC=1"]
    print("  patamar de 8 MB a 2 GB, sem flag e glc: %s a %s GB/s (%s a %s do pico)"
          % (br(min(plat), 1), br(max(plat), 1), pct(min(plat) / PICO), pct(max(plat) / PICO)))
    s128 = [v for (w, fl), v in V.items() if w >= 128 * MB and fl == "GLC=1 SLC=1"]
    print("  glc slc de 128 MB a 2 GB: %s a %s GB/s" % (br(min(s128), 1), br(max(s128), 1)))
    var = [r for r in R if r["parte"] == "B" and int(r["ws_bytes"]) == 1024 * MB
           and r["kernel"] != "sweep_noflag"]
    bm = max(var, key=g)
    print("  %d variantes em 1 GB (%d a %d B em voo por wavefront): melhor mediana %s (%s, GS=%s),"
          " melhor execucao %s, pior mediana %s"
          % (len(var), min(int(r["bytes_em_voo_por_wave"]) for r in var),
             max(int(r["bytes_em_voo_por_wave"]) for r in var), br(g(bm), 1), bm["kernel"],
             bm["gs"], br(max(float(r["gbs_max"]) for r in var), 1), br(min(map(g, var)), 1)))
    padrao = r"sweep_glc_slc\s+GLC=1 SLC=1\s+[\d.]+\s+([\d.]+)"
    with open(os.path.join(RES, "slc_l2_reexecucao.txt"), encoding="utf-8") as f:
        reexec = [float(x) for x in re.findall(padrao, f.read())][0::2]  # so 1 MB
    novas = [g(r) for r in R if r["kernel"] == "sweep_glc_slc" and int(r["ws_bytes"]) == MB]
    todas = [O[(MB, "GLC=1 SLC=1")]] + reexec + novas
    vo, vv = O[(MB, "GLC=1 SLC=1")], V[(MB, "GLC=1 SLC=1")]
    po = (O[(64 * MB, "GLC=0 SLC=0")] + O[(64 * MB, "GLC=1 SLC=0")]) / 2   # patamar original
    pv = statistics.mean(v for (w, fl), v in V.items() if w >= 8 * MB and fl == "GLC=1 SLC=0")
    print("  1 MB glc slc: original %s = %sx o pico e %sx o patamar de 64 MB (%s);"
          " varredura %s = %sx o pico e %sx o patamar (%s)"
          % (br(vo, 1), br(vo / PICO, 2), br(vo / po, 2), br(po, 1),
             br(vv, 1), br(vv / PICO, 2), br(vv / pv, 2), br(pv, 1)))
    print("  1 MB glc slc, todas as medicoes: n=%d (1 original, %d reexecucoes, %d na varredura,"
          " dez delas em buffers distintos), %s a %s, mediana %s; menor/pico = %s"
          % (len(todas), len(reexec), len(novas), br(min(todas), 1), br(max(todas), 1),
             br(statistics.median(todas), 1), br(min(todas) / PICO, 2) + "x"))
    C = {int(r["ws_bytes"]): g(r) for r in R if r["parte"] == "C" and r["flags"] == "GLC=1 SLC=1"}
    d = max(abs(C[w] / V[(w, "GLC=1 SLC=1")] - 1) for w in C)
    print("  L2 ocupado antes de cada execucao (parte C), glc slc: maior diferenca para a"
          " varredura = %s em %d tamanhos" % (pct(d), len(C)))
    T = varredura_slc()
    print("  razao glc slc / glc (tempos): " + ", ".join(
        "%d MB %s" % (w // MB, br(T[(w, "GLC=1 SLC=0")][0] / T[(w, "GLC=1 SLC=1")][0], 3))
        for w in sorted(w for w, fl in T if fl == "GLC=1 SLC=1" and 2 * MB <= w <= 64 * MB)))
    erros = []
    for w in sorted(w for w, fl in V if fl == "GLC=1 SLC=1" and 8 * MB <= w <= 128 * MB):
        h = 2 * MB / w
        erros.append(V[(w, "GLC=1 SLC=1")] * (h / vv + (1 - h) / pv) - 1)
    print("  modelo 1/v = h/%s + (1-h)/%s, h = 2 MB/W, de 8 a 128 MB: maior erro %s"
          % (br(vv, 1), br(pv, 1), pct(max(map(abs, erros)))))
    h = 2 / 64
    prev = 1 / (h / vo + (1 - h) / po)
    print("  mesmo modelo na campanha original, 64 MB: previsto %s, medido %s"
          % (br(prev, 1), br(O[(64 * MB, "GLC=1 SLC=1")], 1)))


def conferir():
    """Imprime os valores das Tabelas 2 e 3 recalculados do CSV do Exp. 1."""
    amostras = collections.defaultdict(list)
    for r in ler("matmul_cache_flags.csv"):
        amostras[r["kernel"]].append(float(r["tempo_ms"]))
    gflop = 2.0 * 1024 ** 3 / 1e9
    est = {}
    print("Tabela 2 (GPU, N=1024): media | mediana | cv% | GFLOPS (2N^3 / mediana)")
    for k, rot in CENARIOS + [("matmul_direto_VRAM", "3'. replica do 3")]:
        v = amostras[k]
        m, md, dp = statistics.mean(v), statistics.median(v), statistics.stdev(v)
        est[k] = (m, dp / math.sqrt(len(v)))
        print("  %-44s n=%d  %s | %s | %s | %s" % (rot.split(" & ")[0], len(v), br(m, 3),
              br(md, 3), br(100 * dp / m, 2), br(gflop / (md / 1000), 1)))
    cpu = [r for r in ler("matmul_cpu_sweep.csv") if r["N"] == "1024"][0]
    print("  CPU (matmul_cpu_sweep.csv, %s execucoes): %s | %s | %s | %s" % (
        cpu["runs"], br(float(cpu["media_ms"]), 1), br(float(cpu["mediana_ms"]), 1),
        br(float(cpu["cv_pct"]), 2), br(float(cpu["gflops"]), 1)))
    print("\nTabela 3: razao de medias, IC95 pelo metodo delta, t de Welch")
    for a, b, rot in [("matmul_via_L2", "matmul_direto_VRAM", "3 vs replica"),
                      ("matmul_via_L1", "matmul_asm_baseline", "1 -> 2"),
                      ("matmul_asm_baseline", "matmul_bypass_L1", "2 -> 4"),
                      ("matmul_bypass_L1", "matmul_bypass_L1_L2", "4 -> 5"),
                      ("matmul_bypass_L1", "matmul_via_L2", "3 vs 4 (3/4)")]:
        (ma, ea), (mb, eb) = est[a], est[b]
        r = mb / ma
        rel = math.sqrt((ea / ma) ** 2 + (eb / mb) ** 2)
        t = (mb - ma) / math.sqrt(ea ** 2 + eb ** 2)
        print("  %-14s %s  [%s; %s]  t=%s" % (rot, br(r, 3), br(r * (1 - 1.96 * rel), 3),
              br(r * (1 + 1.96 * rel), 3), br(t, 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", default=os.path.join(RAIZ, "Documentação"))
    ap.add_argument("--conferir", action="store_true")
    a = ap.parse_args()
    if a.conferir:
        conferir()
        conferir_slc()
        return
    os.makedirs(a.saida, exist_ok=True)
    tabelas = {
        "tab_slc_original.tex": tab_slc_original(),
        "tab_slc.tex": tab_slc(),
        "tab_cpu_ref.tex": tab_cpu_ref(),
        "tab_medianas_artigo.tex": tab_medianas_artigo(),
        "tab_loc_glc.tex": tab_loc("glc"),
        "tab_loc_glcslc.tex": tab_loc("glc slc"),
        "tab_layout.tex": tab_layout(),
    }
    for nome, conteudo in tabelas.items():
        with open(os.path.join(a.saida, nome), "w", encoding="utf-8") as f:
            f.write(conteudo)
        print("gravado:", os.path.join(a.saida, nome))


if __name__ == "__main__":
    main()
