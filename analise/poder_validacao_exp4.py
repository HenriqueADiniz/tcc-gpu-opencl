# -*- coding: utf-8 -*-
# Poder de detecção da validação do Experimento 4, com o preenchimento real do código.
# Critério do código: falha se |c - esp| > tol*|esp| + tol, tol = 0,02*N/1024.
#   linha_coluna.cpp:  A[i] = (i % 251)*0,125 + 1;  esperado = soma de linha/coluna/linha deslocada
#   matmul_layout.cpp: B[i] = (i % 173)*0,0625 + 1; esperado = (A·B)[i][j] em 4.096 posições
# Pergunta: um erro de endereçamento (ler o padrão errado, ou B sem transpor) seria reprovado?
import numpy as np

fA = lambda t: (t % 251) * 0.125 + 1.0
fB = lambda t: (t % 173) * 0.0625 + 1.0
ciclo = np.array([fA(t) for t in range(251)])
P = np.concatenate([[0.0], np.cumsum(ciclo)])
S = lambda m: (m // 251) * ciclo.sum() + P[m % 251]      # soma de fA(0..m-1)

def detecta(errado, esp, tol):
    return bool(np.any(np.abs(errado - esp) > tol * np.abs(esp) + tol))

print("N      tol    linha<->coluna  linha->dispersa  B sem transpor | dif. rel. max (lc, ld, B)")
rng = np.random.default_rng(2027)
for N in [1024, 2048, 4096, 8192, 16384]:
    tol = 0.02 * N / 1024
    i = np.arange(N, dtype=np.int64)
    sl = S(i * N + N) - S(i * N)                          # soma da linha i
    k = np.arange(N, dtype=np.int64)
    sc_res = np.array([fA((k * (N % 251) + j) % 251).sum() for j in range(251)])
    sc = sc_res[np.arange(N) % 251]                       # soma da coluna j (depende de j mod 251)
    # linha <-> coluna: em (i,j), o kernel devolveria sc[j] no lugar de sl[i]; testa os extremos
    lc = max(detecta(sc.max(), sl, tol), detecta(sc.min(), sl, tol))
    rel_lc = max(np.abs(sc.max() - sl).max(), np.abs(sc.min() - sl).max()) / sl.min()
    # linha -> dispersa: devolveria sl[(i+j) % N], que percorre todas as linhas quando j varia
    ld = max(detecta(sl.max(), sl, tol), detecta(sl.min(), sl, tol))
    rel_ld = (sl.max() - sl.min()) / sl.min()
    # multiplicação: o kernel transposto lendo B sem transpor calcularia (A·B^T)[i][j]
    amostra = rng.integers(0, N * N, 4096)
    rel_B, det_B = 0.0, False
    for bloco in np.array_split(amostra, 64):
        ii, jj = bloco // N, bloco % N
        a = fA(ii[:, None] * N + k[None, :])
        b_col = fB(k[None, :] * N + jj[:, None])          # coluna j de B (correto)
        b_lin = fB(jj[:, None] * N + k[None, :])          # linha j de B (erro)
        esp = (a * b_col).sum(1); err = (a * b_lin).sum(1)
        rel_B = max(rel_B, (np.abs(err - esp) / esp).max())
        det_B = det_B or detecta(err, esp, tol)
    sim = lambda x: "reprova" if x else "PASSA"
    print("%-6d %5.0f%%  %-14s  %-15s  %-14s | %.2f%%, %.2f%%, %.2f%%" %
          (N, 100 * tol, sim(lc), sim(ld), sim(det_B), 100 * rel_lc, 100 * rel_ld, 100 * rel_B))

# Exatidão: em float32, as somas de linha/coluna (múltiplos de 0,125, até 16384*32,25)
# cabem na mantissa de 24 bits, então a saída correta é exata e o erro esperado é zero.
print("maior soma em N=16384: %.0f  (limite exato em float32 com passo 0,125: %d)" %
      (16384 * 32.25, 2**24 // 8))
