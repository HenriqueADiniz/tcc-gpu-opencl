# -*- coding: utf-8 -*-
# Tolerância dos Exps. 1 e 3 (matmul.cpp): |c - ref| > 0,01*N/1024, absoluta.
# Preenchimento real: A[i] = (i % 100)/100, B[i] = ((i + 37) % 100)/100.
# Compara a tolerância com o erro de arredondamento da soma em float32 e com o
# desvio que um erro de endereçamento (ler B transposta) produziria.
import numpy as np
rng = np.random.default_rng(1)
print("N      tol     |C|~     erro float32 (mediana / max)   tol/erro_max   erro de indice (min)")
for N in [1024, 4096, 16384]:
    tol = 0.01 * N / 1024
    k = np.arange(N, dtype=np.int64)
    erros, desv, mag = [], [], []
    for _ in range(200):
        i, j = rng.integers(0, N, 2)
        a = ((i * N + k) % 100 / 100.0).astype(np.float32)
        b = (((k * N + j) + 37) % 100 / 100.0).astype(np.float32)
        bt = (((j * N + k) + 37) % 100 / 100.0).astype(np.float32)      # B lida transposta
        exato = np.dot(a.astype(np.float64), b.astype(np.float64))
        f32 = np.cumsum((a * b).astype(np.float32), dtype=np.float32)[-1]  # soma sequencial
        erros.append(abs(float(f32) - exato)); mag.append(exato)
        desv.append(abs(np.dot(a.astype(np.float64), bt.astype(np.float64)) - exato))
    e = np.array(erros)
    print("%-6d %.3f  %7.0f   %.4f / %.4f              %6.1f         %.2f" %
          (N, tol, np.median(mag), np.median(e), e.max(), tol / e.max(), min(desv)))
