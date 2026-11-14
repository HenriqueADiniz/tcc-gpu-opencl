# Impacto da Hierarquia de Memória em GPU AMD sob OpenCL

Código, dados e artigo do Trabalho de Conclusão de Curso de Henrique de Almeida
Diniz (Ciência da Computação, PUC Minas), orientado por Pedro Henrique Ramos Costa.

O trabalho mede quanto o *cache* L1 de uma GPU AMD GCN 4 (RX 580, gfx803) vale
para algoritmos paralelos em OpenCL, tirando-o do caminho das leituras com as
*flags* GLC/SLC emitidas por *inline assembly* e verificadas na ISA.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `experimentos/` | programas anfitrião (`.cpp`) e *kernels* OpenCL (`.cl` ou embutidos) |
| `resultados/` | CSVs e saídas de texto de cada campanha de medição |
| `analise/` | scripts Python que geram as tabelas (`gera_tabelas.py`) e as figuras do artigo |
| `isa/` | binários compilados e desmontagens que comprovam as *flags* emitidas |
| `tools/` | scripts das campanhas longas |
| `setup/` | instalação do ROCm 5.2 em Polaris e variáveis de ambiente |
| `Documentação/` | artigo (`main.tex`, formato SBC), tabelas geradas e figuras |

## Reproduzir

O manual completo, com o comando de cada experimento e o arquivo que ele grava,
está em [`COMO_RODAR.md`](COMO_RODAR.md). Em resumo:

```bash
source setup/env_rocm.sh
g++ -Wall -O2 experimentos/matmul.cpp -o output/matmul -lOpenCL
./output/matmul        # sempre a partir da raiz do repositório
```

Os programas gravam nos CSVs de `resultados/`. Para remedir sem misturar com as
campanhas do artigo, leia antes a seção correspondente do `COMO_RODAR.md`.

## Ambiente das medições

- GPU AMD Radeon RX 580 8 GB (Ellesmere, GCN 4, gfx803), *clocks* no
  gerenciamento dinâmico padrão (não fixados)
- Ubuntu 20.04.6 LTS, *kernel* 5.15
- ROCm 5.2.0 (`ROC_ENABLE_PRE_VEGA=1`), compilador de *kernel* AMD clang 14.0.0
  (`comgr`)
- GCC 9.4.0 no anfitrião; ISA desmontada com `llvm-objdump` (LLVM 12)
