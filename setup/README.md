# Setup do ambiente Linux — TCC Hierarquia de Memória em GPU AMD

Ambiente real: **Ubuntu 20.04.6 LTS (focal)**, kernel 5.15, **AMD RX 580 (Ellesmere / Polaris / gfx803)**.

> O `CONTEXTO_LINUX.md` assume Ubuntu 22.04 — a máquina é 20.04. Os passos abaixo são
> os que efetivamente funcionaram nesta configuração.

## Scripts (executar em ordem, uma vez só)

Todos precisam de root. Como o `sudo` exige senha e não há TTY no agente, use `pkexec`
com **caminho absoluto** (o `pkexec` reseta o diretório de trabalho):

```bash
cp setup/01_base.sh /tmp/ && pkexec bash /tmp/01_base.sh
cp setup/02_fix_clover_libclc.sh /tmp/ && pkexec bash /tmp/02_fix_clover_libclc.sh
cp setup/03_rocm_opencl.sh /tmp/ && pkexec bash /tmp/03_rocm_opencl.sh
```

| Script | O que faz |
|---|---|
| `01_base.sh` | `build-essential`, `cmake`, headers OpenCL, loader ICD, `clinfo`, Mesa Clover, PoCL |
| `02_fix_clover_libclc.sh` | Corrige por symlink o bug de caminho do libclc no `mesa-opencl-icd` do focal |
| `03_rocm_opencl.sh` | Runtime OpenCL do ROCm 5.2 (só `rocm-opencl` + `rocminfo`, ~500 MB) |

Depois é preciso **logout/login** para o grupo `render` valer na sessão gráfica.
Antes disso, prefixe os comandos com `sg render -c '...'`.

## Rodando os experimentos

```bash
source setup/env_rocm.sh
g++ -Wall -O2 experimentos/matmul.cpp -o output/matmul -lOpenCL
./output/matmul
```

A variável **`ROC_ENABLE_PRE_VEGA=1` é obrigatória** — sem ela o ROCm 5.2 enumera a
plataforma AMD com zero dispositivos, porque Polaris foi depreciado.

## Runtimes OpenCL disponíveis

| Plataforma | Versão | Dispositivo | Observações |
|---|---|---|---|
| **AMD APP (ROCm 5.2)** | OpenCL 2.1 | `gfx803` — 36 CU, 1350 MHz, 8 GB, cache 16 KB, LDS 64 KB | Runtime principal. Exige `ROC_ENABLE_PRE_VEGA=1`. Reporta L1 e LDS corretamente. |
| **Clover (Mesa 21.2.6)** | OpenCL 1.1 | Radeon RX 580 (POLARIS10, LLVM 12) | Segundo runtime LLVM, útil para comparação. Rejeita `-cl-std=CL1.2`; reporta LDS errado (32 KB) e cache 0. |
| **PoCL 1.4** | OpenCL 1.2 | CPU (Core Ultra 5 245KF) | Só controle/sanidade — não é GPU. |

## `setup/asm_probe.cpp`

Sonda que percorre todas as plataformas e tenta compilar um kernel com
`__asm__ volatile("flat_load_dword %0, %1 glc")`. Build:

```bash
g++ -Wall -O2 setup/asm_probe.cpp -o output/asm_probe -lOpenCL
source setup/env_rocm.sh && ./output/asm_probe
```

**Resultado:** o `__asm__ glc` **compila** no ROCm e no Clover — ao contrário do driver
Windows (AMDOCL), que retorna `use of undeclared identifier '__asm__'`.

> Ressalva metodológica: compilar não prova GLC=1. É obrigatório confirmar o encoding
> `DC510000` no ISA gerado pelo ROCm (via `clGetProgramInfo(CL_PROGRAM_BINARIES)`) antes
> de afirmar qualquer coisa no artigo. O PoCL também "compilou" o mesmo kernel num alvo
> x86, o que mostra que sucesso de compilação isoladamente não valida nada.

## Armadilhas conhecidas

- **Clover, erro `-11` com build log vazio**: quase sempre flag de build inválida
  (`-cl-std=CL1.2` num runtime OpenCL 1.1) ou libclc ausente.
- **`pkexec` com caminho relativo**: falha com exit 127.
- **Disco**: só ~6 GB livres em `/`. Nunca instalar o meta-pacote `rocm` (>20 GB).
