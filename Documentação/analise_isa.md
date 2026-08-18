# Análise ISA — Efeito do `volatile` nas Instruções de Cache
## Evidência Experimental Completa

---

## Objetivo

Verificar se o qualificador `volatile` nos ponteiros de kernel OpenCL altera as instruções de acesso à memória global no nível da ISA (Instruction Set Architecture), especificamente as flags de controle de cache.

---

## Metodologia

### 1. Extração do binário compilado (GFX8 — RX 580)

O driver AMD Adrenalin compila os kernels OpenCL em tempo de execução via `clBuildProgram()`. O binário resultante foi extraído usando a API OpenCL:

```c
clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, ...);
clGetProgramInfo(program, CL_PROGRAM_BINARIES, ...);
```

O ficheiro resultante (`matmul_kernel_dump.bin`, 39 KB) contém os 4 kernels compilados para a GPU.

### 2. Estrutura do binário

```
matmul_kernel_dump.bin
  └─ ELF externo (32-bit) — container do driver AMD
       └─ ELF interno (64-bit) — HSA Code Object
            ├─ .hsatext — código de máquina GPU (3232 bytes)
            ├─ .symtab  — tabela de símbolos (nomes dos kernels)
            └─ .note    — metadados HSA
```

### 3. Ferramentas de análise

| Ferramenta | Tipo | O que faz |
|-----------|------|-----------|
| **isa_decoder.cpp** | Programa nosso | Parseia o ELF, decodifica instruções MUBUF, extrai bits GLC/SLC |
| **CLRadeonExtender (clrxdisasm)** | Ferramenta open-source de terceiros | Disassembly completo do binário AMD OpenCL |
| **Radeon GPU Analyzer (RGA)** | Ferramenta oficial da AMD | Compilação offline para múltiplos targets (GFX9+) |

---

## Kernels Testados

Mesmo algoritmo (multiplicação de matrizes 1024×1024), diferindo apenas no uso de `volatile`:

```c
/* Kernel 1 — sem volatile (acesso normal via cache) */
__kernel void matmul_via_L1(
    __global const float* restrict A, ...)

/* Kernel 2 — com volatile (suposto bypass de L1) */
__kernel void matmul_via_L2(
    __global volatile const float* A, ...)

/* Kernel 3 — com volatile (suposto bypass de L1+L2) */
__kernel void matmul_direto_VRAM(
    __global volatile const float* A, ...)
```

---

## Resultado 1 — GFX8 (RX 580, driver Adrenalin, Windows)

### Formato das instruções de memória no GFX8

As leituras de memória global usam instruções **MUBUF** (`buffer_load_dword`). O formato binário possui dois campos de controle de cache:

- **GLC** (bit 14) — Globally Coherent: se 1, bypass do L1
- **SLC** (bit 17) — System Level Coherent: se 1, bypass do L2

### Saída do clrxdisasm (CLRadeonExtender)

**`matmul_via_L1`** (sem volatile):
```asm
buffer_load_dword v5, v5, s[8:11], 0 offen    ← sem flags
buffer_load_dword v6, v6, s[8:11], 0 offen    ← sem flags
buffer_store_dword v2, v0, s[8:11], 0 offen   ← sem flags
```

**`matmul_via_L2`** (com volatile):
```asm
buffer_load_dword v5, v5, s[8:11], 0 offen    ← sem flags
buffer_load_dword v6, v6, s[8:11], 0 offen    ← sem flags
buffer_store_dword v2, v0, s[8:11], 0 offen   ← sem flags
```

**`matmul_direto_VRAM`** (com volatile):
```asm
buffer_load_dword v5, v5, s[8:11], 0 offen    ← sem flags
buffer_load_dword v6, v6, s[8:11], 0 offen    ← sem flags
buffer_store_dword v2, v0, s[8:11], 0 offen   ← sem flags
```

### Saída do isa_decoder (programa nosso)

```
Kernel: matmul_via_L1       → GLC=0 SLC=0 (cache normal)
Kernel: matmul_via_L2       → GLC=0 SLC=0 (cache normal)
Kernel: matmul_direto_VRAM  → GLC=0 SLC=0 (cache normal)

CONCLUSAO: TODAS as instrucoes usam cache normal (GLC=0, SLC=0).
O compilador NAO gerou flags de bypass, mesmo com volatile.
```

### Confirmação por hex (driver Adrenalin, MUBUF)

Os bytes das instruções de load são idênticos nos 3 kernels:
```
e0501000 80020505   ← buffer_load_dword, GLC=0, SLC=0
```

### Confirmação por hex (RGA offline, FLAT)

```
via_L1:       DC500000 07000007   ← flat_load_dword, GLC=0
via_L2:       DC500000 0A000003   ← flat_load_dword, GLC=0 (IDÊNTICO)
glc_asm:      DC510000 05000005   ← flat_load_dword, GLC=1 (inline asm)
```

A diferença está no primeiro dword: `DC50` (GLC=0) vs `DC51` (GLC=1). O bit GLC é visível diretamente no encoding.

### Compreensão refinada do `volatile`

O `volatile` **não é ignorado pelo compilador** — ele atua no **modelo de memória da linguagem**:
- Impede o reuso de valores em registradores (VGPR allocation diferente: 10 vs 11 VGPRs)
- Impede a fusão e reordenação de loads pelo otimizador
- **Mas não emite flags de cache (GLC/SLC)** — o caminho de memória é o mesmo

O efeito do volatile é de **inibição de otimização**, não de controle de microarquitetura. Isto explica porque os tempos de execução são quase iguais: o overhead de um registrador adicional é insignificante comparado ao custo do compute.

### Inline asm: o único caminho para GLC=1 no GFX8

O kernel `matmul_glc_asm` usa inline assembly com o modificador `glc` explícito:
```c
__asm__ volatile(
    "flat_load_dword %0, %1 glc\n"
    "s_waitcnt vmcnt(0)\n"
    : "=v"(a) : "v"(pa));
```

Resultado no ISA: `DC510000` — **GLC=1 confirmado**.

**Ressalva**: o `s_waitcnt vmcnt(0)` após cada load serializa os acessos e impede sobreposição de latência. Este kernel é mais lento por construção (serialização), não necessariamente pelo GLC=1. Para uma comparação justa de performance L1 vs bypass, seria necessário remover os waitcnt e deixar o compilador agendar.

---

## Resultado 2 — Validação cruzada com RGA (outras gerações)

Os mesmos kernels foram compilados via Radeon GPU Analyzer para GFX9 (Vega) e GFX9.5+ (CDNA3), usando o ficheiro `kernels.cl`.

### GFX9 — gfx900 (Vega)

**Sem volatile** (`matmul_via_L1`):
```asm
global_load_dword  v10, v[1:2], off         ← sem flags
```

**Com volatile** (`matmul_via_L2` e `matmul_direto_VRAM`):
```asm
global_load_dword  v10, v[1:2], off glc     ← flag glc presente
```

O volatile gera **`glc` apenas** — bypass do L1, mantém o L2.

### GFX9.5+ — gfx950 (MI300, CDNA3)

**Sem volatile** (`matmul_via_L1`):
```asm
global_load_dword  v1, v[4:5], off          ← sem flags
```

**Com volatile** (`matmul_via_L2` e `matmul_direto_VRAM`):
```asm
global_load_dword  v7, v[2:3], off sc0 sc1  ← ambas as flags
```

O volatile gera **`sc0 sc1`** — bypass do L1 e do L2.

---

## Tabela Comparativa — 3 Gerações × 2 Compiladores

### Por compilador e target (com volatile)

| Target | Driver Adrenalin (runtime) | RGA (offline) |
|--------|--------------------------|---------------|
| **gfx803** (Polaris) | `buffer_load_dword` — sem flags | `flat_load_dword` — **sem flags** |
| **gfx900** (Vega) | — (sem hardware) | `global_load_dword` **glc** |
| **gfx950** (CDNA3) | — (sem hardware) | `global_load_dword` **sc0 sc1** |

### Resumo por geração

| Geração | Target | Sem volatile | Com volatile | Efeito do volatile |
|---------|--------|-------------|-------------|-------------------|
| **GFX8** (Polaris) | gfx803 | — | — | **Ignorado** |
| **GFX9** (Vega) | gfx900 | — | `glc` | Bypass L1 |
| **GFX9.5+** (CDNA3) | gfx950 | — | `sc0 sc1` | Bypass L1 + L2 |

Onde `—` = sem flags de cache (acesso normal via L1 → L2 → VRAM).

### O que foi isolado (e o que não foi)

O teste do gfx803 via RGA (compilador offline) **elimina o driver Windows e o SO** como causa: o compilador offline também não gera flags para esse target, descartando a hipótese de ser um comportamento específico do ambiente de execução runtime.

**Porém**, o mapeamento `volatile` → ISA é uma **decisão de codegen do backend LLVM**, não uma propriedade diretamente observável do hardware. O compilador escolhe diferentes estratégias para realizar a semântica `volatile` conforme o modelo de coerência de cada target ISA. Assim, as diferenças observadas entre targets refletem a lógica do backend, que pode variar entre versões do LLVM.

**Limitação importante**: como o target gfx803 foi compilado com uma **versão anterior do RGA** (que ainda incluía suporte a Polaris) e os targets gfx900/gfx950 com a **versão atual**, as versões do backend LLVM diferem. Isso introduz uma variável confundidora: não é possível atribuir as diferenças exclusivamente à arquitetura. Para um isolamento completo, seria necessário compilar todos os targets com a mesma versão do LLVM.

### Observações adicionais

- Em todas as gerações, `matmul_via_L2` e `matmul_direto_VRAM` geram ISA **idêntica** — o OpenCL não permite controlar os níveis de cache separadamente
- A evolução das flags reflete a mudança de formato ISA: GFX8 usa MUBUF/FLAT (`buffer_load_dword`/`flat_load_dword`), enquanto GFX9+ usa instruções GLOBAL (`global_load_dword` com flags glc/sc0/sc1)
- O GFX8 foi verificado com **4 métodos**: medição de performance, análise bit a bit (isa_decoder), CLRadeonExtender, e RGA offline

---

## Confirmação via Performance (Etapa 1)

Os tempos de execução na RX 580 corroboram a análise ISA:

| Kernel | Tempo médio | GFLOPS |
|--------|-------------|--------|
| `matmul_via_L1` (normal) | 6.20 ms | 346 |
| `matmul_via_L2` (volatile) | 6.11 ms | 351 |
| `matmul_direto_VRAM` (volatile) | 6.21 ms | 346 |
| `matmul_tiled` (LDS) | **2.07 ms** | **1038** |

Os 3 primeiros kernels têm performance **indistinguível** (~6.2 ms), consistente com binários idênticos. Apenas o tiled (que usa LDS, não volatile) mostrou diferença.

---

## Conclusões

1. **O `volatile` em OpenCL não controla a hierarquia de cache no GFX8** — atua no modelo de memória da linguagem (impede reuso de registradores, fusão e reordenação de loads), mas não emite flags GLC/SLC. Verificado por encoding hex: `DC500000` (GLC=0) tanto com como sem volatile.

2. **O driver Windows e o SO não são a causa** — o compilador offline (RGA) também não gera flags para gfx803, eliminando o ambiente de execução como variável.

3. **O mapeamento `volatile` → ISA é uma decisão de codegen do backend LLVM** e varia por target: sem flags (GFX8/gfx803), `glc` (GFX9/gfx900), `sc0 sc1` (GFX9.5+/gfx950). No GFX9+, a forma de garantir a semântica volatile ("sempre tocar a memória, sem reuso") passou a incluir flags de coerência; no GFX8, o backend não emite essas flags.
   - **Limitação**: foram usadas versões diferentes do RGA (e portanto do LLVM) para gfx803 e gfx900/gfx950, o que impede atribuir as diferenças exclusivamente à arquitetura.

4. **O único mecanismo funcional para GLC=1 no GFX8 é inline assembly** com o modificador `glc` explícito (`flat_load_dword %0, %1 glc`), confirmado pelo encoding `DC510000`.

5. **A variação do codegen por target não é documentada pela AMD** — constitui um achado verificável do presente trabalho, embora deva ser apresentada como comportamento observado do compilador e não como propriedade intrínseca do hardware.

---

## Resultado 3 — GFX8 via RGA (prova de isolamento)

Utilizando uma versão anterior do RGA com suporte ao target gfx803, os mesmos kernels foram compilados **offline** para a arquitetura Polaris:

**`matmul_via_L1`** (sem volatile):
```asm
flat_load_dword  v7,  v[7:8]      ← sem flags
flat_load_dword  v8,  v[3:4]      ← sem flags
```

**`matmul_via_L2`** (com volatile):
```asm
flat_load_dword  v10, v[3:4]      ← sem flags
flat_load_dword  v7,  v[7:8]      ← sem flags
```

**`matmul_direto_VRAM`** (com volatile):
```asm
flat_load_dword  v10, v[3:4]      ← sem flags
flat_load_dword  v7,  v[7:8]      ← sem flags
```

**`matmul_glc_asm`** (inline asm com `glc` explícito):
```asm
flat_load_dword  v5, v[5:6] glc   ← GLC=1! (encoding DC510000)
flat_load_dword  v6, v[7:8] glc   ← GLC=1! (encoding DC510000)
s_waitcnt vmcnt(0)                 ← serializa (nota: distorce performance)
```

O volatile não gera flags no gfx803 mas gera no gfx900 e gfx950. O inline asm com `glc` explícito é o único mecanismo que força GLC=1 no GFX8.

**Nota sobre versões do RGA**: o target gfx803 foi compilado com uma versão anterior do RGA (que ainda incluía suporte a Polaris); os targets gfx900/gfx950 foram compilados com a versão atual. As versões do backend LLVM diferem entre elas.

---

## Nota Metodológica

A comparação inicial entre o driver Adrenalin (runtime, Windows) e o RGA (offline) envolvia duas variáveis: o ambiente de compilação e a arquitetura alvo. Com a adição do teste via RGA para gfx803, **a variável "driver/SO" foi isolada** — o compilador offline também não gera flags para gfx803, descartando o ambiente de execução como causa.

**Porém, a variável "versão do compilador" não foi completamente isolada**: como foram usadas versões diferentes do RGA para os targets, não é possível separar com certeza o efeito da arquitetura do efeito da evolução do backend LLVM. O que se pode afirmar é que (a) o driver Windows não é a causa, e (b) o mapeamento `volatile` → flags de coerência varia conforme o target no compilador AMD/LLVM, refletindo a evolução conjunta da arquitetura e do compilador.

---

## Suporte a Inline Assembly em OpenCL — Investigação

O kernel `matmul_glc_asm` demonstrou que inline assembly (`__asm__ volatile`) com o modificador `glc` é o único mecanismo capaz de forçar GLC=1 no GFX8. No entanto, este mecanismo **só funciona em compiladores offline** — não é possível executá-lo na GPU real via driver Windows.

### Compatibilidade por ambiente

| Ambiente | `__asm__` em kernel OpenCL | Motivo |
|----------|--------------------------|--------|
| **RGA** (compilador offline AMD) | **Funciona** | Backend Clang/LLVM com suporte nativo a inline asm GCN |
| **ROCm OpenCL** (Linux) | **Provável** | Usa Clang/LLVM, mas ROCm não suporta Polaris (GFX8) |
| **HIP** (Linux/ROCm) | **Funciona** | Caminho recomendado pela AMD para controle de ISA |
| **Driver Adrenalin** (Windows, AMDOCL) | **Não funciona** | Compilador proprietário, não expõe `__asm__` do Clang |

### Erro obtido no driver Windows

```
error: use of undeclared identifier '__asm__'
```

O driver AMD Windows (AMDOCL) utiliza um compilador OpenCL proprietário que não implementa as extensões de inline assembly do Clang. Como o RGA e o ROCm utilizam o Clang/LLVM diretamente, o suporte a `__asm__` existe nesses ambientes mas não no runtime Windows.

Não existe documentação oficial da AMD que explicite esta limitação — é uma consequência de o compilador proprietário do driver não expor funcionalidades que são específicas do frontend Clang.

### Implicação para o trabalho

No ambiente GFX8/OpenCL/Windows, **não existe mecanismo suportado para controlar flags de cache em tempo de execução**:
- `volatile` → não gera GLC/SLC (apenas inibe otimizações de registradores)
- `__asm__` → não compila no driver Windows
- Binary patching → crasha por inconsistência de metadados HSA
- HIP SDK → não suporta Polaris (mínimo: RDNA2)

A única forma de verificar o efeito do GLC=1 foi via compilação offline (RGA), confirmando que a instrução **existe** na ISA do GFX8 mas não é acessível por software no ambiente disponível.

### Fontes

- [AMD Community: Inline assembly on OpenCL](https://community.amd.com/thread/188702)
- [ROCm Issue #201: Inline GCN Assembly in OpenCL Kernel](https://github.com/RadeonOpenCompute/ROCm/issues/201)
- [HIP FAQ — Inline Assembly](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.0/how-to/faq.html)
- [ROCm Compiler Reference — amdclang++](https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html)

---

## Ficheiros e Ferramentas

| Ficheiro | Descrição |
|---------|-----------|
| `isa/matmul_kernel_dump.bin` | Binário ISA extraído do driver (GFX8) |
| `isa/clrx_disassembly.txt` | Disassembly completo via CLRadeonExtender |
| `isa/kernels.cl` | Kernels OpenCL standalone para análise com RGA |
| `isa/gfx950_matmul_via_L1_isa_output.txt` | Saída RGA — GFX950 sem volatile |
| `isa/gfx950_matmul_via_L2_isa_output.txt` | Saída RGA — GFX950 com volatile |
| `isa/gfx950_matmul_direto_VRAM_isa_output.txt` | Saída RGA — GFX950 com volatile |
| `isa/isa_decoder.cpp` | Decodificador de binários ISA (nosso) |
| `tools/clrxdisasm.exe` | CLRadeonExtender compilado |
| `experimentos/matmul.cpp` | Código fonte dos 4 kernels + benchmark |
