# Contexto para o Agente no Linux — TCC: Hierarquia de Memória em GPU AMD

## Quem sou e o que estou fazendo

Sou estudante de Ciência da Computação na PUC Minas, no TCC II. Meu trabalho:

**"Análise do Impacto da Hierarquia de Memória em GPU AMD Utilizando OpenCL"**

Artigo em LaTeX formato SBC, 10-16 páginas. Já tenho resultados preliminares de 4 experimentos feitos no Windows. Agora estou no Linux para o **Experimento 5: verificação multiplataforma do inline assembly**.

---

## Por que estou no Linux

No Windows, o compilador OpenCL do driver AMD (AMDOCL) é **proprietário** e **não reconhece `__asm__`** em kernels OpenCL:

```
error: use of undeclared identifier '__asm__'
```

Porém, na compilação **offline** via RGA (Radeon GPU Analyzer, que usa Clang/LLVM), o mesmo kernel com `__asm__` compila e gera a instrução correta:

```
flat_load_dword v5, v[5:6] glc   ← GLC=1 (encoding DC510000)
```

O GLC=1 faz bypass do cache L1 — é exatamente o que preciso para medir o efeito direto do cache.

**O problema**: o RGA só compila offline, não executa na GPU. Preciso de um compilador OpenCL baseado em **Clang/LLVM** que rode em **runtime** para executar esse kernel na GPU real. No Linux, o **ROCm OpenCL** usa Clang/LLVM e deve suportar `__asm__`.

---

## Hardware

- **GPU**: AMD Ellesmere RX 580 8GB (Polaris, GCN 4ª geração, **GFX8/gfx803**)
  - 36 Compute Units, 1350 MHz
  - L1: 16 KB/CU (write-through) | L2: 2 MB compartilhado | LDS: 64 KB/CU
  - VRAM: 8 GB GDDR5
- **Placa-mãe**: nova (trocada recentemente), verificar suporte IOMMU se necessário
- **SO**: Ubuntu 22.04 LTS (dual boot com Windows 11)
- **Objetivo de software**: ROCm 5.2 (última versão com suporte a gfx803)

---

## O que preciso fazer neste ambiente Linux

### Passo 1: Verificar se a GPU é reconhecida
```bash
lspci | grep -i amd
sudo dmesg | grep -i amdgpu
```

### Passo 2: Instalar ROCm 5.2
Seguir a documentação AMD para Ubuntu 22.04. O meta-pacote `rocm` inclui o compilador OpenCL Clang/LLVM. Adicionar o usuário aos grupos `render` e `video`.

### Passo 3: Verificar OpenCL
```bash
clinfo
# Deve mostrar a RX 580 / Ellesmere como dispositivo OpenCL
```

### Passo 4: Compilar e rodar os kernels
O código principal está em `experimentos/matmul.cpp`. O kernel crítico é o `matmul_glc_asm`:

```c
__asm__ volatile(
    "flat_load_dword %0, %1 glc\n"
    "s_waitcnt vmcnt(0)\n"
    : "=v"(a) : "v"(pa));
```

Compilação no Linux (ajustar paths conforme instalação do ROCm):
```bash
g++ -Wall -O2 -I /opt/rocm/include experimentos/matmul.cpp -o matmul -lOpenCL
```

### Passo 5: Comparar resultados

O que espero observar:

| Kernel | Windows (AMDOCL) | Linux (ROCm) |
|--------|-----------------|---------------|
| matmul_via_L1 (normal) | 6.20 ms | ? |
| matmul_via_L2 (volatile) | 6.11 ms | ? |
| matmul_glc_asm (__asm__ glc) | **NÃO COMPILA** | **DEVE COMPILAR** |
| matmul_tiled (LDS) | 2.07 ms | ? |

Se o `matmul_glc_asm` compilar e rodar:
- **Caso A**: tempo diferente do matmul_via_L1 → prova direta de que GLC=1 afeta o desempenho (bypass de L1 confirmado)
- **Caso B**: tempo igual → o bypass de L1 não tem impacto mensurável neste algoritmo (matmul é compute-bound, não memory-bound para esta dimensão)

Ambos os resultados são publicáveis e interessantes.

### Passo 6: Extrair e comparar o ISA
```bash
# Extrair o binário ISA compilado pelo ROCm
# (o matmul.cpp já tem código para dump via clGetProgramInfo)
# Comparar com o binário gerado pelo driver Windows
```

Verificar se o ROCm gera `DC510000` (GLC=1) para o kernel com `__asm__ glc`, e `DC500000` (GLC=0) para os demais — confirmando que o inline assembly funciona em runtime.

---

## Arquivos do projeto

```
projeto/
├── experimentos/
│   ├── matmul.cpp          ← 4 kernels matmul (PRINCIPAL para Exp 5)
│   ├── mem_hierarchy.cpp   ← benchmarks de padrões de acesso (Exp 3)
│   └── pchase.cpp          ← P-chase microbenchmark (Exp 4)
├── isa/
│   ├── kernels.cl          ← kernels standalone (inclui matmul_glc_asm)
│   ├── matmul_kernel_dump.bin  ← ISA do Windows (para comparação)
│   └── clrx_disassembly.txt   ← disassembly do binário Windows
├── Documentação/
│   ├── main.tex            ← artigo LaTeX (formato SBC)
│   ├── analise_isa.md      ← documentação da análise ISA
│   └── apresentacao_orientador.md
├── CLAUDE.md               ← contexto completo do projeto
└── CONTEXTO_LINUX.md       ← este arquivo
```

---

## Resultados já obtidos no Windows (referência)

### Matmul 1024×1024
| Kernel | Tempo | GFLOPS |
|--------|-------|--------|
| Naive (normal) | 6.20 ms | 346 |
| volatile | 6.11 ms | 351 |
| volatile agressivo | 6.21 ms | 346 |
| Tiled LDS | **2.07 ms** | **1038** |

### Análise ISA — volatile por target (RGA offline)
| Target | Sem volatile | Com volatile |
|--------|-------------|-------------|
| gfx803 (Polaris) | sem flags | **sem flags** |
| gfx900 (Vega) | sem flags | **glc** |
| gfx950 (CDNA3) | sem flags | **sc0 sc1** |

### Padrões de acesso (mem_hierarchy.cpp)
| Padrão | Bandwidth |
|--------|-----------|
| Sequencial | 383 GB/s |
| Aleatório | 7.4 GB/s |
| **Diferença** | **51×** |

### P-chase
- L1 confirmado: 16 KB (salto de latência em 20 KB)
- L2 confirmado: ~2 MB (salto com 36 CUs simultâneos)

---

## Achados-chave da análise ISA

1. **`volatile` no GFX8**: atua como barreira de otimização (impede reuso de registradores, 10→11 VGPRs), mas **NÃO gera flags GLC/SLC**. Encoding: `DC500000` (GLC=0) com e sem volatile.

2. **`__asm__` com `glc`**: é o **único mecanismo** que gera GLC=1 no GFX8. Encoding confirmado: `DC510000`. Mas só funciona em compiladores Clang/LLVM, não no AMDOCL proprietário do Windows.

3. **O mapeamento volatile→ISA é decisão de codegen do backend LLVM**, varia por target. Não é propriedade do hardware diretamente. Foram usadas versões diferentes do RGA para gfx803 vs gfx900/gfx950, o que é uma limitação metodológica (versões diferentes do LLVM).

---

## Comando de compilação no Windows (referência)

```bash
g++ -Wall -O2 -D__USE_MINGW_ANSI_STDIO=1 "-Wl,--enable-stdcall-fixup" \
    -I "C:\Program Files (x86)\OCL_SDK_Light\include" \
    arquivo.cpp -o output/saida.exe "C:\Windows\System32\OpenCL.dll"
```

No Linux será diferente — usar os headers e biblioteca do ROCm.

---

## Possíveis problemas no Linux

1. **ROCm 5.2 pode não suportar gfx803 oficialmente**: Polaris foi deprecated. Se `clinfo` não mostrar a GPU, tentar ROCm 4.5 ou o driver AMDGPU-PRO.

2. **Se ROCm não funcionar, alternativas**:
   - **AMDGPU-PRO** (driver proprietário Linux): suporta Polaris, mas usa compilador proprietário — pode ter o mesmo problema do Windows com `__asm__`
   - **Mesa Rusticl**: implementação OpenCL open-source baseada em LLVM, suporta Polaris nativamente em kernels recentes do Linux. Pode ser o caminho mais viável para `__asm__`.

3. **Se `__asm__` não funcionar em nenhum compilador runtime**: o resultado ainda é publicável — documenta-se que o inline assembly GCN é exclusivo de compilação offline (RGA/HIP) e não acessível via OpenCL runtime em nenhuma plataforma para GFX8.

---

## Diretrizes para o agente

1. **Linguagem científica precisa**: "barreira de otimização", "decisão de codegen do backend LLVM", "cache miss inferido pela bandwidth efetiva". Evitar "o compilador ignora volatile" ou "bypass do cache" sem evidência direta.
2. **Compatibilidade GFX8/gfx803**: não sugerir recursos de RDNA, GFX9+, ou HIP.
3. **Documentar tudo quantitativamente**: tempos, bandwidth, encodings hex.
4. **Comparar com os resultados do Windows**: o valor está na comparação multiplataforma.
5. **Se algo falhar, documentar o porquê**: resultados negativos são resultados válidos.
