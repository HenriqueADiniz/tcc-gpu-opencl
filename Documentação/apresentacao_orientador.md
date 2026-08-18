# Apresentação para Orientador — TCC I (4º Ponto de Controle)
## Análise do Impacto da Hierarquia de Memória em GPU AMD (OpenCL/Windows)
### Henrique de Almeida Diniz — PUC Minas

---

## Ambiente Experimental

- **GPU**: AMD Ellesmere RX 580 8GB (Polaris/GCN 4ª gen, GFX8/gfx803)
  - 36 Compute Units | 1350 MHz
  - L1 vCache: 16 KB/CU (write-through) | L2: 2 MB compartilhado | LDS: 64 KB/CU
  - VRAM: 8 GB GDDR5 (~256 GB/s teóricos)
- **SO**: Windows 11 | **Framework**: OpenCL 1.2 (driver AMD Adrenalin) | **Compilador host**: MinGW GCC

---

## Percurso Experimental (6 etapas)

### Etapa 1 — Hipótese do `volatile` (matmul.cpp)

**Hipótese**: usar `volatile` nos ponteiros OpenCL forçaria o compilador a gerar instruções com bypass de cache (flags GLC/SLC).

4 kernels de multiplicação de matrizes 1024×1024:

| Kernel | Tempo | GFLOPS |
|--------|-------|--------|
| Naive (normal) | 6.20 ms | 346 |
| Com `volatile` (hipótese: bypass L1) | 6.11 ms | 351 |
| Com `volatile` agressivo (hipótese: VRAM) | 6.21 ms | 346 |
| **Tiled com LDS** | **2.07 ms** | **1038** |

**Resultado**: os 3 primeiros são praticamente idênticos (~6.2 ms). Apenas o tiled com LDS fez diferença (**3× mais rápido**). O `volatile` não afetou o desempenho de forma mensurável.

---

### Etapa 2 — Análise do Binário ISA (4 ferramentas independentes)

O binário compilado pelo driver foi extraído via `clGetProgramInfo(CL_PROGRAM_BINARIES)` e analisado com **4 métodos complementares**:

1. **Medição de desempenho** (Etapa 1) — tempos idênticos entre kernels com/sem volatile
2. **Decodificador ISA próprio** (isa_decoder.cpp) — parse da estrutura ELF aninhada (ELF 32-bit → ELF 64-bit HSA Code Object → seção `.hsatext`), decodificação das instruções MUBUF, extração dos campos GLC (bit 14) e SLC (bit 17)
3. **CLRadeonExtender** (clrxdisasm) — disassembler open-source, verificação independente
4. **Radeon GPU Analyzer** (RGA) — compilador offline da AMD para múltiplos targets

**Resultado — todos os 4 métodos confirmam**: GLC=0, SLC=0 em TODOS os kernels, com e sem `volatile`. Binários idênticos byte-a-byte.

**Encoding hexadecimal**:
```
Sem volatile:  DC500000  →  flat_load_dword, GLC=0
Com volatile:  DC500000  →  flat_load_dword, GLC=0 (IDÊNTICO)
Com __asm__ glc: DC510000  →  flat_load_dword, GLC=1 (único mecanismo)
```

#### Compreensão refinada do `volatile`

O `volatile` **não é ignorado** pelo compilador — atua no **modelo de memória da linguagem**:
- Impede o reuso de valores em registradores (10 vs 11 VGPRs)
- Impede a fusão e reordenação de loads pelo otimizador
- **Mas não emite flags de cache (GLC/SLC)** — o caminho de memória na microarquitetura é o mesmo

O efeito do volatile é de **inibição de otimização**, não de controle de microarquitetura.

---

### Etapa 3 — Validação Cruzada entre Gerações (RGA)

Os mesmos kernels foram compilados via RGA para 3 targets diferentes:

| Geração | Target | Sem volatile | Com volatile | Efeito |
|---------|--------|-------------|-------------|--------|
| **GFX8** (Polaris) | gfx803 | sem flags | **sem flags** | Nenhum |
| **GFX9** (Vega) | gfx900 | sem flags | **glc** | Bypass L1 |
| **GFX9.5+** (CDNA3) | gfx950 | sem flags | **sc0 sc1** | Bypass L1+L2 |

#### O que foi isolado (e o que não foi)

Utilizando uma versão anterior do RGA (com suporte a gfx803) e a versão atual (para gfx900/gfx950), observou-se:
- gfx803: volatile → **sem flags**
- gfx900: volatile → **glc**
- gfx950: volatile → **sc0 sc1**

Isto **elimina o driver Windows e o SO** como causa — o compilador offline também não gera flags para gfx803.

**Porém**: o mapeamento `volatile` → ISA é uma **decisão de codegen do backend LLVM**, não uma propriedade diretamente observável do hardware. O compilador escolhe diferentes estratégias para realizar a semântica `volatile` conforme o modelo de coerência de cada target ISA. No GFX9+, a forma de garantir "sempre tocar a memória, sem reuso" passou a incluir flags de coerência; no GFX8, o backend opta por não emitir essas flags.

**Limitação metodológica**: como foram usadas **versões diferentes do RGA** (e portanto versões diferentes do backend LLVM) para gfx803 e gfx900/gfx950, não é possível atribuir as diferenças exclusivamente à arquitetura — a evolução do compilador é uma variável confundidora. Para isolar completamente a variável, seria necessário compilar todos os targets com a mesma versão do LLVM.

#### Variação do codegen por target

O que se pode afirmar com segurança: o mapeamento `volatile` → ISA **varia por target** no compilador AMD/LLVM, passando de nenhuma flag de coerência (GFX8) a `glc` (GFX9) a `sc0 sc1` (GFX9.5+). Essa variação reflete a evolução conjunta da arquitetura e do backend do compilador. Este comportamento específico não é documentado pela AMD.

---

### Etapa 4 — Tentativas Alternativas de Controle de Cache

| Abordagem | Resultado | Motivo |
|-----------|-----------|--------|
| Inline assembly (`__asm__` com `glc`) | Compila no RGA, **não compila no driver Windows** | Driver proprietário AMDOCL não expõe `__asm__` do Clang |
| Binary patching (editar bits GLC/SLC) | Crasha na execução | Metadados HSA ficam inconsistentes |
| HIP SDK Windows | Não disponível para Polaris | GPU mínima: RDNA2 (gfx1030) |

**Achado sobre inline asm**: o `__asm__` com `glc` explícito **funciona no RGA** (confirmado pelo encoding `DC510000` = GLC=1), mas o driver Windows (AMDOCL) retorna `error: use of undeclared identifier '__asm__'`. Isto ocorre porque o RGA usa Clang/LLVM diretamente, enquanto o driver Windows usa um compilador proprietário.

**Conclusão**: não existe caminho suportado para controle direto de cache no GFX8/OpenCL/Windows em tempo de execução. A única forma de verificar o efeito do GLC=1 foi via compilação offline.

---

### Etapa 5 — Padrões de Acesso à Memória (mem_hierarchy.cpp)

Nova abordagem: demonstrar o impacto da hierarquia via **padrões de acesso**, sem necessidade de controle direto de cache.

**Parte A — Benchmarks de redução** (16 MB, 262.144 threads, 128 iterações):

| Padrão | Tempo | Bandwidth | Tipo |
|--------|-------|-----------|------|
| Sequencial (contíguo) | 0.35 ms | 383 GB/s | Cache-friendly |
| LDS (memória local) | 0.34 ms | 396 GB/s | Cache-friendly |
| Stride (salto 67) | 17.94 ms | 7.5 GB/s | Cache-hostile |
| Aleatório (xorshift) | 18.03 ms | 7.4 GB/s | Cache-hostile |

**Diferença de 51× entre cache-friendly e cache-hostile.**

**Parte B — Stencil 2D blur 3×3** (4096×4096):

| Modo | Tempo |
|------|-------|
| Memória global (via L1) | 0.71 ms |
| Memória local (LDS) | 0.85 ms |

L1 é tão eficiente para padrões com alta localidade espacial que o LDS não compensa o overhead de carregamento e sincronização.

---

### Etapa 6 — P-Chase: Mapeamento da Hierarquia (pchase.cpp)

Técnica de Wong et al. (2010) e Mei & Chu (2017): lista encadeada aleatória com working set variável para medir latência por nível de cache.

**P-Chase v1 — Single work-item** (latência pura, sem contenção):

| Working Set | Latência | Nível |
|-------------|----------|-------|
| 1–16 KB | ~39 ns | **L1** |
| 20 KB | ~59 ns (+52%) | L1→L2 |
| 32–512 KB | 86–140 ns | **L2** |
| 1–8 MB | ~137 ns | **VRAM** |

Confirmou L1 = 16 KB. Transição L2→VRAM quase invisível (~2%) com 1 thread.

**P-Chase v2 — Multi work-item** (1 por CU, 36 CUs simultâneos):

| Working Set/CU | Single (ns) | Multi (ns) | Footprint Total |
|-----------------|-------------|------------|-----------------|
| 16 KB | 39 | 127 | 576 KB |
| 64 KB | 108 | 203 | 2.3 MB ≈ L2 |
| 256 KB | 127 | **301** | 9 MB >> L2 |
| 8 MB | 137 | **314** | 288 MB |

Transição L2→VRAM agora clara: **salto de 48%** (203→301 ns) quando o footprint total ultrapassa o L2 de 2 MB. VRAM sob contenção real: **2.3× mais lenta** que com 1 thread.

---

## Achados Principais

1. **O `volatile` em OpenCL atua como barreira de otimização, não como controle de cache** — impede reuso de registradores e reordenação de loads, mas não emite flags GLC/SLC no GFX8. Verificado pelo encoding hex (`DC500000` = GLC=0) com 4 ferramentas independentes.

2. **O driver Windows e o SO não são a causa** — o compilador offline (RGA) também não gera flags para gfx803, eliminando o ambiente de execução como variável.

3. **O mapeamento `volatile` → ISA é decisão de codegen do backend LLVM e varia por target** — sem flags (GFX8), `glc` (GFX9), `sc0 sc1` (GFX9.5+). Reflete a evolução conjunta da arquitetura e do compilador. Como foram usadas versões diferentes do RGA para os targets, a variável "versão do LLVM" não foi completamente isolada. Comportamento não documentado pela AMD.

4. **Inline asm é o único mecanismo para GLC=1 no GFX8** — confirmado por `DC510000`, mas incompatível com o driver Windows (compilador proprietário AMDOCL vs Clang/LLVM do RGA).

5. **L1 = 16 KB e L2 ≈ 2 MB confirmados experimentalmente** via P-chase com saltos de latência mensuráveis.

6. **51× de diferença** entre padrões cache-friendly e cache-hostile em bandwidth efetiva.

7. **LDS nem sempre compensa**: stencil 2D com boa localidade espacial é mais rápido via L1 que via LDS.

8. **Controle direto de cache impossível** no GFX8/OpenCL/Windows — 4 abordagens tentadas, nenhuma viável em runtime.

---

## Estrutura Atual do Artigo (4º PC entregue)

- Introdução completa (problema, relevância, objetivos)
- Referencial Teórico (GPGPU, OpenCL, hierarquia AMD com MUBUF/FLAT, GLC/SLC, sc0/sc1)
- Trabalhos Relacionados (10 trabalhos analisados + tabela comparativa)
- Metodologia detalhada (4 subsecções: matmul, análise ISA com 3 ferramentas + isolamento + mecanismos alternativos, padrões de acesso, P-chase)
- Resultados Esperados (com dados preliminares e evidência de encoding hex)
- Cronograma do TCC II (Ago–Dez)

**Referências**: 12 no total (mínimo exigido: 10)

---

## Próximos Passos (TCC II — Ago a Dez)

1. Refinamento dos experimentos existentes e coleta de dados finais
2. Novos experimentos (FFT, scan paralelo, outros algoritmos)
3. Análise e discussão aprofundada dos resultados
4. Redação das secções de Resultados e Conclusão
5. Revisão geral, formatação final e preparação da defesa

---

## Pontos para Discussão com o Orientador

- A descoberta sobre o `volatile` (barreira de otimização vs controle de cache, com prova por encoding hex e isolamento de variável via RGA) tem valor como contribuição científica?
- A evolução progressiva entre gerações (GFX8→GFX9→GFX9.5+) merece uma análise mais aprofundada?
- O fato de inline asm funcionar no RGA mas não no driver Windows é relevante como contribuição adicional?
- Devo tentar mais algum experimento? (ex: FFT, scan paralelo, diferentes tamanhos de tile)
- Sugestões de referências adicionais, especialmente sobre comportamento de cache em GPUs AMD?
