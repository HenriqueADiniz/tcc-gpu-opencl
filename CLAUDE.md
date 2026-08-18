# Contexto do Projeto TCC - Análise de Hierarquia de Memória em GPU AMD

## Sobre mim e o trabalho

Sou estudante de Ciência da Computação na PUC Minas, atualmente cursando TCC I. Meu trabalho de conclusão de curso tem o seguinte tema:

**"Análise do impacto da hierarquia de memória no desempenho de algoritmos paralelos executados em GPU AMD utilizando OpenCL em ambiente Windows"**

Estou no 3º Ponto de Controle do TCC I. O artigo deve seguir o formato SBC (Sociedade Brasileira de Computação), entre 10 e 16 páginas, escrito em LaTeX.

## Hardware e ambiente

- **GPU**: AMD Ellesmere (RX 470/480/570/580) — arquitetura Polaris/GCN 4ª geração (GFX8/gfx803)
  - Cache L1: 16 KB por CU (vL1, write-through)
  - Cache L2: 2 MB compartilhado
  - LDS (Local Data Share): 64 KB por CU
  - VRAM: GDDR5, bandwidth teórica ~224-256 GB/s
- **Sistema**: Windows 10/11
- **Framework**: OpenCL 1.2 (driver AMD Adrenalin)
- **SDK**: OCL SDK Light em `C:\Program Files (x86)\OCL_SDK_Light`
- **Compilador host**: MinGW GCC (g++)
- **IDE**: VS Code

## Comando de compilação que funciona

```
g++ -Wall -O2 -D__USE_MINGW_ANSI_STDIO=1 "-Wl,--enable-stdcall-fixup" -I "C:\Program Files (x86)\OCL_SDK_Light\include" arquivo.cpp -o output\saida.exe "C:\Windows\System32\OpenCL.dll"
```

**Pontos críticos da compilação**:
- Linkar diretamente contra `C:\Windows\System32\OpenCL.dll`, NÃO contra o `.lib` do SDK (formato MSVC incompatível com MinGW)
- `-D__USE_MINGW_ANSI_STDIO=1` para suportar `%zu` no printf
- `"-Wl,--enable-stdcall-fixup"` precisa de aspas no PowerShell
- Definir `#define CL_USE_DEPRECATED_OPENCL_1_2_APIS` antes do include OpenCL para suprimir warnings de `clCreateCommandQueue`

## O que já foi feito (percurso científico)

### Etapa 1 - Tentativa com volatile
Hipótese inicial: usar `volatile` nos ponteiros do kernel OpenCL forçaria o compilador a gerar instruções com bypass de cache via flags GLC/SLC.

Implementei `matmul.cpp` com 4 kernels para multiplicação de matrizes 1024x1024:
- Kernel naive (via L1)
- Kernel com volatile (suposto bypass para L2)
- Kernel com volatile mais agressivo (suposto VRAM)
- Kernel tiled com __local memory (LDS)

**Resultado**: kernels 1, 2 e 3 tiveram tempo idêntico (~6.2ms, ~346 GFLOPS). Apenas o tiled foi 3x mais rápido (2.07ms, 1038 GFLOPS).

### Etapa 2 - Análise do binário ISA
Extraí o binário compilado pelo driver via `clGetProgramInfo(CL_PROGRAM_BINARIES)`, gerando `matmul_kernel_dump.bin`. Fiz parse manual:
- ELF externo (32-bit) → ELF interno (64-bit, HSA Code Object)
- Localização da seção `.hsatext`
- Decodificação bit-a-bit das instruções MUBUF (`buffer_load_dword`)
- Análise dos campos GLC (bit 13) e SLC (bit 17)

**Resultado**: TODOS os kernels foram gerados com GLC=0 e SLC=0. Os kernels "via L2" e "VRAM" são IDÊNTICOS byte-a-byte ao kernel normal. Prova que o compilador OpenCL da AMD no GFX8/Polaris **ignora** o qualificador `volatile` para geração de flags de cache.

### Etapa 3 - Validação cruzada com RGA
RGA (Radeon GPU Analyzer) não suporta Polaris (lista começa em GFX900/Vega). Compilei os mesmos kernels para GFX950 (MI300) como referência.

**Resultado**: na arquitetura GFX9+, o `volatile` GERA flags `sc0+sc1` (equivalentes modernas ao GLC+SLC). Confirma que o comportamento é específico da geração da GPU.

### Etapa 4 - Tentativas alternativas (todas falharam)

1. **Inline assembly (`__asm__`)**: não compila no OpenCL do Windows. Erro: `use of undeclared identifier '__asm__'`. Só funciona via HIP/ROCm.

2. **Binary patching**: implementei `patch_cache.cpp` que edita os bits GLC/SLC no binário. O driver carrega mas crasha ao executar (provavelmente por inconsistência de metadados HSA).

3. **HIP SDK Windows**: não suporta Polaris. GPU mínima é RDNA2 (gfx1030).

### Etapa 5 - Nova abordagem (sucesso): mem_hierarchy.cpp
Em vez de forçar bypass de cache, implementei benchmarks que demonstram o impacto da hierarquia via padrões de acesso. 

**Parte A - Benchmarks de redução** (16 MB, 262.144 threads, 128 iterações):
- `reduce_sequential` - acesso contíguo → 0.35 ms, 383 GB/s
- `reduce_local` (LDS) - 0.34 ms, 396 GB/s
- `reduce_stride` (salto 67) - 17.94 ms, 7.5 GB/s
- `reduce_random` (xorshift) - 18.03 ms, 7.4 GB/s
- **Diferença de 51x** entre cache-friendly e cache-hostile

**Parte B - Stencil 2D blur 3x3** (4096x4096):
- `stencil_global` - 0.71 ms, ~949 GB/s efetivos
- `stencil_local` - 0.85 ms, ~794 GB/s
- L1 é tão eficiente para padrões espaciais que LDS não compensa o overhead

## Limitação metodológica importante (a ser tratada)

**Não tenho garantia direta** de que os acessos stride/aleatórios estão "pulando" o cache. O que tenho é evidência indireta via bandwidth efetiva. A análise ISA mostrou que TODAS as instruções têm GLC=0/SLC=0, então tecnicamente todos os acessos passam pelo L1/L2 — apenas o cache não consegue ajudar quando não há localidade.

**Linguagem correta a usar**: "padrão cache-hostile", "baixa localidade espacial/temporal", "cache miss frequente inferido pela bandwidth efetiva". Evitar termos como "bypass do L1/L2" que sugerem controle direto.

**Possível adição futura**: implementar microbenchmark P-chase (pointer chasing, técnica de Wong et al. 2010 e Mei & Chu 2016) para mapear experimentalmente os tamanhos do L1 e L2 da Ellesmere e provar diretamente quando cada nível é usado.

## Estrutura de arquivos do projeto

```
projeto/
├── experimentos/              # Código dos experimentos que funcionaram
│   ├── matmul.cpp             # 4 kernels matmul (etapa 1)
│   ├── mem_hierarchy.cpp      # Benchmarks de padrões de acesso (etapa 5)
│   └── pchase.cpp             # P-chase microbenchmark (etapa 6)
├── tentativas/                # Código de abordagens que falharam (etapa 4)
│   ├── asm_test.cpp           # Inline asm (não suportado no OpenCL Windows)
│   ├── patch_cache.cpp        # Binary patching (crasha na execução)
│   └── debug_patch.cpp        # Versão debug do binary patching
├── isa/                       # Binários ISA e análises de compilador
│   ├── matmul_kernel_dump.bin # Binário ISA extraído do driver
│   ├── kernel_dump.bin        # Binário ISA do main.cpp original
│   ├── kernels.cl             # Kernels standalone para análise RGA
│   ├── gfx950_*.txt           # Saídas RGA para GFX950 (validação cruzada)
│   ├── debug_original.bin     # Binário original (debug)
│   └── debug_patched.bin      # Binário patchado (debug)
├── output/                    # Executáveis compilados
├── resultados/                # Saídas dos benchmarks
│   ├── resultado.txt          # Resultados do matmul
│   ├── resultado_patch.txt    # Resultados do binary patching
│   └── debug_log.txt          # Log do debug_patch
├── Documentação/              # Artigo LaTeX (formato SBC)
│   ├── main.tex               # Artigo principal
│   ├── sbc-template.bib       # 12 referências bibliográficas
│   ├── sbc-template.sty       # Estilo SBC
│   └── sbc.bst                # Estilo de bibliografia SBC
├── CLAUDE.md                  # Contexto do projeto
└── Regulamento_TCC_CC (1).pdf # Regulamento do TCC
```

## Bibliografia (12 referências no projeto)

1. Fang et al. (2011) - "A Comprehensive Performance Comparison of CUDA and OpenCL" - IEEE ICPP
2. Seo et al. (2011) - "Performance Characterization of NAS Parallel Benchmarks in OpenCL" - IEEE IISWC
3. Bernal et al. (2012) - "Stochastic Memory Hierarchy Model" - ACM PPoPP
4. Vidye et al. (2023) - "OpenCL Performance Evaluation on Multiple OSs" - IJSRCSEIT
5. Fraire et al. (2013) - "OpenCL Overview, Implementation, and Performance Comparison" - IEEE Latin America
6. Memeti et al. (2017) - "Benchmarking OpenCL/OpenACC/OpenMP/CUDA" - ACM ARMS-CC
7. Balen et al. (2022) - "Performance Evaluation of Linux Operating Systems" - IEEE SoftCOM
8. Gupta et al. (1991) - "Impact of OS Scheduling Policies on Parallel Applications" - ACM SIGMETRICS
9. Chen & Bershad (1993) - "Impact of OS Structure on Memory System Performance" - ACM SOSP
10. Wong et al. (2010) - "Demystifying GPU Microarchitecture through Microbenchmarking" - IEEE ISPASS
11. Mei & Chu (2017) - "Dissecting GPU Memory Hierarchy through Microbenchmarking" - IEEE TPDS
12. Jia et al. (2018) - "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" - arXiv

## Cronograma do TCC I

- **2º PC (concluído)**: introdução, motivações, problema, objetivos
- **3º PC (06/05, atual)**: + revisão de literatura com análise de 10 trabalhos correlatos
- **4º PC (03/06)**: + metodologia + resultados esperados + cronograma
- **6º PC (24/06)**: + resultados preliminares

## O que pode me ajudar a fazer

- Implementar novos benchmarks OpenCL (FFT, redução, ordenação, P-chase)
- Refatorar/melhorar código existente
- Analisar binários ISA AMD GCN
- Escrever seções do artigo LaTeX
- Gerar tabelas e gráficos de resultados
- Resolver problemas de compilação MinGW/OpenCL
- Discutir metodologia experimental
- Sugerir próximos experimentos cientificamente relevantes

## Diretrizes importantes

1. **Sempre usar linguagem científica precisa** sobre comportamento de cache (sem afirmar "bypass" sem prova direta)
2. **Manter compatibilidade com Ellesmere/GFX8** — não sugerir HIP, RDNA, ou recursos não disponíveis
3. **Testar no Windows com MinGW** — não assumir Linux ou MSVC
4. **Seguir formato SBC** para qualquer texto destinado ao artigo
5. **Documentar resultados quantitativamente** com tabelas claras (tempo, bandwidth, speedup)
6. **Citar a literatura existente** quando fizer sentido (especialmente Mei & Chu 2016, Wong et al. 2010 para metodologia de microbenchmarking)

Por favor, leia este contexto antes de começar qualquer tarefa. Se eu pedir algo que entre em conflito com a metodologia ou que não faça sentido para a Ellesmere/Windows, me avise antes de implementar.
