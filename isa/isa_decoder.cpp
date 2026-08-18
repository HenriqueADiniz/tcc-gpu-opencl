/*
 * isa_decoder.cpp — Decodificador de binarios ISA AMD GCN (GFX8/Polaris)
 *
 * Le um binario extraido via clGetProgramInfo(CL_PROGRAM_BINARIES),
 * parseia a estrutura ELF aninhada, localiza a secao .hsatext,
 * e decodifica as instrucoes MUBUF mostrando as flags GLC e SLC.
 *
 * Compilar:
 *   g++ -Wall -O2 isa_decoder.cpp -o ..\output\isa_decoder.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Estruturas ELF simplificadas ---- */

#pragma pack(push, 1)

struct Elf32_Ehdr {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint32_t e_entry;
    uint32_t e_phoff;
    uint32_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf32_Shdr {
    uint32_t sh_name;
    uint32_t sh_type;
    uint32_t sh_flags;
    uint32_t sh_addr;
    uint32_t sh_offset;
    uint32_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint32_t sh_addralign;
    uint32_t sh_entsize;
};

struct Elf64_Ehdr {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf64_Shdr {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
};

struct Elf64_Sym {
    uint32_t st_name;
    uint8_t  st_info;
    uint8_t  st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
};

#pragma pack(pop)

/* ---- Decodificacao de instrucoes GCN ---- */

struct MUBUFInfo {
    uint32_t offset;
    uint32_t word0;
    uint32_t word1;
    int glc;
    int slc;
    int offen;
    int idxen;
    int lds;
    int op;
    int vaddr;
    int vdata;
    int srsrc;
    int soffset;
};

static const char* mubuf_op_name(int op) {
    switch (op) {
        case 0:  return "buffer_load_format_x";
        case 1:  return "buffer_load_format_xy";
        case 2:  return "buffer_load_format_xyz";
        case 3:  return "buffer_load_format_xyzw";
        case 4:  return "buffer_store_format_x";
        case 5:  return "buffer_store_format_xy";
        case 6:  return "buffer_store_format_xyz";
        case 7:  return "buffer_store_format_xyzw";
        case 8:  return "buffer_load_ubyte";
        case 9:  return "buffer_load_sbyte";
        case 10: return "buffer_load_ushort";
        case 11: return "buffer_load_sshort";
        case 12: return "buffer_load_dword";
        case 13: return "buffer_load_dwordx2";
        case 14: return "buffer_load_dwordx4";
        case 15: return "buffer_load_dwordx3";
        case 24: return "buffer_store_byte";
        case 26: return "buffer_store_short";
        case 28: return "buffer_store_dword";
        case 29: return "buffer_store_dwordx2";
        case 30: return "buffer_store_dwordx4";
        case 31: return "buffer_store_dwordx3";
        case 61: return "buffer_atomic_swap";
        case 62: return "buffer_atomic_cmpswap";
        case 63: return "buffer_atomic_add";
        default: return "buffer_???";
    }
}

static int decode_mubuf(const uint8_t* ptr, MUBUFInfo* info) {
    info->word0 = *(const uint32_t*)ptr;
    info->word1 = *(const uint32_t*)(ptr + 4);

    /* MUBUF encoding (GCN/GFX8):
     * Word 0: [31:26]=111000 [25:18]=op [17]=SLC [16]=lds
     *         [15:14]=00 [13]=GLC [12]=idxen [11]=offen [10:0]=offset
     * Word 1: [31:24]=soffset [23:21]=srsrc [20:18]=000
     *         [17:10]=vdata [9:1]=vaddr [0]=... */

    uint32_t enc = (info->word0 >> 26) & 0x3F;
    if (enc != 0x38) return 0; /* 111000 = MUBUF */

    info->op     = (info->word0 >> 18) & 0x7F;
    info->slc    = (info->word0 >> 17) & 1;
    info->lds    = (info->word0 >> 16) & 1;
    info->glc    = (info->word0 >> 14) & 1;
    info->idxen  = (info->word0 >> 12) & 1;
    info->offen  = (info->word0 >> 11) & 1;

    info->vaddr  = (info->word1 >>  0) & 0xFF;
    info->vdata  = (info->word1 >>  8) & 0xFF;
    info->srsrc  = (info->word1 >> 16) & 0x1F;
    info->soffset= (info->word1 >> 24) & 0xFF;

    return 1;
}

/* ---- Busca de ELF interno (64-bit HSA Code Object) ---- */

static const uint8_t* find_inner_elf(const uint8_t* data, size_t size, size_t* out_size) {
    const uint8_t elf_magic[] = { 0x7f, 'E', 'L', 'F' };
    for (size_t i = 1; i < size - 64; i++) {
        if (memcmp(data + i, elf_magic, 4) == 0 && data[i + 4] == 2) {
            /* ELF 64-bit encontrado */
            const Elf64_Ehdr* ehdr = (const Elf64_Ehdr*)(data + i);
            size_t end = i + ehdr->e_shoff + (size_t)ehdr->e_shnum * ehdr->e_shentsize;
            if (end > size) end = size;
            *out_size = end - i;
            return data + i;
        }
    }
    return NULL;
}

/* ---- Funcao principal de analise ---- */

static void analyze_binary(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Erro: nao foi possivel abrir '%s'\n", filename);
        return;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* data = (uint8_t*)malloc(fsize);
    fread(data, 1, fsize, fp);
    fclose(fp);

    printf("=== ISA Decoder para AMD GCN (GFX8) ===\n");
    printf("Arquivo: %s (%ld bytes)\n\n", filename, fsize);

    /* Verificar ELF externo */
    if (fsize < 52 || memcmp(data, "\x7f""ELF", 4) != 0) {
        fprintf(stderr, "Erro: arquivo nao e um ELF valido\n");
        free(data);
        return;
    }

    int elf_class = data[4]; /* 1=32bit, 2=64bit */
    printf("[ELF Externo] Classe: %d-bit\n", elf_class == 1 ? 32 : 64);

    /* Procurar ELF interno (HSA Code Object, 64-bit) */
    size_t inner_size = 0;
    const uint8_t* inner = find_inner_elf(data, fsize, &inner_size);

    if (!inner) {
        printf("\n[AVISO] ELF interno (64-bit) nao encontrado.\n");
        printf("Tentando analisar como ELF unico...\n\n");
        inner = data;
        inner_size = fsize;
    } else {
        printf("[ELF Interno] HSA Code Object encontrado em offset 0x%lx (%lu bytes)\n",
               (unsigned long)(inner - data), (unsigned long)inner_size);
    }

    /* Parsear ELF interno (64-bit) */
    const Elf64_Ehdr* ehdr = (const Elf64_Ehdr*)inner;
    if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0) {
        printf("[AVISO] ELF interno sem section headers\n");
        free(data);
        return;
    }

    const Elf64_Shdr* shdr_table = (const Elf64_Shdr*)(inner + ehdr->e_shoff);
    const char* shstrtab = NULL;
    if (ehdr->e_shstrndx < ehdr->e_shnum) {
        shstrtab = (const char*)(inner + shdr_table[ehdr->e_shstrndx].sh_offset);
    }

    /* Listar secoes */
    printf("\n--- Secoes do ELF interno ---\n");
    for (int i = 0; i < ehdr->e_shnum; i++) {
        const char* name = shstrtab ? shstrtab + shdr_table[i].sh_name : "???";
        printf("  [%2d] %-20s  offset=0x%06llx  size=0x%06llx\n",
               i, name,
               (unsigned long long)shdr_table[i].sh_offset,
               (unsigned long long)shdr_table[i].sh_size);
    }

    /* Procurar secao .hsatext */
    const Elf64_Shdr* hsatext = NULL;
    for (int i = 0; i < ehdr->e_shnum; i++) {
        const char* name = shstrtab ? shstrtab + shdr_table[i].sh_name : "";
        if (strcmp(name, ".hsatext") == 0) {
            hsatext = &shdr_table[i];
            break;
        }
    }

    if (!hsatext) {
        printf("\n[AVISO] Secao .hsatext nao encontrada\n");
        free(data);
        return;
    }

    printf("\n--- Secao .hsatext (codigo GPU) ---\n");
    printf("Offset: 0x%llx | Tamanho: %llu bytes\n\n",
           (unsigned long long)hsatext->sh_offset,
           (unsigned long long)hsatext->sh_size);

    /* Procurar simbolos (nomes dos kernels) */
    const Elf64_Shdr* symtab = NULL;
    const Elf64_Shdr* strtab_shdr = NULL;
    for (int i = 0; i < ehdr->e_shnum; i++) {
        if (shdr_table[i].sh_type == 2) { /* SHT_SYMTAB */
            symtab = &shdr_table[i];
            if (symtab->sh_link < (uint32_t)ehdr->e_shnum)
                strtab_shdr = &shdr_table[symtab->sh_link];
            break;
        }
    }

    /* Mapear simbolos na .hsatext */
    struct KernelRange {
        const char* name;
        uint64_t start;
        uint64_t end;
    };
    KernelRange kernels[32];
    int n_kernels = 0;

    if (symtab && strtab_shdr) {
        const char* strtab = (const char*)(inner + strtab_shdr->sh_offset);
        int n_syms = (int)(symtab->sh_size / symtab->sh_entsize);
        const Elf64_Sym* syms = (const Elf64_Sym*)(inner + symtab->sh_offset);

        for (int i = 0; i < n_syms && n_kernels < 32; i++) {
            /* Tipo STT_FUNC ou STT_AMDGPU_HSA_KERNEL (10) */
            uint8_t type = syms[i].st_info & 0xF;
            if ((type == 2 || type == 10) && syms[i].st_size > 0) {
                kernels[n_kernels].name = strtab + syms[i].st_name;
                kernels[n_kernels].start = syms[i].st_value;
                kernels[n_kernels].end = syms[i].st_value + syms[i].st_size;
                n_kernels++;
            }
        }
    }

    /* Decodificar instrucoes MUBUF na .hsatext */
    const uint8_t* code = inner + hsatext->sh_offset;
    size_t code_size = (size_t)hsatext->sh_size;

    int total_mubuf = 0;
    int total_glc = 0;
    int total_slc = 0;

    for (int k = 0; k < (n_kernels > 0 ? n_kernels : 1); k++) {
        uint64_t scan_start = 0;
        uint64_t scan_end = code_size;
        const char* kernel_name = "(todo o .hsatext)";

        if (n_kernels > 0) {
            kernel_name = kernels[k].name;
            scan_start = kernels[k].start - hsatext->sh_addr;
            scan_end = kernels[k].end - hsatext->sh_addr;
            if (scan_end > code_size) scan_end = code_size;
        }

        printf("====================================================\n");
        printf("Kernel: %s\n", kernel_name);
        printf("====================================================\n");

        int k_mubuf = 0;
        for (uint64_t off = scan_start; off + 8 <= scan_end; off += 4) {
            MUBUFInfo info;
            info.offset = (uint32_t)off;
            if (decode_mubuf(code + off, &info)) {
                k_mubuf++;
                total_mubuf++;
                if (info.glc) total_glc++;
                if (info.slc) total_slc++;

                printf("  0x%04x: %-30s v%d, v%d, s[%d:%d]",
                       info.offset, mubuf_op_name(info.op),
                       info.vdata, info.vaddr,
                       info.srsrc * 4, info.srsrc * 4 + 3);

                if (info.glc || info.slc) {
                    printf("  ");
                    if (info.glc) printf(" GLC=1");
                    if (info.slc) printf(" SLC=1");
                    printf("  <-- CACHE BYPASS");
                } else {
                    printf("   GLC=0 SLC=0  (cache normal)");
                }
                printf("\n");

                off += 4; /* MUBUF tem 8 bytes, pular a segunda word */
            }
        }

        if (k_mubuf == 0) {
            printf("  (nenhuma instrucao MUBUF encontrada)\n");
        }
        printf("\n");
    }

    /* Resumo */
    printf("====================================================\n");
    printf("RESUMO\n");
    printf("====================================================\n");
    printf("Total de instrucoes MUBUF: %d\n", total_mubuf);
    printf("  Com GLC=1 (bypass L1):    %d\n", total_glc);
    printf("  Com SLC=1 (bypass L2):    %d\n", total_slc);
    printf("  Sem flags (cache normal): %d\n", total_mubuf - total_glc);

    if (total_glc == 0 && total_slc == 0 && total_mubuf > 0) {
        printf("\n>>> CONCLUSAO: TODAS as instrucoes usam cache normal (GLC=0, SLC=0).\n");
        printf(">>> O compilador NAO gerou flags de bypass, mesmo com volatile.\n");
    } else if (total_glc > 0) {
        printf("\n>>> Encontradas instrucoes com bypass de cache.\n");
    }

    free(data);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Uso: isa_decoder <arquivo.bin>\n");
        printf("Exemplo: isa_decoder matmul_kernel_dump.bin\n");
        return 1;
    }

    analyze_binary(argv[1]);
    return 0;
}
