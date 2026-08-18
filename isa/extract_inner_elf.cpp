/*
 * extract_inner_elf.cpp — Extrai o ELF interno (HSA Code Object) de um
 * binario OpenCL AMD para uso com clrxdisasm.
 *
 * Compilar:
 *   g++ -Wall -O2 extract_inner_elf.cpp -o ..\output\extract_inner_elf.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Uso: extract_inner_elf <input.bin> <output.elf>\n");
        return 1;
    }

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "Erro ao abrir %s\n", argv[1]); return 1; }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* data = (uint8_t*)malloc(fsize);
    fread(data, 1, fsize, fp);
    fclose(fp);

    const uint8_t elf_magic[] = { 0x7f, 'E', 'L', 'F' };

    for (long i = 1; i < fsize - 64; i++) {
        if (memcmp(data + i, elf_magic, 4) == 0 && data[i + 4] == 2) {
            long remaining = fsize - i;
            printf("ELF interno (64-bit) encontrado em offset 0x%lx\n", i);
            printf("Salvando %ld bytes em %s\n", remaining, argv[2]);

            FILE* out = fopen(argv[2], "wb");
            fwrite(data + i, 1, remaining, out);
            fclose(out);
            free(data);
            return 0;
        }
    }

    fprintf(stderr, "ELF interno nao encontrado\n");
    free(data);
    return 1;
}
