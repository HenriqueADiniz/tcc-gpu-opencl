#!/usr/bin/env python3
"""
Analisa um disassembly gerado por llvm-objdump para gfx803, separando por kernel
e comparando as instrucoes de memoria e as flags de cache.

Uso:
    llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx803 BIN > d.txt
    python3 isa/analyze_disasm.py d.txt
    python3 isa/analyze_disasm.py d.txt --diff matmul_via_L1 matmul_via_L2

Encoding FLAT no GFX8 (Vol. "AMD GCN3 ISA", secao FLAT):
    bits [31:26] = 0b110111  -> prefixo 0xDC
    bit 16       = GLC       -> 1 = contorna o L1 (globally coherent)
    bit 17       = SLC       -> 1 = contorna o L2 (system level coherent)
Portanto:
    0xDC500000 = flat_load_dword  GLC=0 SLC=0
    0xDC510000 = flat_load_dword  GLC=1 SLC=0
"""
import re
import sys

MEM_RE = re.compile(r'\b(flat|buffer|global)_(load|store|atomic)\w*')
# "  <asm>   // <addr>: <hex> <hex> [<simbolo+0xNN>]"
LINE_RE = re.compile(r'^\s+(.*?)\s*//\s*([0-9A-Fa-f]+):\s*(.*)$')
HDR_RE = re.compile(r'^([0-9a-f]+)\s+<([^>]+)>:')


def parse(path):
    """Devolve {kernel: [(asm, addr, [dwords], resto)]} preservando a ordem."""
    kernels, cur = {}, None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            m = HDR_RE.match(line)
            if m:
                cur = m.group(2)
                kernels[cur] = []
                continue
            if cur is None:
                continue
            m = LINE_RE.match(line.rstrip('\n'))
            if m:
                asm, addr, rest = m.group(1), m.group(2), m.group(3)
                # llvm-objdump anexa alvos de salto ao hex; guarda so os dwords
                dwords = [h for h in rest.split() if re.fullmatch(r'[0-9A-Fa-f]{8}', h)]
                kernels[cur].append((asm.strip(), addr, dwords, rest))
    return kernels


def inner_loop(insts, name):
    """Localiza o corpo do laco interno pelo ultimo salto condicional para tras.

    O tamanho do laco importa: um bloco __asm__ e opaco para o otimizador e
    impede a reducao de forca do calculo de endereco (strength reduction), o
    que adiciona instrucoes de ALU por iteracao. Sem contabilizar isso, essa
    sobrecarga e confundida com o efeito das flags de cache.
    """
    back = tgt_addr = None
    for i, (asm, addr, _d, rest) in enumerate(insts):
        m = re.search(r'<' + re.escape(name) + r'\+0x([0-9a-f]+)>', rest)
        if m and asm.startswith('s_cbranch'):
            tgt = int(insts[0][1], 16) + int(m.group(1), 16)
            if tgt <= int(addr, 16):
                back, tgt_addr = i, tgt
    if back is None:
        return None
    start = next(i for i, (_a, ad, _d, _r) in enumerate(insts)
                 if int(ad, 16) >= tgt_addr)
    return insts[start:back + 1]


def flags(dwords):
    """Extrai GLC/SLC do primeiro dword de uma instrucao FLAT."""
    if not dwords:
        return None
    w = int(dwords[0], 16)
    if (w >> 26) != 0b110111:      # nao e FLAT
        return None
    return (w >> 16) & 1, (w >> 17) & 1


def summarize(kernels):
    print(f"{'kernel':24s} {'inst':>5s} {'laco':>5s} {'loads':>6s} "
          f"{'vmcnt':>6s}  flags de cache")
    print("-" * 78)
    for k, insts in kernels.items():
        mem = [(a, ad, d) for (a, ad, d, _r) in insts if MEM_RE.search(a)]
        real = [i for i in insts if not i[0].startswith('s_nop')]
        loop = inner_loop(insts, k)
        loads = [(a, ad, d) for (a, ad, d) in mem if '_load' in a]
        fl = {flags(d) for (_a, _ad, d) in loads if flags(d)}
        if fl == {(0, 0)}:
            tag = "GLC=0 SLC=0 (usa L1)"
        elif fl == {(1, 0)}:
            tag = "GLC=1 SLC=0 (contorna L1)"
        elif fl == {(1, 1)}:
            tag = "GLC=1 SLC=1 (contorna L1 e L2)"
        else:
            tag = f"misto {sorted(fl)}"
        lw = (sum(1 for a, _, _, _ in loop
                  if a.startswith('s_waitcnt') and 'vmcnt' in a)
              if loop else -1)
        print(f"{k:24s} {len(real):5d} {len(loop) if loop else 0:5d} "
              f"{len(loads):6d} {lw:6d}  {tag}")

    print("\ndetalhe dos acessos a memoria global:")
    for k, insts in kernels.items():
        print(f"\n  === {k} ===")
        for asm, addr, d, _r in insts:
            if not MEM_RE.search(asm):
                continue
            f = flags(d)
            t = f"GLC={f[0]} SLC={f[1]}" if f else "(nao-FLAT)"
            print(f"    {addr}  {d[0] if d else '':10s}  {t:15s}  {asm}")


def loops(kernels):
    """Imprime o corpo do laco interno de cada kernel."""
    for k, insts in kernels.items():
        body = inner_loop(insts, k)
        if not body:
            print(f"\n=== {k}: laco interno nao localizado ===")
            continue
        nl = sum(1 for a, _, _, _ in body if '_load' in a)
        nw = sum(1 for a, _, _, _ in body
                 if a.startswith('s_waitcnt') and 'vmcnt' in a)
        print(f"\n=== {k}: {len(body)} instrucoes | {nl} loads | {nw} waitcnt(vmcnt) ===")
        for asm, _ad, _d, _r in body:
            print(f"    {asm}")


def diff(kernels, a, b):
    """Compara duas listas de instrucoes ignorando enderecos e alocacao de registrador."""
    ka = [i[0] for i in kernels[a] if not i[0].startswith('s_nop')]
    kb = [i[0] for i in kernels[b] if not i[0].startswith('s_nop')]
    print(f"\n=== DIFF {a} ({len(ka)} inst)  vs  {b} ({len(kb)} inst) ===")
    import difflib
    n = 0
    for line in difflib.unified_diff(ka, kb, lineterm='', n=1,
                                     fromfile=a, tofile=b):
        print("  " + line)
        n += 1
    if n == 0:
        print("  (sequencias de instrucoes IDENTICAS)")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    path = args[0]
    kernels = parse(path)
    if '--diff' in args:
        i = args.index('--diff')
        diff(kernels, args[i + 1], args[i + 2])
    elif '--loops' in args:
        loops(kernels)
    else:
        summarize(kernels)
    return 0


if __name__ == '__main__':
    sys.exit(main())
