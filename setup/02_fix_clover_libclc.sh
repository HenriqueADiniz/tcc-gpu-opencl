#!/usr/bin/env bash
# Corrige o bug de caminho do libclc no mesa-opencl-icd 21.2.6 do Ubuntu 20.04.
#
# Sintoma:
#   fatal error: cannot open file '/usr/local//usr/lib/clc/polaris10-amdgcn-mesa-mesa3d.bc'
#
# Duas incompatibilidades entre o Mesa empacotado e o libclc-12 do focal:
#   1) prefixo: o Clover procura em /usr/local/usr/lib/clc, o libclc instala em /usr/lib/clc
#   2) triple:  o Clover pede <dev>-amdgcn-mesa-mesa3d.bc, o libclc gera <dev>-amdgcn--.bc
#
# A correcao e apenas por symlink — nao altera nem substitui nenhum arquivo do sistema.
set -euo pipefail

CLC=/usr/lib/clc

if [ ! -d "$CLC" ]; then
    echo "ERRO: $CLC nao existe (libclc-12 instalado?)" >&2
    exit 1
fi

echo "### 1) alias de triple: <dev>-amdgcn--.bc  ->  <dev>-amdgcn-mesa-mesa3d.bc"
n=0
for f in "$CLC"/*-amdgcn--.bc; do
    [ -e "$f" ] || continue
    base=$(basename "$f" -amdgcn--.bc)
    alias_name="$CLC/${base}-amdgcn-mesa-mesa3d.bc"
    if [ ! -e "$alias_name" ]; then
        ln -s "$f" "$alias_name"
        n=$((n+1))
    fi
done
echo "    $n symlinks de triple criados"

echo "### 2) prefixo: /usr/local/usr/lib/clc -> $CLC"
mkdir -p /usr/local/usr/lib
if [ ! -e /usr/local/usr/lib/clc ]; then
    ln -s "$CLC" /usr/local/usr/lib/clc
    echo "    symlink de prefixo criado"
else
    echo "    ja existia"
fi

echo "### verificacao"
ls -l /usr/local/usr/lib/clc/polaris10-amdgcn-mesa-mesa3d.bc

echo "### CORRECAO DO CLOVER CONCLUIDA"
