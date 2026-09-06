#!/bin/bash
# Analise de sensibilidade ao tamanho, do 4096 ao 16384, com saida visivel no terminal.
#
# O passo de validacao dos lotes e uma trava: N=16384 so roda se o lancamento
# fatiado provar que produz resultado identico ao lancamento unico. Os lotes
# existem porque o amdgpu reseta a GPU quando um job de compute passa de 60 s,
# e em N=16384 um lancamento unico chegaria a ~58 s — dentro da margem de erro
# desse limite, numa maquina cuja unica GPU tambem move o video.
PROJ="/home/henrique/Área de Trabalho/tcc-gpu-opencl"
cd "$PROJ" || exit 1
source setup/env_rocm.sh
CSV=resultados/matmul_sweep.csv
exec > >(tee -a resultados/cadeia_log.txt) 2>&1

echo "############################################################"
echo "#  ANALISE DE SENSIBILIDADE AO TAMANHO — matmul — RX 580 (gfx803)         #"
echo "#  100 execucoes por cenario, 10 cenarios por tamanho       #"
echo "############################################################"
echo "inicio: $(date '+%d/%m %H:%M:%S')"
echo

completos() { LC_ALL=C awk -F, -v n="$1" 'NR>1 && $1==n' "$CSV" 2>/dev/null | wc -l; }

for N in 4096 8192; do
    if [ "$(completos $N)" -ge 10 ]; then
        echo ">>> N=$N ja completo, pulando"; continue
    fi
    echo "=========== N=$N  inicio $(date +%H:%M:%S) ==========="
    ./output/matmul 100 "$N" 5 | grep -vE "^  Run "
    echo "=========== N=$N  fim    $(date +%H:%M:%S) ==========="
    echo
done

echo ">>> validando lancamento em lotes (N=2048, lotes de 512 linhas)"
echo ">>> o 16384 so roda se esta etapa passar"
cd /tmp || exit 1
SAIDA=$("$PROJ/output/matmul" 20 2048 3 512 2>&1)
cd "$PROJ" || exit 1
if echo "$SAIDA" | grep -q "FALHOU"; then
    echo "!!! ABORTADO: o modo em lotes produziu resultado incorreto"
    echo "$SAIDA" | grep -B2 FALHOU
    echo; echo "Pressione ENTER para fechar."; read _; exit 1
fi
echo ">>> lotes OK — resultado identico ao lancamento unico"
echo "$SAIDA" | grep -E "Lotes:|Validacao" | head -4
echo

echo "=========== N=16384  inicio $(date +%H:%M:%S) ==========="
echo "    100 execucoes, lotes de 2048 linhas (8 lancamentos por execucao)"
echo "    ~3 GB de VRAM, ~4 GB de RAM, estimativa de ~11 h"
echo
./output/matmul 100 16384 3 2048 | grep -vE "^  Run "
echo "=========== N=16384  fim $(date +%H:%M:%S) ==========="
echo
echo "############### ANALISE CONCLUIDA $(date '+%d/%m %H:%M:%S') ###############"
echo
LC_ALL=C awk -F, 'NR>1{gsub(/matmul_/,"",$2);
    printf "%-6s %-20s %-13s %6s %10.3f ms %8.1f GF  %s\n",$1,$2,$3,$4,$6,$13,$14}' "$CSV"
echo
echo "Resultados: resultados/matmul_sweep.csv"
echo "Pressione ENTER para fechar esta janela."
read _
