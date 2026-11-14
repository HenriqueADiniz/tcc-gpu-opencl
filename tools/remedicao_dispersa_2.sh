#!/bin/bash
# Segunda remedição do padrão dispersa em N=8192 e 16384 (anomalia do slc no
# Exp. 4). Mesmas condições da campanha original (27/09) e da 1ª remedição
# (14/10): mesmo .cl, lançamento único, orçamento de 4000 ms (5 execuções por
# ponto nesses tamanhos). Saída em CSV PRÓPRIO (regra 2): não acrescenta ao
# linha_coluna_remedicao.csv da 1ª remedição e recusa rodar se o seu já existir.
cd "/home/henrique/Área de Trabalho/tcc-gpu-opencl" || exit 1
source setup/env_rocm.sh
CSV=resultados/linha_coluna_remedicao2.csv
LOG=resultados/linha_coluna_remedicao2.txt
if [ -e "$CSV" ]; then
    echo "ERRO: $CSV já existe; não sobrescrevo nem acrescento."
    read -r _; exit 1
fi
exec > >(tee -a "$LOG") 2>&1
echo "############ 2ª remedição iniciada $(date '+%d/%m %H:%M:%S') ############"
echo "dispersa glc slc / GLC=0 — original: N=8192 3,376x  N=16384 2,111x"
echo "                           1ª remedição: N=8192 3,016x  N=16384 2,500x"
echo "Duração esperada: ~1 h. Lançamentos de até ~4 min: a tela pode engasgar."
df -h / | tail -1
echo
./output/linha_coluna "8192 16384" 4000 dispersa "$CSV" &
PID=$!
echo "PID do linha_coluna: $PID"
wait "$PID"
echo "código de saída: $?"
echo "############ concluída $(date '+%d/%m %H:%M:%S') ############"
echo "Pode fechar esta janela."
read -r _
