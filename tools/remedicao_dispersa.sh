#!/bin/bash
# Remedição do padrão dispersa em N=8192 e 16384 (anomalia do slc no Exp. 5).
# Mesmas condições da campanha original: mesmo .cl, lançamento único, orçamento
# de 4000 ms. Saída em CSV próprio, para não misturar campanhas.
cd "/home/henrique/Área de Trabalho/tcc-gpu-opencl" || exit 1
source setup/env_rocm.sh
LOG=resultados/linha_coluna_remedicao.txt
exec > >(tee -a "$LOG") 2>&1
echo "############ remedição iniciada $(date '+%d/%m %H:%M:%S') ############"
echo "Campanha original (resultados/linha_coluna.csv), dispersa glc slc / GLC=0:"
echo "  N=8192: 3,376x    N=16384: 2,111x   <- o ponto em questão"
echo
./output/linha_coluna "8192 16384" 4000 dispersa resultados/linha_coluna_remedicao.csv
echo "############ concluída $(date '+%d/%m %H:%M:%S') ############"
echo "Pode fechar esta janela."
read
