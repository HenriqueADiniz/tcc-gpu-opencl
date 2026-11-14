#!/bin/bash
# Roda N=16384 do experimento de layout assim que a varredura menor terminar.
#
# A espera é por PID, não por padrão de texto: pgrep -f casa com a própria linha
# de comando de quem o invoca, e um encadeamento anterior neste projeto ficou
# 45 min travado exatamente por isso.
ALVO=$1
cd "/home/henrique/Área de Trabalho/tcc-gpu-opencl" || exit 1
source setup/env_rocm.sh
exec >> resultados/matmul_layout.txt 2>&1
while kill -0 "$ALVO" 2>/dev/null; do sleep 30; done
echo
echo "############ N=16384 iniciado $(date '+%d/%m %H:%M:%S') ############"
# lotes de 1024 linhas: o kernel mais lento fica em ~12,6 s por lançamento,
# com folga larga diante do watchdog de 60 s do amdgpu.
./output/matmul_layout "16384" 6000 1024
echo "############ concluído $(date '+%d/%m %H:%M:%S') ############"
