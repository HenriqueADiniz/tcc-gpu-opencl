#!/bin/bash
# Espera a varredura 1024-8192 terminar e roda o N=16384 em seguida.
# Lotes de 1024 linhas: o kernel mais lento (dispersa glc slc) deve ficar em
# ~260 s por execução, e 16 lançamentos de ~16 s mantêm folga larga diante do
# watchdog de 60 s do amdgpu.
cd "/home/henrique/Área de Trabalho/tcc-gpu-opencl" || exit 1
source setup/env_rocm.sh
exec >> resultados/linha_coluna_sweep.txt 2>&1
while pgrep -f "linha_coluna 1024" > /dev/null; do sleep 20; done
echo
echo "############ N=16384 iniciado $(date '+%d/%m %H:%M:%S') ############"
./output/linha_coluna "16384" 4000 1024
echo "############ concluído $(date '+%d/%m %H:%M:%S') ############"
