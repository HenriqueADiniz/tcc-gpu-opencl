#!/bin/bash
# Referencia em CPU sobre todos os tamanhos, com saida visivel no terminal.
cd "/home/henrique/Área de Trabalho/tcc-gpu-opencl" || exit 1
exec > >(tee -a resultados/cpu_sweep_log.txt) 2>&1
echo "inicio: $(date '+%d/%m %H:%M:%S')"
echo "estimativa: ~80 min (N=16384 sozinho leva ~75 min)"
echo
./output/matmul_cpu "1024 2048 4096 8192 16384" 60
echo
echo "fim: $(date '+%d/%m %H:%M:%S')"
echo
column -s, -t < resultados/matmul_cpu_sweep.csv
echo
echo "Pressione ENTER para fechar."
read _
