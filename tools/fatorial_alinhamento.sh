#!/bin/bash
# Desenho fatorial 4x2: magnitude do problema x alinhamento do stride.
#
# Motivacao: no sweep anterior todos os tamanhos eram potencias de dois, de modo
# que o crescimento do conjunto de trabalho e o agravamento das condicoes de
# conflict miss avancaram juntos. Nao da para dizer qual dos dois produz a queda
# do custo de glc (2,47x -> 1,65x).
#
# O stride entre linhas de B vale 4N bytes; o numero de fatores 2 desse valor
# determina quantos acessos colidem no mesmo conjunto do cache. Os pares abaixo
# tem magnitude equivalente e alinhamento oposto:
#
#     magnitude   alto (2^k)              baixo (6 fatores 2)
#     ~1024       1024  -> 4N = 2^12      1008 = 16*63   -> 4N = 2^6 * 63
#     ~2048       2048  -> 4N = 2^13      2032 = 16*127  -> 4N = 2^6 * 127
#     ~4096       4096  -> 4N = 2^14      4080 = 16*255  -> 4N = 2^6 * 255
#     ~8192       8192  -> 4N = 2^15      8176 = 16*511  -> 4N = 2^6 * 511
#
# Todos multiplos de 16, exigencia do work-group 16x16 (o global_work_size do
# OpenCL 1.2 precisa ser divisivel pelo local_work_size).
#
# Os pares rodam adjacentes no tempo, para que a comparacao pareada sofra o
# minimo de deriva termica, e a ordem dentro do par alterna, para que "primeiro
# do par" nao coincida sempre com o mesmo nivel de alinhamento.
#
# A campanha inteira e nova: os pontos alinhados sao remedidos em vez de
# reaproveitados do sweep anterior, porque tempos de campanhas distintas nao sao
# comparaveis entre si (ha 2,8% de diferenca medida para o mesmo kernel em
# N=1024 entre as duas campanhas ja existentes).
#
# Sem lotes: o kernel mais lento em N=8192 leva ~6 s, folgado ante o watchdog
# de 60 s do amdgpu.

PROJ="/home/henrique/Área de Trabalho/tcc-gpu-opencl"
cd "$PROJ" || exit 1
source setup/env_rocm.sh

CSV=resultados/matmul_fatorial.csv
LOG=resultados/fatorial_log.txt
SWEEP=resultados/matmul_sweep.csv
GUARDA=resultados/matmul_sweep_campanha1.csv
RUNS=100
WARMUP=5

# O programa grava sempre em resultados/matmul_sweep.csv. Para nao misturar a
# campanha nova com a publicada, guarda-se a antiga durante a execucao e
# devolve-se ao final -- inclusive se o script for interrompido.
restaurar() {
    [ -f "$SWEEP" ] && mv "$SWEEP" "$CSV"
    [ -f "$GUARDA" ] && mv "$GUARDA" "$SWEEP"
    echo
    echo "CSV do fatorial: $CSV"
    echo "sweep original devolvido a: $SWEEP"
}
[ -f "$SWEEP" ] && mv "$SWEEP" "$GUARDA"
[ -f "$CSV" ] && mv "$CSV" "$SWEEP"     # retoma campanha interrompida
trap restaurar EXIT

exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "#  FATORIAL 4x2 — magnitude x alinhamento — RX 580 (gfx803) #"
echo "#  $RUNS execucoes por cenario, 10 cenarios por tamanho             #"
echo "#  estimativa: ~3,5 h                                      #"
echo "############################################################"
echo "inicio: $(date '+%d/%m %H:%M:%S')"
echo

completos() { LC_ALL=C awk -F, -v n="$1" 'NR>1 && $1==n' "$SWEEP" 2>/dev/null | wc -l; }

# pares adjacentes, ordem alternada dentro do par
for N in 1024 1008  2032 2048  4096 4080  8176 8192; do
    if [ "$(completos "$N")" -ge 10 ]; then
        echo ">>> N=$N ja completo, pulando"
        continue
    fi
    echo "=========== N=$N  inicio $(date +%H:%M:%S) ==========="
    ./output/matmul "$RUNS" "$N" "$WARMUP" | grep -vE "^  Run "
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "!!! N=$N terminou com codigo $rc — interrompendo"
        break
    fi
    echo "=========== N=$N  fim    $(date +%H:%M:%S) ==========="
    echo
done

echo "fim: $(date '+%d/%m %H:%M:%S')"
echo
echo "Pressione ENTER para fechar."
read -r
