#!/bin/bash
# Painel ao vivo da analise de tamanhos. Somente leitura: nao toca na GPU
# nem nos arquivos de resultado, so exibe o que ja foi gravado.
cd "/home/henrique/Área de Trabalho/tcc-gpu-opencl"
CSV=resultados/matmul_sweep.csv
LOG=resultados/sweep_log.txt
while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║  ANALISE DE SENSIBILIDADE AO TAMANHO — matmul / hierarquia de memoria — RX 580 gfx803   ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    printf "  %s\n\n" "$(date '+%d/%m %H:%M:%S')"

    if pgrep -f "output/matmul_sweep" > /dev/null; then
        P=$(pgrep -f "output/matmul_sweep" | head -1)
        echo "  ESTADO: rodando (pid $P, ha $(ps -o etime= -p $P 2>/dev/null | tr -d ' '))"
    elif pgrep -f "output/matmul " > /dev/null; then
        P=$(pgrep -f "output/matmul " | head -1)
        echo "  ESTADO: rodando N=16384 (pid $P, ha $(ps -o etime= -p $P | tr -d ' '))"
    else
        echo "  ESTADO: nenhum processo de medicao ativo"
    fi
    echo

    if [ -f "$CSV" ]; then
        echo "  ── RESULTADOS GRAVADOS ────────────────────────────────────────────────"
        LC_ALL=C awk -F, 'NR>1{
            gsub(/matmul_/,"",$2)
            printf "  %-6s %-20s %-12s %5s %9.3f ms %7.1f GF  %s\n",$1,$2,$3,$4,$6,$13,$14
        }' "$CSV" | tail -30
        echo
        printf "  %d linhas gravadas\n\n" "$(($(wc -l < "$CSV")-1))"
    fi

    if [ -f "$LOG" ]; then
        echo "  ── LOG ────────────────────────────────────────────────────────────────"
        grep -E "^=====|adaptativo|CONCLUIDO|FALHOU" "$LOG" | tail -6 | sed 's/^/  /'
    fi
    echo
    echo "  (atualiza a cada 5 s — Ctrl+C para fechar, nao interrompe a medicao)"
    sleep 5
done
