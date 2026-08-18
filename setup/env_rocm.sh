#!/usr/bin/env bash
# Ambiente para rodar os experimentos na RX 580 via ROCm.
#
# Uso:   source setup/env_rocm.sh
#        ./output/matmul
#
# ROC_ENABLE_PRE_VEGA=1 e OBRIGATORIO: o ROCm 5.2 depreciou Polaris/gfx803 e,
# sem essa variavel, a plataforma "AMD Accelerated Parallel Processing" e
# enumerada com ZERO dispositivos (o rocminfo ve o agente HSA, mas o runtime
# OpenCL nao expoe o device).

export PATH=/opt/rocm/bin:$PATH
export ROC_ENABLE_PRE_VEGA=1

# Restringe a plataforma AMD, evitando que o programa pegue Clover ou PoCL
# quando ele simplesmente usa a plataforma de indice 0.
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors

if ! id -nG | grep -qw render; then
    echo "AVISO: a sessao atual nao esta no grupo 'render' (necessario para /dev/kfd)."
    echo "       Faca logout/login para aplicar, ou prefixe os comandos com:  sg render -c '...'"
fi
