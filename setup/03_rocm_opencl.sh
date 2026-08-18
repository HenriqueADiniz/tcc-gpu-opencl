#!/usr/bin/env bash
# Fase 3 - Runtime OpenCL do ROCm 5.2 (compilador Clang/LLVM em runtime).
#
# Instala APENAS o subconjunto OpenCL, nao o meta-pacote `rocm` completo:
#   rocm-opencl -> rocm-ocl-icd, comgr (LLVM embutido), hsa-rocr, rocm-device-libs
# Isso da ~400 MB instalados em vez de ~20 GB. O `rocm-llvm` (compilador offline)
# NAO e necessario: o comgr ja embute o LLVM usado para compilar kernels em runtime.
#
# ROCm 5.2 e a ultima versao com chance de enumerar gfx803 (Polaris). O suporte
# a pre-Vega e gated por ROC_ENABLE_PRE_VEGA=1 no runtime.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

ROCM_VER=5.2

echo "### liberando espaco: cache de .deb ja instalados"
df -h / | tail -1
apt-get clean
df -h / | tail -1

echo "### dependencias do repositorio"
apt-get install -y --no-install-recommends gnupg ca-certificates libnuma-dev

echo "### chave GPG do repositorio AMD"
mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor --yes -o /etc/apt/keyrings/rocm.gpg
chmod a+r /etc/apt/keyrings/rocm.gpg

echo "### fonte apt do ROCm ${ROCM_VER}"
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER} ubuntu main" \
    > /etc/apt/sources.list.d/rocm.list

# Prioridade baixa: impede que pacotes do ROCm substituam pacotes do Ubuntu
# (ex.: bibliotecas LLVM/clang do sistema) por acidente.
cat > /etc/apt/preferences.d/rocm-pin-600 <<'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

echo "### apt update"
apt-get update -qq

echo "### instalando runtime OpenCL do ROCm (sem rocm-llvm)"
apt-get install -y --no-install-recommends rocm-opencl rocminfo

echo "### registrando as bibliotecas do ROCm no ld.so"
echo "/opt/rocm/lib"     >  /etc/ld.so.conf.d/rocm.conf
echo "/opt/rocm/lib64"   >> /etc/ld.so.conf.d/rocm.conf
ldconfig

echo "### grupos render/video"
usermod -aG render,video henrique

echo "### ICDs registrados:"
ls -l /etc/OpenCL/vendors/

echo "### FASE 3 CONCLUIDA"
