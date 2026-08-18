#!/usr/bin/env bash
# Fase 1 - Toolchain de build + stack OpenCL dos repositorios oficiais do Ubuntu 20.04
# Nao instala ROCm (isso e a Fase 2). Tudo aqui e leve e reversivel.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "### apt update"
apt-get update -qq

echo "### instalando toolchain de build"
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config

echo "### instalando stack OpenCL (loader ICD + headers + clinfo)"
apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo

echo "### instalando runtimes OpenCL"
# Mesa Clover: runtime OpenCL open-source baseado em Clang/LLVM, suporta Polaris/gfx803.
# E o candidato mais provavel para aceitar __asm__ em kernel, alem do ROCm.
apt-get install -y --no-install-recommends mesa-opencl-icd
# PoCL: runtime de CPU, usado apenas como controle/sanidade (nao roda na GPU).
apt-get install -y --no-install-recommends pocl-opencl-icd

echo "### adicionando o usuario aos grupos render e video"
groupadd -f render
usermod -aG render,video henrique

echo "### FASE 1 CONCLUIDA"
