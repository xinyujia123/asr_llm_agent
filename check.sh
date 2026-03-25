#!/usr/bin/env bash

echo "================ AI 硬件信息检查 ================"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo

echo "===== 1. CPU ====="
lscpu | egrep 'Architecture|Model name|Socket|Core|Thread|CPU\(s\)'
echo

echo "===== 2. 内存 ====="
free -h
echo

echo "===== 3. 磁盘 ====="
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
echo
df -h /
echo

echo "===== 4. GPU ====="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv
else
    echo "未检测到 nvidia-smi（可能没装 NVIDIA 驱动，或者不是 NVIDIA GPU）"
fi
echo

echo "===== 5. CUDA ====="
if command -v nvcc >/dev/null 2>&1; then
    nvcc -V
else
    echo "未检测到 nvcc"
fi
echo

echo "===== 6. PCI 显卡信息 ====="
lspci | egrep 'VGA|3D|NVIDIA'
echo

echo "================ 检查完成 ================"
