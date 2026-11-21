#!/bin/sh
# Regenerates gpu_cuda.ptx from gpu.cu, and gpu_cuda_det.ptx from gpu_det.cu (the deterministic kernels, compiled for
# IEEE float32: no fused multiply-adds unless written, correctly rounded division and square roots, subnormals kept),
# with nvcc, in NVIDIA's CUDA container (no GPU needed), so only the developer who changes the .cu files needs
# Docker; builds and users never do. Run: go generate ./engine
set -e
cd "$(dirname "$0")"
IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
DET="-fmad=false -ftz=false -prec-div=true -prec-sqrt=true"
if command -v nvcc >/dev/null 2>&1; then
	nvcc -ptx -arch=compute_75 -O3 -o gpu_cuda.ptx gpu.cu
	nvcc -ptx -arch=compute_75 -O3 $DET -o gpu_cuda_det.ptx gpu_det.cu
else
	docker run --rm -v "$PWD":/src -w /src "$IMAGE" sh -c "nvcc -ptx -arch=compute_75 -O3 -o gpu_cuda.ptx gpu.cu &&
		nvcc -ptx -arch=compute_75 -O3 $DET -o gpu_cuda_det.ptx gpu_det.cu" >/dev/null
fi
echo "gpu_cuda.ptx: $(grep -c '^.visible .entry' gpu_cuda.ptx) kernels"
echo "gpu_cuda_det.ptx: $(grep -c '^.visible .entry' gpu_cuda_det.ptx) kernels"
