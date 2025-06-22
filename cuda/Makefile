# CUDA Eulerian Video Magnification - Enhanced Build System
# ========================================================

BUILD_DIR := build
TARGET := $(BUILD_DIR)/evmpipeline
CPU_TARGET := ../cpp/build/main

# Default target
.DEFAULT_GOAL := all

# Help
.PHONY: help
help:
	@echo "CUDA Eulerian Video Magnification - Enhanced Build System"
	@echo "========================================================"
	@echo ""
	@echo "Build Commands:"
	@echo "  make               - Build the CUDA EVM program"
	@echo "  make clean         - Remove build directory"
	@echo ""
	@echo "Presentation Presets:"
	@echo "  make gaussian      - Run Gaussian mode on face.mp4 (level=4, alpha=50, ω=0.8333-1.0)"
	@echo "  make laplacian     - Run Laplacian mode on baby.mp4 (level=4, alpha=15, λ=16, ω=0.4-3.0)"
	@echo "  make reference     - Run CPU reference implementation"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench_gauss   - Benchmark Gaussian mode with timing"
	@echo "  make bench_lap     - Benchmark Laplacian mode with timing"
	@echo "  make bench_all     - Run both benchmarks"
	@echo ""
	@echo "Custom Usage:"
	@echo "  $(TARGET) --input=video.mp4 --output=result.avi --mode=gaussian --alpha=50"

# Build target (uses CMake for reliable compilation)
.PHONY: all
all: $(TARGET)

$(TARGET):
	@echo "🔨 Building CUDA EVM pipeline..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make -j$(shell nproc)
	@echo "✅ Build complete: $(TARGET)"

# Clean
.PHONY: clean
clean:
	@echo "🧹 Cleaning..."
	@rm -rf $(BUILD_DIR)
	@echo "✅ Clean complete"

# Presentation presets
.PHONY: gaussian
gaussian: $(TARGET)
	@echo "🚀 Running Gaussian EVM on face.mp4 (presentation preset)..."
	@echo "   Parameters: level=4, alpha=50, attenuation=1, ω=0.8333-1.0"
	./$(TARGET) --input=../data/face.mp4 --output=face_gaussian_presentation.avi \
		--mode=gaussian --level=4 --alpha=50 --chrom_attenuation=1 \
		--fl=0.8333 --fh=1.0 --timing

.PHONY: laplacian  
laplacian: $(TARGET)
	@echo "🚀 Running Laplacian EVM on baby.mp4 (presentation preset)..."
	@echo "   Parameters: level=4, alpha=15, λ=16, attenuation=0.1, ω=0.4-3.0"
	./$(TARGET) --input=../data/baby.mp4 --output=baby_laplacian_presentation.avi \
		--mode=laplacian --level=4 --alpha=15 --chrom_attenuation=0.1 \
		--fl=0.4 --fh=3.0 --timing

.PHONY: reference
reference:
	@echo "🖥️  Running CPU reference implementation..."
	@if [ ! -f "$(CPU_TARGET)" ]; then \
		echo "Building CPU reference first..."; \
		cd ../cpp && mkdir -p build && cd build && cmake .. && make -j$(shell nproc); \
	fi
	@echo "Running CPU Gaussian on face.mp4..."
	cd ../cpp/build && ./main --input=../../data/face.mp4 --output=../../cuda/face_cpu_reference.avi \
		--mode=gaussian --level=4 --alpha=50 --fl=0.8333 --fh=1.0
	@echo "Running CPU Laplacian on baby.mp4..."
	cd ../cpp/build && ./main --input=../../data/baby.mp4 --output=../../cuda/baby_cpu_reference.avi \
		--mode=laplacian --level=4 --alpha=15 --fl=0.4 --fh=3.0

# Benchmarking targets
.PHONY: bench_gauss
bench_gauss: $(TARGET)
	@echo "⏱️  Benchmarking Gaussian mode..."
	time ./$(TARGET) --input=../data/face.mp4 --output=face_benchmark.avi \
		--mode=gaussian --level=4 --alpha=50 --fl=0.8333 --fh=1.0 --timing --gpu-sync

.PHONY: bench_lap
bench_lap: $(TARGET)
	@echo "⏱️  Benchmarking Laplacian mode..."
	time ./$(TARGET) --input=../data/baby.mp4 --output=baby_benchmark.avi \
		--mode=laplacian --level=4 --alpha=15 --fl=0.4 --fh=3.0 --timing --gpu-sync

.PHONY: bench_all
bench_all: bench_gauss bench_lap
	@echo "✅ All benchmarks complete"

# Advanced targets
.PHONY: debug
debug: NVCC_FLAGS += -g -G -O0
debug: $(TARGET)

.PHONY: check
check:
	@echo "🔍 Checking CUDA installation..."
	@nvcc --version || (echo "❌ CUDA not found" && exit 1)
	@nvidia-smi || (echo "❌ NVIDIA driver not found" && exit 1)
	@echo "✅ CUDA check passed"