#!/bin/bash

# Define input file
INPUT_FILE="data/baby.mp4"

# Define output directories
CPU_OUTPUT_DIR="cpp/build/output"
CUDA_OUTPUT_DIR="cuda/build/output"

# Create output directories if they don't exist
mkdir -p "$CPU_OUTPUT_DIR"
mkdir -p "$CUDA_OUTPUT_DIR"

# Define parameters (same for both implementations)
LEVELS=4
ALPHA=20
CUTOFF=20
FREQ_LOW=0.05
FREQ_HIGH=0.4
CHROM_ATT=0.1

echo "========================================="
echo "Benchmarking EVM CPU vs CUDA Implementation"
echo "========================================="
echo "Input: $INPUT_FILE"
echo "Pyramid Levels: $LEVELS"
echo "Alpha: $ALPHA"
echo "Lambda Cutoff: $CUTOFF"
echo "Frequency Range: [$FREQ_LOW, $FREQ_HIGH]"
echo "Chrominance Attenuation: $CHROM_ATT"
echo "========================================="

# Run CPU implementation with timing
echo "Running CPU implementation..."
CPU_OUTPUT="$CPU_OUTPUT_DIR/baby_cpu.mp4"
CPU_START=$(date +%s.%N)
./cpp/build/evmpipeline --input "$INPUT_FILE" --output "$CPU_OUTPUT" --level "$LEVELS" --alpha "$ALPHA" --lambda_cutoff "$CUTOFF" --fl "$FREQ_LOW" --fh "$FREQ_HIGH" --chrom_atten "$CHROM_ATT" --mode laplacian
CPU_END=$(date +%s.%N)
CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)
echo "CPU Time: $CPU_TIME seconds"

# Run CUDA implementation with timing
echo "Running CUDA implementation..."
CUDA_OUTPUT="$CUDA_OUTPUT_DIR/baby_cuda.mp4"
CUDA_START=$(date +%s.%N)
./cuda/build/evm_cuda -i "$INPUT_FILE" -o "$CUDA_OUTPUT" -l "$LEVELS" -a "$ALPHA" -c "$CUTOFF" -fl "$FREQ_LOW" -fh "$FREQ_HIGH" -ca "$CHROM_ATT"
CUDA_END=$(date +%s.%N)
CUDA_TIME=$(echo "$CUDA_END - $CUDA_START" | bc)
echo "CUDA Time: $CUDA_TIME seconds"

# Calculate speedup
SPEEDUP=$(echo "$CPU_TIME / $CUDA_TIME" | bc -l)
echo "========================================="
echo "CUDA Speedup: $SPEEDUP x"
echo "========================================="

# Check if output files were created
echo "CPU output file size: $(du -h "$CPU_OUTPUT" 2>/dev/null | cut -f1) ($(stat -c%s "$CPU_OUTPUT" 2>/dev/null || echo "failed") bytes)"
echo "CUDA output file size: $(du -h "$CUDA_OUTPUT" 2>/dev/null | cut -f1) ($(stat -c%s "$CUDA_OUTPUT" 2>/dev/null || echo "failed") bytes)"