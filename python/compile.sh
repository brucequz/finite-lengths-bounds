#!/bin/bash

INPUT_FILE="lib/kernel.cu"
OUTPUT_FILE="lib/libtrellis.so"

echo "Compiling $INPUT_FILE for modern architecture..."

# -gencode arch=compute_XX,code=sm_XX tells nvcc which hardware to support
# sm_75 covers RTX 20-series and T4 GPUs. 
# Use sm_80 for A100 or sm_86 for RTX 30-series.
nvcc -shared -Xcompiler -fPIC -O3 \
    -gencode arch=compute_89,code=sm_89 \
    "$INPUT_FILE" -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Compilation successful: $OUTPUT_FILE"
else
    echo "Compilation failed."
    exit 1
fi