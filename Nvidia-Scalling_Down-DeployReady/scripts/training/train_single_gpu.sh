#!/bin/bash

echo "=== Quick Single GPU Training ==="
echo "Using only GPU 0 for faster training"

# Use only first GPU
export CUDA_VISIBLE_DEVICES=0

# Add environment variables for better performance
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# Run training with optimized settings
python bert_training.py \
    --dataset phrasebank \
    --data_portion "${1:-1.0}" \
    --output_report ../../results

echo "Single GPU training completed!"
