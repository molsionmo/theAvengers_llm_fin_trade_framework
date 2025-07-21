#!/bin/bash

echo "=== GPU Performance Comparison Script ==="
echo "This script will help you compare single vs multi-GPU performance."
echo

# Function to run training with specific GPU settings
run_training_test() {
    local gpu_config=$1
    local test_name=$2
    local data_portion=0.01  # Small dataset for quick testing
    
    echo "🚀 Running $test_name..."
    echo "GPU Config: $gpu_config"
    
    export CUDA_VISIBLE_DEVICES=$gpu_config
    
    start_time=$(date +%s)
    timeout 60s python bert_training.py \
        --dataset phrasebank \
        --data_portion $data_portion \
        --output_report ../../results \
        > "test_${test_name}.log" 2>&1
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "✅ $test_name completed in ${duration}s"
    echo "Check test_${test_name}.log for details"
    echo
}

echo "Starting performance comparison tests..."
echo "Each test will run for max 60 seconds with 1% of data"
echo

# Test 1: Single GPU (GPU 0)
run_training_test "0" "single_gpu_0"

# Test 2: Single GPU (GPU 1) 
run_training_test "1" "single_gpu_1"

# Test 3: Both GPUs
run_training_test "0,1" "dual_gpu"

echo "=== Performance Test Summary ==="
echo "Check the generated log files:"
echo "- test_single_gpu_0.log"
echo "- test_single_gpu_1.log" 
echo "- test_dual_gpu.log"
echo
echo "Look for training speed metrics in each log file to compare performance."

# Optional: Show GPU utilization
echo "Current GPU status:"
nvidia-smi
