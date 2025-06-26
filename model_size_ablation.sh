#!/bin/bash

# Model Size Ablation Script for TDMPC2
# This script runs ablation experiments on DMC environments with different model sizes
# Uses the original TDMPC2 implementation (Adam optimizer) for comparison

# ======================== USER CONFIGURATION ========================

# Set your model sizes for ablation (1, 5, 19, 48, 317 million parameters)
MODEL_SIZES=(
     1 
    # 5 
    19 
    48 
    #317
)

# Set which GPUs to use (specify GPU IDs you want to use)
# Example: GPU_IDS=(0 1 2) to use GPUs 0, 1, and 2
# Example: GPU_IDS=(1 3 5) to use GPUs 1, 3, and 5
# Comment/uncomment GPU IDs as needed
GPU_IDS=(
    #0
    1
    2
    3
    # 4
    # 5
    # 6
    # 7
)

# Set your wandb entity and project
WANDB_ENTITY="stablegradients"
WANDB_PROJECT="tdmpc2_model_size_ablation"

# Set number of training steps
STEPS=4000000

# Set observation type (state or rgb)
OBS_TYPE="state"

# Seeds to run (3 seeds as requested)
SEEDS=(
    #42 
    123 
    #456
    #789
)

# ======================== DMC ENVIRONMENTS ========================
# Environments as requested: dog-run, dog-walk, dog-trot, humanoid-walk, humanoid-run

DMC_ENVS=(
    #"dog-run"
    #"dog-walk"
    #"dog-trot"
    #"humanoid-walk"
    "humanoid-run"
)

# ======================== FUNCTIONS ========================

# Function to get number of available GPUs (for reference only)
get_num_gpus() {
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
}

# Function to validate GPU IDs
validate_gpu_ids() {
    local available_gpus=$(get_num_gpus)
    local invalid_gpus=()
    
    for gpu_id in "${GPU_IDS[@]}"; do
        if [ $gpu_id -ge $available_gpus ]; then
            invalid_gpus+=($gpu_id)
        fi
    done
    
    if [ ${#invalid_gpus[@]} -gt 0 ]; then
        echo "Error: Invalid GPU IDs specified: ${invalid_gpus[@]}"
        echo "Available GPUs: 0 to $((available_gpus - 1))"
        exit 1
    fi
}

# Function to run a single experiment
run_experiment() {
    local env=$1
    local seed=$2
    local model_size=$3
    local gpu_id=$4
    local log_file=$5
    
    echo "🚀 Starting: $env, seed=$seed, model_size=${model_size}M, GPU=$gpu_id"
    
    # Create experiment name for grouping
    exp_name="${env}_model${model_size}M"
    
    # Run the experiment with GPU isolation using original TDMPC2 (Adam optimizer)
    CUDA_VISIBLE_DEVICES=$gpu_id python tdmpc2/train.py \
        task=$env \
        seed=$seed \
        optimizer=Adam \
        model_size=$model_size \
        steps=$STEPS \
        obs=$OBS_TYPE \
        wandb_entity=$WANDB_ENTITY \
        wandb_project=$WANDB_PROJECT \
        exp_name=$exp_name \
        enable_wandb=true \
        save_video=false \
        compile=false
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed: $env, seed=$seed, model_size=${model_size}M, GPU=$gpu_id"
    else
        echo "❌ Failed: $env, seed=$seed, model_size=${model_size}M, GPU=$gpu_id (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# ======================== MAIN EXECUTION ========================

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA is required for this script."
    exit 1
fi

# Check if the tdmpc2 directory exists
if [ ! -d "tdmpc2" ]; then
    echo "Error: tdmpc2 directory not found. Please run this script from the project root."
    exit 1
fi

# Validate GPU IDs
validate_gpu_ids

# Get number of selected GPUs
NUM_SELECTED_GPUS=${#GPU_IDS[@]}
TOTAL_AVAILABLE_GPUS=$(get_num_gpus)

# Print configuration
echo "======================== CONFIGURATION ========================"
echo "MODEL SIZES: ${MODEL_SIZES[@]} (million parameters)"
echo "SELECTED GPUs: ${GPU_IDS[@]} (using $NUM_SELECTED_GPUS out of $TOTAL_AVAILABLE_GPUS available)"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "OPTIMIZER: Adam (original TDMPC2)"
echo "STEPS: $STEPS"
echo "OBS_TYPE: $OBS_TYPE"
echo "SEEDS: ${SEEDS[@]}"
echo "ENVIRONMENTS: ${DMC_ENVS[@]}"
echo "TOTAL EXPERIMENTS: $((${#DMC_ENVS[@]} * ${#SEEDS[@]} * ${#MODEL_SIZES[@]}))"
echo "PARALLEL JOBS: Up to $NUM_SELECTED_GPUS (number of selected GPUs)"
echo "============================================================"

# Ask for confirmation
read -p "Do you want to proceed with the model size ablation? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Aborted."
    exit 0
fi

# Create results directory
mkdir -p results/model_size_ablation

# Create job list
job_list=()
for env in "${DMC_ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for model_size in "${MODEL_SIZES[@]}"; do
            job_list+=("$env:$seed:$model_size")
        done
    done
done

total_jobs=${#job_list[@]}
completed_jobs=0
failed_jobs=0

echo "Starting $total_jobs experiments with parallel execution..."

# Launch all jobs in parallel, but limit to NUM_SELECTED_GPUS concurrent jobs
job_index=0
declare -a job_pids=()
declare -a job_info=()
declare -a available_gpus=("${GPU_IDS[@]}")  # Copy of GPU_IDS for tracking availability

# Function to wait for any job to complete and get its result
wait_for_job_completion() {
    for i in "${!job_pids[@]}"; do
        pid=${job_pids[$i]}
        if ! kill -0 $pid 2>/dev/null; then
            # Job completed
            wait $pid
            exit_code=$?
            
            info=${job_info[$i]}
            IFS=':' read -r env seed model_size gpu_id <<< "$info"
            
            if [ $exit_code -eq 0 ]; then
                ((completed_jobs++))
            else
                ((failed_jobs++))
            fi
            
            # Remove completed job from arrays
            unset job_pids[$i]
            unset job_info[$i]
            
            return $gpu_id  # Return freed GPU ID
        fi
    done
    return -1  # No jobs completed
}

# Function to get next available GPU from selected GPUs
get_next_gpu() {
    for gpu_id in "${GPU_IDS[@]}"; do
        # Check if this GPU is currently in use
        local in_use=false
        for info in "${job_info[@]}"; do
            if [[ $info == *":$gpu_id" ]]; then
                in_use=true
                break
            fi
        done
        if [ "$in_use" = false ]; then
            return $gpu_id
        fi
    done
    return -1  # No available GPU
}

# Launch initial batch of jobs
echo "Launching initial batch..."
while [ $job_index -lt $total_jobs ] && [ ${#job_pids[@]} -lt $NUM_SELECTED_GPUS ]; do
    job=${job_list[$job_index]}
    IFS=':' read -r env seed model_size <<< "$job"
    
    get_next_gpu
    gpu_id=$?
    
    if [ $gpu_id -ge 0 ]; then
        log_file="results/model_size_ablation/${env}_seed${seed}_model${model_size}M_gpu${gpu_id}.log"
        
        # Launch job in background
        run_experiment "$env" "$seed" "$model_size" "$gpu_id" "$log_file" &
        pid=$!
        
        job_pids+=($pid)
        job_info+=("$env:$seed:$model_size:$gpu_id")
        
        echo "Launched job $((job_index + 1))/$total_jobs: $env, seed=$seed, model_size=${model_size}M on GPU $gpu_id (PID: $pid)"
        
        ((job_index++))
    else
        break  # No available GPU
    fi
done

echo "Initial batch launched. ${#job_pids[@]} jobs running in parallel."

# Process remaining jobs
while [ $job_index -lt $total_jobs ]; do
    # Wait for a job to complete
    wait_for_job_completion
    freed_gpu=$?
    
    if [ $freed_gpu -ge 0 ]; then
        # Launch next job on freed GPU
        job=${job_list[$job_index]}
        IFS=':' read -r env seed model_size <<< "$job"
        
        log_file="results/model_size_ablation/${env}_seed${seed}_model${model_size}M_gpu${freed_gpu}.log"
        
        # Launch job in background
        run_experiment "$env" "$seed" "$model_size" "$freed_gpu" "$log_file" &
        pid=$!
        
        job_pids+=($pid)
        job_info+=("$env:$seed:$model_size:$freed_gpu")
        
        echo "Launched job $((job_index + 1))/$total_jobs: $env, seed=$seed, model_size=${model_size}M on GPU $freed_gpu (PID: $pid)"
        ((job_index++))
    else
        sleep 2  # Wait a bit before checking again
    fi
    
    # Print progress
    echo "Progress: $((completed_jobs + failed_jobs))/$total_jobs completed, ${#job_pids[@]} running"
done

# Wait for all remaining jobs to complete
echo "Waiting for remaining jobs to complete..."
while [ ${#job_pids[@]} -gt 0 ]; do
    wait_for_job_completion
    echo "Progress: $((completed_jobs + failed_jobs))/$total_jobs completed, ${#job_pids[@]} running"
    sleep 2
done

# Final summary
echo "======================== SUMMARY ========================"
echo "Total experiments: $total_jobs"
echo "Completed: $completed_jobs"
echo "Failed: $failed_jobs"
echo "========================================================"

if [ $failed_jobs -gt 0 ]; then
    echo "Some experiments failed. Check the logs in results/model_size_ablation/ for details."
    exit 1
else
    echo "All experiments completed successfully!"
    exit 0
fi 