#!/bin/bash

# DMC Environments Ablation Script - Multi-Rho Parallel Version
# This script runs ablation experiments on DMC environments with TRUE parallel execution
# Supports multiple rho values for comprehensive ablation studies

# ======================== USER CONFIGURATION ========================

# Set your rho values for ablation (you can add multiple values)
RHO_VALUES=(
            0.001 
            #0.0025 
            #0.005 
            #0.01 
            #0.025 
            #0.05 
            
            )
# Example for multiple rho values:
# RHO_VALUES=(0.1 0.3 0.5 0.7 0.9)

# Set your wandb entity and project
WANDB_ENTITY="stablegradients"
WANDB_PROJECT="dmc_ablation"

# Set model size (options: 1, 5, 19, 48, 317)
MODEL_SIZE=5

# Set number of training steps
STEPS=2000000

# Set observation type (state or rgb)
OBS_TYPE="state"

# Seeds to run
SEEDS=(
        42 
        123 
        456 
        789
       )

# ======================== DMC ENVIRONMENTS ========================
# Comment out environments you don't want to run by adding # at the beginning

DMC_ENVS=(
    "dog-walk"
    #"dog-run"
    #"dog-trot"
    #"humanoid-walk"
    #"humanoid-run"
    
    # Walker tasks
    # "walker-stand"
    # "walker-walk" 
    # "walker-run"
    
    # # Cheetah tasks
    # "cheetah-run"
    
    # # Reacher tasks
    # "reacher-easy"
    # "reacher-hard"
    
    # # Hopper tasks
    # "hopper-stand"
    # "hopper-hop"
    
    # # Pendulum tasks
    # "pendulum-swingup"
    
    # # Cartpole tasks
    # "cartpole-balance"
    # "cartpole-balance-sparse"
    # "cartpole-swingup"
    # "cartpole-swingup-sparse"
    
    # # Acrobot tasks
    # "acrobot-swingup"
    
    # # Ball in cup tasks
    # "cup-catch"
    
    # # Finger tasks
    # "finger-spin"
    # "finger-turn-easy"
    # "finger-turn-hard"
    
    # # Fish tasks
    # "fish-swim"
)

# ======================== FUNCTIONS ========================

# Function to get number of available GPUs
get_num_gpus() {
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
}

# Function to run a single experiment
run_experiment() {
    local env=$1
    local seed=$2
    local rho=$3
    local gpu_id=$4
    local log_file=$5
    
    echo "🚀 Starting: $env, seed=$seed, rho=$rho, GPU=$gpu_id"
    
    # Create experiment name for grouping
    exp_name="${env}_rho${rho}"
    
    # Run the experiment with GPU isolation
    CUDA_VISIBLE_DEVICES=$gpu_id python tdmpc2/train.py \
        task=$env \
        seed=$seed \
        sam_rho=$rho \
        optimizer=SAM \
        model_size=$MODEL_SIZE \
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
        echo "✅ Completed: $env, seed=$seed, rho=$rho, GPU=$gpu_id"
    else
        echo "❌ Failed: $env, seed=$seed, rho=$rho, GPU=$gpu_id (exit code: $exit_code)"
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

# Get number of GPUs
NUM_GPUS=$(get_num_gpus)
echo "Detected $NUM_GPUS GPUs"

# Count active environments (not commented out)
active_envs=()
for env in "${DMC_ENVS[@]}"; do
    if [[ ! $env =~ ^#.*$ ]]; then
        active_envs+=("$env")
    fi
done

# Print configuration
echo "======================== CONFIGURATION ========================"
echo "RHO VALUES: ${RHO_VALUES[@]}"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "STEPS: $STEPS"
echo "OBS_TYPE: $OBS_TYPE"
echo "SEEDS: ${SEEDS[@]}"
echo "ENVIRONMENTS: ${#active_envs[@]} active environments"
echo "TOTAL EXPERIMENTS: $((${#active_envs[@]} * ${#SEEDS[@]} * ${#RHO_VALUES[@]}))"
echo "PARALLEL JOBS: Up to $NUM_GPUS (number of GPUs)"
echo "============================================================"

# Ask for confirmation
read -p "Do you want to proceed with the ablation? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Aborted."
    exit 0
fi

# Create results directory
mkdir -p results/dmc_ablation_multi_rho

# Create job list
job_list=()
for env in "${active_envs[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for rho in "${RHO_VALUES[@]}"; do
            job_list+=("$env:$seed:$rho")
        done
    done
done

total_jobs=${#job_list[@]}
completed_jobs=0
failed_jobs=0

echo "Starting $total_jobs experiments with parallel execution..."

# Launch all jobs in parallel, but limit to NUM_GPUS concurrent jobs
job_index=0
declare -a job_pids=()
declare -a job_info=()

# Function to wait for any job to complete and get its result
wait_for_job_completion() {
    for i in "${!job_pids[@]}"; do
        pid=${job_pids[$i]}
        if ! kill -0 $pid 2>/dev/null; then
            # Job completed
            wait $pid
            exit_code=$?
            
            info=${job_info[$i]}
            IFS=':' read -r env seed rho gpu_id <<< "$info"
            
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

# Launch initial batch of jobs
echo "Launching initial batch..."
gpu_id=0
while [ $job_index -lt $total_jobs ] && [ $gpu_id -lt $NUM_GPUS ]; do
    job=${job_list[$job_index]}
    IFS=':' read -r env seed rho <<< "$job"
    
    log_file="results/dmc_ablation_multi_rho/${env}_seed${seed}_rho${rho}_gpu${gpu_id}.log"
    
    # Launch job in background
    run_experiment "$env" "$seed" "$rho" "$gpu_id" "$log_file" &
    pid=$!
    
    job_pids+=($pid)
    job_info+=("$env:$seed:$rho:$gpu_id")
    
    echo "Launched job $((job_index + 1))/$total_jobs: $env, seed=$seed, rho=$rho on GPU $gpu_id (PID: $pid)"
    
    ((job_index++))
    ((gpu_id++))
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
        IFS=':' read -r env seed rho <<< "$job"
        
        log_file="results/dmc_ablation_multi_rho/${env}_seed${seed}_rho${rho}_gpu${freed_gpu}.log"
        
        # Launch job in background
        run_experiment "$env" "$seed" "$rho" "$freed_gpu" "$log_file" &
        pid=$!
        
        job_pids+=($pid)
        job_info+=("$env:$seed:$rho:$freed_gpu")
        
        echo "Launched job $((job_index + 1))/$total_jobs: $env, seed=$seed, rho=$rho on GPU $freed_gpu (PID: $pid)"
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
    echo "Some experiments failed. Check the logs in results/dmc_ablation_multi_rho/ for details."
    exit 1
else
    echo "All experiments completed successfully!"
    exit 0
fi 