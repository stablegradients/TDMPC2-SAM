#!/bin/bash

# Checkpoint Saving Test Script
# This script tests buffer saving and model/optimizer checkpoint saving functionality
# Runs shorter experiments to verify checkpoint creation and loading

# ======================== USER CONFIGURATION ========================

# Set your wandb entity and project
WANDB_ENTITY="stablegradients"
WANDB_PROJECT="checkpoint_test"

# Set model size (options: 1, 5, 19, 48, 317)
MODEL_SIZE=5

# Set number of training steps (shorter for testing)
STEPS=1000000

# Set observation type (state or rgb)
OBS_TYPE="state"

# Set evaluation frequency (more frequent for testing checkpoints)
EVAL_FREQ=100000

# Seeds to run (fewer for quick testing)
SEEDS=(
        #42 
        #123
        0 
       )

# ======================== DMC ENVIRONMENTS ========================
# Comment out environments you don't want to run by adding # at the beginning

DMC_ENVS=(
    #"walker-walk"
    #"cheetah-run"
    #"dog-walk"
    "dog-run"
    #"humanoid-walk"
    "humanoid-run"
    # Additional environments for more thorough testing
    #"reacher-easy"
    #"hopper-stand"
    #"pendulum-swingup"
    #"cartpole-balance"
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
    local gpu_id=$3
    local log_file=$4
    
    echo "🚀 Starting: $env, seed=$seed, GPU=$gpu_id"
    
    # Create experiment name for grouping
    exp_name="${env}_checkpoint_test"
    
    # Create save directory for this experiment
    save_dir="/zfsauton2/home/shrinivr/tdmpc2/buffer_logging/checkpoints/${exp_name}_seed${seed}"
    mkdir -p "$save_dir"
    
    # Run the experiment with GPU isolation
    CUDA_VISIBLE_DEVICES=$gpu_id python tdmpc2/train.py \
        task=$env \
        seed=$seed \
        model_size=$MODEL_SIZE \
        steps=$STEPS \
        obs=$OBS_TYPE \
        eval_freq=$EVAL_FREQ \
        wandb_entity=$WANDB_ENTITY \
        wandb_project=$WANDB_PROJECT \
        exp_name=$exp_name \
        save_path=$save_dir \
        enable_wandb=true \
        save_video=false \
        save_agent=true \
        compile=false
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed: $env, seed=$seed, GPU=$gpu_id"
        echo "📁 Checkpoints saved to: $save_dir"
        
        # Verify checkpoint files were created
        verify_checkpoints "$save_dir" "$env" "$seed"
    else
        echo "❌ Failed: $env, seed=$seed, GPU=$gpu_id (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Function to verify checkpoint files were created
verify_checkpoints() {
    local save_dir=$1
    local env=$2
    local seed=$3
    
    echo "🔍 Verifying checkpoints for $env, seed=$seed..."
    
    # Check for train buffers
    if [ -d "$save_dir/train" ]; then
        train_files=$(find "$save_dir/train" -name "*.pt" | wc -l)
        echo "  📊 Train buffers: $train_files files"
    else
        echo "  ⚠️  No train buffer directory found"
    fi
    
    # Check for test buffers
    if [ -d "$save_dir/test" ]; then
        test_files=$(find "$save_dir/test" -name "*.pt" | wc -l)
        echo "  📊 Test buffers: $test_files files"
    else
        echo "  ⚠️  No test buffer directory found"
    fi
    
    # Check for agent checkpoints
    if [ -d "$save_dir/agent" ]; then
        agent_files=$(find "$save_dir/agent" -name "*.pt" | wc -l)
        echo "  🧠 Agent checkpoints: $agent_files files"
        
        # Show latest checkpoint info
        latest_checkpoint=$(find "$save_dir/agent" -name "*.pt" -exec basename {} \; | sort -n | tail -1)
        if [ -n "$latest_checkpoint" ]; then
            echo "  📄 Latest checkpoint: $latest_checkpoint"
        fi
    else
        echo "  ⚠️  No agent checkpoint directory found"
    fi
    
    echo "  ✅ Verification complete"
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
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "STEPS: $STEPS"
echo "EVAL_FREQ: $EVAL_FREQ"
echo "OBS_TYPE: $OBS_TYPE"
echo "SEEDS: ${SEEDS[@]}"
echo "ENVIRONMENTS: ${#active_envs[@]} active environments"
echo "TOTAL EXPERIMENTS: $((${#active_envs[@]} * ${#SEEDS[@]}))"
echo "PARALLEL JOBS: Up to $NUM_GPUS (number of GPUs)"
echo "CHECKPOINT DIR: ./checkpoints/"
echo "============================================================"

# Ask for confirmation
read -p "Do you want to proceed with the checkpoint test? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Aborted."
    exit 0
fi

# Create results directory
mkdir -p results/checkpoint_test
mkdir -p checkpoints

# Create job list
job_list=()
for env in "${active_envs[@]}"; do
    for seed in "${SEEDS[@]}"; do
        job_list+=("$env:$seed")
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
            IFS=':' read -r env seed gpu_id <<< "$info"
            
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
    IFS=':' read -r env seed <<< "$job"
    
    log_file="results/checkpoint_test/${env}_seed${seed}_gpu${gpu_id}.log"
    
    # Launch job in background
    run_experiment "$env" "$seed" "$gpu_id" "$log_file" &
    pid=$!
    
    job_pids+=($pid)
    job_info+=("$env:$seed:$gpu_id")
    
    echo "Launched job $((job_index + 1))/$total_jobs: $env, seed=$seed on GPU $gpu_id (PID: $pid)"
    # Wait 5 seconds between job launches
    sleep 5
    
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
        IFS=':' read -r env seed <<< "$job"
        
        log_file="results/checkpoint_test/${env}_seed${seed}_gpu${freed_gpu}.log"
        
        # Launch job in background
        run_experiment "$env" "$seed" "$freed_gpu" "$log_file" &
        pid=$!
        
        job_pids+=($pid)
        job_info+=("$env:$seed:$freed_gpu")
        
        echo "Launched job $((job_index + 1))/$total_jobs: $env, seed=$seed on GPU $freed_gpu (PID: $pid)"
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

# Final checkpoint verification
echo "🔍 Final checkpoint verification..."
for env in "${active_envs[@]}"; do
    for seed in "${SEEDS[@]}"; do
        exp_name="${env}_checkpoint_test"
        save_dir="checkpoints/${exp_name}_seed${seed}"
        if [ -d "$save_dir" ]; then
            echo "📁 $save_dir:"
            find "$save_dir" -name "*.pt" -exec echo "  {}" \; | sort
        fi
    done
done

if [ $failed_jobs -gt 0 ]; then
    echo "Some experiments failed. Check the logs in results/checkpoint_test/ for details."
    exit 1
else
    echo "All experiments completed successfully!"
    echo "📊 Checkpoint files saved to ./checkpoints/"
    exit 0
fi 