#!/bin/bash

# SAM Ablation Script for TDMPC2
# This script runs ablation experiments with different SAM rho values
# Supports checkpoint saving to the buffer_logging directory structure

# ======================== USER CONFIGURATION ========================

# Set your rho values for ablation
RHO_VALUES=(
   0.0000005
   0.0000001
)

# You can also use scientific notation in comments or define them differently:
# RHO_VALUES=(1e-6 1e-5 1e-4 1e-3 1e-2 5e-2 1e-1 5e-1)

# Set your wandb entity and project
WANDB_ENTITY="stablegradients"
WANDB_PROJECT="sam_ablation_tdmpc2_buffer_neurips-rebuttal"

# Set model size (options: 1, 5, 19, 48, 317)
MODEL_SIZE=5

# Set number of training steps
STEPS=1000000  # 10M steps

# Set observation type (state or rgb)
OBS_TYPE="state"

# Set evaluation frequency
EVAL_FREQ=50000

# Save frequency for checkpoints (set to same as eval_freq for regular saving)
SAVE_FREQ=200000  # Save every 300k steps

# Seeds to run
SEEDS=(
    #0
    #42
    123
    456
    #789
)

# Base directory for saving checkpoints
CHECKPOINT_BASE_DIR="/zfsauton2/home/shrinivr/tdmpc2/neurips_rebuttal/sam_checkpoints"

# ======================== ENVIRONMENTS ========================
# Comment out environments you don't want to run by adding # at the beginning

ENVS=(
    #"dog-run"
    #"dog-walk"
    #"dog-trot"
    #"humanoid-walk"
    "humanoid-run"
    
    # Walker tasks
    #"walker-stand"
    #"walker-walk" 
    #"walker-run"
    
    # Cheetah tasks
    #"cheetah-run"
    
    # Reacher tasks
    #"reacher-easy"
    #"reacher-hard"
    
    # Hopper tasks
    #"hopper-stand"
    #"hopper-hop"
    
    # Other tasks...
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
    exp_name="${env}_sam_rho${rho}_seed${seed}"
    
    # Create save directory for this experiment
    save_dir="${CHECKPOINT_BASE_DIR}/${exp_name}"
    mkdir -p "$save_dir"
    
    # Log start time
    start_time=$(date +%s)
    
    # Run the experiment with GPU isolation
    CUDA_VISIBLE_DEVICES=$gpu_id python tdmpc2/train.py \
        task=$env \
        seed=$seed \
        sam_rho=$rho \
        optimizer=SAM \
        model_size=$MODEL_SIZE \
        steps=$STEPS \
        obs=$OBS_TYPE \
        eval_freq=$EVAL_FREQ \
        save_freq=$SAVE_FREQ \
        wandb_entity=$WANDB_ENTITY \
        wandb_project=$WANDB_PROJECT \
        exp_name=$exp_name \
        save_path=$save_dir \
        enable_wandb=true \
        save_video=false \
        save_agent=true \
        save_buffer=true \
        compile=false \
        2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed: $env, seed=$seed, rho=$rho, GPU=$gpu_id (Duration: $(format_duration $duration))"
        echo "📁 Checkpoints saved to: $save_dir"
        
        # Verify checkpoint files were created
        verify_checkpoints "$save_dir" "$env" "$seed" "$rho"
    else
        echo "❌ Failed: $env, seed=$seed, rho=$rho, GPU=$gpu_id (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Function to format duration in human-readable format
format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Function to verify checkpoint files were created
verify_checkpoints() {
    local save_dir=$1
    local env=$2
    local seed=$3
    local rho=$4
    
    echo "🔍 Verifying checkpoints for $env, seed=$seed, rho=$rho..."
    
    # Check for train buffers
    if [ -d "$save_dir/train" ]; then
        train_files=$(find "$save_dir/train" -name "*.pt" | wc -l)
        echo "  📊 Train buffers: $train_files files"
        
        # Show latest train buffer
        latest_train=$(find "$save_dir/train" -name "*.pt" -exec basename {} \; | sort -n | tail -1)
        if [ -n "$latest_train" ]; then
            echo "     Latest: $latest_train"
        fi
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
            echo "     Latest: $latest_checkpoint"
            
            # Get file size
            checkpoint_path="$save_dir/agent/$latest_checkpoint"
            if [ -f "$checkpoint_path" ]; then
                size=$(du -h "$checkpoint_path" | cut -f1)
                echo "     Size: $size"
            fi
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
for env in "${ENVS[@]}"; do
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
echo "EVAL_FREQ: $EVAL_FREQ"
echo "SAVE_FREQ: $SAVE_FREQ"
echo "OBS_TYPE: $OBS_TYPE"
echo "SEEDS: ${SEEDS[@]}"
echo "ENVIRONMENTS: ${#active_envs[@]} active (${active_envs[@]})"
echo "TOTAL EXPERIMENTS: $((${#active_envs[@]} * ${#SEEDS[@]} * ${#RHO_VALUES[@]}))"
echo "PARALLEL JOBS: Up to $NUM_GPUS (number of GPUs)"
echo "CHECKPOINT DIR: $CHECKPOINT_BASE_DIR"
echo "============================================================"

# Ask for confirmation
read -p "Do you want to proceed with the SAM ablation? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Aborted."
    exit 0
fi

# Create results directory
mkdir -p results/sam_ablation
mkdir -p "$CHECKPOINT_BASE_DIR"

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
start_time=$(date +%s)

echo "Starting $total_jobs experiments with parallel execution..."
echo "Start time: $(date)"

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
    
    log_file="results/sam_ablation/${env}_seed${seed}_rho${rho}_gpu${gpu_id}.log"
    
    # Launch job in background
    run_experiment "$env" "$seed" "$rho" "$gpu_id" "$log_file" &
    pid=$!
    
    job_pids+=($pid)
    job_info+=("$env:$seed:$rho:$gpu_id")
    
    echo "Launched job $((job_index + 1))/$total_jobs: $env, seed=$seed, rho=$rho on GPU $gpu_id (PID: $pid)"
    
    # Wait a bit between job launches to avoid resource conflicts
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
        IFS=':' read -r env seed rho <<< "$job"
        
        log_file="results/sam_ablation/${env}_seed${seed}_rho${rho}_gpu${freed_gpu}.log"
        
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
    
    # Print progress with time estimate
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    completed=$((completed_jobs + failed_jobs))
    
    if [ $completed -gt 0 ]; then
        avg_time_per_job=$((elapsed / completed))
        remaining_jobs=$((total_jobs - completed - ${#job_pids[@]}))
        estimated_remaining=$((remaining_jobs * avg_time_per_job / NUM_GPUS))
        
        echo "Progress: $completed/$total_jobs completed, ${#job_pids[@]} running"
        echo "  Elapsed: $(format_duration $elapsed)"
        echo "  Estimated remaining: $(format_duration $estimated_remaining)"
    else
        echo "Progress: $completed/$total_jobs completed, ${#job_pids[@]} running"
    fi
done

# Wait for all remaining jobs to complete
echo "Waiting for remaining jobs to complete..."
while [ ${#job_pids[@]} -gt 0 ]; do
    wait_for_job_completion
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    echo "Progress: $((completed_jobs + failed_jobs))/$total_jobs completed, ${#job_pids[@]} running"
    echo "  Elapsed: $(format_duration $elapsed)"
    sleep 2
done

# Final summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo "======================== SUMMARY ========================"
echo "Total experiments: $total_jobs"
echo "Completed: $completed_jobs"
echo "Failed: $failed_jobs"
echo "Total duration: $(format_duration $total_duration)"
echo "End time: $(date)"
echo "========================================================"

# Final checkpoint verification summary
echo ""
echo "📊 Checkpoint Summary by Rho Value:"
for rho in "${RHO_VALUES[@]}"; do
    echo ""
    echo "Rho = $rho:"
    for env in "${active_envs[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp_name="${env}_sam_rho${rho}_seed${seed}"
            save_dir="${CHECKPOINT_BASE_DIR}/${exp_name}"
            if [ -d "$save_dir/agent" ]; then
                latest=$(find "$save_dir/agent" -name "*.pt" -exec basename {} \; | sort -n | tail -1)
                if [ -n "$latest" ]; then
                    echo "  ✓ $env (seed $seed): checkpoint $latest"
                else
                    echo "  ✗ $env (seed $seed): no checkpoints found"
                fi
            else
                echo "  ✗ $env (seed $seed): directory not found"
            fi
        done
    done
done

if [ $failed_jobs -gt 0 ]; then
    echo ""
    echo "⚠️  Some experiments failed. Check the logs in results/sam_ablation/ for details."
    exit 1
else
    echo ""
    echo "✅ All experiments completed successfully!"
    echo "📁 All checkpoints saved to: $CHECKPOINT_BASE_DIR"
    exit 0
fi 