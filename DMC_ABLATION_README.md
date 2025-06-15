# DMC Environments Multi-Rho Ablation Script

This script automates running ablation experiments on DeepMind Control (DMC) environments with TD-MPC2, supporting multiple rho values for comprehensive ablation studies.

## Features

- **Multi-Rho Ablation**: Supports multiple rho values for comprehensive ablation studies
- **True Parallel Execution**: Runs exactly N experiments simultaneously (where N = number of GPUs)
- **Automated GPU Management**: Automatically detects and assigns GPUs to experiments
- **Wandb Grouping**: Groups experiments by environment and rho (seeds are separate runs in same group)
- **User-configurable Parameters**: Set multiple rho values, wandb settings, and other hyperparameters
- **Flexible Environment Selection**: Comment out environments you don't want to run
- **Progress Tracking**: Shows real-time progress with running/pending job counts
- **Error Handling**: Tracks successful and failed experiments with detailed logging
- **Optimized for Speed**: Video logging disabled by default to reduce overhead

## Quick Start

1. **Configure the script** by editing the USER CONFIGURATION section in `dmc_ablation_simple.sh`:
   ```bash
   # Set your rho values for ablation (you can add multiple values)
   RHO_VALUES=(0.5)
   # Example for multiple rho values:
   # RHO_VALUES=(0.1 0.3 0.5 0.7 0.9)
   
   # Set your wandb entity and project
   WANDB_ENTITY="your_entity_name"
   WANDB_PROJECT="dmc_ablation"
   ```

2. **Select environments** by commenting out environments you don't want to run:
   ```bash
   DMC_ENVS=(
       "walker-stand"
       "walker-walk" 
       # "walker-run"  # This will be skipped
       "cheetah-run"
       # ... more environments
   )
   ```

3. **Run the script**:
   ```bash
   ./dmc_ablation_simple.sh
   ```

## Configuration Options

### Core Parameters
- `RHO_VALUES`: Array of rho parameters for your ablation study (e.g., `(0.1 0.3 0.5 0.7 0.9)`)
- `WANDB_ENTITY`: Your wandb entity name
- `WANDB_PROJECT`: Your wandb project name
- `MODEL_SIZE`: Model size (1, 5, 19, 48, 317)
- `STEPS`: Number of training steps
- `OBS_TYPE`: Observation type ("state" or "rgb")
- `SEEDS`: Array of seeds to run (default: 42, 123, 456, 789)

### Available DMC Environments

The script includes all standard DMC environments:

**Core Environments:**
- Walker: `walker-stand`, `walker-walk`, `walker-run`
- Cheetah: `cheetah-run`
- Reacher: `reacher-easy`, `reacher-hard`
- Hopper: `hopper-stand`, `hopper-hop`
- Pendulum: `pendulum-swingup`
- Cartpole: `cartpole-balance`, `cartpole-swingup` (and sparse variants)
- Acrobot: `acrobot-swingup`
- Ball in Cup: `cup-catch`
- Finger: `finger-spin`, `finger-turn-easy`, `finger-turn-hard`
- Fish: `fish-swim`

**Custom Environments (commented out by default):**
- Custom cheetah tasks: `cheetah-run-backwards`, `cheetah-jump`, etc.
- Custom walker tasks: `walker-walk-backwards`, etc.
- Custom hopper tasks: `hopper-hop-backwards`
- Custom reacher tasks: `reacher-three-easy`, `reacher-three-hard`
- Custom pendulum/cup tasks: `pendulum-spin`, `cup-spin`

## GPU Management

The script automatically:
- Detects the number of available GPUs
- Runs up to N experiments in parallel (where N = number of GPUs)
- Assigns one GPU per experiment
- Monitors job completion and frees up GPUs for new experiments
- Tracks GPU usage to prevent conflicts

## Output

- **Progress tracking**: Shows running/pending job counts and overall progress
- **Results directory**: Creates `results/dmc_ablation_multi_rho/` directory
- **Individual logs**: Each experiment gets its own log file with format `{env}_seed{seed}_rho{rho}_gpu{gpu}.log`
- **Experiment naming**: Uses format `{env}_rho{rho}` (seeds become different runs in same group)
- **Wandb grouping**: Groups experiments as `{task}-{env}_rho{rho}` (seeds grouped together, separate groups for different rho values)
- **Final summary**: Shows total completed and failed experiments

## Example Usage

```bash
# 1. Edit the script to set your parameters
vim dmc_ablation_simple.sh

# 2. Run the ablation
./dmc_ablation_simple.sh

# Example output:
# Detected 4 GPUs
# ======================== CONFIGURATION ========================
# RHO VALUES: 0.1 0.5 0.9
# WANDB_ENTITY: stablegradients
# WANDB_PROJECT: dmc_ablation
# MODEL_SIZE: 5
# STEPS: 1000000
# OBS_TYPE: state
# SEEDS: 42 123 456 789
# ENVIRONMENTS: 1 active environments
# TOTAL EXPERIMENTS: 12
# PARALLEL JOBS: Up to 4 (number of GPUs)
# ============================================================
```

## Notes

- **True parallel execution**: Exactly N experiments run simultaneously (N = number of GPUs), new jobs launch immediately when others complete
- **Wandb grouping**: Experiments with the same environment and rho are grouped together in wandb, with different seeds as separate runs within the group
- **Individual logging**: Each experiment gets its own log file for detailed debugging
- **Failed experiments**: Tracked and reported in the final summary with exit codes
- **Job management**: Uses bash job control to monitor background processes
- **Video logging**: Disabled by default (`save_video=false`) to reduce training overhead

## Troubleshooting

- **GPU issues**: Make sure `nvidia-smi` is available and GPUs are accessible
- **Environment not found**: Check that the environment name matches exactly
- **Memory issues**: Reduce `MODEL_SIZE` or ensure sufficient GPU memory
- **Wandb issues**: Verify your wandb credentials and project settings 