#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import sys

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
tdmpc2_dir = os.path.join(project_root, 'tdmpc2')
sys.path.insert(0, tdmpc2_dir)

from tdmpc2 import TDMPC2
from common.buffer import Buffer
from common.parser import parse_cfg

def compute_model_error(agent_checkpoint, buffer_checkpoint, cfg_path=None, task=None, horizon=3, device='cuda:0'):
    """
    Compute the average model error (MSE) on a test buffer using a trained agent.
    
    Args:
        agent_checkpoint: Path to agent checkpoint (.pt file)
        buffer_checkpoint: Path to buffer checkpoint (.pt file)
        cfg_path: Optional path to config file. If not provided, will try to load from checkpoint
        task: Task name (required if not in config)
        horizon: Prediction horizon (default: 3)
        device: Device to run on (default: 'cuda:0')
    
    Returns:
        Dictionary with error statistics
    """
    
    # Load configuration
    if cfg_path:
        cfg = OmegaConf.load(cfg_path)
    else:
        # Try to infer config from checkpoint
        print("No config path provided. Using default configuration...")
        cfg = OmegaConf.create({
            'task': task or 'dog-run',
            'obs': 'state',
            'model_size': 5,
            'horizon': horizon,
            'batch_size': 256,
            'buffer_size': 1_000_000,
            'steps': 10_000_000,
            'seed': 1,
            'multitask': False,
            'compile': False,
            'episodic': False,
            'lr': 3e-4,
            'enc_lr_scale': 0.3,
            'grad_clip_norm': 20,
            'tau': 0.01,
            'discount_denom': 5,
            'discount_min': 0.95,
            'discount_max': 0.995,
            'exp_name': 'default',
            'iterations': 6,
            'num_samples': 512,
            'num_elites': 64,
            'num_pi_trajs': 24,
            'min_std': 0.05,
            'max_std': 2,
            'temperature': 0.5,
            'num_bins': 101,
            'vmin': -10,
            'vmax': 10,
            'consistency_coef': 20,
            'reward_coef': 0.1,
            'value_coef': 0.1,
            'termination_coef': 1,
            'rho': 0.5,
            'num_q': 5,
            'entropy_coef': 1e-4,
            'log_std_min': -10,
            'log_std_max': 2,
            'dropout': 0.01,
            'simnorm_dim': 8,
            'num_channels': 32,
            'enc_dim': 256,
            'mlp_dim': 512,
            'latent_dim': 512,
            'num_enc_layers': 2,
            'mpc': True,
        })
    
    # Set device
    cfg.device = device
    
    # Get environment parameters by creating a temporary env
    from envs import make_env
    
    # Create temporary config for environment
    temp_cfg = OmegaConf.create({'task': cfg.task, 'obs': cfg.obs, 'seed': cfg.seed, 'multitask': False})
    
    # Set virtual display to avoid display errors
    os.environ['MUJOCO_GL'] = 'egl'
    
    temp_env = make_env(temp_cfg)
    
    # Get action dim and episode length from environment
    cfg.action_dim = temp_env.action_space.shape[0]
    cfg.episode_length = temp_env.max_episode_steps
    cfg.obs_shape = {cfg.obs: temp_env.observation_space.shape}
    
    # Try to close environment if it has a close method
    try:
        temp_env.close()
    except AttributeError:
        pass
    
    # Parse config manually without Hydra
    # Add required fields that parse_cfg would normally add
    cfg.work_dir = Path.cwd() / 'logs' / cfg.task / str(cfg.seed) / cfg.get('exp_name', 'default')
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.multitask = False  # Single task for now
    cfg.tasks = [cfg.task]
    cfg.task_dim = 0  # No task embedding for single task
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    
    # Model size configuration
    from common import MODEL_SIZE
    if cfg.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
    
    # Convert to dataclass
    from common.parser import cfg_to_dataclass
    cfg = cfg_to_dataclass(cfg)
    
    # Initialize agent
    print(f"Initializing agent with model size {cfg.model_size}...")
    agent = TDMPC2(cfg)
    
    # Load agent checkpoint
    print(f"Loading agent checkpoint from {agent_checkpoint}...")
    agent.load(agent_checkpoint)
    agent.to(device)
    agent.eval()
    
    # Initialize buffer
    print(f"Initializing buffer...")
    buffer = Buffer(cfg)
    
    # Load buffer checkpoint
    print(f"Loading buffer checkpoint from {buffer_checkpoint}...")
    num_episodes = buffer.load_buffer_from_disk(buffer_checkpoint)
    print(f"Loaded {num_episodes} episodes from buffer")
    
    # Compute model error
    print(f"Computing model error with horizon {cfg.horizon}...")
    
    total_loss = 0.0
    total_samples = 0
    batch_errors = []
    
    # Check if buffer has data
    if num_episodes == 0:
        print("Warning: Buffer is empty. No data to compute model error.")
        return {
            'avg_mse': 0.0,
            'std_mse': 0.0,
            'min_mse': 0.0,
            'max_mse': 0.0,
            'num_batches': 0,
            'num_episodes': 0,
            'horizon': cfg.horizon,
        }
    
    # Process multiple batches to get stable statistics
    num_batches = min(100, max(1, num_episodes // cfg.batch_size))  # Use up to 100 batches, at least 1
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Sample a batch from the buffer
            obs, action, reward, terminated, task = buffer.sample()
            
            # Move to device
            obs = obs.to(device)
            action = action.to(device)
            
            # Encode observations
            next_z = agent.model.encode(obs[1:], task)  # Shape: [horizon, batch_size, latent_dim]
            
            # Latent rollout (same as in _update method)
            z = agent.model.encode(obs[0], task)  # Shape: [batch_size, latent_dim]
            
            batch_consistency_loss = 0.0
            for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
                # Predict next latent state
                z = agent.model.next(z, _action, task)
                
                # Compute MSE loss (without rho weighting for clearer interpretation)
                mse_loss = F.mse_loss(z, _next_z)
                batch_consistency_loss += mse_loss
                
                # Also track per-timestep errors
                if batch_idx == 0:  # Only print for first batch
                    print(f"  Timestep {t+1} MSE: {mse_loss.item():.6f}")
            
            # Average over horizon
            batch_consistency_loss = batch_consistency_loss / cfg.horizon
            batch_errors.append(batch_consistency_loss.item())
            
            total_loss += batch_consistency_loss.item()
            total_samples += 1
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx+1}/{num_batches}, Running avg MSE: {total_loss / (batch_idx + 1):.6f}")
    
    # Compute statistics
    avg_error = total_loss / total_samples
    batch_errors_tensor = torch.tensor(batch_errors)
    std_error = batch_errors_tensor.std().item()
    min_error = batch_errors_tensor.min().item()
    max_error = batch_errors_tensor.max().item()
    
    results = {
        'avg_mse': avg_error,
        'std_mse': std_error,
        'min_mse': min_error,
        'max_mse': max_error,
        'num_batches': num_batches,
        'num_episodes': num_episodes,
        'horizon': cfg.horizon,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute model error on test buffer')
    parser.add_argument('--agent-checkpoint', required=True, help='Path to agent checkpoint (.pt file)')
    parser.add_argument('--buffer-checkpoint', required=True, help='Path to test buffer checkpoint (.pt file)')
    parser.add_argument('--config', help='Path to config.yaml file (optional)')
    parser.add_argument('--task', default='dog-run', help='Task name (default: dog-run)')
    parser.add_argument('--horizon', type=int, default=3, help='Prediction horizon (default: 3)')
    parser.add_argument('--device', default='cuda:0', help='Device to run on (default: cuda:0)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.agent_checkpoint):
        print(f"Error: Agent checkpoint not found: {args.agent_checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.buffer_checkpoint):
        print(f"Error: Buffer checkpoint not found: {args.buffer_checkpoint}")
        sys.exit(1)
    
    if args.config and not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Compute model error
    results = compute_model_error(
        agent_checkpoint=args.agent_checkpoint,
        buffer_checkpoint=args.buffer_checkpoint,
        cfg_path=args.config,
        task=args.task,
        horizon=args.horizon,
        device=args.device
    )
    
    # Print results
    print("\n" + "="*50)
    print("MODEL ERROR RESULTS")
    print("="*50)
    print(f"Average MSE:     {results['avg_mse']:.6f}")
    print(f"Std Dev MSE:     {results['std_mse']:.6f}")
    print(f"Min MSE:         {results['min_mse']:.6f}")
    print(f"Max MSE:         {results['max_mse']:.6f}")
    print(f"Num Batches:     {results['num_batches']}")
    print(f"Num Episodes:    {results['num_episodes']}")
    print(f"Horizon:         {results['horizon']}")
    print("="*50)
    
    # Save results to file
    result_path = args.agent_checkpoint.replace('.pt', '_model_error.txt')
    with open(result_path, 'w') as f:
        f.write("Model Error Results\n")
        f.write("="*50 + "\n")
        f.write(f"Agent Checkpoint: {args.agent_checkpoint}\n")
        f.write(f"Buffer Checkpoint: {args.buffer_checkpoint}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Device: {args.device}\n")
        f.write("="*50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main() 