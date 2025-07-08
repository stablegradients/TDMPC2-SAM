#!/usr/bin/env python3

import os
import sys
import torch
from omegaconf import OmegaConf
from pathlib import Path

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
tdmpc2_dir = os.path.join(project_root, 'tdmpc2')
sys.path.insert(0, tdmpc2_dir)

from tdmpc2 import TDMPC2
from common.buffer import Buffer
from common.parser import parse_cfg
from envs import make_env
from common import MODEL_SIZE
from tensordict import TensorDict

def generate_test_data(agent_checkpoint, output_path, task='dog-run', num_episodes=10, device='cuda:0'):
    """
    Generate test data by running agent in environment.
    """
    
    # Create configuration
    cfg = OmegaConf.create({
        'task': task,
        'obs': 'state',
        'model_size': 5,
        'horizon': 3,
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
    
    # Set virtual display
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Create environment
    print(f"Creating environment for {task}...")
    env = make_env(cfg)
    
    # Get environment parameters
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.obs_shape = {cfg.obs: env.observation_space.shape}
    cfg.work_dir = Path.cwd() / 'logs' / cfg.task / str(cfg.seed) / cfg.get('exp_name', 'default')
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.multitask = False
    cfg.tasks = [cfg.task]
    cfg.task_dim = 0
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    
    # Model size configuration
    if cfg.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
    
    # Convert to dataclass
    from common.parser import cfg_to_dataclass
    cfg = cfg_to_dataclass(cfg)
    
    # Initialize agent
    print(f"Loading agent from {agent_checkpoint}...")
    agent = TDMPC2(cfg)
    agent.load(agent_checkpoint)
    agent.to(device)
    agent.eval()
    
    # Initialize buffer
    buffer = Buffer(cfg)
    
    # Generate episodes
    print(f"Generating {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs = env.reset()
        episode_data = []
        done = False
        step = 0
        
        while not done and step < cfg.episode_length:
            # Get action from agent
            with torch.no_grad():
                action = agent.act(obs, t0=(step==0), eval_mode=True)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            episode_data.append({
                'obs': obs,
                'action': action,
                'reward': torch.tensor(reward, dtype=torch.float32),
                'terminated': torch.tensor(info.get('terminated', False), dtype=torch.float32),
            })
            
            obs = next_obs
            step += 1
        
        # Convert episode to tensordict and add to buffer
        episode_td = TensorDict({
            'obs': torch.stack([t['obs'] for t in episode_data]),
            'action': torch.stack([t['action'] for t in episode_data]),
            'reward': torch.stack([t['reward'] for t in episode_data]),
            'terminated': torch.stack([t['terminated'] for t in episode_data]),
        }, batch_size=(len(episode_data),))
        
        buffer.add(episode_td)
        print(f"  Episode {ep+1}/{num_episodes} completed, length: {len(episode_data)}")
    
    # Save buffer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    buffer.save(os.path.dirname(output_path), 100000, 'test')
    print(f"Buffer saved to {output_path}")
    
    try:
        env.close()
    except AttributeError:
        pass
    
    return buffer.num_eps


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test data for model error computation')
    parser.add_argument('--agent-checkpoint', required=True, help='Path to agent checkpoint')
    parser.add_argument('--output-path', required=True, help='Path to save test buffer')
    parser.add_argument('--task', default='dog-run', help='Task name')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of episodes to generate')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    
    args = parser.parse_args()
    
    generate_test_data(
        agent_checkpoint=args.agent_checkpoint,
        output_path=args.output_path,
        task=args.task,
        num_episodes=args.num_episodes,
        device=args.device
    ) 