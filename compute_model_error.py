#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gc

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
tdmpc2_dir = os.path.join(project_root, 'tdmpc2')
sys.path.insert(0, tdmpc2_dir)

from tdmpc2 import TDMPC2
from common.buffer import Buffer
from common.parser import parse_cfg
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage


class HessianEigenspectrum:
    """
    Computes the top eigenvalues and eigenvectors of the Hessian matrix using the Lanczos algorithm.
    Adapted for TDMPC2 model loss computation.
    """
    def __init__(self, agent, buffer, cfg, max_iter=100, tol=1e-6):
        self.agent = agent
        self.buffer = buffer
        self.cfg = cfg
        self.max_iter = max_iter
        self.tol = tol
        self.device = cfg.device
        
        # Get all parameters that require gradients from the model (encoder + dynamics)
        # First, identify which parameters are actually used by doing a dummy forward pass
        self.agent.model.eval()
        with torch.no_grad():
            # Sample a batch to do a test forward pass
            obs, action, _, _, task = self.buffer.sample()
            obs = obs.to(self.device)
            action = action.to(self.device)
            
        # Enable gradients temporarily to check which params are used
        with torch.enable_grad():
            # Create a dummy loss to identify used parameters
            z = self.agent.model.encode(obs[0], task)
            z_next = self.agent.model.next(z, action[0], task)
            dummy_loss = z_next.sum()
            
            # Get gradients to see which parameters are actually used
            grads = torch.autograd.grad(dummy_loss, self.agent.model.parameters(), 
                                       allow_unused=True, retain_graph=False)
            
            # Only keep parameters that have non-None gradients (i.e., are used)
            self.params = [p for p, g in zip(self.agent.model.parameters(), grads) 
                          if g is not None and p.requires_grad]
        
        self.n_params = sum(p.numel() for p in self.params)
        print(f"Analyzing Hessian for {self.n_params} parameters (out of {sum(p.numel() for p in self.agent.model.parameters())} total)")
        print(f"Initialized HessianEigenspectrum with {len(self.params)} parameter groups")
        
    def _flatten_grad(self, grads):
        """Flatten and concatenate gradients."""
        return torch.cat([g.reshape(-1) for g in grads])

    def _get_hvp(self, v_list):
        """
        Compute Hessian-vector product using the R-operator.
        Computes the Hessian of the model consistency loss.
        """
        self.agent.model.eval()
        self.agent.model.zero_grad()
        
        # Sample a batch from the buffer
        obs, action, reward, terminated, task = self.buffer.sample()
        
        # Move to device
        obs = obs.to(self.device)
        action = action.to(self.device)
        
        # Encode observations for consistency loss
        next_z_target = self.agent.model.encode(obs[1:], task)  # Shape: [horizon, batch_size, latent_dim]
        
        # Latent rollout
        z = self.agent.model.encode(obs[0], task)  # Shape: [batch_size, latent_dim]
        
        # Compute consistency loss (same as in the model error computation)
        consistency_loss = 0.0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z_target.unbind(0))):
            # Predict next latent state
            z = self.agent.model.next(z, _action, task)
            
            # Compute MSE loss
            mse_loss = F.mse_loss(z, _next_z)
            consistency_loss += mse_loss
        
        # Average over horizon
        consistency_loss = consistency_loss / self.cfg.horizon
        
        # First-order gradients
        grad_params = torch.autograd.grad(consistency_loss, self.params, create_graph=True, allow_unused=True)
        
        # Ensure v_list is on the right device
        v_list_device = []
        for i, v in enumerate(v_list):
            if v.device != self.device:
                v = v.to(self.device)
            v_list_device.append(v)
        
        # Compute gradient-vector product, handling None gradients for unused params
        grad_v_prod = 0.0
        for g, v in zip(grad_params, v_list_device):
            if g is not None:
                grad_v_prod += torch.sum(g * v)
        
        # Second-order gradients (Hessian-vector product)
        hvp = torch.autograd.grad(grad_v_prod, self.params, allow_unused=True)
        
        # Move results to CPU to save GPU memory, handling None values
        hvp_cpu = [h.detach().cpu() if h is not None else torch.zeros_like(p).cpu() 
                   for h, p in zip(hvp, self.params)]
        
        # Clean up to save memory
        del grad_params, grad_v_prod, consistency_loss, obs, action
        torch.cuda.empty_cache()
        
        return hvp_cpu
    
    def lanczos_algorithm(self, num_eigenvals=20):
        """
        Lanczos algorithm for finding the top eigenvalues and eigenvectors
        of the Hessian matrix.
        """
        # Ensure we don't try to compute more eigenvalues than parameters
        num_eigenvals = min(num_eigenvals, self.n_params)
        
        # Initialize random vector on CPU
        v_list = [torch.randn_like(p.detach().cpu()) for p in self.params]
        
        # Normalize v
        v_flat = self._flatten_grad(v_list)
        v_flat = v_flat / torch.norm(v_flat)
        
        # Reshape v back to parameter shapes
        start = 0
        for i, p in enumerate(self.params):
            v_size = p.numel()
            v_list[i] = v_flat[start:start+v_size].reshape(p.shape)
            start += v_size
        
        # Initialize Lanczos
        alpha = torch.zeros(self.max_iter)
        beta = torch.zeros(self.max_iter)
        
        # Store all vectors for reorthogonalization
        q_vectors = [v_list]
        
        # First iteration
        v_old_list = [torch.zeros_like(p.detach().cpu()) for p in self.params]
        
        # Move v_list to device for the first HVP computation
        v_list_device = [v.to(self.device) for v in v_list]
        w_list = self._get_hvp(v_list_device)
        del v_list_device  # Clean up device tensors
        
        # All w_list elements are on CPU after _get_hvp
        alpha[0] = sum(torch.sum(w * v) for w, v in zip(w_list, v_list))
        
        for i in range(len(w_list)):
            w_list[i] = w_list[i] - alpha[0] * v_list[i]
        
        # Keep all operations on CPU to avoid device mismatches
        for j in range(1, self.max_iter):
            # Check for convergence
            if j >= num_eigenvals * 2:
                # Early stopping if we've computed enough iterations
                # for our desired number of eigenvalues
                break
            
            # Reorthogonalize w_list against all previous q_vectors
            for q in q_vectors:
                dot_prod = sum(torch.sum(w_list[i] * q[i]) for i in range(len(w_list)))
                for i in range(len(w_list)):
                    w_list[i] -= dot_prod * q[i]
            
            # Get beta (all on CPU)
            beta[j-1] = torch.sqrt(sum(torch.sum(w * w) for w in w_list))
            print(f"Iteration {j}: beta = {beta[j-1]:.6e}")
            
            if beta[j-1] < self.tol:
                # We've reached numerical precision limit
                print(f"Lanczos converged after {j} iterations (beta < {self.tol})")
                j = j - 1  # Adjust j to reflect actual iterations
                break
            
            # Update v_old and v (all on CPU)
            for i in range(len(v_list)):
                v_old_list[i] = v_list[i].clone()
                v_list[i] = w_list[i] / beta[j-1]
            
            # Store the new vector for future reorthogonalization
            q_vectors.append([v.clone() for v in v_list])
            
            # Move v_list to device for HVP computation
            v_list_device = [v.to(self.device) for v in v_list]
            
            # Get the HVP (result will be on CPU)
            w_list = self._get_hvp(v_list_device)
            del v_list_device  # Clean up device tensors
            
            # Calculate alpha (all on CPU)
            alpha[j] = sum(torch.sum(w * v) for w, v in zip(w_list, v_list))
            
            # Update w (all on CPU)
            for i in range(len(w_list)):
                w_list[i] = w_list[i] - alpha[j] * v_list[i] - beta[j-1] * v_old_list[i]
        
        # Truncate alpha and beta to actual iterations
        actual_iter = j + 1
        alpha = alpha[:actual_iter]
        beta = beta[:actual_iter-1] if actual_iter > 1 else torch.tensor([])
        
        # Construct tri-diagonal matrix
        if actual_iter == 1:
            T = torch.tensor([[alpha[0]]])
        else:
            T = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
        
        # Get eigenvalues and eigenvectors of T
        eigenvalues, eigenvectors = torch.linalg.eigh(T)
        
        # Sort eigenvalues in descending order
        indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        
        return eigenvalues[:num_eigenvals], eigenvectors[:, :num_eigenvals]
    
    def compute_spectrum(self, num_eigenvals=20, output_dir=None, prefix="tdmpc2_hessian"):
        """
        Compute and visualize the eigenspectrum of the Hessian matrix.
        """
        print(f"Computing Hessian eigenspectrum with {num_eigenvals} eigenvalues...")
        start_time = time.time()
        eigenvalues, eigenvectors = self.lanczos_algorithm(num_eigenvals)
        elapsed_time = time.time() - start_time
        print(f"Hessian eigenspectrum computation completed in {elapsed_time:.2f} seconds")
        
        # Convert to numpy for easier handling
        if isinstance(eigenvalues, torch.Tensor):
            eigenvalues = eigenvalues.cpu().numpy()
        
        # Save the eigenvalues to file
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f"{prefix}_eigenvalues.npy"), eigenvalues)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', markersize=8)
            plt.xlabel('Index', fontsize=12)
            plt.ylabel('Eigenvalue', fontsize=12)
            plt.title(f'Hessian Eigenspectrum - {self.cfg.task}', fontsize=14)
            if len(eigenvalues) > 1 and eigenvalues[0] > 0:
                plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_eigenspectrum.png"), dpi=300)
            
            # Also plot the eigenvalue distribution
            plt.figure(figsize=(10, 6))
            plt.hist(eigenvalues, bins=min(50, len(eigenvalues)), alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Eigenvalue', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Hessian Eigenvalue Distribution - {self.cfg.task}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_eigenvalue_hist.png"), dpi=300)
            
            # Save text summary
            summary_path = os.path.join(output_dir, f"{prefix}_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Hessian Eigenspectrum Analysis\n")
                f.write(f"="*50 + "\n")
                f.write(f"Task: {self.cfg.task}\n")
                f.write(f"Model parameters: {self.n_params}\n")
                f.write(f"Eigenvalues computed: {len(eigenvalues)}\n")
                f.write(f"Computation time: {elapsed_time:.2f} seconds\n")
                f.write(f"\nTop 10 eigenvalues:\n")
                for i, eig in enumerate(eigenvalues[:10]):
                    f.write(f"λ_{i+1} = {eig:.6e}\n")
                
                if len(eigenvalues) > 1:
                    f.write(f"\nCondition number (λ_max/λ_min): {eigenvalues[0]/eigenvalues[-1]:.2e}\n")
            
            print(f"Saved eigenspectrum results to {output_dir}")
        
        plt.close('all')
        return eigenvalues


def compute_model_error(agent_checkpoint, buffer_checkpoint, cfg_path=None, task=None, horizon=3, device='cuda:0', 
                       compute_eigenvalues=False, num_eigenvals=20, eigenvalue_output_dir=None):
    """
    Compute the average model error (MSE) on a test buffer using a trained agent.
    
    Args:
        agent_checkpoint: Path to agent checkpoint (.pt file)
        buffer_checkpoint: Path to buffer checkpoint (.pt file)
        cfg_path: Optional path to config file. If not provided, will try to load from checkpoint
        task: Task name (required if not in config)
        horizon: Prediction horizon (default: 3)
        device: Device to run on (default: 'cuda:0')
        compute_eigenvalues: Whether to compute Hessian eigenvalue spectrum (default: False)
        num_eigenvals: Number of eigenvalues to compute (default: 20)
        eigenvalue_output_dir: Directory to save eigenvalue results (default: None)
    
    Returns:
        Dictionary with error statistics (and eigenvalues if requested)
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
    # Create a custom buffer class that forces CPU storage to avoid device mismatch
    class CPUBuffer(Buffer):
        def _init(self, tds):
            """Initialize the replay buffer with CPU storage."""
            print(f'Buffer capacity: {self._capacity:,}')
            bytes_per_step = sum([
                    (v.numel()*v.element_size() if not isinstance(v, TensorDict) \
                    else sum([x.numel()*x.element_size() for x in v.values()])) \
                for v in tds.values()
            ]) / len(tds)
            total_bytes = bytes_per_step*self._capacity
            print(f'Storage required: {total_bytes/1e9:.2f} GB')
            print(f'Using CPU memory for storage (forced for evaluation).')
            self._storage_device = torch.device('cpu')
            return self._reserve_buffer(
                LazyTensorStorage(self._capacity, device=self._storage_device)
            )
    
    buffer = CPUBuffer(cfg)
    
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
            with torch.no_grad():
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
    
    # Compute Hessian eigenvalue spectrum if requested
    if compute_eigenvalues:
        print("\n" + "="*50)
        print("COMPUTING HESSIAN EIGENVALUE SPECTRUM")
        print("="*50)
        
        # Enable gradients for the model
        for param in agent.model.parameters():
            param.requires_grad_(True)
        
        # Create HessianEigenspectrum instance
        hessian = HessianEigenspectrum(agent, buffer, cfg)
        
        # Set output directory if not provided
        if eigenvalue_output_dir is None:
            eigenvalue_output_dir = os.path.dirname(agent_checkpoint)
        
        # Compute eigenvalues
        eigenvalues = hessian.compute_spectrum(
            num_eigenvals=num_eigenvals,
            output_dir=eigenvalue_output_dir,
            prefix=f"hessian_{os.path.basename(agent_checkpoint).replace('.pt', '')}"
        )
        
        # Add eigenvalues to results
        results['eigenvalues'] = eigenvalues.tolist() if isinstance(eigenvalues, np.ndarray) else eigenvalues
        results['num_eigenvalues'] = len(eigenvalues)
        
        # Print top eigenvalues
        print(f"\nTop {min(10, len(eigenvalues))} eigenvalues:")
        for i, eig in enumerate(eigenvalues[:10]):
            print(f"λ_{i+1} = {eig:.6e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute model error on test buffer')
    parser.add_argument('--agent-checkpoint', required=True, help='Path to agent checkpoint (.pt file)')
    parser.add_argument('--buffer-checkpoint', required=True, help='Path to test buffer checkpoint (.pt file)')
    parser.add_argument('--config', help='Path to config.yaml file (optional)')
    parser.add_argument('--task', default='dog-run', help='Task name (default: dog-run)')
    parser.add_argument('--horizon', type=int, default=3, help='Prediction horizon (default: 3)')
    parser.add_argument('--device', default='cuda:0', help='Device to run on (default: cuda:0)')
    parser.add_argument('--eigenvalue', action='store_true', help='Compute Hessian eigenvalue spectrum')
    parser.add_argument('--num-eigenvals', type=int, default=20, help='Number of eigenvalues to compute (default: 20)')
    parser.add_argument('--eigenvalue-output-dir', help='Directory to save eigenvalue results (default: same as agent checkpoint)')
    
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
        device=args.device,
        compute_eigenvalues=args.eigenvalue,
        num_eigenvals=args.num_eigenvals,
        eigenvalue_output_dir=args.eigenvalue_output_dir
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
    
    if 'eigenvalues' in results:
        print("\nHESSIAN EIGENVALUE RESULTS")
        print(f"Num Eigenvalues: {results['num_eigenvalues']}")
        if results['eigenvalues']:
            print(f"Max Eigenvalue:  {results['eigenvalues'][0]:.6e}")
            print(f"Min Eigenvalue:  {results['eigenvalues'][-1]:.6e}")
            if results['eigenvalues'][-1] != 0:
                print(f"Condition Number: {results['eigenvalues'][0]/results['eigenvalues'][-1]:.2e}")
    
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
        f.write(f"Eigenvalue Analysis: {args.eigenvalue}\n")
        f.write("="*50 + "\n")
        
        # Write main results
        for key, value in results.items():
            if key != 'eigenvalues':  # Skip the full eigenvalue list
                f.write(f"{key}: {value}\n")
        
        # Write eigenvalue summary if available
        if 'eigenvalues' in results and results['eigenvalues']:
            f.write("\nEigenvalue Summary:\n")
            f.write(f"Top 10 eigenvalues:\n")
            for i, eig in enumerate(results['eigenvalues'][:10]):
                f.write(f"  λ_{i+1} = {eig:.6e}\n")
    
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main() 