import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from common.sam import SAM
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)

		# --- (FIX) SELECTIVE OPTIMIZER SETUP ---
		# We separate the world model parameters into two groups:
		# 1. Predictive components (encoder, dynamics) which benefit from SAM's regularization.
		# 2. Supervisory heads (Q-funcs, reward) which need sharp Adam updates.
		
		# Parameters for the predictive model (encoder + dynamics)
		predictive_model_params = [
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		]

		# Parameters for the supervisory heads (Q-functions, reward, termination)
		head_params = [
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()}
		]

		if hasattr(self.cfg, 'optimizer') and self.cfg.optimizer == 'SAM':
			print(f'Using SAM optimizer for predictive model (rho={self.cfg.sam_rho}) and Adam for heads.')
			base_optimizer = torch.optim.Adam
			# SAM for the predictive components
			self.model_optim = SAM(predictive_model_params, base_optimizer, lr=self.cfg.lr, rho=self.cfg.sam_rho, capturable=True)
			# Standard Adam for the heads
			self.head_optim = torch.optim.Adam(head_params, lr=self.cfg.lr, capturable=True)
		else:
			print('Using Adam optimizer for all world model components.')
			# If not using SAM, a single Adam optimizer for everything is fine.
			world_model_params = predictive_model_params + head_params
			self.model_optim = torch.optim.Adam(world_model_params, lr=self.cfg.lr, capturable=True)
			self.head_optim = None # Not used

		# Optimizer for the policy network remains separate
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		# --- END OF FIX ---

		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20)
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def get_value(self, obs, task=None, num_samples=256):
		"""
		Estimate the value of a given observation by averaging Q-values
		over actions sampled from the policy. V(s) = E_{a ~ pi(s)} [Q(s, a)]
		"""
		# Move observation to the correct device and add a batch dimension.
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		
		# Encode the observation into the latent space.
		z = self.model.encode(obs, task)

		# Repeat the latent state to create a batch for sampling multiple actions.
		z_repeated = z.repeat(num_samples, 1)

		# Sample a batch of actions from the policy for the same latent state.
		actions_from_pi, _ = self.model.pi(z_repeated, task)

		# Evaluate the Q-function for the state and all sampled actions.
		# Use the ensemble method specified in the config.
		q_values = self.model.Q(z_repeated, actions_from_pi, task, return_type=self.cfg.eval_q_ensemble_method)

		# The returned shape depends on the ensemble method, so we average correctly.
		# 'all' returns (num_q, num_samples, 1), others return (num_samples, 1)
		# We average over all dimensions to get a single scalar value.
		value = q_values.mean()

		return value

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		for _ in range(self.cfg.iterations):
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		self.model.train()

		# --- (FIX) LOSS CALCULATION AND GRADIENT APPLICATION ---
		# This section is now generalized for both Adam and selective SAM.
		# For SAM, this logic is called twice.
		def calculate_loss(zs, next_z_target):
			# Predictions
			_zs = zs[:-1]
			qs = self.model.Q(_zs, action, task, return_type='all')
			reward_preds = self.model.reward(_zs, action, task)
			
			# Latent consistency loss
			consistency_loss = F.mse_loss(zs[1:], next_z_target, reduction='none').mean(dim=(1,2))
			consistency_loss = (consistency_loss * (self.cfg.rho ** torch.arange(self.cfg.horizon, device=self.device))).mean()

			# Value and reward losses
			reward_loss, value_loss = 0, 0
			for t, (rew_pred_t, rew_t, td_targets_t, qs_t) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
				rho_t = self.cfg.rho ** t
				reward_loss += math.soft_ce(rew_pred_t, rew_t, self.cfg).mean() * rho_t
				for q_t in qs_t.unbind(0):
					value_loss += math.soft_ce(q_t, td_targets_t, self.cfg).mean() * rho_t
			
			# Termination loss
			termination_loss = 0.
			if self.cfg.episodic:
				termination_pred = self.model.termination(zs[1:], task, unnormalized=True)
				termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
			
			value_loss /= self.cfg.num_q

			# Combine losses
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.termination_coef * termination_loss +
				self.cfg.value_coef * value_loss
			)
			
			# For logging purposes
			all_losses = {
				'consistency_loss': consistency_loss.detach(),
				'reward_loss': reward_loss.detach() / self.cfg.horizon,
				'value_loss': value_loss.detach() / self.cfg.horizon,
				'termination_loss': termination_loss.detach() if self.cfg.episodic else 0.,
				'total_loss': total_loss.detach()
			}
			return total_loss, all_losses

		# Latent rollout
		z = self.model.encode(obs[0], task)
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		zs[0] = z
		for t in range(self.cfg.horizon):
			zs[t+1] = self.model.next(zs[t], action[t], task)

		# Calculate loss and update
		if hasattr(self.cfg, 'optimizer') and self.cfg.optimizer == 'SAM':
			# First step of SAM on predictive model
			total_loss, losses = calculate_loss(zs.clone(), next_z.detach())
			total_loss.backward()
			self.model_optim.first_step(zero_grad=True)
			
			# Update heads with standard Adam
			# Grads are already present from the first backward, so just step.
			torch.nn.utils.clip_grad_norm_(self.model._reward.parameters(), self.cfg.grad_clip_norm)
			torch.nn.utils.clip_grad_norm_(self.model._Qs.parameters(), self.cfg.grad_clip_norm)
			if self.cfg.episodic:
				torch.nn.utils.clip_grad_norm_(self.model._termination.parameters(), self.cfg.grad_clip_norm)
			self.head_optim.step()
			self.head_optim.zero_grad(set_to_none=True)

			# Second step of SAM on predictive model
			z_prime = self.model.encode(obs[0], task)
			zs_prime = torch.empty_like(zs)
			zs_prime[0] = z_prime
			for t in range(self.cfg.horizon):
				zs_prime[t+1] = self.model.next(zs_prime[t], action[t], task)
			
			total_loss_prime, _ = calculate_loss(zs_prime, next_z.detach())
			total_loss_prime.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model._encoder.parameters(), self.cfg.grad_clip_norm)
			grad_norm += torch.nn.utils.clip_grad_norm_(self.model._dynamics.parameters(), self.cfg.grad_clip_norm)
			self.model_optim.second_step(zero_grad=True)

		else: # Standard Adam
			total_loss, losses = calculate_loss(zs.clone(), next_z.detach())
			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
			self.model_optim.step()
			self.model_optim.zero_grad(set_to_none=True)

		# Policy update
		with torch.no_grad():
			policy_zs = self.model.encode(obs[:self.cfg.horizon], task)
		pi_info = self.update_pi(policy_zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		losses['grad_norm'] = grad_norm
		info = TensorDict(losses)
		info.update(pi_info)
		return info.detach().mean()


	def update(self, buffer):
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)