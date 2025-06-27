from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent and log detailed value metrics."""
		
		# Lists to store metrics from each evaluation episode
		all_ep_rewards, all_ep_successes, all_ep_lengths = [], [], []
		all_estimated_values, all_true_discounted_returns, all_value_errors = [], [], []

		for i in range(self.cfg.eval_episodes):
			obs, done, t = self.env.reset(), False, 0
			ep_rewards = []
			
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i == 0))

			# 1. Estimate discounted value at s_0
			estimated_value_s0 = 0.0
			if self.cfg.get('eval_value', False):
				estimated_value_s0 = self.agent.get_value(obs).item()

			# Run the full episode
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=(t==0), eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_rewards.append(reward.item())
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			
			# --- After episode is done, calculate and log metrics ---

			# 2. Calculate the True Discounted Return
			rewards_tensor = torch.tensor(ep_rewards, dtype=torch.float32)
			discount_factors = self.agent.discount ** torch.arange(len(rewards_tensor))
			true_discounted_return = (rewards_tensor * discount_factors).sum().item()

			# 3. Calculate Value Prediction Error
			value_prediction_error = abs(estimated_value_s0 - true_discounted_return)

			# Store metrics for this episode
			all_ep_rewards.append(np.sum(ep_rewards))
			all_ep_successes.append(info['success'])
			all_ep_lengths.append(t)
			if self.cfg.get('eval_value', False):
				all_estimated_values.append(estimated_value_s0)
				all_true_discounted_returns.append(true_discounted_return)
				all_value_errors.append(value_prediction_error)
			
			# 6. Log metrics for this specific episode to wandb
			individual_metrics = {
				'step': self._step,
				f'episode_reward_ep{i}': np.sum(ep_rewards),
				f'episode_length_ep{i}': t,
			}
			if self.cfg.get('eval_value', False):
				individual_metrics[f'estimated_value_s0_ep{i}'] = estimated_value_s0
				individual_metrics[f'true_discounted_return_ep{i}'] = true_discounted_return
				individual_metrics[f'value_prediction_error_ep{i}'] = value_prediction_error
			
			if self.logger._wandb: # Only log if wandb is enabled
				self.logger.log(individual_metrics, 'eval')

			if self.cfg.save_video:
				self.logger.video.save(self._step, key=f'videos/eval_video_ep{i}')

		# --- After all evaluation episodes are done, calculate and log summary statistics ---

		summary_metrics = dict(
			episode_reward=np.nanmean(all_ep_rewards),
			episode_success=np.nanmean(all_ep_successes),
			episode_length=np.nanmean(all_ep_lengths),
		)
		if self.cfg.get('eval_value', False):
			summary_metrics.update(dict(
				avg_estimated_value_s0=np.nanmean(all_estimated_values),
				avg_true_discounted_return=np.nanmean(all_true_discounted_returns),
				avg_value_prediction_error=np.nanmean(all_value_errors),
				std_value_prediction_error=np.nanstd(all_value_errors),
			))
		
		return summary_metrics

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)
