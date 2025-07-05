# import make_env and buffer from common
from common.buffer import Buffer
from envs import make_env

class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.test_env = make_env(cfg)
		self.test_buffer = Buffer(cfg)
		self.logger = logger
		print('Architecture:', self.agent.model)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
