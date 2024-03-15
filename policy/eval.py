#!/usr/bin/env python3

import warnings
import os

from pathlib import Path

import hydra
import numpy as np
import torch

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec[cfg.obs_type].shape
	cfg.action_shape = action_spec.shape
	return hydra.utils.instantiate(cfg)

class Workspace:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.env.observation_spec(),
								self.env.action_spec(), cfg.agent)

		self.expert_replay_loader = make_expert_replay_loader(
			self.cfg.expert_dataset, self.cfg.batch_size,  self.cfg.obs_type, 
			self.cfg.suite.height, self.cfg.suite.width, self.cfg.train_test_ratio)
		self.expert_replay_iter = iter(self.expert_replay_loader)

		if repr(self.agent) == 'bet':
			actions = []
			for episode in self.expert_replay_loader.dataset._episodes:
				actions.extend(episode['action'])
			self.agent.compute_action_bins(actions)

		self.timer = utils.Timer()
		self._global_step = 0
		
	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		self.env = hydra.utils.call(self.cfg.suite.task_make_fn)

		self.expert_replay_iter = None

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)

	@property
	def global_step(self):
		return self._global_step

	def eval(self):
		step, episode, total_reward = 0, 0, 0
		eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
		episode_length = []

		while eval_until_episode(episode):
			step = 0
			# Sample test traj
			start_state, goal_state = self.expert_replay_loader.dataset.sample_test()

			time_step = self.env.reset(start_state=start_state, reset_goal=True, 
							  		   goal_state=np.array(goal_state[-1]))
			if episode == 0:
				self.video_recorder.init(self.env, enabled=True)
			while not time_step.last():
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(time_step.observation[self.cfg.obs_type],
							 				goal_state[min(step, len(goal_state)-1)],
											self.global_step)
				time_step = self.env.step(action)
				self.video_recorder.record(self.env)
				total_reward += time_step.reward
				step += 1

			episode += 1
			episode_length.append(step)
		self.video_recorder.save(f'{self.global_step}.mp4')
		
		with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
			log('episode_reward', total_reward / episode)
			log('episode_length', np.mean(episode_length))
			log('step', self.global_step)

	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)

@hydra.main(config_path='cfgs', config_name='config_eval')
def main(cfg):
	from eval import Workspace as W
	root_dir = Path.cwd()
	workspace = W(cfg)

	# weight
	weight_path = Path(cfg.weight)
	if weight_path.exists():
		print(f'Loading weight from {weight_path}')
		workspace.load_snapshot(weight_path)
	else:
		raise FileNotFoundError(f'Weight not found at {weight_path}')
	
	workspace.eval()


if __name__ == '__main__':
	main()
