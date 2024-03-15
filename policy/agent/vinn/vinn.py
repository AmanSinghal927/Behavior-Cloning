import torch
from torch import nn
import torchvision.transforms as T
# from sklearn.neighbors import KNeighborsRegressor
import utils
import random


class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, stddev_schedule, 
	      		 stddev_clip, use_tb, obs_type):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.use_tb = use_tb
		# actor parameters
		self._act_dim = action_shape[0]
		# self.knn = KNeighborsRegressor(n_neighbors=1)
		self.actions_observations = None


	def __repr__(self):
		return "vinn"
	
	def get_action(self, obs, k = 1):
		all_actions, all_obs =  self.actions_observations
		# sample from the k-nearest neighbors
		distance = (obs - all_obs)**2
		distance = torch.sum(distance, -1)
		min_distance = torch.min(distance)
		# find indexes where this minimum distance exists
		min_indexes = (distance==min_distance).nonzero(as_tuple = False) # gives all non-zero or all false values
		
		random_element_index = random.randint(0, min_indexes.size(0) - 1)
		sampled_idx = min_indexes[random_element_index]
		sampled_actions = all_actions[sampled_idx[0], sampled_idx[1]]
		return torch.tensor(sampled_actions, dtype = torch.float32)

	def compute_action_bins(self, obs, actions):
		# Compute nbins bin centers using knn algorithm 
		actions = torch.as_tensor(actions, device=self.device).float().unsqueeze(0)
		obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
		self.actions_observations = (actions, obs)

	def act(self, obs, goal, step): # not using the goal here
		# convert to tensor and add batch dimension
		obs = torch.as_tensor(obs, device=self.device).float()
		action = self.get_action(obs)
		# print (action)
		return action.cpu().numpy()


	def save_snapshot(self):
		keys_to_save = ['actor']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v
