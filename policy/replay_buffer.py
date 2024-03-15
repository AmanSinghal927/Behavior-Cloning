import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import pickle

class ExpertReplayBuffer(IterableDataset):
    def __init__(self, dataset_path, obs_type, height, width, train_test_ratio):
        self._height = height
        self._width = width
        self._train_test_ratio = train_test_ratio
        
        # Read data
        with open(dataset_path, 'rb') as f:
            if obs_type == 'pixels':
                obses, _, actions, _, goals = pickle.load(f)
            elif obs_type == 'features':
                _, obses, actions, _, goals = pickle.load(f)
        
        self._episodes = []
        for i in range(len(obses)):
            episode = dict(
                        observation=obses[i],
                        action=actions[i],
                        goal=goals[i])
            self._episodes.append(episode)
        
        # Compute max episode length
        self._max_episode_len = max([len(episode) for episode in self._episodes])

        # Train set
        self._train_episodes_till = int(len(self._episodes) * self._train_test_ratio)

    def _sample_episode(self):
        episode = random.choice(self._episodes[:self._train_episodes_till])
        return episode

    def _sample(self):
        episodes = self._sample_episode() # all episodes for a goal
        observation = episodes['observation']
        action = episodes['action']
        goal = np.array(episodes['goal'])

        # Sample obs, action
        sample_idx = np.random.randint(0, len(observation))
        sampled_obs = observation[sample_idx]
        sampled_action = action[sample_idx]
        goal = goal[sample_idx]

        sampled_obs = np.array(sampled_obs)
        sampled_obs[0], sampled_obs[1] = sampled_obs[0] / self._height, sampled_obs[1] / self._width
        goal = np.array(goal)
        goal[0], goal[1] = goal[0] / self._height, goal[1] / self._width

        return (sampled_obs, sampled_action, goal)
    
    def sample_test(self):
        episode = random.choice(self._episodes[self._train_episodes_till:])
        observation = episode['observation']
        goal = np.array(episode['goal'])
        
        # Set prompt as same observation
        start_obs = observation[0]
        start_obs = np.array(start_obs)
        start_obs[0], start_obs[1] = start_obs[0] / self._height, start_obs[1] / self._width

        goal = np.array(goal)
        goal[:,0], goal[:,1] = goal[:,0] / self._height, goal[:,1] / self._width
        
        return start_obs, goal

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def make_expert_replay_loader(replay_dir, batch_size, obs_type, height, width, train_test_ratio):
    iterable = ExpertReplayBuffer(replay_dir, obs_type, height, width, train_test_ratio)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
