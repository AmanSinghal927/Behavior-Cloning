import gym
from gym import spaces
import cv2
import numpy as np

class ParticleEnv(gym.Env):
	def __init__(self, height=84, width=84, step_size=10, reward_type='dense', 
	      		 reward_scale=None, block=None, fixed_goal=False, num_goals=1):
		super(ParticleEnv, self).__init__()

		self.height = height
		self.width = width
		self.step_size = step_size
		self.reward_type = reward_type
		self.reward_scale = np.sqrt(height**2 + width**2) if reward_scale is None else reward_scale
		self.block = block

		self.goals = []
		self.fixed_goal = fixed_goal
		self.num_goals = num_goals
		
		'''
		Define observation space which blocked in between.
		0: Traversable blocks
		1: Blocked
		2: Goal
		'''
		self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), 
									   		high = np.array([self.height-1, self.width-1],dtype=np.float32),
									  		dtype = np.float32)
		
		self.action_space = spaces.Box(low = np.array([-step_size, -step_size],dtype=np.float32), 
									   high = np.array([step_size, step_size],dtype=np.float32),
									   dtype = np.float32)
	
	def step(self, action):
		prev_state = self.state
		# self.state = np.array([int(self.state[0] + self.step_size * action[0]), int(self.state[1] + self.step_size * action[1])], dtype=np.float32)
		self.state = np.array([self.state[0] + self.step_size * action[0], self.state[1] + self.step_size * action[1]], dtype=np.float32)
		
		if self.state[0]<0 or self.state[0]>=self.height or self.state[1]<0 or self.state[1]>=self.width:
			reward = -10 if self.reward_type == 'dense' else 0 #-1
			self.state = prev_state
			done = False
		elif self.observation[int(self.state[0]), int(self.state[1])]==1:
			reward = -10 if self.reward_type == 'dense' else 0 #-1
			self.state = prev_state
			done = False
		elif self.observation[int(self.state[0]), int(self.state[1])] >= 2:
			if self.observation[int(self.state[0]), int(self.state[1])] == self.num_goals_reached + 2:
				self.num_goals_reached += 1
				reward = 1 #0
			else:
				reward = -10 if self.reward_type == 'dense' else 0 #-1
			done = True if self.num_goals_reached == self.num_goals else False
		else:
			reward = 0 #-1 if self.reward_type =='sparse' else -np.sqrt((self.goals[-1][0]-self.state[0])**2 + (self.goals[-1][0]-self.state[1])**2) / self.reward_scale
			done = False
		self._step += 1
		
		info = {}
		info['is_success'] = 1 if reward==0 else 0 

		# Normalize state
		state = np.array([self.state[0] / self.height, self.state[1] / self.width]).astype(np.float32)

		return state, reward, done, info
	
	def reset(self, start_state=None, reset_goal=False, goal_state=None, demo=None):
		# set start state
		if start_state is None:
			self.state = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
		else:
			start_state[0], start_state[1] = start_state[0] * self.height, start_state[1] * self.width
			self.state = np.array(start_state).astype(np.int32)

		# set goal
		if len(self.goals) == 0:
			if goal_state is None:
				goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
				while (self.state == goal).all():
					goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
				self.goals.append(goal)
			else:
				if self.num_goals == 1 or demo is None:
					goal_state[0], goal_state[1] = goal_state[0] * self.height, goal_state[1] * self.width
					self.goals.append(np.array(goal_state).astype(np.int32))
				elif self.num_goals > 1:
					intermediate_goal_idx = [(i+1)*len(demo)//self.num_goals for i in range(self.num_goals-1)]
					self.goals.append(np.array([demo[intermediate_goal_idx[i]][0] * self.height, demo[intermediate_goal_idx[i]][1] * self.width]).astype(np.int32))
					# Add final goal
					goal_state[0], goal_state[1] = goal_state[0] * self.height, goal_state[1] * self.width
					self.goals.append(np.array(goal_state).astype(np.int32))
		else:
			if reset_goal:
				self.goals = []
				if goal_state is None:
					goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
					while (self.state == goal).all():
						goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
					self.goals.append(goal)
				else:
					if self.num_goals == 1:
						goal_state[0], goal_state[1] = goal_state[0] * self.height, goal_state[1] * self.width
						self.goals.append(np.array(goal_state).astype(np.int32))
					elif self.num_goals > 1:
						intermediate_goal_idx = [(i+1)*len(demo)//self.num_goals for i in range(self.num_goals-1)]
						for idx in intermediate_goal_idx:
							self.goals.append(np.array([demo[idx][0] * self.height, demo[idx][1] * self.width]).astype(np.int32))
						# Add final goal
						goal_state[0], goal_state[1] = goal_state[0] * self.height, goal_state[1] * self.width
						self.goals.append(np.array(goal_state).astype(np.int32))

		# if self.goal is None or (reset_goal and not self.fixed_goal):
		# 	if goal_state is None:
		# 		self.goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
		# 		while (self.state == self.goal).all():
		# 			self.goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
		# 	else:
		# 		goal_state[0], goal_state[1] = goal_state[0] * self.height, goal_state[1] * self.width
		# 		self.goal = np.array(goal_state).astype(np.int32)

		# observation image
		self.observation = np.zeros((self.height, self.width)).astype(np.uint8)		
		# Set blocked regions
		if self.block is not None:
			for region in self.block:
				block_hmin, block_hmax = int(region[0]), int(region[1])
				block_wmin, block_wmax = int(region[2]), int(region[3])
				for h in range(block_hmin, block_hmax+1):
					for w in range(block_wmin, block_wmax+1):
						self.observation[h, w] = 1
		# Set goal regions
		for idx, goal in enumerate(self.goals):
			goal_hmin, goal_hmax = int(goal[0]-10), int(goal[0]+10)
			goal_wmin, goal_wmax = int(goal[1]-10), int(goal[1]+10)
			goal_hmin, goal_hmax = max(0, goal_hmin), min(self.height-1, goal_hmax)
			goal_wmin, goal_wmax = max(0, goal_wmin), min(self.width-1, goal_wmax)
			for h in range(goal_hmin, goal_hmax+1):
				for w in range(goal_wmin, goal_wmax+1):
					self.observation[h,w] = 2 + idx
		self.num_goals_reached = 0
		# goal_hmin, goal_hmax = int(self.goal[0]-10), int(self.goal[0]+10)
		# goal_wmin, goal_wmax = int(self.goal[1]-10), int(self.goal[1]+10)
		# goal_hmin, goal_hmax = max(0, goal_hmin), min(self.height-1, goal_hmax)
		# goal_wmin, goal_wmax = max(0, goal_wmin), min(self.width-1, goal_wmax)
		# for h in range(goal_hmin, goal_hmax+1):
		# 	for w in range(goal_wmin, goal_wmax+1):
		# 		self.observation[h,w] = 1
		# self.observation[self.goal[0], self.goal[1]] = 1

		self._step = 0

		# Normalize state
		state = np.array([self.state[0] / self.height, self.state[1] / self.width]).astype(np.float32)
		return state


	def render(self, mode='', width=None, height=None):
		img = np.ones(self.observation.shape).astype(np.uint8) * 255
		# Identify blocked region
		blocked = np.where(self.observation == 1)
		img[blocked] = 0

		# Identify goal region
		# hmin, hmax = max(0, self.goal[0]-10), min(self.height-1, self.goal[0] + 10)
		# wmin, wmax = max(0, self.goal[1]-10), min(self.width-1, self.goal[1] + 10)
		for idx, goal in enumerate(self.goals):
			# hmin, hmax = max(0, goal[0]-5*(idx+1)), min(self.height-1, goal[0] + 5*(idx+1))
			# wmin, wmax = max(0, goal[1]-5*(idx+1)), min(self.width-1, goal[1] + 5*(idx+1))
			hmin, hmax = max(0, goal[0]-10), min(self.height-1, goal[0] + 10)
			wmin, wmax = max(0, goal[1]-10), min(self.width-1, goal[1] + 10)
			hmin, hmax, wmin, wmax = int(hmin), int(hmax), int(wmin), int(wmax)
			img[hmin:hmax, wmin:wmax] = 64

		# Mark state
		img[max(0, int(self.state[0])-5):min(self.height-1, int(self.state[0])+5), max(0, int(self.state[1])-5):min(self.width-1, int(self.state[1])+5)] = 128

		if width is not None and height is not None:
			dim = (int(width), int(height))
			img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		img = img[..., None]

		if mode=='rgb_array':
			return img
		else:
			cv2.imshow("Render", img)
			cv2.waitKey(5)

# Code to test the environment
if __name__ == '__main__':
	env = ParticleEnv(height=640, width=640, step_size=10, reward_type='dense', reward_scale=None, start=None, goal=None, block=None)

	for i in range(10):
		state = env.reset()
		done = False
		while not done:
			action = env.action_space.sample()
			next_state, reward, done, info = env.step(action)
			env.render()
			print("State: ", state, "Action: ", action, "Next State: ", next_state, "Reward: ", reward, "Done: ", done, "Info: ", info)
			state = next_state
		print("Episode: ", i)
		print("Final State: ", state)
		print("Final Reward: ", reward)
		print("Final Done: ", done)
		print("Final Info: ", info)
		print("\n\n\n")