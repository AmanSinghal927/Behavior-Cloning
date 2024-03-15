import gym
from gym import spaces
import cv2
import numpy as np

class ParticleEnv(gym.Env):
	def __init__(self, height=84, width=84, step_size=10, reward_type='dense', 
	      		 reward_scale=None, block=None):
		super(ParticleEnv, self).__init__()

		self.height = height
		self.width = width
		self.step_size = step_size
		self.reward_type = reward_type
		self.reward_scale = np.sqrt(height**2 + width**2) if reward_scale is None else reward_scale
		self.block = block

		# self.goals = []
		# self.num_goals = num_goals
		
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
		
		# Set initial start
		self.state = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)

		# Set initial goal
		goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
		while (self.state == goal).all():
			goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
		self.goal = goal
	
	def step(self, action):
		prev_state = self.state
		self.state = np.array([self.state[0] + self.step_size * action[0], self.state[1] + self.step_size * action[1]], dtype=np.float32)
		
		if self.state[0]<0 or self.state[0]>=self.height or self.state[1]<0 or self.state[1]>=self.width:
			reward = -10 if self.reward_type == 'dense' else 0 #-1
			self.state = prev_state
			done = False
		elif self.observation[int(self.state[0]), int(self.state[1])]==1:
			reward = -10 if self.reward_type == 'dense' else 0 #-1
			self.state = prev_state
			done = False
		elif self.observation[int(self.state[0]), int(self.state[1])] == 2:
			reward = 1
			done = True
		else:
			reward = 0
			done = False
		self._step += 1
		
		info = {}
		info['is_success'] = 1 if reward==0 else 0 

		# Normalize state
		state = np.array([self.state[0] / self.height, self.state[1] / self.width]).astype(np.float32)

		return state, reward, done, info
	
	def reset(self, start_state=None, reset_goal=False, goal_state=None): #, demo=None):
		start_state = np.array(start_state).astype(np.float32) if start_state is not None else None
		goal_state = np.array(goal_state).astype(np.float32) if goal_state is not None else None

		# set start state
		if start_state is None:
			self.state = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
		else:
			start_state[0], start_state[1] = start_state[0] * self.height, start_state[1] * self.width
			self.state = np.array(start_state).astype(np.int32)

		# set goal
		if reset_goal:
			if goal_state is not None:
				goal_state[0], goal_state[1] = goal_state[0] * self.height, goal_state[1] * self.width
				self.goal = np.array(goal_state).astype(np.int32)
			else:
				goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
				while (self.state == goal).all():
					goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
				self.goal = goal

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
		goal_hmin, goal_hmax = int(self.goal[0]-10), int(self.goal[0]+10)
		goal_wmin, goal_wmax = int(self.goal[1]-10), int(self.goal[1]+10)
		goal_hmin, goal_hmax = max(0, goal_hmin), min(self.height-1, goal_hmax)
		goal_wmin, goal_wmax = max(0, goal_wmin), min(self.width-1, goal_wmax)
		for h in range(goal_hmin, goal_hmax+1):
			for w in range(goal_wmin, goal_wmax+1):
				self.observation[h,w] = 2

		self._step = 0

		# Normalize state
		state = np.array([self.state[0] / self.height, self.state[1] / self.width]).astype(np.float32)
		return state


	def render(self, mode='', width=None, height=None):
		img = np.ones(self.observation.shape).astype(np.uint8) * 255
		# Identify blocked region
		blocked = np.where(self.observation == 1)
		img[blocked] = 0

		hmin, hmax = max(0, self.goal[0]-10), min(self.height-1, self.goal[0] + 10)
		wmin, wmax = max(0, self.goal[1]-10), min(self.width-1, self.goal[1] + 10)
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