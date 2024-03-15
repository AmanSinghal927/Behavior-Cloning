from gym.envs.registration import register 

register(
	id='particle-v0',
	entry_point='particle_envs.envs:ParticleEnv',
	max_episode_steps=300,
	)