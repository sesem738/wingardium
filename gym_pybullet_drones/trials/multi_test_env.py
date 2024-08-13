from gym_pybullet_drones.envs.MultiWaypointAviary import MultiWaypointAviary
from gym_pybullet_drones.envs.TrialAviary import TrialAviary

test_env = MultiWaypointAviary(gui=True)
obs, info = test_env.reset(seed=42, options={})

print(obs)