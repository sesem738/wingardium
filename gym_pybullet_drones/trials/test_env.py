from gym_pybullet_drones.envs.WaypointAviary import WaypointAviary
from gym_pybullet_drones.envs.TrialAviary import TrialAviary

test_env = WaypointAviary(gui=True)
obs, info = test_env.reset(seed=42, options={})

print(obs)