import os 
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.TrialAviary import TrialAviary
from gym_pybullet_drones.envs.WaypointAviary import WaypointAviary
from gym_pybullet_drones.utils.utils import sync

DEFAULT_GUI = True

def run(gui=DEFAULT_GUI):

    test_env = WaypointAviary(gui=gui)

    hover_RPM = test_env.HOVER_RPM

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action = np.array([-1,1,-1,1]).reshape(1,4)
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(obs)
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated or truncated:  # Check for both terminated and truncated
            obs, info = test_env.reset(seed=42, options={})  # Reset if either condition is met

    test_env.close()

if __name__== '__main__':
    run(DEFAULT_GUI)