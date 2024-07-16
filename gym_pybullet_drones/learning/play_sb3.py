import os 
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.TrialAviary import TrialAviary
from gym_pybullet_drones.utils.utils import sync

DEFAULT_GUI = True

def run(gui=DEFAULT_GUI):

    test_env = TrialAviary(gui=gui)

    model = PPO.load("final_hover.zip")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})

    test_env.close()

if __name__== '__main__':
    run(DEFAULT_GUI)