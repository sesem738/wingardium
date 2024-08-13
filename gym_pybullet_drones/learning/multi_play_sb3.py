import os 
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.MultiWaypointAviary import MultiWaypointAviary
from gym_pybullet_drones.utils.utils import sync

DEFAULT_GUI = True
NUM_DRONES = 3  # Set the number of drones you want to use

def run(gui=DEFAULT_GUI):
    # Initialize the MultiWaypointAviary environment
    test_env = MultiWaypointAviary(
        num_drones=NUM_DRONES,
        gui=gui
    )

    # Load the trained model for drone
    models = [PPO.load('best_model/best_model.zip') for i in range(NUM_DRONES)]

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        # Predict actions for all drones
        actions = []
        for drone in range(NUM_DRONES):
            action, _states = models[drone].predict(
                obs[drone],
                deterministic=True
            )
            actions.append(action)
        
        # Step the environment
        obs, rewards, terminated, truncated, info = test_env.step(actions)
        test_env.render()

        print(f"Step {i}: Terminated: {terminated}, Rewards: {rewards}")
        
        sync(i, start, test_env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            obs, info = test_env.reset(seed=42, options={})

    test_env.close()

if __name__ == '__main__':
    run(DEFAULT_GUI)