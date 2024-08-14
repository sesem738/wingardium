import os
import time
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.callbacks import EvalCallback

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.WaypointAviary import WaypointAviary

DEFAULT_GUI = True

def run(gui=DEFAULT_GUI):
    train_env = make_vec_env(WaypointAviary,
                             n_envs=16,
                             seed=0)
    
    eval_env = WaypointAviary()
    
    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch=dict(pi=[240, 240], vf=[240, 240]))
    
    model = PPO('MlpPolicy',
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1)
    
    eval_callback = EvalCallback(eval_env,
                                 verbose=1,
                                 best_model_save_path='./best_model_circle/',
                                 log_path='./logs/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    
    model.learn(total_timesteps=5_000_000, callback=eval_callback, log_interval=100)

    #### Save the model ########################################
    model.save('final_circle.zip')



if __name__ == '__main__':
    run(DEFAULT_GUI)
