import os
import time
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.callbacks import EvalCallback

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.WaypointAviary import WaypointAviary

DEFAULT_GUI = True
CHECKPOINT_PATH = ''  # Update this with your checkpoint path

def run(gui=DEFAULT_GUI):
    # Create and normalize the training environment
    train_env = make_vec_env(WaypointAviary, n_envs=128, seed=0)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Create and normalize the evaluation environment
    eval_env = make_vec_env(WaypointAviary, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    # Copy the normalization statistics from the training environment to the evaluation environment
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Load the model from checkpoint or create a new one ####
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Loading model from checkpoint: {CHECKPOINT_PATH}")
        model = PPO.load(CHECKPOINT_PATH, env=train_env)
        print("[INFO] Model loaded successfully")
    else:
        print("[INFO] No checkpoint found. Creating a new model.")
        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                             net_arch=dict(pi=[240, 240], vf=[240, 240]))
        
        model = PPO('MlpPolicy',
                    train_env,
                    learning_rate=1.5e-4,
                    gamma=0.995,
                    clip_range=0.2,
                    batch_size=128,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./logs/tensorboard/",
                    verbose=1)

    #### Continue training the model ###########################
    eval_callback = EvalCallback(eval_env,
                                 verbose=1,
                                 best_model_save_path='./best_model_lr/',
                                 log_path='./logs/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    
    model.learn(total_timesteps=10_000_000, callback=eval_callback, log_interval=100, reset_num_timesteps=False)

    #### Save the model ########################################
    model.save('final_eureka_lr.zip')

if __name__ == '__main__':
    run(DEFAULT_GUI)