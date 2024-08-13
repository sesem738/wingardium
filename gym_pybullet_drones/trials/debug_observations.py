import gymnasium as gym
from gym_pybullet_drones.envs.WaypointAviary import WaypointAviary
import numpy as np

class ObservationDebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._check_and_print_observation(observation)
        return observation, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._check_and_print_observation(observation)
        return observation, reward, terminated, truncated, info
    
    def _check_and_print_observation(self, observation):
        if observation.shape != (1, 31):
            print(f"Unexpected observation shape: {observation.shape}")
            print("Observation content:")
            print(observation)
            print("\nObservation details:")
            print(f"Type: {type(observation)}")
            print(f"Shape: {observation.shape}")
            print(f"Data type: {observation.dtype}")
            if isinstance(observation, np.ndarray):
                print(f"Min value: {observation.min()}")
                print(f"Max value: {observation.max()}")
                print(f"Mean value: {observation.mean()}")

def main():
    env = WaypointAviary()
    env = ObservationDebugWrapper(env)
    
    observation, info = env.reset()
    
    for _ in range(1000):  # Run for 100 steps
        action = env.action_space.sample()  # Your agent would choose an action here
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main()