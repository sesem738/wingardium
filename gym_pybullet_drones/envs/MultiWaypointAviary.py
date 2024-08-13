# MultiWaypointAviary.py
import random
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.waypoints import WaypointGenerator
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class MultiWaypointAviary(BaseRLAviary):
    """Multi-agent RL problem: Waypoints Tracking."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=3,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 waypoints=None
                 ):
        """Initialization of a multi-agent RL environment.

        Parameters
        ----------
        num_drones : int, optional
            The number of drones to use in the environment.
        waypoints : list of lists, optional
            A list of waypoints for each drone. If None, random waypoints will be generated.
        [... other parameters remain the same ...]
        """

        self.EPISODE_LEN_SEC = 8
        self.waypt_gen = WaypointGenerator()
        self.waypoints = waypoints
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    def reset(self, 
              seed: int = None,
              options: dict = None):

        self.INIT_XYZS = np.array([np.random.uniform(-5, 5, 3) for _ in range(self.NUM_DRONES)])
        self.INIT_XYZS[:, 2] = np.random.uniform(1, 5, self.NUM_DRONES)

        self.target_cnt = [0] * self.NUM_DRONES
        self.collision = [False] * self.NUM_DRONES

        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self._startVideoRecording()

        # Load Waypoints
        self.waypt_cnt = [1] * self.NUM_DRONES
        self.waypt_threshold = 0.1
        if self.waypoints is None:
            self.waypoints = [self.waypt_gen.generate_random_trajectory(
                self.INIT_XYZS[i, 0], self.INIT_XYZS[i, 1], self.INIT_XYZS[i, 2]
            ) for i in range(self.NUM_DRONES)]

        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    def _computeObs(self):
        """Computes the observation for each drone."""
        if self.OBS_TYPE == ObservationType.KIN:  # Kinematic observation
            obs = np.zeros((self.NUM_DRONES, 15))
            for i in range(self.NUM_DRONES):
                drone_state = self._getDroneStateVector(i)  # Get drone state vector
                current_pos = drone_state[0:3]  # Extract current position
                waypoint = np.asarray(self.waypoints[i][self.waypt_cnt[i]]).reshape(3,)  # Extract current waypoint
                pos_diff = waypoint - current_pos  # Calculate difference between current position and waypoint
                # Construct observation vector: [pos, rpy, vel, ang_vel, pos_diff]
                obs[i, :] = np.hstack([drone_state[0:3], drone_state[7:10], drone_state[10:13], drone_state[13:16], pos_diff])

            ret = obs.astype('float32')  # Convert to float32
            # Add action buffer to observation
            for i in range(self.ACTION_BUFFER_SIZE):
                # Reshape the action buffer part to be 2D
                action_buffer_part = np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)]).reshape(self.NUM_DRONES, -1)
                ret = np.hstack([ret, action_buffer_part])

            return ret
        else:
            print("[ERROR] in MultiWaypointAviary._computeObs()")

    def _computeReward(self):
        """Computes the reward for all drones as a single scalar value."""
        total_reward = 0  # Initialize total reward
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)  # Get drone state vector
            pos = state[0:3]  # Extract current position
            vel = state[10:13]  # Extract current velocity
            current_waypoint = (self.waypoints[i][self.waypt_cnt[i]]).reshape(3,)  # Extract current waypoint

            # Calculate distance to waypoint
            distance = np.linalg.norm(current_waypoint - pos)
            # Calculate direction to waypoint
            direction_to_waypoint = np.divide(current_waypoint - pos, distance + 1e-8)
            # Calculate velocity alignment with direction to waypoint
            velocity_alignment = np.dot(vel, direction_to_waypoint)

            # Base reward based on distance and velocity alignment
            reward = -distance + velocity_alignment

            # Penalize collisions
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[i])
            if len(contact_points) > 0:
                self.collision[i] = True
                reward -= 100

            # Reward for reaching waypoint
            if distance < self.waypt_threshold:
                reward += 10
                print("Hurrrraaaaayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                self.waypt_cnt[i] += 1  # Move to the next waypoint
                # Reward for completing all waypoints
                if self.waypt_cnt[i] >= len(self.waypoints[i]):
                    reward += 50
                    self.waypt_cnt[i] = 0  # Reset to the first waypoint

            total_reward += reward  # Accumulate reward for each drone

        return total_reward  # Return the total reward as a scalar

    def _computeTerminated(self):
        return any(self.collision)

    def _computeTruncated(self):
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
                return True
            if min([np.linalg.norm(state[0:3] - wp) for wp in self.waypoints[i]]) > 10:
                return True
        
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        return False

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years