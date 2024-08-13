import random
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.waypoints import WaypointGenerator
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

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
        self.waypt_cnt = [0] * self.NUM_DRONES
        self.waypt_threshold = 0.1
        if self.waypoints is None:
            self.waypoints = [self.waypt_gen.generate_random_trajectory(
                self.INIT_XYZS[i, 0], self.INIT_XYZS[i, 1], self.INIT_XYZS[i, 2]
            ) for i in range(self.NUM_DRONES)]

        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN:
            obs = np.zeros((self.NUM_DRONES, 15))
            for i in range(self.NUM_DRONES):
                drone_state = self._getDroneStateVector(i)
                current_pos = drone_state[0:3]
                waypoint = np.asarray(self.waypoints[i][self.waypt_cnt[i]]).reshape(3,)
                pos_diff = waypoint - current_pos
                obs[i, :] = np.hstack([drone_state[0:3], drone_state[7:10], drone_state[10:13], drone_state[13:16], pos_diff])
            
            ret = obs.astype('float32')
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            
            return ret
        else:
            print("[ERROR] in MultiWaypointAviary._computeObs()")

    def _computeReward(self):
        rewards = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            vel = state[10:13]
            current_waypoint = (self.waypoints[i][self.waypt_cnt[i]]).reshape(3,)
            
            distance = np.linalg.norm(current_waypoint - pos)
            direction_to_waypoint = np.divide(current_waypoint - pos, distance + 1e-8)
            velocity_alignment = np.dot(vel, direction_to_waypoint)     

            rewards[i] = -distance + velocity_alignment

            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[i])
            if len(contact_points) > 0:
                self.collision[i] = True
                rewards[i] -= 100 

            if distance < self.waypt_threshold:
                rewards[i] += 10
                self.waypt_cnt[i] += 1
                if self.waypt_cnt[i] >= len(self.waypoints[i]):
                    rewards[i] += 50
                    self.waypt_cnt[i] = 0
        
        return rewards

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
        return {"drone_{}".format(i): {
            "waypoint": self.waypt_cnt[i],
            "collision": self.collision[i]
        } for i in range(self.NUM_DRONES)}