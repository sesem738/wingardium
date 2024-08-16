import random
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.waypoints import WaypointGenerator
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class WaypointAviary(BaseRLAviary):
    """X agent RL problem: Waypoints Tracking."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """

        self.EPISODE_LEN_SEC = 8
        self.waypt_gen = WaypointGenerator()

        super().__init__(drone_model=drone_model,
                         num_drones=1,
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
        
    ################################################################################

    def reset(self, 
              seed: int = None,
              options: dict = None):

        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z = np.random.uniform(1, 5)
        self.INIT_XYZS = np.array([x, y, z]).reshape(1,3)

        self.target_cnt = 0
        self.collision = False

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()

        # Load Waypoints
        self.waypt_cnt = 1
        self.waypt_threshold = 0.1
        self.waypoints = self.waypt_gen.generate_random_trajectory(self.INIT_XYZS[:,0], self.INIT_XYZS[:,1], self.INIT_XYZS[:,2])

        # Visualize Waypoint (Without Collision)
        self.target_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 0, 0, 0.8], # Red with alpha transparency
            physicsClientId=self.CLIENT
        )
        self.target_body_id = p.createMultiBody(
            baseMass=0,  # Mass of 0 makes it non-collidable
            baseVisualShapeIndex=self.target_visual,
            basePosition=self.waypoints[self.waypt_cnt],
            physicsClientId=self.CLIENT
        )

        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info


    ##################################################################################


    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 15
            #### Observation vector ### X    Y     Z      R       P       Y       VX       VY       VZ       WX       WY       WZ     DX      DY      DZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo, lo, lo, lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi, hi, hi, hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    

    #################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,15) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 15
            obs_15 = np.zeros((self.NUM_DRONES,15))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                current_pos = obs[0:3]
                waypoint = np.asarray(self.waypoints[self.waypt_cnt]).reshape(3,)
                pos_diff = current_pos - waypoint
                obs_15[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], pos_diff]).reshape(15,)
            ret = np.array([obs_15[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

    #################################################################################
    
    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        ang_vel = state[13:16]

        current_waypoint = self.waypoints[self.waypt_cnt].reshape(3,) # Assuming self.waypoints and self.waypt_cnt are defined

        squared_distance = (pos[0] - current_waypoint[0])**2 + (pos[1] - current_waypoint[1])**2 + (pos[2] - current_waypoint[2])**2

        linvel_magnitude = np.sqrt(np.sum(vel**2))
        angvel_magnitude = np.sqrt(np.sum(ang_vel**2))

        # Change temperature parameters and re-scale
        sq_distance_temperature = 1.0
        linvel_temperature = 0.075  # Adjusted from 0.05 to 0.075
        angvel_temperature = 0.005  # Adjusted from 0.01 to 0.005
        angvel_penalty_temperature = 0.05  # Adjusted from 0.1 to 0.05

        # Compute the reward components and apply transformations
        sq_distance_reward = np.exp(-sq_distance_temperature * squared_distance)
        linvel_reward = np.exp(-linvel_temperature * (linvel_magnitude - 0.5)) - 1.0  # Modified to encourage variance in linvel_reward
        angvel_reward = np.exp(-angvel_temperature * angvel_magnitude)
        angvel_penalty = -np.exp(angvel_penalty_temperature * (angvel_magnitude - 0.5))  # Modified to encourage variance in angvel_penalty

        # Check success condition
        success_condition = 1 if squared_distance < 0.1 else 0
        success_reward = 15 * success_condition  # Increased the success reward from 10 to 15

        # Waypoint reached reward
        waypoint_reward = 0
        if squared_distance**0.5 < self.waypt_threshold:
            self.waypt_cnt += 1
            print("Hurrrrrrrrrrrayyyyyyyyyyyyyyyyy")
            if self.waypt_cnt >= len(self.waypoints):
                reward += 50  # Reduced from 50
                self.waypt_cnt = 0

        p.resetBasePositionAndOrientation(
                self.target_body_id,
                posObj=self.waypoints[self.waypt_cnt],
                ornObj=[0, 0, 0, 1], # No rotation
                physicsClientId=self.CLIENT)

        # Calculate the total reward
        total_reward = success_reward + sq_distance_reward + linvel_reward + angvel_reward + angvel_penalty + waypoint_reward

        return total_reward

    def _computeTerminated(self):
        """Computes whether the episode should terminate."""
        
        if self.collision == True:
            return True
        
        state = self._getDroneStateVector(0)
        current_waypoint = self.waypoints[self.waypt_cnt].reshape(3,)
        
        # Terminate if drone is too far from the CURRENT waypoint
        distance = np.linalg.norm(current_waypoint - state[0:3])
        if distance > 10:  # Adjust the threshold (10) as needed
            return True
        
        return False

    def _computeTruncated(self):
        """Computes whether the episode should be truncated."""
        state = self._getDroneStateVector(0)

        # Truncate if drone is too tilted
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        
        # Truncate if episode is too long
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        return False
        ################################################################################
        
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years