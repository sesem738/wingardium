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
                 ctrl_freq: int = 30,
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

        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    #################################################################################


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
                pos_diff = waypoint - current_pos
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
        """Computes the current reward value."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        current_waypoint = (self.waypoints[self.waypt_cnt]).reshape(3,)
        
        # Distance to current waypoint
        distance = np.linalg.norm(current_waypoint - pos)
        
        # Reward for moving towards the waypoint
        direction_to_waypoint = np.divide(current_waypoint - pos, distance + 1e-8)  # Normalized direction
        velocity_alignment = np.dot(vel, direction_to_waypoint)     

        # Base reward
        reward = -distance + velocity_alignment

        # Penalize for colllision
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
        if len(contact_points) > 0:
            self.collision = True
            reward -= 100 

        # Reward for reaching waypoint
        if distance < self.waypt_threshold:
            reward += 10
            self.waypt_cnt += 1
            if self.waypt_cnt >= len(self.waypoints):
                reward += 50  # Bonus for completing all waypoints
                self.waypt_cnt = 0  # Reset to first waypoint
        
        return reward

    def _computeTerminated(self):
        """Computes whether the episode should terminate."""
        
        if self.collision == True:
            return True
        
        state = self._getDroneStateVector(0)
        
        # Terminate if drone is too far from any waypoint
        if min([np.linalg.norm(state[0:3] - wp) for wp in self.waypoints]) > 10:
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