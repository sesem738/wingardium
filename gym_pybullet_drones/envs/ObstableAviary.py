import random
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.waypoints import WaypointGenerator
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class WaypointAviary(BaseRLAviary):
    """Single agent RL problem: Waypoints Tracking."""

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
                 obs: ObservationType=ObservationType.RGB,
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

        self.seq_len = 3
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
        
        self.depth_buffer = deque(maxlen=self.seq_len)

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

        # Load Environment
        self._load_environment()

        # Load Waypoints
        self.waypt_cnt = 1
        self.waypt_threshold = 0.01
        self.waypoints = self.waypt_gen.generate_circle(self.INIT_XYZS[:,0], self.INIT_XYZS[:,1], self.INIT_XYZS[:,2])

        # Clear and pre-fill the depth image buffer
        for _ in range(self.seq_len):
            _, depth, _ = self._getDroneImages(0)
            self.depth_buffer.append(depth)

        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    #################################################################################


    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        #### Observation vector ### X        Y        Z      R       P       Y       VX       VY       VZ       WX       WY       WZ       TX      TY     TZ
        obs_lower_bound = np.array([-np.inf, -np.inf, 0., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
        return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                     high=obs_upper_bound,
                                                                     dtype=np.float32
                                                                     ),
                                                 "dep": spaces.Box(low=.01,
                                                                   high=1000.,
                                                                   shape=(
                                                                       self.seq_len,
                                                                       self.IMG_RES[1],
                                                                       self.IMG_RES[0]
                                                                   ),
                                                                   dtype=np.float32
                                                                   )
                                                 }) for i in range(self.NUM_DRONES)})
    

    #################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix,
        "rgb", "dep", and "seg" are matrices containing POV camera captures.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        obs = {}
        for i in range(self.NUM_DRONES):
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                self.depth_buffer.append(self.dep[i])

                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.DEP, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                      img_input=self.dep[i],
                                      path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
                    
            obs_kin = self._getDroneStateVector(i)
            waypoint = np.asarray(self.waypoints[self.waypt_cnt]).reshape(3,)
            obs_kin = np.hstack([obs_kin[0:3], obs_kin[7:10], obs_kin[10:13], obs_kin[13:16], waypoint]).reshape(15,)
            obs_kin = obs_kin.astype('float32')

            obs[str(i)] = {"state": obs_kin, \
                           "dep": np.array(self.depth_buffer).astype('float32'), \
                           }
        return obs

    #################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        fin = 0
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        waypt = self.waypoints[self.waypt_cnt]
        direction = waypt - pos

        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
        if len(contact_points) > 0:
            print("Collision detected!")
            self.collision = True
            for point in contact_points:
                print(f"Contact with body {point[2]} at link {point[4]}")
                
        
        dist = -np.linalg.norm(direction)
        vel_reward = np.dot(vel, direction / dist)
        collision = 100 if self.collision == True else 0

        if dist < self.waypt_threshold:
            self.target_cnt += 1
        else:
            self.target_cnt = 0

        if self.waypt_cnt > len(self.waypoints):
            fin = 50
            self.waypt_cnt = 0

        ret = vel_reward + dist + collision + fin


        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """

        if self.collision == True:
            return True

        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.waypoints[self.waypt_cnt]-state[0:3]) > 5:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
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


    def _load_environment(self):
        # Define tree properties (adjust as needed)
        tree_width = 0.5
        tree_height_min = 2
        tree_height_max = 5

        # Create trees randomly within a specified area
        for _ in range(50):
            # Randomize tree position (adjust area as needed)
            x = random.uniform(-10, 10) 
            y = random.uniform(-10, 10)
            tree_height = random.uniform(tree_height_min, tree_height_max)

            # Create a cuboid as a tree
            tree_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[tree_width/2, tree_width/2, tree_height/2], rgbaColor=[0.4, 0.2, 0, 1]) # Brown color
            tree_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tree_width/2, tree_width/2, tree_height/2]) 
            tree_id = p.createMultiBody(baseMass=0,  # Make trees static (or adjust mass)
                                    baseCollisionShapeIndex=tree_collision,
                                    baseVisualShapeIndex=tree_visual,
                                    basePosition=[x, y, tree_height/2])
            