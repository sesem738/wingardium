import os
import numpy as np
import pybullet as p
from gymnasium import spaces


from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.waypoints import WaypointGenerator
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class WaypointAviary(BaseRLAviary):
    # Single RL Agent For Obstable Avoidance

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
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

        self.EPISODE_LEN_SEC = 8
        self.way_cnt = 0
        self.waygen= WaypointGenerator(0, 0)
        self.waypoints = self.waygen.generate_circle(20)

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
        
        ###############################################################################################

        # def _addObstacles(self):
        #     pass

        ###############################################################################################

        def _observationSpace(self):
            """Returns the observation space of the environment.

            Returns
            -------
            dict[str, dict[str, ndarray]]
                A Dict with NUM_DRONES entries indexed by Id in string format,
                each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

            """
            #### Observation vector ### X        Y        Z     R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            obs_lower_bound = np.array([-np.inf, -np.inf, 0., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
            obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
            return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                        high=obs_upper_bound,
                                                                        dtype=np.float32
                                                                        ),
                                                    "dep": spaces.Box(low=.01,
                                                                    high=1000.,
                                                                    shape=(self.IMG_RES[1],
                                                                        self.IMG_RES[0]),
                                                                    dtype=np.float32
                                                                    )
                                                    }) for i in range(self.NUM_DRONES)})
            

        ###############################################################################################

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
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                        img_input=self.rgb[i],
                                        path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                        frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                        )
                state = self._getDroneStateVector(i)        
                stateObs = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]]).reshape(12,)

                obs[str(i)] = {"state": stateObs, \
                            "dep": self.dep[i], \
                            }
            return obs

        ###############################################################################################

        def _computeRewards(self):
            state = self._getDroneStateVector(0)
            dist = -1 * np.linalg.norm(self.waypoint[self.way_cnt]-state[0:3])**2
            col = collide is huge_negative
            alive = alive + 1

            

        ###############################################################################################

        def _computeTerminated(self):
            """Computes the current done value.

            Returns
            -------
            bool
                Whether the current episode is done.

            """
            state = self._getDroneStateVector(0)
            contact_points = p.getContactPoints(self.DRONE_IDS)
            if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
                return True
            elif len(contact_points) > 0:
                print("Collision detected!")
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