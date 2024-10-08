from gymnasium.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:CtrlAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VelocityAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:HoverAviary',
)

register(
    id='multihover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:MultiHoverAviary',
)

register(
    id='trialhover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:TrialAviary',
)

register(
    id='waypoint-aviary-v0',
    entry_point='gym_pybullet_drones.envs:WaypointAviary',
)

register(
    id='multi-waypoint-aviary-v0',
    entry_point='gym_pybullet_drones.envs:MultiWaypointAviary',
)