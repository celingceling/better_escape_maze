# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# """ references 
# C:\Users\LICF\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\
#     manager_based\navigation\config\anymal_c\navigation_env_cfg.py
#     """

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg

from . import mdp # from init

##
# Pre-defined configs
##

from isaaclab_assets.robots.ant import ANT_CFG  # isort:skip
import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
# from better_escape_maze.assets.ant_env_cfg_modified import AntEnvCfg

##
# Scene definition
##

LOW_LEVEL_ENV_CFG = AntEnvCfg()
# "C:\Users\LICF\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\ant\ant_env_cfg.py"

@configclass
class BetterEscapeMazeSceneCfg(InteractiveSceneCfg):
    """Configuration for a ant scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = ANT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ANT_CFG.spawn.replace( # spawn modifies PhysX and USD behavior
            activate_contact_sensors=True,
        ),
        ) # replace path
    
    # contact forces
    # add a contact sensor to every prim under Robot
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # maze
    maze_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Maze",
        spawn=sim_utils.UsdFileCfg(
            usd_path=r"C:\Users\LICF\isaac_demo\better_escape_maze\source\maze.usd", # what is raw string
            scale=(1.0,1.0,1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
        )
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


    
    
@configclass
class EventCfg: # isaaclab_tasks.manager_based.navigation.config.anymal.navigation_env_cfg
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )



##
# MDP settings
##


@configclass
class ActionsCfg: 
    """Action specifications for the MDP.
        celing brainstorm: i think i want action space = 3, vx vy wz
        and so i really don't need joint effort, maybe something new 
        
        maybe i can use OperationalSpaceControllerAction
        
        well i don't need OSC for the commands do i, use pretrained policy
        
        use UniformVelocityCommand?
        
        nvm i found something
        
        OK actions are still joint actions, it's just that they're pretrained already so we use this
    """
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=r"source\better_escape_maze\better_escape_maze\tasks\manager_based\better_escape_maze\ant_policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_effort,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )
    
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """
        Observations for policy group.
        other group is low level ant stuff
        
        for this navigation one it's just base_lin_vel (3), gravity (1), velocity_cmd (3?)
        """

        # observation terms (order preserved)
        # obs terms  contain information about obs function to call (obs = properties that go into a policy)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity_command"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    
        might want to add distance to goal?
    
    """

    # early termination bad
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    
    # track position
    velocity_tracking = RewTerm(
        func=mdp.velocity_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "velocity_command"},
    )
    
    # track position lower std?
    velocity_tracking_fine_grained = RewTerm(
        func=mdp.velocity_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "velocity_command"},
    )
    
    # # track heading # no heading control
    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_error_abs,
    #     weight=-0.2,
    #     params={"command_name": "velocity_command"},
    # )

@configclass
class CommandsCfg: # might wanna change to velocity
    """Command terms for the MDP."""

    velocity_command = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        heading_command = False,
        ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.5, 1.5),
            ang_vel_z=(-2.0, 2.0),
        )
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Hit a wall 
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=['torso', 'front_left_leg', 
                            'front_left_foot', 'front_right_leg', 
                            'front_right_foot', 'left_back_leg', 
                            'left_back_foot', 'right_back_leg', 
                            'right_back_foot']), 
            "threshold": 1.0},
    )


##
# Environment configuration
##


@configclass
class BetterEscapeMazeEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings (inherit most from ant)
    # also referenced from polict_in_inference_isd.py
    # scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    
    scene: BetterEscapeMazeSceneCfg = BetterEscapeMazeSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # # import maze
    # scene.terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="usd",
    #     usd_path=f"C:\Users\LICF\isaac_demo\escape_maze\maze_spot.usd"
    # )
    
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        
        # general settings
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10 # incrase decimation because nav doesn't need that much
        self.episode_length_s = self.commands.velocity_command.resampling_time_range[1] # chagne to vel also what is this
        self.sim.device = "cuda:0"   # or "cuda"
        
        # what is this, do i need to add sensors to ant
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = (
        #         self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
        #     )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        
        
    class BetterEscapeMazeEnvCfg_PLAY(ManagerBasedRLEnvCfg):
        def __post_init__(self) -> None:
            # post init of parent
            super().__post_init__()

            # make a smaller scene for play
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
            # disable randomization for play
            self.observations.policy.enable_corruption = False