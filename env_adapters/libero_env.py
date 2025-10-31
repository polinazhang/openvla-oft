from typing import Dict, Any, Tuple, Iterable
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from . import register_env
from .base_env import BaseEnv

import draccus
import numpy as np
import tqdm
# from libero.libero import benchmark
import os, sys
sys.path.insert(0, os.path.expanduser("/nethome/xzhang3205/LIBERO"))
from libero.libero import benchmark

from .config_general import GenerateConfig, log_message

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    invert_gripper_action,
    normalize_gripper_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

@dataclass(frozen=True)
class TaskHandle:
    task_id: int
    description: str
    suite_name: str
    metadata: dict

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


def validate_libero_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    # assert cfg.num_trials_per_task <= 10, "num_trials_per_task can't exceed 10 for libero. If more trials are needed, change how initial states are loaded for each episode."
    # No, it can exceed 10 under custom settings, which is what used by the original openvla repo

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"



@register_env("libero")
class LiberoEnv(BaseEnv):
    def __init__(self, cfg: GenerateConfig, log_file=None):
        """
        config: environment-specific parameters (yaml-driven).
        Should include sim type, task name, camera setup, etc.
        """
        validate_libero_config(cfg)
        self.config = cfg

        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[cfg.task_suite_name]()
        self.num_tasks = self.task_suite.n_tasks
        self.log_file = log_file
        log_message(f"Task suite: {cfg.task_suite_name}", log_file)

        self.env = None
        self.initial_states = None
        self.all_initial_states = None

    def get_task_iterator(self) -> Iterable[Any]:
        """
        Returns an iterable of opaque task items.
        Examples:
          - LIBERO: [Task, Task, ...]
          - CALVIN: [(initial_state, sequence), ...]
        The evaluator will pass each task_item back to reset()/get_task_description().
        """

        task_handles = []
        for task_id in range(self.num_tasks):
            task_item = self.task_suite.get_task(task_id)
            handle = TaskHandle(
                task_id=task_id,
                description=task_item.language,
                suite_name=self.config.task_suite_name,
                metadata={}
                # metadata={"task_item": task_item},  # optional caching
            )
            task_handles.append(handle)

        return task_handles

    def get_task_description(self, task_item: TaskHandle) -> str:
        """
        Optional. Human-readable label for logging / videos.
        LIBERO: task.language
        CALVIN: ' -> '.join(sequence)
        """
        return task_item.description
    
    # def get_observation(self):
    #     """
    #     Might not need this.
    #     """
    #     return self.env.get_observation()

#############  To do: move this to the model side ###############
    def process_action(self, action, model_family):
        """Pre-process action before sending to environment."""
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        action = normalize_gripper_action(action, binarize=True)

        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        if model_family == "openvla":
            action = invert_gripper_action(action)

        return action

    # def reset(self, task_id):

    #     self.initial_states, self.all_initial_states = load_initial_states(self.cfg, self.task_suite, task_id, self.log_file)
    #     task_item = self.task_suite.get_task(task_id)

    #     # Initialize environment and get task description
    #     self.env, self.task_description = get_libero_env(task_item, self.cfg.model_family, resolution=self.cfg.env_img_res)

    def reset(self, task_handle: TaskHandle, episode_index: int, max_retries: int = 3):
        ## logid inside openvla:run_task()
        ## Libero specific expert demo checking
        if self.cfg.initial_states_path != "DEFAULT":
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_handle.description.replace(" ", "_")
            episode_key = f"demo_{episode_index}"

            # Skip episode if expert demonstration failed to complete the task
            if not self.all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_handle.task_id} episode {episode_index} due to failed expert demo!", self.log_file)
            return False

        for attempt in range(max_retries):
            done_during_settle = self._reset_once(task_handle, episode_index)

            if not done_during_settle:
                return True

            log_message(
                f"Done during settle (attempt {attempt+1}/{max_retries}), retrying environment reset...", 
                self.log_file
            )
            return False

        raise RuntimeError(
            f"Failed to produce non-terminal initial state after {max_retries} attempts "
            f"for task {task_handle.description} trial {episode_index}"
        )


    def _reset_once(self, task_handle: TaskHandle, episode_index: int):
        task = self.task_suite.get_task(task_handle.task_id)
        if self.initial_states is None:
            self.initial_states, self.all_initial_states = load_initial_states(self.config, self.task_suite, task_handle.task_id, self.log_file)

        # pick initial state
        if self.all_initial_states is not None:
            key = task_handle.description.replace(" ", "_")
            ep_key = f"demo_{episode_index}"
            self.init_state = self.all_initial_states[key][ep_key]
        else:
            # It was an assertion in the original repo too! Not on me
            assert episode_index < len(self.initial_states), "episode_index exceeds available default initial states"
            self.init_state = self.initial_states[episode_index]


        # (re)create env
        self.env, _ = get_libero_env(task, self.config.model_family, resolution=self.config.env_img_res)

        # set seed and apply initial state
        self.env.seed(self.config.seed)

        ######## logic inside openvla: run_episode() #########
        self.env.reset()
            # Set initial state if provided
        if self.init_state is not None:
            self.init_obs = self.env.set_init_state(self.init_state)
        else:
            self.init_obs = self.env.get_observation()

        # if self.cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        #     print(f"WARNING: cfg.num_open_loop_steps ({self.cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
        #         f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
        #         "both speed and success rate), we recommend executing the full action chunk.")
        # self.action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
        # Action queue should belong to the model side instead of libero side

        # let scene settle # also logic inside run_episode()
        for _ in range(self.config.num_steps_wait):
            obs, reward, done, info = self.env.step(get_libero_dummy_action(self.cfg.model_family))
        self.init_obs = obs
        return done
    
    def get_initial_observation(self):
        return self.init_obs 

    def step(self, action):
        ### in progress
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
        # # LIBERO success is exposed in info["success"]; keep it explicit & boolean
        # info = dict(info)
        # info.setdefault("success", bool(info.get("success", False)))
        # info.setdefault("truncation", False)
        # return obs, done, info

    # def is_success(self, info: Dict[str, Any]) -> bool:
    #     return bool(info.get("success", False))

    def max_steps(self, task_handle: TaskHandle) -> int:
        return TASK_MAX_STEPS[self.cfg.task_suite_name]
    
    def action_spec(self) -> Dict[str, Any]:
        # For LIBERO: 7-DoF pose delta + gripper scalar in [-1, +1] (after your processing)
        return {"shape": (8,), "dtype": np.float32, "components": ["dx","dy","dz","dax","day","daz","daw","gripper"]}

    def observation_spec(self) -> Dict[str, Any]:
        return {
            "rgb": {"keys": ["full_image", "wrist_image"], "size": (self.config.img_h, self.config.img_w)},
            "proprio": {"shape": (7,), "components": ["eef_x","eef_y","eef_z","ax","ay","az","gripper_qpos"]},
        }


    # def step(self, action: Any) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    #     """
    #     Step the env with a model-produced action.
    #     Returns: (observation, done, info)
    #     """


    def render(self, mode: str = "rgb_array") -> Any:
        """Video generation should NOT exist within the class, but outside the class and separate for every environment."""

    def close(self):
        """Clean shutdown for sim/real connection."""
        pass

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None
    







    

    
@dataclass
class LiberoGenerateConfig(GenerateConfig):
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)