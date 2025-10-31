from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Protocol, Tuple, TypeVar, Generic


TaskT = TypeVar("TaskT")
Observation = Dict[str, Any]
Info = Dict[str, Any]
Action = Any
StepResult = Tuple[Observation, float, bool, Info]


class BaseEnv(ABC, Generic[TaskT]):
    """Abstract base class implementing :class:`EnvAdapter`.

    Subclasses are responsible for simulator/robot specific behavior while this
    base class standardizes shared attributes and sensible defaults.
    """

    def __init__(self, config: Any, log_file: Any | None = None) -> None:
        self.config = config
        self.log_file = log_file

    @property
    def cfg(self) -> Any:
        """Alias retained for legacy adapters that still reference ``self.cfg``."""
        return self.config

    # --- Benchmark / task lifecycle -------------------------------------------------
    @abstractmethod
    def get_task_iterator(self) -> Iterable[TaskT]:
        """Return an iterable of task handles consumed by the evaluator."""

    def get_task_description(self, task_handle: TaskT) -> str:
        """Optional: override to return a human-readable task description."""
        return str(task_handle)

    @abstractmethod
    def reset(self, task_handle: TaskT, episode_index: int) -> bool:
        """Reset the environment for the given task/episode. Return False to skip."""

    @abstractmethod
    def max_steps(self, task_handle: TaskT | None = None) -> int:
        """Maximum number of control steps for the given task."""

    # # --- Core control loop ----------------------------------------------------------
    # @abstractmethod
    # def get_observation(self) -> Observation:
    #     """Current observation dictionary exposed to the model adapter."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Advance the simulator with the provided action."""

    # --- Optional adapter hooks -----------------------------------------------------
    def process_action(self, action: Action, model_family: str | None = None) -> Action:
        """Override if the environment requires per-model action preprocessing."""
        return action

    def action_spec(self) -> Mapping[str, Any] | None:
        """Override to surface a structured action specification."""
        return None

    def observation_spec(self) -> Mapping[str, Any] | None:
        """Override to surface a structured observation specification."""
        return None

    # --- Utilities ------------------------------------------------------------------
    def render(self, mode: str = "rgb_array") -> Any:
        """Optional rendering hook for visualization."""
        return None

    def close(self) -> None:
        """Clean shutdown for sim/real connections."""
        return None
