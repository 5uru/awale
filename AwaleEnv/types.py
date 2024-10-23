from typing import TYPE_CHECKING, NamedTuple, Generic

import chex
from typing_extensions import TypeAlias

from jumanji.types import StepType

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

# Define a type alias for a board, which is an array.
Board: TypeAlias = chex.Array


class State(NamedTuple):
    board: Board
    step_count: chex.Numeric
    action_space: chex.Array
    key: chex.PRNGKey
    score: chex.Array
    current_player: chex.Numeric


class Observation(NamedTuple):
    state: Board
    action_space: chex.Array
