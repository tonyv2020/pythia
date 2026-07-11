"""Honest baselines every model must beat (or honestly report it doesn't)."""

from .last_return import LastReturn
from .random_walk import RandomWalk
from .raptor_p_move import RaptorPMoveStub

__all__ = ["RandomWalk", "LastReturn", "RaptorPMoveStub"]
