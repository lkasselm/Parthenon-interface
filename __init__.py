from .interface import *
from .interface import BaseSimulation
from .output_types import (
    YTFieldFrame,
    FieldFrame,
    EnergyTransferResult,
    OutputFrame,
    OutputType,
)
from .physics import PhysicsMixin


class Simulation(PhysicsMixin, BaseSimulation):
    """BaseSimulation (problem-agnostic core) + PhysicsMixin (physics methods)."""
