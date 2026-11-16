from .interface import *
from .interface import BaseSimulation
from .physics import PhysicsMixin


class Simulation(PhysicsMixin, BaseSimulation):
    """BaseSimulation (problem-agnostic core) + PhysicsMixin (physics methods)."""
