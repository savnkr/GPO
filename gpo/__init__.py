"""Gaussian process operators with exact inference, stochastic dual descent and Matheron posterior sampling."""
from .model import GPO
from .conditioning import SensorUpdate, covariance_columns, greedy_sensors
from .solvers import KernelOperator, cg, sdd

__version__ = "2.0.0"
__all__ = ["GPO", "SensorUpdate", "covariance_columns", "greedy_sensors", "KernelOperator", "cg", "sdd"]
