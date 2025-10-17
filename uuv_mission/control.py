"""Simple controller module for the UUV project.

Provides a PD controller with a small, well-documented API used by ClosedLoop.
"""
from __future__ import annotations

from typing import Optional


class PDController:
    """Proportional-Derivative controller.

    The controller maintains the previous error internally so it can compute
    a discrete derivative term. The controller API is intentionally small:

    - control(reference, observation) -> action
    - reset() to clear internal state
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6, dt: float = 1.0):
        self.kp = float(kp)
        self.kd = float(kd)
        self.dt = float(dt)
        self._last_error: Optional[float] = None

    def control(self, reference: float, observation: float) -> float:
        """Compute control action u = Kp*e + Kd*(e - e_last)/dt.

        The first call yields a derivative of zero (no prior error available).
        """
        error = float(reference) - float(observation)
        if self._last_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._last_error) / self.dt

        u = self.kp * error + self.kd * derivative
        self._last_error = error
        return float(u)

    def reset(self) -> None:
        """Reset the internal state (previous error)."""
        self._last_error = None
