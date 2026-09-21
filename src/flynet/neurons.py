"""Neuron models for spiking neural networks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NeuronModel(ABC):
    """Abstract base class for all neuron models."""

    @abstractmethod
    def step(self, input_current: float, dt: float) -> bool:
        """Advance one timestep. Returns True if the neuron fires a spike."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the neuron to its resting state."""

    @abstractmethod
    def get_potential(self) -> float:
        """Return the current membrane potential."""


class LIFNeuron(NeuronModel):
    """Leaky Integrate-and-Fire neuron.

    Dynamics:  dv/dt = (-(v - v_rest) + R_mem * I) / tau_mem

    During the refractory period the voltage is clamped to ``v_reset``
    and the neuron cannot fire.  On a spike the voltage goes to
    ``v_reset`` and the refractory timer starts.
    """

    def __init__(
        self,
        tau_mem: float = 20.0,
        v_rest: float = -70.0,
        v_thresh: float = -55.0,
        v_reset: float = -75.0,
        t_refrac: float = 3.0,
        R_mem: float = 10.0,
    ) -> None:
        self.tau_mem = tau_mem
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.t_refrac = t_refrac
        self.R_mem = R_mem

        self._v: float = v_rest
        self._refrac_remaining: float = 0.0

    def step(self, input_current: float, dt: float) -> bool:
        if self._refrac_remaining > 0.0:
            self._refrac_remaining -= dt
            return False

        dv = (-(self._v - self.v_rest) + self.R_mem * input_current) / self.tau_mem
        self._v += dv * dt

        if self._v >= self.v_thresh:
            self._v = self.v_reset
            self._refrac_remaining = self.t_refrac
            return True
        return False

    def reset(self) -> None:
        self._v = self.v_rest
        self._refrac_remaining = 0.0

    def get_potential(self) -> float:
        return self._v


class SpikingNeuron(NeuronModel):
    """Simple integer-threshold spiking neuron (SNN training style).

    Potential accumulates via weighted spike input, decays by ``leak``
    each step when above ``rest``, and spikes when it reaches
    ``threshold``.  After a spike the neuron enters a refractory period
    of ``refrac_time`` steps.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        rest: float = 0.0,
        min_pot: float = -500.0,
        leak: float = 1.0,
        refrac_time: int = 30,
    ) -> None:
        self.threshold = threshold
        self.rest = rest
        self.min_pot = min_pot
        self.leak = leak
        self.refrac_time = refrac_time

        self._potential: float = rest
        self._refrac_counter: int = 0

    def step(self, input_current: float, dt: float) -> bool:
        """Advance one timestep.

        ``input_current`` is treated as the dot-product of weights and
        incoming spikes (already computed by the caller).
        """
        if self._refrac_counter > 0:
            self._refrac_counter -= 1
            return False

        self._potential += input_current
        if self._potential > self.rest:
            self._potential -= self.leak

        if self._potential >= self.threshold:
            self._potential = self.rest
            self._refrac_counter = self.refrac_time
            return True
        return False

    def reset(self) -> None:
        self._potential = self.rest
        self._refrac_counter = 0

    def get_potential(self) -> float:
        return self._potential

    def inhibit(self) -> None:
        """Set potential to ``min_pot`` (lateral inhibition)."""
        self._potential = self.min_pot
