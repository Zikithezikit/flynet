"""Spike train encoding module.

Based on rate coding from the Spiking-Neural-Network project.
Provides functions to convert analog potential values into binary spike trains.
"""

from __future__ import annotations

import numpy as np


def rate_encode(
    potentials: np.ndarray,
    t_steps: int = 200,
    scale: float = 1.0,
) -> np.ndarray:
    """Convert analog potential values to binary spike trains via rate coding.

    Spike probability at each timestep is proportional to the potential value.
    Higher potential = higher firing rate.

    Parameters
    ----------
    potentials : np.ndarray
        2D array of shape ``(n_neurons,)`` or ``(n_neurons, spatial_dim)``
        containing analog membrane potential values.
    t_steps : int
        Number of simulation timesteps (spikes array has ``t_steps + 1`` columns).
    scale : float
        Multiplier applied to the normalised probability. Values > 1 increase
        overall firing rate.

    Returns
    -------
    np.ndarray
        Binary array of shape ``(n_total, t_steps + 1)`` where 1 = spike.
    """
    potentials = np.asarray(potentials, dtype=np.float64)
    if potentials.ndim == 1:
        potentials = potentials.reshape(-1, 1)
    if potentials.ndim != 2:
        raise ValueError(
            f"potentials must be 1D or 2D, got {potentials.ndim}D"
        )

    n_neurons, spatial = potentials.shape
    n_total = n_neurons * spatial
    flat = potentials.ravel()

    p_min = flat.min()
    p_max = flat.max()
    eps = np.finfo(np.float64).eps
    normed = (flat - p_min) / (p_max - p_min + eps) * scale
    normed = np.clip(normed, 0.0, 1.0)

    spikes = (np.random.uniform(size=(n_total, t_steps + 1)) < normed[:, np.newaxis]).astype(np.int8)
    return spikes


def poisson_encode(
    rates: np.ndarray,
    t_steps: int = 200,
) -> np.ndarray:
    """Generate Poisson spike trains from firing rates.

    Parameters
    ----------
    rates : np.ndarray
        Firing rates in Hz. Can be 1D ``(n_neurons,)`` or 2D.
    t_steps : int
        Number of simulation timesteps.

    Returns
    -------
    np.ndarray
        Binary array of shape ``(n_total, t_steps + 1)``.
    """
    rates = np.asarray(rates, dtype=np.float64)
    original_shape = rates.shape
    flat = rates.ravel()
    n_total = flat.size

    # Spike probability per bin assuming dt = 1 unit
    prob = 1.0 - np.exp(-flat / 1000.0)
    prob = np.clip(prob, 0.0, 1.0)

    spikes = (np.random.uniform(size=(n_total, t_steps + 1)) < prob[:, np.newaxis]).astype(np.int8)
    return spikes


class SpikeTrain:
    """Wrapper around a binary spike train array.

    Parameters
    ----------
    data : np.ndarray
        Binary array of shape ``(n_neurons, t_steps + 1)``.
    """

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.int8)
        if self._data.ndim != 2:
            raise ValueError(
                f"data must be 2D (n_neurons, t_steps+1), got shape {self._data.shape}"
            )

    @property
    def data(self) -> np.ndarray:
        """Underlying binary spike array."""
        return self._data

    @property
    def n_neurons(self) -> int:
        """Number of neurons."""
        return self._data.shape[0]

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps (excluding the initial state at t=0)."""
        return self._data.shape[1] - 1

    def spike_times(self, neuron_idx: int) -> list[int]:
        """Return the list of timesteps where neuron *neuron_idx* spiked.

        Parameters
        ----------
        neuron_idx : int
            Index of the neuron to query.

        Returns
        -------
        list[int]
            Sorted list of spike times.
        """
        if neuron_idx < 0 or neuron_idx >= self.n_neurons:
            raise IndexError(
                f"neuron_idx {neuron_idx} out of range [0, {self.n_neurons})"
            )
        return np.nonzero(self._data[neuron_idx])[0].tolist()

    def rates(self) -> np.ndarray:
        """Average firing rate per neuron (spikes per timestep).

        Returns
        -------
        np.ndarray
            1D array of shape ``(n_neurons,)``.
        """
        return self._data.mean(axis=1)

    def is_active(self, t: int) -> np.ndarray:
        """Boolean mask indicating which neurons fire at timestep *t*.

        Parameters
        ----------
        t : int
            Timestep index (0 to ``n_timesteps``).

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_neurons,)``.
        """
        if t < 0 or t > self.n_timesteps:
            raise IndexError(
                f"t={t} out of range [0, {self.n_timesteps}]"
            )
        return self._data[:, t].astype(bool)
