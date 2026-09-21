"""Lateral inhibition and dynamic threshold management for SNN layers.

Provides winner-takes-all and soft inhibition modes, plus adaptive
threshold computation based on spike train statistics.
"""

from __future__ import annotations

import numpy as np


class LateralInhibition:
    """Lateral inhibition module supporting winner-takes-all and soft modes."""

    def __init__(self, mode: str = "wta", strength: float | None = None) -> None:
        """Initialize lateral inhibition.

        Args:
            mode (str): "wta" for winner-takes-all, "soft" for soft inhibition
            strength (float | None): potential reduction for losing neurons in soft mode
        """
        if mode not in ("wta", "soft"):
            raise ValueError(f"mode must be 'wta' or 'soft', got '{mode}'")
        self.mode = mode
        self.strength = strength if strength is not None else (0.5 if mode == "soft" else 0.0)
        self.Pmin = -500.0

    def apply(self, potentials: np.ndarray, fired: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Apply inhibition and return mask of neurons allowed to fire.

        Args:
            potentials (np.ndarray): membrane potentials, shape (n_neurons,)
            fired (np.ndarray): boolean array of which neurons spiked this step, shape (n_neurons,)
            threshold (float | None): firing threshold; neurons below this cannot fire

        Returns:
            np.ndarray: Boolean mask of neurons allowed to fire after inhibition.
        """
        mask = np.zeros_like(fired, dtype=bool)

        if self.mode == "wta":
            candidates = np.where(fired)[0]
            if len(candidates) == 0:
                return mask
            potentials_candidates = potentials[candidates]
            if threshold is not None:
                above = potentials_candidates >= threshold
                if not np.any(above):
                    return mask
                candidates = candidates[above]
                potentials_candidates = potentials_candidates[above]
            winner_idx = candidates[np.argmax(potentials_candidates)]
            mask[winner_idx] = True

        elif self.mode == "soft":
            candidates = np.where(fired)[0]
            if len(candidates) == 0:
                return mask
            if threshold is not None:
                above = potentials[candidates] >= threshold
                candidates = candidates[above]
            if len(candidates) == 0:
                return mask
            mask[candidates] = True
            winner_idx = candidates[np.argmax(potentials[candidates])]
            losers = candidates[candidates != winner_idx]
            potentials[losers] -= self.strength

        return mask

    def inhibit_losers(self, potentials: np.ndarray, winner_idx: int) -> np.ndarray:
        """Suppress all neurons except the winner.

        Args:
            potentials (np.ndarray): membrane potentials, shape (n_neurons,)
            winner_idx (int): index of the winning neuron

        Returns:
            np.ndarray: Modified potentials array (in-place).
        """
        potentials[:] = self.Pmin
        potentials[winner_idx] = 0.0
        return potentials


class ThresholdManager:
    """Dynamic threshold manager based on activity statistics.

    Supports either a fixed threshold or a dynamically computed threshold
    derived from the maximum number of simultaneous spikes in a spike train.
    """

    def __init__(self) -> None:
        """Initialize the threshold manager with no fixed threshold set."""
        self._fixed_threshold: float | None = None

    def compute(self, spike_train: np.ndarray) -> float:
        """Compute dynamic threshold from a spike train.

        Args:
            spike_train (np.ndarray): binary spike train, shape (n_neurons, t_steps)

        Returns:
            float: Threshold = max simultaneous spikes / 3.
        """
        simultaneous = np.sum(spike_train, axis=0)
        return float(np.max(simultaneous)) / 3.0

    def set_fixed(self, threshold: float) -> None:
        """Set a fixed threshold value.

        Args:
            threshold (float): fixed threshold to use
        """
        self._fixed_threshold = threshold

    def get_threshold(self, spike_train: np.ndarray | None = None) -> float:
        """Get current threshold (fixed if set, otherwise dynamic).

        Args:
            spike_train (np.ndarray | None): used for dynamic computation if no fixed threshold is set

        Returns:
            float: Current threshold value.
        """
        if self._fixed_threshold is not None:
            return self._fixed_threshold
        if spike_train is None:
            raise ValueError("spike_train required when no fixed threshold is set")
        return self.compute(spike_train)
