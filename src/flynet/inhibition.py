import numpy as np


class LateralInhibition:
    """Lateral inhibition module supporting winner-takes-all and soft modes."""

    def __init__(self, mode: str = "wta", strength: float = None):
        """Initialize lateral inhibition.

        Args:
            mode: "wta" for winner-takes-all, "soft" for soft inhibition
            strength: potential reduction for losing neurons in soft mode
        """
        if mode not in ("wta", "soft"):
            raise ValueError(f"mode must be 'wta' or 'soft', got '{mode}'")
        self.mode = mode
        self.strength = strength if strength is not None else (0.5 if mode == "soft" else 0.0)
        self.Pmin = -500.0

    def apply(self, potentials: np.ndarray, fired: np.ndarray, threshold: float = None) -> np.ndarray:
        """Apply inhibition and return mask of neurons allowed to fire.

        Args:
            potentials: membrane potentials, shape (n_neurons,)
            fired: boolean array of which neurons spiked this step, shape (n_neurons,)
            threshold: firing threshold; neurons below this cannot fire

        Returns:
            Boolean mask of neurons allowed to fire after inhibition.
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
            potentials: membrane potentials, shape (n_neurons,)
            winner_idx: index of the winning neuron

        Returns:
            Modified potentials array (in-place).
        """
        potentials[:] = self.Pmin
        potentials[winner_idx] = 0.0
        return potentials


class ThresholdManager:
    """Dynamic threshold manager based on activity statistics."""

    def __init__(self):
        self._fixed_threshold: float = None

    def compute(self, spike_train: np.ndarray) -> float:
        """Compute dynamic threshold from a spike train.

        Args:
            spike_train: binary spike train, shape (n_neurons, t_steps)

        Returns:
            Threshold = max simultaneous spikes / 3.
        """
        simultaneous = np.sum(spike_train, axis=0)
        return float(np.max(simultaneous)) / 3.0

    def set_fixed(self, threshold: float) -> None:
        """Set a fixed threshold value.

        Args:
            threshold: fixed threshold to use
        """
        self._fixed_threshold = threshold

    def get_threshold(self, spike_train: np.ndarray = None) -> float:
        """Get current threshold (fixed if set, otherwise dynamic).

        Args:
            spike_train: used for dynamic computation if no fixed threshold is set

        Returns:
            Current threshold value.
        """
        if self._fixed_threshold is not None:
            return self._fixed_threshold
        if spike_train is None:
            raise ValueError("spike_train required when no fixed threshold is set")
        return self.compute(spike_train)
