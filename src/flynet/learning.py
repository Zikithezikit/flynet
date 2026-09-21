"""Training pipeline for spiking networks.

Combines reward-gated Hebbian learning (from working-brain/main.py)
and STDP training (from Spiking-Neural-Network/training/learning.py).
"""

from __future__ import annotations

import numpy as np

from flynet.neurons import SpikingNeuron
from flynet.stdp import STDPRule
from flynet.synapses import SynapseList


class RewardHebbian:
    """Reward-gated three-factor Hebbian plasticity.

    Synapses that are co-active get strengthened when reward is positive.
    Only the weights change; the connectivity structure stays fixed.
    """

    def __init__(self, eta: float = 0.6, reward_bonus: float = 6.0) -> None:
        """Initialize the Hebbian learner.

        Args:
            eta: Learning rate.
            reward_bonus: Extra reward for reaching the goal.
        """
        self.eta = eta
        self.reward_bonus = reward_bonus

    def update(self, network, events: list, W=None) -> None:
        """Apply Hebbian updates to the network weights.

        Args:
            network: SpikingNetwork instance with ``rows`` and ``cols``
                attributes (COO views of the synapse matrix).
            events: List of ``(rho_right, rho_left, activity_vector)``
                tuples from successful games.
            W: Weight matrix to update.  Default: ``network.W``.

        Returns:
            None.  Updates *W* in place.
        """
        if W is None:
            W = network.W
        rows, cols = network.rows, network.cols
        dW = np.zeros(rows.shape[0], dtype=np.float64)

        for rho_right, rho_left, act in events:
            reward = (rho_right + rho_left) + self.reward_bonus
            if reward <= 0:
                continue
            if isinstance(act, tuple):
                idx, vals = act
                a = np.zeros(len(network.ids))
                a[idx] = vals
            else:
                a = act
            dW += reward * (a[rows] * a[cols])

        if dW.max() <= 0:
            return
        W.data += self.eta * dW / dW.max()


class STDPTrainer:
    """STDP-based training loop for spiking networks.

    Runs the full training loop: encode inputs, simulate LIF neurons,
    apply lateral inhibition, detect spikes, apply STDP weight updates.
    """

    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        stdp_rule: STDPRule | None = None,
        neuron_params: dict | None = None,
    ) -> None:
        """Initialize the STDP trainer.

        Args:
            n_neurons: Number of output neurons.
            n_inputs: Number of input features.
            stdp_rule: STDPRule instance.  Default: standard parameters.
            neuron_params: Dict of SpikingNeuron parameters.
        """
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.stdp_rule = stdp_rule if stdp_rule is not None else STDPRule()

        params = neuron_params if neuron_params is not None else {}
        self._neuron_defaults = {
            "threshold": params.get("threshold", 5.0),
            "rest": params.get("rest", 0.0),
            "min_pot": params.get("min_pot", -500.0),
            "leak": params.get("leak", 1.0),
            "refrac_time": params.get("refrac_time", 30),
        }

    def train_step(
        self,
        spike_train: np.ndarray,
        synapses: SynapseList,
        threshold: float | None = None,
    ) -> dict:
        """Run one training step on a spike train.

        Args:
            spike_train: Binary array ``(n_inputs, t_steps + 1)``.
            synapses: SynapseList to update.
            threshold: Firing threshold.  Auto-computed if ``None``.

        Returns:
            Dict with ``'spikes'``, ``'potentials'``, ``'winner'`` info.
        """
        n = self.n_neurons
        m = self.n_inputs
        T = spike_train.shape[1] - 1
        neurons = [SpikingNeuron(**self._neuron_defaults) for _ in range(n)]

        if threshold is None:
            threshold = self._neuron_defaults["threshold"]

        pot_arrays = [[] for _ in range(n)]
        spike_record = np.zeros((n, T + 1), dtype=np.int8)
        active_pot = np.zeros(n)
        fired_flag = False
        winner_idx = None

        for t in range(T + 1):
            for j in range(n):
                if neurons[j]._refrac_counter <= 0:
                    current_input = float(np.dot(synapses.get_row(j), spike_train[:, t]))
                    neurons[j].step(current_input, dt=1.0)
                active_pot[j] = neurons[j].get_potential()
                pot_arrays[j].append(neurons[j].get_potential())

            if not fired_flag:
                high_pot = active_pot.max()
                if high_pot > threshold:
                    fired_flag = True
                    winner_idx = int(np.argmax(active_pot))
                    for s in range(n):
                        if s != winner_idx:
                            neurons[s].inhibit()

            for j in range(n):
                pot = neurons[j].get_potential()
                if pot >= neurons[j].threshold:
                    spike_record[j, t] = 1
                    neurons[j].reset()
                    neurons[j]._refrac_counter = neurons[j].refrac_time
                    self._apply_stdp(j, t, spike_train, synapses, T)

        for p in range(m):
            if spike_train[p].sum() == 0 and winner_idx is not None:
                old_w = synapses.get_row(winner_idx)[p]
                new_w = old_w - 0.06
                new_w = max(new_w, self.stdp_rule.w_min)
                synapses.set_weight(winner_idx, p, new_w)

        return {
            "spikes": spike_record,
            "potentials": np.array(pot_arrays),
            "winner": winner_idx,
            "threshold": threshold,
        }

    def _apply_stdp(
        self,
        neuron_idx: int,
        spike_time: int,
        spike_train: np.ndarray,
        synapses: SynapseList,
        T: int,
    ) -> None:
        """Apply STDP updates after a neuron fires.

        Checks input spikes in the time window around the spike and
        updates weights using the STDP rule.
        """
        t_back = -20
        t_fore = 20

        for h in range(self.n_inputs):
            for t1 in range(-2, t_back - 1, -1):
                t_check = spike_time + t1
                if 0 <= t_check < T + 1 and spike_train[h, t_check] == 1:
                    dt = t1
                    old_w = synapses.get_row(neuron_idx)[h]
                    new_w = self.stdp_rule.update_weight(old_w, dt)
                    synapses.set_weight(neuron_idx, h, new_w)

            for t1 in range(2, t_fore + 1, 1):
                t_check = spike_time + t1
                if 0 <= t_check < T + 1 and spike_train[h, t_check] == 1:
                    dt = t1
                    old_w = synapses.get_row(neuron_idx)[h]
                    new_w = self.stdp_rule.update_weight(old_w, dt)
                    synapses.set_weight(neuron_idx, h, new_w)

    def train_epoch(
        self,
        images: list,
        synapses: SynapseList,
        epochs: int = 12,
        t_steps: int = 200,
    ) -> list[dict]:
        """Run multiple epochs over a set of images.

        Args:
            images: List of 2D arrays (one per image).
            synapses: SynapseList to train.
            epochs: Number of full passes.
            t_steps: Number of simulation timesteps.

        Returns:
            List of result dicts, one per image per epoch.
        """
        from flynet.encoding import rate_encode

        all_results = []
        for _ in range(epochs):
            for img in images:
                potentials = img.ravel().astype(np.float64)
                spike_train = rate_encode(potentials, t_steps=t_steps)
                result = self.train_step(spike_train, synapses)
                all_results.append(result)
        return all_results


class TrainingLogger:
    """Log training progress for visualization."""

    def __init__(self) -> None:
        self.episode_rewards: list[float] = []
        self.episode_steps: list[int] = []
        self.weight_history: list = []

    def log_episode(
        self, reward: float, steps: int, W=None
    ) -> None:
        """Record one episode.

        Args:
            reward: Total reward for the episode.
            steps: Number of timesteps taken.
            W: Optional weight matrix snapshot.
        """
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        if W is not None:
            self.weight_history.append(W.copy())

    def get_learning_curve(self) -> tuple[list[int], list[float]]:
        """Return episode numbers and average steps for plotting.

        Returns:
            ``(episode_numbers, avg_steps)``.
        """
        n = len(self.episode_steps)
        episodes = list(range(1, n + 1))
        return episodes, list(self.episode_steps)

    def get_reward_curve(self) -> tuple[list[int], list[float]]:
        """Return episode numbers and rewards for plotting.

        Returns:
            ``(episode_numbers, rewards)``.
        """
        n = len(self.episode_rewards)
        episodes = list(range(1, n + 1))
        return episodes, list(self.episode_rewards)
