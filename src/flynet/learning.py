"""Training pipeline for spiking networks.

Combines reward-gated Hebbian learning (from working-brain/main.py)
and STDP training (from Spiking-Neural-Network/training/learning.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from flynet.neurons import SpikingNeuron
from flynet.stdp import STDPRule
from flynet.synapses import SynapseList

if TYPE_CHECKING:
    from flynet.network import SpikingNetwork

__all__ = [
    "MotorRewardHebbian",
    "RewardHebbian",
    "STDPTrainer",
    "TrainingLogger",
]


class MotorRewardHebbian:
    """Reward-modulated plasticity on the motor projection.

    The steering command is the difference between the two motor neurons'
    activities, and a motor's activity is
    ``sum_pre a[pre] * W[pre, motor]``.  The synapses that can therefore
    change the command are the ones *into* the motors, and the
    first-order sensitivity of the command to each of them is proportional
    to the presynaptic activity.  This rule reinforces exactly those
    synapses, scaled by a signed reward, so a large brain learns through
    the handful of synapses that actually drive its output instead of the
    hundreds of thousands that do not.

    Attributes:
        eta (float): Maximum absolute weight change per update.
        w_max (float): Upper bound on an updated weight.
    """

    def __init__(self, eta: float = 0.05, w_max: float = 1.5) -> None:
        """Initialize the motor-projection learner.

        Args:
            eta (float): Maximum absolute weight change per update.
            w_max (float): Upper bound on an updated weight.
        """
        self.eta = eta
        self.w_max = w_max

    def update(self, network: "SpikingNetwork", events: list, W: Any = None) -> None:
        """Reinforce or weaken the synapses that project into the motors.

        Args:
            network (SpikingNetwork): Network whose ``rows``/``cols`` COO
                views and ``motors`` define the update.
            events (list): ``(rho_right, rho_left, activity, reward)``
                tuples.  *activity* is a dense vector or a sparse
                ``(indices, values)`` pair; *reward* is signed, so steps
                that moved towards the goal strengthen the active input
                synapses and steps that moved away weaken them.
            W (np.ndarray | None): Weight matrix to update.  Default:
                ``network.W``.

        Returns:
            None: Updates *W* in place.
        """
        if W is None:
            W = network.W
        rows, cols = network.rows, network.cols

        motor_cols = [network._ix[b] for b in network.motors[:2]]
        mask = np.zeros(cols.shape[0], dtype=bool)
        for m in motor_cols:
            mask |= cols == m
        if not mask.any():
            return

        pre_rows = rows[mask]
        dW = np.zeros(int(mask.sum()), dtype=np.float64)
        n = len(network.ids)

        for event in events:
            if len(event) == 4:
                _, _, act, reward = event
            else:
                _, _, act = event
                reward = 1.0
            if isinstance(act, tuple):
                idx, vals = act
                a = np.zeros(n)
                a[idx] = vals
            else:
                a = act
            dW += reward * a[pre_rows]

        peak = float(np.abs(dW).max()) if dW.size else 0.0
        if peak <= 0.0:
            return
        W.data[mask] = np.clip(
            W.data[mask] + self.eta * dW / peak, -self.w_max, self.w_max
        )


class RewardHebbian:
    """Reward-gated three-factor Hebbian plasticity.

    Synapses that are co-active get strengthened when reward is positive.
    Only the weights change; the connectivity structure stays fixed.
    """

    def __init__(self, eta: float = 0.6, reward_bonus: float = 6.0) -> None:
        """Initialize the Hebbian learner.

        Args:
            eta (float): Learning rate.
            reward_bonus (float): Extra reward for reaching the goal.
        """
        self.eta = eta
        self.reward_bonus = reward_bonus

    def update(self, network: "SpikingNetwork", events: list, W: Any = None) -> None:
        """Apply Hebbian updates to the network weights.

        Each event is ``(rho_right, rho_left, activity)``.  A fourth
        element may be supplied as an explicit signed reward, which
        overrides the default ``rho_right + rho_left + reward_bonus``.
        Negative rewards depress the co-active synapses, which turns the
        rule into a proper three-factor learner: synapses active during a
        good decision are strengthened, synapses active during a bad one
        are weakened.

        Args:
            network (SpikingNetwork): SpikingNetwork instance with ``rows``
                and ``cols`` attributes (COO views of the synapse matrix).
            events (list): List of ``(rho_right, rho_left, activity_vector)``
                tuples from successful games, optionally with a trailing
                signed reward.  *activity* is either a dense vector or a
                sparse ``(indices, values)`` pair.
            W (np.ndarray | None): Weight matrix to update.
                Default: ``network.W``.

        Returns:
            None: Updates *W* in place.
        """
        if W is None:
            W = network.W
        rows, cols = network.rows, network.cols
        dW = np.zeros(rows.shape[0], dtype=np.float64)
        n = len(network.ids)

        for event in events:
            if len(event) == 4:
                rho_right, rho_left, act, reward = event
            else:
                rho_right, rho_left, act = event
                reward = (rho_right + rho_left) + self.reward_bonus
            if isinstance(act, tuple):
                idx, vals = act
                a = np.zeros(n)
                a[idx] = vals
            else:
                a = act
            dW += reward * (a[rows] * a[cols])

        # Normalise by the largest magnitude so that an all-negative batch
        # depresses instead of being skipped.
        peak = float(np.abs(dW).max()) if dW.size else 0.0
        if peak <= 0.0:
            return
        W.data += self.eta * dW / peak


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
            n_neurons (int): Number of output neurons.
            n_inputs (int): Number of input features.
            stdp_rule (STDPRule | None): STDPRule instance.
                Default: standard parameters.
            neuron_params (dict | None): Dict of SpikingNeuron parameters.
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
            spike_train (np.ndarray): Binary array
                ``(n_inputs, t_steps + 1)``.
            synapses (SynapseList): SynapseList to update.
            threshold (float | None): Firing threshold.
                Auto-computed if ``None``.

        Returns:
            dict: Dict with ``'spikes'``, ``'potentials'``,
                ``'winner'`` info.
        """
        n = self.n_neurons
        m = self.n_inputs
        T = spike_train.shape[1] - 1
        neurons = [SpikingNeuron(**self._neuron_defaults) for _ in range(n)]

        if threshold is None:
            threshold = self._neuron_defaults["threshold"]

        pot_arrays: list[list[Any]] = [[] for _ in range(n)]
        spike_record = np.zeros((n, T + 1), dtype=np.int8)
        fired_flag = False
        winner_idx = None

        for t in range(T + 1):
            pre_pot = np.zeros(n)
            spiked = np.zeros(n, dtype=bool)
            for j in range(n):
                current_input = float(np.dot(synapses.get_row(j), spike_train[:, t]))
                pre_pot[j] = neurons[j].get_potential()
                # step() integrates, detects the spike, resets, and counts
                # down the refractory period itself.
                spiked[j] = neurons[j].step(current_input, dt=1.0)
                pot_arrays[j].append(neurons[j].get_potential())

            if not fired_flag and spiked.any():
                fired_flag = True
                candidates = np.flatnonzero(spiked)
                # Rank simultaneous spikers by the drive that crossed threshold.
                peaks = [
                    pre_pot[c]
                    + float(np.dot(synapses.get_row(int(c)), spike_train[:, t]))
                    for c in candidates
                ]
                winner_idx = int(candidates[int(np.argmax(peaks))])
                for s in range(n):
                    if s != winner_idx:
                        neurons[s].inhibit()

            for sj in np.flatnonzero(spiked):
                spike_record[sj, t] = 1
                self._apply_stdp(int(sj), t, spike_train, synapses, T)

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

        Args:
            neuron_idx (int): Index of the neuron that fired.
            spike_time (int): Timestep at which the neuron fired.
            spike_train (np.ndarray): Binary array
                ``(n_inputs, t_steps + 1)`` of input spikes.
            synapses (SynapseList): SynapseList whose weights are updated.
            T (int): Total number of simulation timesteps.

        Returns:
            None: Updates synapse weights in place.
        """
        t_back = -20
        t_fore = 20

        for h in range(self.n_inputs):
            # Input spikes BEFORE the post-synaptic spike (t1 < 0):
            # dt = t_post - t_pre = -t1 > 0 -> potentiation.
            for t1 in range(-2, t_back - 1, -1):
                t_check = spike_time + t1
                if 0 <= t_check < T + 1 and spike_train[h, t_check] == 1:
                    dt = -t1
                    old_w = synapses.get_row(neuron_idx)[h]
                    new_w = self.stdp_rule.update_weight(old_w, dt)
                    synapses.set_weight(neuron_idx, h, new_w)

            # Input spikes AFTER the post-synaptic spike (t1 > 0):
            # dt = t_post - t_pre = -t1 < 0 -> depression.
            for t1 in range(2, t_fore + 1, 1):
                t_check = spike_time + t1
                if 0 <= t_check < T + 1 and spike_train[h, t_check] == 1:
                    dt = -t1
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
            images (list): List of 2D arrays (one per image).
            synapses (SynapseList): SynapseList to train.
            epochs (int): Number of full passes.
            t_steps (int): Number of simulation timesteps.

        Returns:
            list[dict]: List of result dicts, one per image per epoch.
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
    """Log training progress for visualization.

    Attributes:
        episode_rewards: List of total rewards per episode.
        episode_steps: List of timestep counts per episode.
        weight_history: List of weight matrix snapshots per episode.
    """

    def __init__(self) -> None:
        """Initialize the training logger with empty history lists."""
        self.episode_rewards: list[float] = []
        self.episode_steps: list[int] = []
        self.weight_history: list = []

    def log_episode(
        self, reward: float, steps: int, W: np.ndarray | None = None
    ) -> None:
        """Record one episode.

        Args:
            reward (float): Total reward for the episode.
            steps (int): Number of timesteps taken.
            W (np.ndarray | None): Optional weight matrix snapshot.
        """
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        if W is not None:
            self.weight_history.append(W.copy())

    def get_learning_curve(self) -> tuple[list[int], list[float]]:
        """Return episode numbers and average steps for plotting.

        Returns:
            tuple[list[int], list[float]]: ``(episode_numbers, avg_steps)``.
        """
        n = len(self.episode_steps)
        episodes = list(range(1, n + 1))
        return episodes, list(self.episode_steps)

    def get_reward_curve(self) -> tuple[list[int], list[float]]:
        """Return episode numbers and rewards for plotting.

        Returns:
            tuple[list[int], list[float]]: ``(episode_numbers, rewards)``.
        """
        n = len(self.episode_rewards)
        episodes = list(range(1, n + 1))
        return episodes, list(self.episode_rewards)
