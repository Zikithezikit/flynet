import numpy as np
from math import exp


class STDPRule:
    """Spike-Timing-Dependent Plasticity learning rule."""

    def __init__(
        self,
        a_plus: float = 0.8,
        a_minus: float = 0.3,
        tau_plus: float = 8.0,
        tau_minus: float = 5.0,
        w_max: float = 1.5,
        w_min: float = -1.2,
        sigma: float = 0.1,
        scale: float = 1.0,
    ):
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min
        self.sigma = sigma
        self.scale = scale

    def curve(self, dt: int) -> float:
        """STDP curve value for a given time difference.

        Args:
            dt: post_spike_time - pre_spike_time

        Returns:
            Negative (depression) if dt > 0, positive (potentiation) if dt <= 0.
        """
        if dt > 0:
            return -self.a_plus * exp(-dt / self.tau_plus)
        else:
            return self.a_minus * exp(dt / self.tau_minus)

    def update_weight(self, w: float, dt: int) -> float:
        """Update a single weight based on STDP.

        Args:
            w: current weight
            dt: post_spike_time - pre_spike_time

        Returns:
            Updated weight clipped to [w_min, w_max].
        """
        delta = self.curve(dt)
        if delta < 0:
            w_new = w + self.sigma * delta * (w - abs(self.w_min)) * self.scale
        else:
            w_new = w + self.sigma * delta * (self.w_max - w) * self.scale
        return float(np.clip(w_new, self.w_min, self.w_max))

    def apply_spike_pair(self, w: float, pre_spike_time: int, post_spike_time: int) -> float:
        """Convenience method for a single pre-post spike pair.

        Args:
            w: current weight
            pre_spike_time: time step of pre-synaptic spike
            post_spike_time: time step of post-synaptic spike

        Returns:
            Updated weight.
        """
        dt = post_spike_time - pre_spike_time
        return self.update_weight(w, dt)


class RewardModulatedSTDP(STDPRule):
    """Reward-modulated STDP with eligibility traces."""

    def __init__(
        self,
        a_plus: float = 0.8,
        a_minus: float = 0.3,
        tau_plus: float = 8.0,
        tau_minus: float = 5.0,
        w_max: float = 1.5,
        w_min: float = -1.2,
        sigma: float = 0.1,
        scale: float = 1.0,
        a_reward: float = 1.0,
        eligibility_trace_decay: float = 0.95,
        n_synapses: int = 0,
    ):
        super().__init__(
            a_plus=a_plus,
            a_minus=a_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            w_max=w_max,
            w_min=w_min,
            sigma=sigma,
            scale=scale,
        )
        self.a_reward = a_reward
        self.eligibility_trace_decay = eligibility_trace_decay
        self.eligibility_traces = np.zeros(n_synapses, dtype=np.float64)

    def update_eligibility(self, synapse_idx: int, dt: int) -> None:
        """Accumulate STDP trace into eligibility trace for a synapse.

        Args:
            synapse_idx: index of the synapse
            dt: post_spike_time - pre_spike_time
        """
        self.eligibility_traces *= self.eligibility_trace_decay
        self.eligibility_traces[synapse_idx] += self.curve(dt)

    def apply_reward(self, reward: float, learning_rate: float = 0.6) -> np.ndarray:
        """Apply reward-modulated weight updates and reset eligibility traces.

        Args:
            reward: scalar reward signal
            learning_rate: scale of reward-driven updates

        Returns:
            Array of weight deltas for each synapse.
        """
        deltas = learning_rate * reward * self.a_reward * self.eligibility_traces
        self.eligibility_traces = np.zeros_like(self.eligibility_traces)
        return deltas
