"""Regression tests for the training pipelines (P0-3, P0-4, P1-9)."""

import numpy as np
import pytest
from flynet import SpikingNeuron, LateralInhibition, STDPRule
from flynet.learning import STDPTrainer
from flynet.synapses import SynapseList


def filled_synapses(n_neurons, n_inputs, rng, low=0.1, high=0.6):
    syn = SynapseList(n_neurons, n_inputs)
    for i in range(n_neurons):
        for j in range(n_inputs):
            syn.set_weight(i, j, float(rng.uniform(low, high)))
    return syn


def snapshot(syn):
    """Copy all weights as a (n_post, n_pre) array."""
    return syn._weights.copy()


class TestSTDPTrainer:
    """P0-3: spike detection, WTA winner, refractory recovery."""

    def drive(self, n_steps=120, n_inputs=8, n_neurons=3, seed=1,
              neuron_params=None):
        rng = np.random.default_rng(seed)
        trainer = STDPTrainer(n_neurons=n_neurons, n_inputs=n_inputs,
                              neuron_params=neuron_params)
        syn = filled_synapses(n_neurons, n_inputs, rng)
        # Strong, sustained drive on the first half of the inputs.
        train = np.zeros((n_inputs, n_steps + 1), dtype=int)
        train[: n_inputs // 2, :] = 1
        result = trainer.train_step(train, syn)
        return trainer, syn, train, result

    def test_strong_input_yields_winner_and_spikes(self):
        _, _, _, result = self.drive()
        spikes = result["spikes"]
        assert result["winner"] is not None
        assert spikes.sum() > 0
        # Recorded spikes must belong to the winner (all other neurons
        # are suppressed by the one-shot WTA inhibition).
        losers = np.delete(np.arange(spikes.shape[0]), result["winner"])
        assert spikes[losers].sum() == 0

    def test_refractory_recovers_within_n_steps(self):
        """P0-3: after a spike the neuron must spike again later.

        The pre-fix code froze neurons in permanent refractoriness
        because the ``_refrac_counter <= 0`` guard prevented the
        counter from ever being decremented.
        """
        params = {"threshold": 5.0, "refrac_time": 10, "leak": 1.0}
        _, _, _, result = self.drive(n_steps=400, neuron_params=params)
        winner = result["winner"]
        assert winner is not None
        spike_times = np.flatnonzero(result["spikes"][winner])
        assert spike_times.size >= 2, (
            f"winner {winner} did not recover from refractoriness: "
            f"spike times {spike_times}"
        )
        # No two spikes may occur inside the refractory window.
        assert np.diff(spike_times).min() >= params["refrac_time"]

    def test_no_spike_within_refractory_window(self):
        """Spikes recorded by train_step must respect the refractory time."""
        params = {"threshold": 5.0, "refrac_time": 10, "leak": 1.0}
        _, _, _, result = self.drive(n_steps=300, n_neurons=1,
                                     neuron_params=params)
        times = np.flatnonzero(result["spikes"][0])
        if times.size > 1:
            assert np.diff(times).min() >= params["refrac_time"]


class TestClassificationStyleLoop:
    """P0-4: the classification example's inline loop must learn."""

    @staticmethod
    def run_epochs(syn, n_epochs=4, t_steps=80, seed=42):
        rng = np.random.default_rng(seed)
        n_inputs = syn._weights.shape[1]
        n_neurons = syn._weights.shape[0]
        stdp = STDPRule()
        inhibition = LateralInhibition(mode="wta")

        patterns = [
            (rng.random(n_inputs) < 0.3).astype(float) for _ in range(3)
        ]
        trains = []
        for pat in patterns:
            train = np.zeros((n_inputs, t_steps + 1), dtype=int)
            for i in range(n_inputs):
                if pat[i] > 0:
                    for t in range(t_steps + 1):
                        if rng.random() < 0.3:
                            train[i, t] = 1
            trains.append(train)

        for _ in range(n_epochs):
            for train in trains:
                neurons = [
                    SpikingNeuron(threshold=5.0, rest=0.0, min_pot=-500,
                                  leak=0.15, refrac_time=30)
                    for _ in range(n_neurons)
                ]
                potentials = np.zeros(n_neurons)
                fired = np.zeros(n_neurons, dtype=bool)

                for t in range(t_steps + 1):
                    # 1. Integrate (counting down refractory timers)
                    for j in range(n_neurons):
                        if neurons[j]._refrac_counter > 0:
                            neurons[j]._refrac_counter -= 1
                            continue
                        potentials[j] += float(
                            np.dot(syn.get_row(j), train[:, t])
                        )
                        if potentials[j] > 0:
                            potentials[j] -= 0.15

                    # 2. Detect spikes first, before inhibition
                    fired[:] = False
                    for j in range(n_neurons):
                        if (neurons[j]._refrac_counter <= 0
                                and potentials[j] >= 5.0):
                            fired[j] = True

                    # 3. WTA inhibition, then suppress non-winners
                    if fired.any():
                        mask = inhibition.apply(potentials, fired,
                                                threshold=5.0)
                        for j in range(n_neurons):
                            if not mask[j]:
                                potentials[j] = -500

                        # 4. Reset winners, start refractory, apply STDP
                        for j in np.flatnonzero(mask):
                            potentials[j] = 0
                            neurons[j]._refrac_counter = 30
                            neurons[j]._potential = 0
                            for h in range(n_inputs):
                                if train[h, t] == 1:
                                    w = syn.get_row(j)[h]
                                    # dt = t_post - t_pre: inputs fired
                                    # before this spike -> potentiation
                                    for offset in range(-2, -21, -1):
                                        t_in = t + offset
                                        if (0 <= t_in < t_steps + 1
                                                and train[h, t_in] == 1):
                                            w = stdp.update_weight(w, -offset)
                                    # inputs fired after -> depression
                                    for offset in range(2, 21):
                                        t_in = t + offset
                                        if (0 <= t_in < t_steps + 1
                                                and train[h, t_in] == 1):
                                            w = stdp.update_weight(w, offset)
                                    syn.set_weight(j, h, w)
        return syn

    def test_weights_change_over_epochs(self):
        rng = np.random.default_rng(7)
        syn = filled_synapses(4, 16, rng, low=0.0, high=0.75)
        before = snapshot(syn)

        self.run_epochs(syn)
        after = snapshot(syn)

        assert not np.allclose(before, after), (
            "classification-style loop left all weights unchanged"
        )
        # Learning must strengthen at least some synapses.
        assert (after > before + 1e-9).any(), (
            "no synapse was strengthened by the classification loop"
        )


class TestRewardHebbianDecay:
    """P1-9: eligibility decay is an explicit per-timestep operation."""

    def test_decay_only_when_called(self):
        from flynet.stdp import RewardModulatedSTDP
        rm = RewardModulatedSTDP(n_synapses=5,
                                 eligibility_trace_decay=0.5)
        rm.update_eligibility(2, dt=4)
        trace = rm.eligibility_traces[2]
        assert trace != 0
        # Repeated eligibility updates alone must not decay it.
        for _ in range(10):
            rm.update_eligibility(3, dt=-4)
        assert rm.eligibility_traces[2] == trace
        rm.decay_eligibility()
        assert rm.eligibility_traces[2] == pytest.approx(trace * 0.5)
