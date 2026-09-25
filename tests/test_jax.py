"""Regression tests for the JAX/gradient stack (P0-5, P1-6)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")

from flynet.connectome import ConnectomeLoader  # noqa: E402
from flynet.jax_network import JaxNetwork  # noqa: E402
from flynet.jax_trainer import GradientTrainer  # noqa: E402
from flynet.network import SpikingNetwork  # noqa: E402


def make_trainer(seed=0, lr=1e-3):
    ids, edges, motors = ConnectomeLoader.synthetic_brain(
        max_edges=40, seed=seed
    )
    net = SpikingNetwork(ids, edges, motor_ids=motors)
    jax_net = JaxNetwork.from_spiking_network(net)
    return GradientTrainer(jax_net, lr=lr, steps_per_turn=12)


class TestJaxNetworkInit:
    """P1-6: normalised weight initialisation."""

    def test_weights_normalised_by_outgoing_strength(self):
        ids, edges, motors = ConnectomeLoader.synthetic_brain(max_edges=40)
        net = SpikingNetwork(ids, edges, motor_ids=motors)
        jax_net = JaxNetwork.from_spiking_network(net)
        W = np.asarray(jax_net._W_data)
        cols = np.asarray(jax_net._sparsity_indices[:, 1])
        colsum = np.bincount(cols, weights=W, minlength=jax_net.n)
        present = np.unique(cols)
        assert W.shape[0] == len(edges) or W.shape[0] > 0
        assert np.allclose(colsum[present], 1.0, atol=1e-5), (
            f"per-pre outgoing sums not 1: {colsum[present].min()}.."
            f"{colsum[present].max()}"
        )

    def test_motor_output_finite_and_nonzero(self):
        """Degenerate (saturated tanh / all-zero) outputs must not occur."""
        ids, edges, motors = ConnectomeLoader.synthetic_brain(max_edges=40)
        net = SpikingNetwork(ids, edges, motor_ids=motors)
        jax_net = JaxNetwork.from_spiking_network(net)
        outputs = [
            float(jax_net.forward(r, l)[0])
            for r, l in [(1.0, 1.0), (1.5, 0.8), (0.5, 1.7), (2.0, 0.2)]
        ]
        assert all(np.isfinite(o) for o in outputs)
        assert max(abs(o) for o in outputs) > 1e-4, (
            f"all motor outputs degenerate: {outputs}"
        )

    def test_forward_sequence_and_gradient_work(self):
        import jax.numpy as jnp
        ids, edges, motors = ConnectomeLoader.synthetic_brain(max_edges=40)
        net = SpikingNetwork(ids, edges, motor_ids=motors)
        jax_net = JaxNetwork.from_spiking_network(net)
        inp = jnp.array([[1.0, 1.0], [1.5, 0.8], [0.5, 1.7]],
                        dtype=jnp.float32)
        motor_diffs, states = jax_net.forward_sequence(inp)
        assert np.all(np.isfinite(np.asarray(motor_diffs)))
        grads = jax_net.gradient(
            inp, jnp.array([0.1, 0.05, -0.05], dtype=jnp.float32)
        )
        assert set(grads) == {"W", "alpha", "b"}
        assert np.all(np.isfinite(grads["W"]))


class TestGradientEvaluate:
    """P0-5: evaluate must reflect the student's own weights."""

    def test_evaluate_is_weight_sensitive(self):
        trainer = make_trainer()
        base = trainer.evaluate(episodes=8, max_steps=80, base_seed=7)

        rng = np.random.default_rng(0)
        scrambled = {
            k: (rng.standard_normal(np.asarray(v).shape).astype("float32") * 5
                if k == "W" else np.asarray(v))
            for k, v in trainer._params.items()
        }
        trainer._params = scrambled
        trainer.network.set_params(scrambled)
        bad = trainer.evaluate(episodes=8, max_steps=80, base_seed=7)

        assert (
            bad["success_rate"] != base["success_rate"]
            or bad["avg_steps_to_food"] != base["avg_steps_to_food"]
        ), f"evaluate insensitive to weights: {base} vs {bad}"

    def test_gradient_training_reduces_loss(self):
        """P1-6: the gradient trainer's own loss must decrease under
        its own optimiser (fixed label batch -> deterministic)."""
        import jax

        trainer = make_trainer()

        # Fixed supervision batch from the expert teacher.
        food = np.array([80.0, 80.0])
        start = np.array([15.0, 20.0])
        labels = []
        for seed in range(30, 40):
            rng = np.random.default_rng(seed)
            pos = start.copy()
            heading = float(rng.uniform(-np.pi, np.pi))
            for _ in range(120):
                if float(np.linalg.norm(pos - food)) < 4.0:
                    break
                new_pos, new_h, rr, rl, steer, fwd = trainer.teacher.step(
                    pos, heading, food, rng, 100.0
                )
                if fwd:
                    labels.append((rr, rl, steer))
                pos, heading = new_pos, new_h
        lab = np.asarray(labels, dtype=np.float32)
        assert lab.shape[0] > 100

        rhoR = jax.numpy.asarray(lab[:, 0])
        rhoL = jax.numpy.asarray(lab[:, 1])
        steer = jax.numpy.asarray(lab[:, 2])

        def loss_fn(params):
            fwd = trainer._make_forward_fn(params)
            preds = jax.vmap(fwd)(rhoR, rhoL)
            return jax.numpy.mean((preds - steer) ** 2)

        initial = float(loss_fn(trainer._params))
        assert np.isfinite(initial)

        opt_state = trainer._optimizer.init(trainer._params)
        params = trainer._params
        for _ in range(80):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = trainer._optimizer.update(
                grads, opt_state, params,
            )
            params = optax.apply_updates(params, updates)

        final = float(loss_fn(params))
        assert np.isfinite(final)
        assert final < initial * 0.75, (
            f"gradient training did not reduce loss: {initial} -> {final}"
        )
