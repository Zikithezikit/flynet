r"""Differentiable spiking network using JAX autograd.

Implements the FLYNN-style leaky integrator dynamics (arxiv 2607.00025)
with trainable weights, biases, and leak rates on top of the real
Drosophila connectome structure.

The key update equation is:

.. math::

    h_{t+1} = \alpha \odot h_t
              + (1 - \alpha) \odot \tanh(W h_t + x_t + b)

where *W* stores the connectome connectivity in ``(post, pre)`` layout so
that ``W @ h`` directly yields the recurrent input to each neuron.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from scipy import sparse as sp

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "JAX is required for JaxNetwork. "
        "Install it with: pip install flynet[gradient]"
    ) from exc

__all__ = ["JaxNetwork"]


# -----------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------
_SETTLE_STEPS: int = 12


class JaxNetwork:
    """Differentiable spiking network using JAX autograd.

    Same connectome structure as :class:`~flynet.network.SpikingNetwork`
    but with trainable weights, biases, and leak rates.  Uses the
    FLYNN-style leaky integrator dynamics.

    The connectivity pattern (sparsity structure) is fixed from the
    connectome; only the nonzero weights, leak rates, and biases are
    learnable.

    Attributes:
        ids (list[int]): Neuron body IDs in index order.
        n (int): Number of neurons.
        motors (list[int]): Body IDs of the two motor neurons.
        sensors (list[int]): Body IDs of the two sensor neurons.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        connectome_ids: list[int],
        edges: list[tuple[int, int, float]],
        sensor_ids: list[int] | None = None,
        motor_ids: list[int] | None = None,
        leak_init: float = 0.5,
    ) -> None:
        """Build from connectome data.

        Args:
            connectome_ids (list[int]): Neuron body IDs (same as
                :class:`~flynet.network.SpikingNetwork`).
            edges (list[tuple[int, int, float]]):
                ``(pre_body, post_body, weight)`` synapse triples.
            sensor_ids (list[int] | None): Sensory input neuron body IDs.
                Auto-detected when ``None``.
            motor_ids (list[int] | None): Motor output neuron body IDs.
                Auto-detected when ``None``.
            leak_init (float): Initial leak rate for all neurons.
                Must be in ``[0, 1]``.

        Raises:
            ValueError: When *edges* is empty, *leak_init* is out of
                range, or fewer than two sensors / motors are found.
        """
        if not 0.0 <= leak_init <= 1.0:
            raise ValueError(f"leak_init must be in [0, 1], got {leak_init}")

        self.ids: list[int] = list(connectome_ids)
        self._ix: dict[int, int] = {body: i for i, body in enumerate(self.ids)}
        self.n: int = len(self.ids)

        if not edges:
            raise ValueError("edges list is empty; cannot build network")

        # Deduplicate edges – BCOO forbids duplicate indices.
        dedup: dict[tuple[int, int], float] = {}
        for pre, post, w in edges:
            dedup[(pre, post)] = dedup.get((pre, post), 0.0) + float(w)
        edges = [(p, q, v) for (p, q), v in dedup.items()]

        # scipy CSR used only for sensor / motor auto-detection.
        self._scipy_W, self._scipy_W_raw = self._build_scipy_matrix(edges)

        # Auto-detect or use provided sensor / motor IDs.
        self.motors: list[int] = (
            motor_ids if motor_ids is not None else self._pick_motors()
        )
        self.sensors: list[int] = (
            sensor_ids if sensor_ids is not None else self._pick_sensors()
        )

        if len(self.motors) < 2:
            raise ValueError(
                f"need at least 2 motor neurons, found {len(self.motors)}: "
                f"{self.motors}"
            )
        if len(self.sensors) < 2:
            raise ValueError(
                f"need at least 2 sensor neurons, found {len(self.sensors)}: "
                f"{self.sensors}"
            )

        # JAX sparse matrix – fixed sparsity pattern, trainable values.
        # Convention: W[post, pre] so that W @ h gives recurrent input.
        rows = np.array(
            [self._ix[post] for _, post, _ in edges], dtype=np.int32
        )
        cols = np.array(
            [self._ix[pre] for pre, _, _ in edges], dtype=np.int32
        )

        self._sparsity_indices: jax.Array = jnp.stack(
            [jnp.asarray(rows), jnp.asarray(cols)], axis=1
        )
        self._sparsity_shape: tuple[int, int] = (self.n, self.n)

        # Trainable parameters – stored as numpy for easy get/set.
        # Each edge is initialised normalised by its pre-synaptic neuron's
        # total outgoing strength (the same convention as
        # SpikingNetwork.W), so W @ h stays O(1) and the tanh nonlinearity
        # is not saturated at initialisation.
        out_sum: dict[int, float] = {}
        for pre, _, w in edges:
            out_sum[pre] = out_sum.get(pre, 0.0) + float(w)
        w_data = np.array(
            [
                (float(w) / out_sum[pre]) if out_sum[pre] > 0 else 0.0
                for pre, _, w in edges
            ],
            dtype=np.float32,
        )
        # May hold either numpy arrays (outside jit) or JAX arrays
        # (while inside a grad trace – see set_params).
        self._W_data: np.ndarray | jax.Array = w_data
        self._alpha: np.ndarray | jax.Array = np.full(
            self.n, leak_init, dtype=np.float32
        )
        self._b: np.ndarray | jax.Array = np.zeros(self.n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scipy_matrix(
        self,
        edges: list[tuple[int, int, float]],
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Build scipy CSR matrices (same convention as SpikingNetwork).

        Builds a row-normalised dynamics matrix ``W`` and a raw
        connectome matrix ``W_raw``.  Both use the ``W[pre, post]``
        layout so that ``W.T @ a`` gives the recurrent input.

        Args:
            edges (list[tuple[int, int, float]]): Synapse triples.

        Returns:
            tuple[sp.csr_matrix, sp.csr_matrix]: ``(W_norm, W_raw)``.
        """
        n = self.n
        r = np.fromiter(
            (self._ix[s] for s, _, _ in edges),
            dtype=np.int64,
            count=len(edges),
        )
        c = np.fromiter(
            (self._ix[t] for _, t, _ in edges),
            dtype=np.int64,
            count=len(edges),
        )
        wt = np.fromiter(
            (w for _, _, w in edges),
            dtype=np.float64,
            count=len(edges),
        )

        rowsum = np.bincount(r, weights=wt, minlength=n)
        norm = np.where(rowsum[r] > 0, wt / rowsum[r], 0.0)

        W = sp.csr_matrix((norm, (r, c)), shape=(n, n))
        W.sum_duplicates()
        W.eliminate_zeros()

        W_raw = sp.csr_matrix((wt, (r, c)), shape=(n, n))
        W_raw.sum_duplicates()
        W_raw.eliminate_zeros()

        return W, W_raw

    # ------------------------------------------------------------------
    # Role auto-detection (mirrors SpikingNetwork)
    # ------------------------------------------------------------------

    def _pick_motors(self) -> list[int]:
        """Auto-detect motor neurons (highest total input weight).

        Motors are the two neurons whose column in ``W_raw`` has the
        largest sum – i.e. they receive the most total synaptic drive.

        Returns:
            list[int]: Body IDs of the detected motor neurons.
        """
        ind = np.asarray(self._scipy_W_raw.sum(axis=0)).ravel()
        order = np.argsort(ind)[::-1]
        return [self.ids[i] for i in order[: min(2, len(self.ids))]]

    def _pick_sensors(self) -> list[int]:
        """Auto-detect sensor neurons from wiring.

        For each motor, the sensor is its strongest presynaptic partner
        (highest raw weight into the motor).  Falls back to neurons with
        the largest total *outgoing* weight when no presynaptic partners
        exist.

        Returns:
            list[int]: Body IDs of the detected sensor neurons.
        """
        sensors: list[int] = []
        for m in self.motors:
            j = self._ix[m]
            col_data = np.asarray(self._scipy_W_raw[:, j].todense()).ravel()
            pres_idx = np.flatnonzero(col_data)
            if len(pres_idx) > 0:
                best = pres_idx[np.argmax(col_data[pres_idx])]
                sensors.append(self.ids[best])
            else:
                sensors.append(None)  # type: ignore[arg-type]

        # Replace None entries with fallback neurons.
        for k, b in enumerate(sensors):
            if b is None:
                ind = np.asarray(self._scipy_W_raw.sum(axis=1)).ravel()
                order = np.argsort(ind)[::-1]
                sensors[k] = next(
                    (
                        x
                        for x in (self.ids[i] for i in order)
                        if x not in self.motors and x not in sensors
                    ),
                    None,
                )

        sensors = [s for s in sensors if s is not None]
        while len(sensors) < 2:
            ind = np.asarray(self._scipy_W_raw.sum(axis=1)).ravel()
            order = np.argsort(ind)[::-1]
            add = [
                x
                for x in (self.ids[i] for i in order)
                if x not in self.motors and x not in sensors
            ][: 2 - len(sensors)]
            if not add:
                break
            sensors += add

        return sensors[:2]

    # ------------------------------------------------------------------
    # JIT-compiled dynamics
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnums=(6,))
    def _forward_jit(
        h: jax.Array,
        x: jax.Array,
        W_data: jax.Array,
        alpha: jax.Array,
        b: jax.Array,
        indices: jax.Array,
        shape: tuple[int, int],
    ) -> jax.Array:
        """Single FLYNN-style settling step (JIT-compiled).

        Computes:
            ``h_new = alpha * h + (1 - alpha) * tanh(W @ h + x + b)``

        Args:
            h (jax.Array): Current hidden state ``(N,)``.
            x (jax.Array): Input vector ``(N,)``.
            W_data (jax.Array): Nonzero weight values ``(E,)``.
            alpha (jax.Array): Per-neuron leak rates ``(N,)``.
            b (jax.Array): Per-neuron biases ``(N,)``.
            indices (jax.Array): Sparse indices ``(E, 2)`` in
                ``[post, pre]`` layout.
            shape (tuple[int, int]): Matrix dimensions ``(N, N)``.
                Static (not traced) because BCOO requires a concrete shape.

        Returns:
            jax.Array: Updated hidden state ``(N,)``.
        """
        W = BCOO((W_data, indices), shape=shape)
        return alpha * h + (1.0 - alpha) * jnp.tanh(W @ h + x + b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        z_right: float,
        z_left: float,
        steps: int = _SETTLE_STEPS,
    ) -> tuple[float, jax.Array]:
        """Run the differentiable forward pass.

        Injects *z_right* / *z_left* into the sensor neurons, runs the
        recurrent dynamics for *steps* iterations, and returns the
        motor-difference readout.

        Args:
            z_right (float): Injection into the right sensor channel.
            z_left (float): Injection into the left sensor channel.
            steps (int): Number of settling iterations.

        Returns:
            tuple[float, jax.Array]: ``(motor_diff, final_state)`` where
                ``motor_diff`` is ``activity[motor0] - activity[motor1]``
                and ``final_state`` is the hidden state vector ``(N,)``.
        """
        x = jnp.zeros(self.n, dtype=jnp.float32)
        x = x.at[self._ix[self.sensors[0]]].set(jnp.float32(z_left))
        x = x.at[self._ix[self.sensors[1]]].set(jnp.float32(z_right))

        params = self.get_params_jax()
        h = jnp.zeros(self.n, dtype=jnp.float32)
        for _ in range(steps):
            h = self._forward_jit(
                h,
                x,
                params["W"],
                params["alpha"],
                params["b"],
                self._sparsity_indices,
                self._sparsity_shape,
            )

        m0 = h[self._ix[self.motors[0]]]
        m1 = h[self._ix[self.motors[1]]]
        return float(m0 - m1), h

    def forward_sequence(
        self,
        inputs: jax.Array,
        steps: int = _SETTLE_STEPS,
    ) -> tuple[jax.Array, jax.Array]:
        """Run forward pass over a sequence of inputs (for training).

        Each timestep is settled independently from a zero hidden state.

        Args:
            inputs (jax.Array): Shape ``(T, 2)`` with
                ``(z_right, z_left)`` per timestep.
            steps (int): Settling iterations per input.

        Returns:
            tuple[jax.Array, jax.Array]:
                ``(motor_diffs, all_states)`` with shapes
                ``(T,)`` and ``(T, N)`` respectively.
        """
        T = inputs.shape[0]
        params = self.get_params_jax()

        # Build all input vectors at once.
        x_all = jnp.zeros((T, self.n), dtype=jnp.float32)
        sL = self._ix[self.sensors[0]]
        sR = self._ix[self.sensors[1]]
        x_all = x_all.at[:, sL].set(inputs[:, 1].astype(jnp.float32))
        x_all = x_all.at[:, sR].set(inputs[:, 0].astype(jnp.float32))

        def _single(x: jax.Array) -> jax.Array:
            """Run the jit'd forward dynamics for one input row.

            Args:
                x (jax.Array): Per-timestep input vector of shape ``(n,)``.

            Returns:
                jax.Array: Final hidden state ``h`` of shape ``(n,)``
                after ``steps`` integration iterations.
            """
            h = jnp.zeros(self.n, dtype=jnp.float32)
            for _ in range(steps):
                h = self._forward_jit(
                    h,
                    x,
                    params["W"],
                    params["alpha"],
                    params["b"],
                    self._sparsity_indices,
                    self._sparsity_shape,
                )
            return h

        all_states = jax.vmap(_single)(x_all)

        m0 = all_states[:, self._ix[self.motors[0]]]
        m1 = all_states[:, self._ix[self.motors[1]]]
        motor_diffs = m0 - m1

        return motor_diffs, all_states

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, np.ndarray | jax.Array]:
        """Return all trainable parameters as a dictionary.

        Values are numpy arrays, or JAX arrays while inside a grad trace.

        Returns:
            dict[str, np.ndarray | jax.Array]: Keys ``'W'`` (nonzero
                weight values, shape ``(E,)``), ``'alpha'`` (leak rates,
                shape ``(N,)``), ``'b'`` (biases, shape ``(N,)``).
        """
        return {
            "W": self._W_data.copy(),
            "alpha": self._alpha.copy(),
            "b": self._b.copy(),
        }

    def get_params_jax(self) -> dict[str, jax.Array]:
        """Return parameters as JAX arrays (for grad computation).

        Returns:
            dict[str, jax.Array]: Same layout as :meth:`get_params`.
        """
        return {
            "W": jnp.asarray(self._W_data),
            "alpha": jnp.asarray(self._alpha),
            "b": jnp.asarray(self._b),
        }

    def set_params(self, params: dict[str, np.ndarray | jax.Array]) -> None:
        """Set parameters from a dict.

        Args:
            params (dict[str, np.ndarray | jax.Array]): Must contain keys ``'W'``,
                ``'alpha'``, ``'b'`` with matching shapes.

        Raises:
            ValueError: When any parameter shape does not match the
                expected shape.
        """
        expected = {
            "W": self._W_data.shape,
            "alpha": self._alpha.shape,
            "b": self._b.shape,
        }
        for key in ("W", "alpha", "b"):
            if key not in params:
                raise KeyError(f"missing required parameter '{key}'")
            # Use JAX-shaped to check shape without converting traced arrays
            actual = params[key].shape
            if actual != expected[key]:
                raise ValueError(
                    f"'{key}' shape mismatch: expected {expected[key]}, "
                    f"got {actual}"
                )

        # Convert to numpy only if not traced (safe outside jit/grad)
        try:
            self._W_data = np.asarray(params["W"], dtype=np.float32)
            self._alpha = np.asarray(params["alpha"], dtype=np.float32)
            self._b = np.asarray(params["b"], dtype=np.float32)
        except jax.errors.TracerArrayConversionError:
            # Inside jax.grad trace – store the JAX arrays directly
            self._W_data = params["W"]
            self._alpha = params["alpha"]
            self._b = params["b"]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_spiking_network(
        cls,
        spiking_net: Any,
        leak_init: float = 0.5,
    ) -> JaxNetwork:
        """Create a JaxNetwork from an existing SpikingNetwork.

        Copies the connectivity structure.  Initial weights are
        normalised by each pre-synaptic neuron's outgoing strength (see
        :meth:`__init__`), regardless of the source matrix's scale.

        Args:
            spiking_net: Source
                :class:`~flynet.network.SpikingNetwork` instance.
            leak_init (float): Initial leak rate for all neurons.

        Returns:
            JaxNetwork: New network with the same connectivity.
        """
        coo = spiking_net.W_raw.tocoo()
        edges = [
            (spiking_net.ids[int(r)], spiking_net.ids[int(c)], float(w))
            for r, c, w in zip(coo.row, coo.col, coo.data)
        ]

        return cls(
            connectome_ids=list(spiking_net.ids),
            edges=edges,
            sensor_ids=list(spiking_net.sensors),
            motor_ids=list(spiking_net.motors),
            leak_init=leak_init,
        )

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def gradient(
        self,
        inputs: jax.Array,
        targets: jax.Array,
        steps: int = _SETTLE_STEPS,
    ) -> dict[str, jax.Array]:
        """Compute parameter gradients of the mean-squared-error loss.

        The loss is ``MSE(motor_diffs, targets)`` where *motor_diffs*
        are produced by :meth:`forward_sequence`.

        Args:
            inputs (jax.Array): Input sequence, shape ``(T, 2)``.
            targets (jax.Array): Target motor differences, shape ``(T,)``.
            steps (int): Settling iterations per input.

        Returns:
            dict[str, jax.Array]: Gradient dict with same keys as
                :meth:`get_params`.
        """
        params = self.get_params_jax()
        loss_fn = self.make_loss_fn(self, inputs, targets, steps)
        return jax.grad(loss_fn)(params)

    @staticmethod
    def make_loss_fn(
        network: JaxNetwork,
        inputs: jax.Array,
        targets: jax.Array,
        steps: int = _SETTLE_STEPS,
    ) -> Any:
        """Build a scalar loss function for gradient computation.

        The returned callable takes a params dict (as returned by
        :meth:`get_params_jax`) and returns a scalar loss value.

        Args:
            network (JaxNetwork): The network whose structure defines
                the forward pass.
            inputs (jax.Array): Input sequence, shape ``(T, 2)``.
            targets (jax.Array): Target motor differences, shape ``(T,)``.
            steps (int): Settling iterations per input.

        Returns:
            Callable[[dict[str, jax.Array]], jax.Array]: Loss function.
        """
        n = network.n
        sL = network._ix[network.sensors[0]]
        sR = network._ix[network.sensors[1]]
        m0_ix = network._ix[network.motors[0]]
        m1_ix = network._ix[network.motors[1]]
        indices = network._sparsity_indices
        shape = network._sparsity_shape

        def loss_fn(params: dict[str, jax.Array]) -> jax.Array:
            """Compute the mean squared steering error over the episode.

            Args:
                params (dict[str, jax.Array]): Candidate network
                    parameters (``'W'``, ``'alpha'``, ``'b'``).

            Returns:
                jax.Array: Scalar mean squared error between predicted
                and target steer commands, weighted by
                ``reward_weight``.
            """
            T = inputs.shape[0]

            # Build all input vectors.
            x_all = jnp.zeros((T, n), dtype=jnp.float32)
            x_all = x_all.at[:, sL].set(inputs[:, 1].astype(jnp.float32))
            x_all = x_all.at[:, sR].set(inputs[:, 0].astype(jnp.float32))

            total = jnp.float32(0.0)
            for t in range(T):
                h = jnp.zeros(n, dtype=jnp.float32)
                for _ in range(steps):
                    h = JaxNetwork._forward_jit(
                        h,
                        x_all[t],
                        params["W"],
                        params["alpha"],
                        params["b"],
                        indices,
                        shape,
                    )
                diff = h[m0_ix] - h[m1_ix]
                total = total + (diff - targets[t]) ** 2

            return total / jnp.float32(T)

        return loss_fn

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> JaxNetwork:
        """Return an independent deep copy of this network.

        Returns:
            JaxNetwork: A fully independent copy.
        """
        import copy as _copy

        return _copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact developer representation of the network.

        Returns:
            str: Summary with neuron/edge counts and sensor/motor IDs.
        """
        return (
            f"JaxNetwork(n={self.n}, edges={self._sparsity_indices.shape[0]}, "
            f"sensors={self.sensors}, motors={self.motors})"
        )
