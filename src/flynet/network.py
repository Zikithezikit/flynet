"""Central simulation engine tying neurons, synapses, and connectome wiring."""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse as sp

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SETTLE = 12            # settling iterations of the network
CAL_PROBE_OFF = 0.0    # calibration probe: channel closed
CAL_PROBE_ON = 1.0     # calibration probe: channel fully on


class SpikingNetwork:
    """A spiking neural network with real connectome wiring.

    Neurons are identified by body IDs. Synapses are stored as a sparse
    weight matrix. The network settles via recurrent dynamics and
    supports calibration of sensor-to-motor responses.
    """

    def __init__(
        self,
        ids: list[int],
        edges: list[tuple[int, int, float]],
        sensor_ids: list[int] | None = None,
        motor_ids: list[int] | None = None,
    ) -> None:
        """Build a spiking network from connectome data.

        Args:
            ids (list[int]): Neuron body IDs participating in the network.
            edges (list[tuple[int, int, float]]): ``(pre_body, post_body, weight)`` synapses.
            sensor_ids (list[int] | None): IDs of sensory input neurons (auto-detected if
                ``None``).
            motor_ids (list[int] | None): IDs of motor output neurons (auto-detected if ``None``).
        """
        self.ids: list[int] = list(ids)
        self._ix: dict[int, int] = {body: i for i, body in enumerate(self.ids)}

        self.W: sp.csr_matrix
        self.W_raw: sp.csr_matrix
        self.W, self.W_raw = self._build_weight_matrix(edges)

        # COO views used by presynaptic / postsynaptic queries
        self._coo = self.W.tocoo()
        self.rows = self._coo.row
        self.cols = self._coo.col
        self.wts = self._coo.data
        self._raw_coo = self.W_raw.tocoo()

        self.motors: list[int] = motor_ids if motor_ids is not None else self._pick_motors()
        self.sensors: list[int] = sensor_ids if sensor_ids is not None else self._pick_sensors()

        self._calib_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Activity after the most recent settle() call
        self._last_activity: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_weight_matrix(
        self, edges: list[tuple[int, int, float]]
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Turn ``(pre, post, weight)`` edges into sparse matrices.

        Builds a row-normalized dynamics matrix ``W`` (each row sums to 1)
        and a raw connectome matrix ``W_raw``.  Both are stored as scipy
        CSR matrices.

        Args:
            edges (list[tuple[int, int, float]]): The synapse triples.

        Returns:
            tuple[sp.csr_matrix, sp.csr_matrix]: ``(W, W_raw)`` – the normalized and raw matrices.
        """
        n = len(self.ids)
        r = np.fromiter(
            (self._ix[s] for s, _, _ in edges), dtype=np.int64, count=len(edges)
        )
        c = np.fromiter(
            (self._ix[t] for _, t, _ in edges), dtype=np.int64, count=len(edges)
        )
        wt = np.fromiter(
            (w for _, _, w in edges), dtype=np.float64, count=len(edges)
        )

        # Row-normalize: divide each row by its total outgoing weight.
        rowsum = np.bincount(r, weights=wt, minlength=n)
        # Guard against zero rows (would produce NaN; leave as zero).
        norm = np.where(rowsum[r] > 0, wt / rowsum[r], 0.0)

        W = sp.csr_matrix((norm, (r, c)), shape=(n, n))
        W.sum_duplicates()
        W.eliminate_zeros()

        W_raw = sp.csr_matrix((wt, (r, c)), shape=(n, n))
        W_raw.sum_duplicates()
        W_raw.eliminate_zeros()

        return W, W_raw

    # ------------------------------------------------------------------
    # Role auto-detection
    # ------------------------------------------------------------------

    def _pick_motors(self) -> list[int]:
        """Auto-detect motor neurons (highest total input weight).

        Motors are the two neurons whose column in ``W_raw`` has the
        largest sum -- i.e. they receive the most total synaptic drive.

        Returns:
            list[int]: Body IDs of the detected motor neurons.
        """
        ind = np.asarray(self.W_raw.sum(axis=0)).ravel()
        order = np.argsort(ind)[::-1]
        return [self.ids[i] for i in order[:min(2, len(self.ids))]]

    def _pick_sensors(self) -> list[int]:
        """Auto-detect sensor neurons from wiring.

        For each motor, the sensor is its strongest presynaptic partner
        (highest raw weight into the motor).  Falls back to neurons with
        the largest total *outgoing* weight if no presynaptic partners
        exist.

        Returns:
            list[int]: Body IDs of the detected sensor neurons.
        """
        sensors: list[int] = []
        for m in self.motors:
            pres = self.presynaptic(m, raw=True)
            if pres:
                sensors.append(max(pres, key=lambda pr: pr[1])[0])
            else:
                sensors.append(None)  # type: ignore[arg-type]

        # Replace any None entries with fallback neurons.
        for k, b in enumerate(sensors):
            if b is None:
                ind = np.asarray(self.W_raw.sum(axis=1)).ravel()
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
            ind = np.asarray(self.W_raw.sum(axis=1)).ravel()
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
    # Settling dynamics
    # ------------------------------------------------------------------

    def settle(
        self,
        z_right: float,
        z_left: float,
        iterations: int = SETTLE,
        activation: str = "tanh",
    ) -> list[np.ndarray]:
        """Relax the network to a fixed point under sensory injection.

        Performs ``a <- activation(W.T @ a + z)`` for *iterations* steps,
        injecting ``z_right`` into the right sensor and ``z_left`` into
        the left sensor.

        Args:
            z_right (float): Injection into the right sensor channel.
            z_left (float): Injection into the left sensor channel.
            iterations (int): Number of settling iterations.
            activation (str): Nonlinearity to apply (``"tanh"`` or ``"relu"``).

        Returns:
            list[np.ndarray]: List of activity vectors (oldest first).

        Raises:
            ValueError: When *activation* is not ``"tanh"`` or ``"relu"``.
        """
        n = len(self.ids)
        z = np.zeros(n)
        sL, sR = self.sensors
        z[self._ix[sL]] = z_left
        z[self._ix[sR]] = z_right

        if activation == "tanh":
            act_fn = np.tanh
        elif activation == "relu":
            act_fn = lambda x: np.maximum(0.0, x)
        else:
            raise ValueError(f"unknown activation: {activation!r}")

        a = np.zeros(n)
        trace: list[np.ndarray] = []
        for _ in range(iterations):
            a = act_fn(self.W.T @ a + z)
            trace.append(a.copy())

        self._last_activity = trace[-1].copy() if trace else a.copy()
        return trace

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _readout(
        self, W: sp.csr_matrix, zr: float, zl: float
    ) -> tuple[float, float]:
        """Settle and return the two motor activities.

        Args:
            W (sp.csr_matrix): Weight matrix to use for settling.
            zr (float): Right sensor injection value.
            zl (float): Left sensor injection value.

        Returns:
            tuple[float, float]: ``(motor0, motor1)`` activity values.
        """
        a = self.settle(zr, zl)[-1]
        m = [self._ix[b] for b in self.motors[:2]]
        if len(m) < 2:
            return float(a[m[0]]), 0.0
        return float(a[m[0]]), float(a[m[1]])

    def calibrate(
        self, W: sp.csr_matrix | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the sensor-to-motor response matrix.

        Probes each sensor independently and reads motor output after
        settling.

        Args:
            W (sp.csr_matrix | None): Weight matrix to use (default: ``self.W``).

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(M, pinv(M))`` where ``M`` is a 2×2 response matrix mapping
            sensor injections to motor differences.
        """
        W = self.W if W is None else W
        cached = self._calib_cache.get(id(W))
        if cached is not None:
            return cached

        base = np.array(self._readout(W, CAL_PROBE_OFF, CAL_PROBE_OFF))
        m1 = np.array(self._readout(W, CAL_PROBE_ON, CAL_PROBE_OFF)) - base
        m2 = np.array(self._readout(W, CAL_PROBE_OFF, CAL_PROBE_ON)) - base
        M = np.column_stack([m1, m2])

        res = (M, np.linalg.pinv(M))
        if len(self._calib_cache) > 12:
            self._calib_cache.clear()
        self._calib_cache[id(W)] = res
        return res

    # ------------------------------------------------------------------
    # Turn
    # ------------------------------------------------------------------

    def turn(
        self,
        rho_right: float,
        rho_left: float,
        W: sp.csr_matrix | None = None,
    ) -> tuple[float, list[np.ndarray]]:
        """Compute steering command for a smell pair.

        Uses calibration to map desired readings to injections, settles
        the network, and returns the motor difference.

        Args:
            rho_right (float): Desired smell reading on the right antenna.
            rho_left (float): Desired smell reading on the left antenna.
            W (sp.csr_matrix | None): Weight matrix to use (default: ``self.W``).

        Returns:
            tuple[float, list[np.ndarray]]: ``(drv, trace)`` – the motor difference and the full
            settling trace.
        """
        W = self.W if W is None else W
        _, Minv = self.calibrate(W=W)
        z = Minv @ np.array([rho_right, rho_left])
        tr = self.settle(z[0], z[1])
        m = [self._ix[b] for b in self.motors[:2]]
        if len(m) < 2:
            drv = float(tr[-1][m[0]])
        else:
            drv = float(tr[-1][m[0]] - tr[-1][m[1]])
        return drv, tr

    # ------------------------------------------------------------------
    # Weight accessors
    # ------------------------------------------------------------------

    def get_weight(self, pre: int, post: int) -> float:
        """Get synapse weight (normalized).

        Args:
            pre (int): Presynaptic neuron body ID.
            post (int): Postsynaptic neuron body ID.

        Returns:
            float: The normalized weight, or ``0.0`` if no connection exists.
        """
        i, j = self._ix[pre], self._ix[post]
        mask = (self.rows == i) & (self.cols == j)
        if not mask.any():
            return 0.0
        return float(self.wts[np.flatnonzero(mask)[0]])

    def set_weight(self, pre: int, post: int, w: float) -> None:
        """Set synapse weight (normalized).

        Args:
            pre (int): Presynaptic neuron body ID.
            post (int): Postsynaptic neuron body ID.
            w (float): New normalized weight value.

        Returns:
            None
        """
        i, j = self._ix[pre], self._ix[post]
        self.W[i, j] = w

    # ------------------------------------------------------------------
    # Wiring queries
    # ------------------------------------------------------------------

    def presynaptic(
        self, body: int, raw: bool = False
    ) -> list[tuple[int, float]]:
        """List partners projecting INTO *body*.

        Args:
            body (int): Postsynaptic body ID.
            raw (bool): Use raw connectome weights instead of normalized.

        Returns:
            list[tuple[int, float]]: ``[(partner_id, weight), ...]`` for every non-zero input.
        """
        j = self._ix[body]
        coo = self._raw_coo if raw else self._coo
        data = coo.data
        mask = coo.col == j
        return [(self.ids[i], float(w)) for i, w in zip(coo.row[mask], data[mask])]

    def postsynaptic(
        self, body: int, raw: bool = False
    ) -> list[tuple[int, float]]:
        """List partners *body* projects INTO.

        Args:
            body (int): Presynaptic body ID.
            raw (bool): Use raw connectome weights instead of normalized.

        Returns:
            list[tuple[int, float]]: ``[(partner_id, weight), ...]`` for every non-zero output.
        """
        i = self._ix[body]
        coo = self._raw_coo if raw else self._coo
        data = coo.data
        mask = coo.row == i
        return [(self.ids[j], float(w)) for j, w in zip(coo.col[mask], data[mask])]

    # ------------------------------------------------------------------
    # Activity queries
    # ------------------------------------------------------------------

    def get_activity(self, threshold: float = 0.001) -> list[int]:
        """Return IDs of neurons above *threshold* after last settle.

        Args:
            threshold (float): Minimum absolute activity to include.

        Returns:
            list[int]: List of body IDs of active neurons.
        """
        if self._last_activity is None:
            return []
        a = self._last_activity
        active = np.flatnonzero(np.abs(a) >= threshold)
        return [self.ids[i] for i in active]

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> SpikingNetwork:
        """Return a deep copy of the network.

        Returns:
            SpikingNetwork: A fully independent copy of this ``SpikingNetwork``.
        """
        return copy.deepcopy(self)
