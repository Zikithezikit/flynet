"""Synaptic weight matrix abstractions for spiking neural networks.

Provides sparse and dense representations of synaptic connectivity
used for simulation and STDP-based training.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["SynapseMatrix", "SynapseList"]


class SynapseMatrix:
    """Sparse synaptic weight matrix backed by scipy CSR.

    Construct with either explicit dimensions (random init) or an
    edge list ``[(pre, post, weight), ...]``.

    Attributes:
        rows: Row (pre-synaptic) indices of stored non-zero entries.
        cols: Column (post-synaptic) indices of stored non-zero entries.
        data: Weight values of stored non-zero entries.
        shape: (n_rows, n_cols) dimensions of the matrix.
        nnz: Number of stored non-zero elements.
    """

    def __init__(
        self,
        n_rows: int | None = None,
        n_cols: int | None = None,
        edges: list[tuple[int, int, float]] | None = None,
    ) -> None:
        """Initialise the synaptic weight matrix.

        Args:
            n_rows (int | None): Number of pre-synaptic neurons (random init mode).
            n_cols (int | None): Number of post-synaptic neurons (random init mode).
            edges (list[tuple[int, int, float]] | None): List of (pre, post, weight)
                tuples.  When provided, ``n_rows`` and ``n_cols`` are ignored.

        Raises:
            ValueError: If neither ``(n_rows, n_cols)`` nor ``edges``
                are supplied.
        """
        if edges is not None:
            self._build_from_edges(edges)
        elif n_rows is not None and n_cols is not None:
            self._build_random(n_rows, n_cols)
        else:
            raise ValueError("Provide either (n_rows, n_cols) or edges")

    def _build_random(self, n_rows: int, n_cols: int) -> None:
        """Build a random dense weight matrix converted to CSR.

        Args:
            n_rows (int): Number of rows (pre-synaptic neurons).
            n_cols (int): Number of columns (post-synaptic neurons).
        """
        dense = np.random.rand(n_rows, n_cols).astype(np.float32)
        self._csr = sparse.csr_matrix(dense)
        self._edge_map: dict[tuple[int, int], int] | None = None

    def _build_from_edges(self, edges: list[tuple[int, int, float]]) -> None:
        """Build the matrix from an explicit edge list.

        Args:
            edges (list[tuple[int, int, float]]): List of (pre, post, weight)
                tuples.
        """
        if not edges:
            self._csr = sparse.csr_matrix((0, 0), dtype=np.float32)
            self._edge_map = {}
            return

        pres = [e[0] for e in edges]
        posts = [e[1] for e in edges]
        weights = np.array([e[2] for e in edges], dtype=np.float32)

        n_rows = max(pres) + 1
        n_cols = max(posts) + 1

        self._edge_map = {}
        for idx, (pre, post) in enumerate(zip(pres, posts)):
            self._edge_map[(pre, post)] = idx

        self._csr = sparse.csr_matrix(
            (weights, (pres, posts)),
            shape=(n_rows, n_cols),
            dtype=np.float32,
        )

    def get_weight(self, pre: int, post: int) -> float:
        """Return the synaptic weight for a single connection.

        Args:
            pre (int): Pre-synaptic neuron index.
            post (int): Post-synaptic neuron index.

        Returns:
            float: The weight of the synapse, or 0.0 if not stored.
        """
        if self._edge_map is not None:
            idx = self._edge_map.get((pre, post))
            if idx is None:
                return 0.0
            return float(self._csr.data[idx])
        return float(self._csr[pre, post])

    def set_weight(self, pre: int, post: int, w: float) -> None:
        """Set the synaptic weight for a single connection.

        If the synapse does not yet exist it is inserted into the
        sparse structure.

        Args:
            pre (int): Pre-synaptic neuron index.
            post (int): Post-synaptic neuron index.
            w (float): New weight value.
        """
        if self._edge_map is not None:
            idx = self._edge_map.get((pre, post))
            if idx is not None:
                self._csr.data[idx] = w
                return
            lil = self._csr.tolil()
            lil[pre, post] = w
            self._csr = lil.tocsr()
            self._edge_map[(pre, post)] = self._csr.nnz - 1
            return
        self._csr[pre, post] = w

    def row_normalize(self) -> SynapseMatrix:
        """Return a new matrix with each row L1-normalised to sum to 1.

        Returns:
            SynapseMatrix: A new ``SynapseMatrix`` with normalised rows.
        """
        row_sums = np.asarray(np.abs(self._csr).sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        diag = sparse.diags(1.0 / row_sums)
        new = SynapseMatrix.__new__(SynapseMatrix)
        new._csr = diag @ self._csr
        new._edge_map = None
        return new

    def to_coo(self) -> sparse.coo_matrix:
        """Convert the internal CSR matrix to COO format.

        Returns:
            sparse.coo_matrix: A ``scipy.sparse.coo_matrix`` view of the data.
        """
        return self._csr.tocoo()

    @property
    def rows(self) -> np.ndarray:
        """Row (pre-synaptic) indices of non-zero entries.

        Returns:
            np.ndarray: 1-D NumPy array of row indices.
        """
        return self.to_coo().row

    @property
    def cols(self) -> np.ndarray:
        """Column (post-synaptic) indices of non-zero entries.

        Returns:
            np.ndarray: 1-D NumPy array of column indices.
        """
        return self.to_coo().col

    @property
    def data(self) -> np.ndarray:
        """Weight values of non-zero entries.

        Returns:
            np.ndarray: 1-D NumPy array of synaptic weights.
        """
        return self.to_coo().data

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions of the matrix (n_rows, n_cols).

        Returns:
            tuple[int, int]: Tuple of (rows, columns).
        """
        return self._csr.shape

    @property
    def nnz(self) -> int:
        """Number of stored non-zero elements.

        Returns:
            int: Count of non-zero synapses.
        """
        return self._csr.nnz

    def copy(self) -> SynapseMatrix:
        """Return a deep copy of this matrix.

        Returns:
            SynapseMatrix: A new ``SynapseMatrix`` with identical data.
        """
        new = SynapseMatrix.__new__(SynapseMatrix)
        new._csr = self._csr.copy()
        new._edge_map = (
            dict(self._edge_map) if self._edge_map is not None else None
        )
        return new

    def apply_delta(self, dW: np.ndarray) -> None:
        """Add *dW* values to the stored synapse data array in-place.

        Args:
            dW (np.ndarray): Array of weight deltas, flattened.  Only the first
                ``nnz`` elements are applied.
        """
        self._csr.data += dW.ravel()[: self._csr.data.size]

    def __repr__(self) -> str:
        """Return a concise string representation.

        Returns:
            str: String of the form ``SynapseMatrix(shape=..., nnz=...)``.
        """
        return f"SynapseMatrix(shape={self.shape}, nnz={self.nnz})"


class SynapseList:
    """Dense 2-D weight array for explicit per-synapse updates (STDP training).

    Stores a weight matrix of shape ``(n_post, n_pre)`` and provides
    row-oriented access for efficient STDP learning.

    Attributes:
        shape: Dimensions of the weight matrix (n_post, n_pre).
    """

    def __init__(self, n_post: int, n_pre: int) -> None:
        """Initialise an empty dense synapse list.

        Args:
            n_post (int): Number of post-synaptic neurons.
            n_pre (int): Number of pre-synaptic neurons.
        """
        self._weights = np.zeros((n_post, n_pre), dtype=np.float64)

    def get_row(self, i: int) -> np.ndarray:
        """Return the weight vector for post-synaptic neuron *i*.

        Args:
            i (int): Post-synaptic neuron index.

        Returns:
            np.ndarray: 1-D NumPy array of weights from all pre-synaptic
                neurons to neuron *i*.
        """
        return self._weights[i]

    def set_weight(self, i: int, j: int, w: float) -> None:
        """Set weight for synapse from pre-synaptic *j* to post-synaptic *i*.

        Args:
            i (int): Post-synaptic neuron index.
            j (int): Pre-synaptic neuron index.
            w (float): New weight value.
        """
        self._weights[i, j] = w

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions of the weight matrix (n_post, n_pre).

        Returns:
            tuple[int, int]: Tuple of (n_post, n_pre).
        """
        return self._weights.shape

    def __repr__(self) -> str:
        """Return a concise string representation.

        Returns:
            str: String of the form ``SynapseList(shape=...)``.
        """
        return f"SynapseList(shape={self.shape})"
