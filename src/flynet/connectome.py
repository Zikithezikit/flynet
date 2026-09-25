"""Load Drosophila brain connectome data from neuPrint.

Provides :class:`ConnectomeLoader` which wraps neuPrint queries with local
CSV/JSON caching and a chunked full-connectome fetch for the ~176k-neuron
male CNS dataset.

Real connectome data is mandatory: when a live fetch fails, the loader
raises :class:`ConnectomeUnavailableError` instead of quietly substituting
synthetic data.  Synthetic test data is available only through the
explicit :meth:`ConnectomeLoader.synthetic_brain` helper (CLI: ``--offline``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

__all__ = ["ConnectomeLoader", "ConnectomeUnavailableError"]

_DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".flynet", "cache")
_SYNTH_N = 41
_SYNTH_ID_BASE = 1000
_SYNTH_HUB_COUNT = 6
_SYNTH_NOISE_WEIGHT = 0.4
_SYNTH_EDGES_PER_BATCH = 8
_FULL_CHUNK = 20000
_FULL_MIN_WEIGHT = 50


class ConnectomeUnavailableError(RuntimeError):
    """Raised when real neuPrint connectome data cannot be obtained.

    flynet never silently substitutes synthetic data for the real brain.
    For synthetic test data use :meth:`ConnectomeLoader.synthetic_brain`
    explicitly (CLI: the ``--offline`` flag).

    Args:
        message (str): Human-readable explanation including how to fix it.
    """


class ConnectomeLoader:
    """Load and cache Drosophila connectome data from neuPrint.

    Wraps neuPrint client queries with local CSV/JSON caching.  Cache
    misses are fetched live from neuPrint; fetch failures raise
    :class:`ConnectomeUnavailableError` (never silent synthetic data).

    Attributes:
        server (str): Base URL of the neuPrint server.
        dataset (str): Dataset identifier (e.g. ``"male-cns:v1.0"``).
        cache_dir (str): Local directory used for CSV/JSON cache files.
        last_source (str): Provenance of the most recent load --
            ``"cache"``, ``"neuprint"``, ``"synthetic"``, or ``""``.
        DATASETS (dict[str, str]): Mapping of dataset identifiers to
            human-readable descriptions.
    """

    DATASETS: dict[str, str] = {
        "hemibrain:v1.2.1":  "adult FEMALE central brain (the classic hemibrain)",
        "male-cns:v0.9":     "male CNS draft (brain + nerve cord + optic lobes)",
        "male-cns:v1.0":     "adult MALE brain + nerve cord + optic lobes (default)",
        "manc:v1.0":         "adult MALE ventral nerve cord (MANC)",
        "manc:v1.2.1":       "adult MALE ventral nerve cord (MANC), updated",
        "manc:v1.2.3":       "adult MALE ventral nerve cord (MANC), current",
        "mushroombody":      "mushroom-body-centric subset (smallest)",
        "optic-lobe:v1.0.1": "adult FEMALE optic lobe",
        "optic-lobe:v1.1":   "adult FEMALE optic lobe (current)",
    }

    def __init__(
        self,
        server: str = "https://neuprint.janelia.org",
        dataset: str = "male-cns:v1.0",
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the ConnectomeLoader.

        Args:
            server (str): Base URL of the neuPrint server.
            dataset (str): Dataset identifier to query.
            cache_dir (str | None): Local directory for cache files. Uses
                ``~/.flynet/cache`` when ``None``.
        """
        self.server = server
        self.dataset = dataset
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._client = None
        self.last_source: str = ""

    # ------------------------------------------------------------------
    # neuPrint client (lazy, cached)
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return a cached neuPrint ``Client`` or ``None``.

        Returns:
            Any: A neuPrint Client instance, or None if unavailable.
        """
        if self._client is not None:
            return self._client
        token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "").strip()
        if not token:
            return None
        try:
            from neuprint import Client

            c = Client(self.server, self.dataset, token=token)
            c.fetch_version()
            self._client = c
            return c
        except Exception as exc:  # noqa: BLE001
            import sys

            sys.stderr.write(
                f"[connectome] neuPrint unavailable: {exc}\n"
            )
            return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _tag(self, neuron_type: str | None = None) -> str:
        """Filesystem-safe suffix derived from the dataset name.

        Args:
            neuron_type (str | None): Optional neuron cell type to include
                in the tag.

        Returns:
            str: Filesystem-safe tag string.
        """
        safe = self.dataset.replace(":", "_").replace("/", "_")
        if neuron_type:
            return f".{safe}.{neuron_type}"
        return f".{safe}"

    def _edge_csv(self, prefix: str, tag: str) -> str:
        """Build the path for the edge CSV cache file.

        Args:
            prefix (str): Filename prefix (e.g. ``"mini_brain"``).
            tag (str): Filesystem tag appended to the cache filename.

        Returns:
            str: Full path to the edge CSV file.
        """
        return os.path.join(self.cache_dir, f"{prefix}_edges{tag}.csv")

    def _meta_json(self, prefix: str, tag: str) -> str:
        """Build the path for the metadata JSON cache file.

        Args:
            prefix (str): Filename prefix (e.g. ``"mini_brain"``).
            tag (str): Filesystem tag appended to the cache filename.

        Returns:
            str: Full path to the metadata JSON file.
        """
        return os.path.join(self.cache_dir, f"{prefix}{tag}.json")

    # ------------------------------------------------------------------
    # mini-brain fetch
    # ------------------------------------------------------------------

    def fetch_mini_brain(
        self,
        neuron_type: str,
        max_edges: int = 40,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]]:
        """Fetch a small brain slice centered on a cell type.

        Args:
            neuron_type (str): Neuron cell type to center the slice on.
            max_edges (int): Maximum number of synaptic edges to return.

        Returns:
            tuple[list[int], list[tuple[int, int, float]], list[int]]:
                ``(ids, edges, motor_bodies)`` where *ids* is the
                sorted list of neuron body IDs, *edges* holds
                ``(pre_body, post_body, weight)`` triples, and *motor_bodies*
                contains candidate motor neuron IDs of the requested cell type.

        Raises:
            RuntimeError: When the neuPrint client is unavailable.
            ValueError: When no neurons of the requested type are found.
        """
        client = self._get_client()
        if client is None:
            raise RuntimeError(
                "neuPrint client unavailable – set "
                "NEUPRINT_APPLICATION_CREDENTIALS"
            )

        from neuprint import fetch_adjacencies, fetch_neurons

        import pandas as pd

        neurons, _ = fetch_neurons(neuron_type, client=client)
        if neurons is None or len(neurons) == 0:
            raise ValueError(f"no cell type '{neuron_type}' found")
        targets = sorted(int(b) for b in neurons["bodyId"].tolist())

        _, edges_in = fetch_adjacencies(
            None, neuron_type, omit_rois=True, client=client
        )
        _, edges_out = fetch_adjacencies(
            neuron_type, omit_rois=True, client=client
        )
        df = pd.concat([edges_in, edges_out], ignore_index=True)
        df = df.drop_duplicates(["bodyId_pre", "bodyId_post"])
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        df = df.head(max_edges)

        ids = sorted(
            set(int(b) for b in df["bodyId_pre"].tolist())
            | set(int(b) for b in df["bodyId_post"].tolist())
        )
        edges = [
            (int(r.bodyId_pre), int(r.bodyId_post), float(r.weight))
            for r in df.itertuples()
        ]
        return ids, edges, targets

    # ------------------------------------------------------------------
    # full connectome fetch
    # ------------------------------------------------------------------

    def fetch_full_connectome(
        self,
        min_weight: int = 50,
        edge_cap: int = 3_000_000,
        chunk_size: int = _FULL_CHUNK,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]]:
        """Fetch the whole fly connectome (strong backbone).

        Args:
            min_weight (int): Minimum synaptic weight to include an edge.
            edge_cap (int): Maximum number of edges to keep after sorting.
            chunk_size (int): Number of neurons per neuPrint query batch.

        Returns:
            tuple[list[int], list[tuple[int, int, float]], list[int]]:
                ``(ids, edges, [])`` with an empty *motor_bodies*
                list.

        Raises:
            RuntimeError: When the neuPrint client is unavailable.
            ValueError: When the neuron fetch or synapse query returns empty.
        """
        client = self._get_client()
        if client is None:
            raise RuntimeError(
                "neuPrint client unavailable – set "
                "NEUPRINT_APPLICATION_CREDENTIALS"
            )

        from neuprint import fetch_adjacencies, fetch_neurons

        import pandas as pd

        res = fetch_neurons(None, client=client, omit_rois=True)
        neurons = res[0] if isinstance(res, tuple) else res
        if neurons is None or len(neurons) == 0:
            raise ValueError("whole connectome neuron fetch came back empty")
        ids = sorted(int(b) for b in neurons["bodyId"].tolist())

        frames: list = []
        for start in range(0, len(ids), chunk_size):
            src = ids[start : start + chunk_size]
            _, df = fetch_adjacencies(
                src,
                None,
                omit_rois=True,
                min_total_weight=min_weight,
                client=client,
            )
            if df is not None and len(df):
                frames.append(df)
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )
        if df.empty:
            raise ValueError("full connectome fetch returned no synapses")
        df = df[df.bodyId_pre != df.bodyId_post]
        df = df.drop_duplicates(["bodyId_pre", "bodyId_post"])
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        df = df.head(edge_cap)

        edges = [
            (int(r.bodyId_pre), int(r.bodyId_post), float(r.weight))
            for r in df.itertuples()
        ]
        ids = sorted(
            set(int(b) for b in df["bodyId_pre"].tolist())
            | set(int(b) for b in df["bodyId_post"].tolist())
        )
        return ids, edges, []

    # ------------------------------------------------------------------
    # caching
    # ------------------------------------------------------------------

    def save_cache(
        self,
        ids: List[int],
        edges: List[Tuple[int, int, float]],
        motor_bodies: List[int] | None = None,
        tag: str = "",
    ) -> None:
        """Persist fetched mini-brain data to *cache_dir*.

        Args:
            ids (list[int]): Sorted list of neuron body IDs.
            edges (list[tuple[int, int, float]]): List of
                ``(pre_body, post_body, weight)`` triples.
            motor_bodies (list[int] | None): Candidate motor neuron IDs,
                or ``None``.
            tag (str): Filesystem tag appended to the cache filename.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        csv_path = self._edge_csv("mini_brain", tag)
        json_path = self._meta_json("mini_brain", tag)
        with open(csv_path, "w") as fh:
            fh.write("bodyId_pre,bodyId_post,weight_syn\n")
            for s, t, w in edges:
                fh.write(f"{s},{t},{w:.2f}\n")
        with open(json_path, "w") as fh:
            json.dump(
                {"ids": ids, "motor_bodies": motor_bodies or []},
                fh,
                indent=2,
            )

    def save_full_cache(
        self,
        ids: List[int],
        edges: List[Tuple[int, int, float]],
        tag: str = "",
    ) -> None:
        """Persist a full-connectome fetch to *cache_dir*.

        Args:
            ids (list[int]): Sorted list of neuron body IDs.
            edges (list[tuple[int, int, float]]): List of
                ``(pre_body, post_body, weight)`` triples.
            tag (str): Filesystem tag appended to the cache filename.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        csv_path = self._edge_csv("full_connectome", tag)
        json_path = self._meta_json("full_connectome", tag)
        with open(csv_path, "w") as fh:
            fh.write("bodyId_pre,bodyId_post,weight\n")
            for s, t, w in edges:
                fh.write(f"{s},{t},{w:.2f}\n")
        with open(json_path, "w") as fh:
            json.dump({"ids": ids}, fh, indent=2)

    def load_cache(
        self,
        tag: str = "",
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]] | None:
        """Load a previously cached mini brain.

        Args:
            tag (str): Filesystem tag appended to the cache filename.

        Returns:
            tuple[list[int], list[tuple[int, int, float]], list[int]] | None:
                ``(ids, edges, motor_bodies)`` or ``None`` when the
                cache is incomplete or missing.
        """
        import csv

        csv_path = self._edge_csv("mini_brain", tag)
        json_path = self._meta_json("mini_brain", tag)
        if not (os.path.exists(csv_path) and os.path.exists(json_path)):
            return None
        edges: list[tuple[int, int, float]] = []
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                wcol = "weight_syn" if "weight_syn" in row else "weight"
                edges.append((int(row["bodyId_pre"]), int(row["bodyId_post"]),
                              float(row[wcol])))
        with open(json_path) as fh:
            meta = json.load(fh)
        return list(meta["ids"]), edges, list(meta.get("motor_bodies", []))

    def load_full_cache(
        self,
        tag: str = "",
    ) -> Tuple[List[int], List[Tuple[int, int, float]]] | None:
        """Load a previously cached full connectome.

        Args:
            tag (str): Filesystem tag appended to the cache filename.

        Returns:
            tuple[list[int], list[tuple[int, int, float]]] | None:
                ``(ids, edges)`` or ``None`` when the cache is
                incomplete or missing.
        """
        import csv

        csv_path = self._edge_csv("full_connectome", tag)
        json_path = self._meta_json("full_connectome", tag)
        if not (os.path.exists(csv_path) and os.path.exists(json_path)):
            return None
        edges: list[tuple[int, int, float]] = []
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                edges.append((int(row["bodyId_pre"]), int(row["bodyId_post"]),
                              float(row["weight"])))
        with open(json_path) as fh:
            meta = json.load(fh)
        return list(meta["ids"]), edges

    # ------------------------------------------------------------------
    # convenience wrappers (try cache -> neuPrint -> synthetic)
    # ------------------------------------------------------------------

    def load_or_fetch_mini(
        self,
        neuron_type: str,
        max_edges: int = 40,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]]:
        """Load the cached mini brain, fetching it live when absent.

        Real connectome data only: a failed fetch raises
        :class:`ConnectomeUnavailableError` rather than silently
        returning a synthetic brain.

        Args:
            neuron_type (str): Neuron cell type to center the slice on.
            max_edges (int): Maximum number of synaptic edges to return.

        Returns:
            tuple[list[int], list[tuple[int, int, float]], list[int]]:
                ``(ids, edges, motor_bodies)`` with the same
                structure as :meth:`fetch_mini_brain`.

        Raises:
            ConnectomeUnavailableError: When the brain is not cached and
                the live neuPrint fetch fails (missing token, missing
                ``neuprint`` extra, network failure, unknown cell type).
        """
        tag = self._tag(neuron_type)
        cached = self.load_cache(tag)
        if cached is not None:
            self.last_source = "cache"
            return cached
        try:
            ids, edges, motors = self.fetch_mini_brain(neuron_type, max_edges)
        except Exception as exc:  # noqa: BLE001
            raise ConnectomeUnavailableError(
                f"could not load the real neuPrint brain for "
                f"{neuron_type!r} (dataset {self.dataset!r}): {exc}\n"
                "flynet requires real connectome data and will not "
                "silently substitute synthetic data. Fix: set "
                "NEUPRINT_APPLICATION_CREDENTIALS (see .env), install the "
                "'neuprint' extra, or explicitly request synthetic test "
                "data via --offline (CLI) or "
                "ConnectomeLoader.synthetic_brain()."
            ) from exc
        self.save_cache(ids, edges, motors, tag)
        self.last_source = "neuprint"
        return ids, edges, motors

    def load_cached_mini(
        self,
        neuron_type: str,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]] | None:
        """Return the cached mini brain, never touching the network.

        Args:
            neuron_type (str): Neuron cell type the cache was built for.

        Returns:
            tuple | None: ``(ids, edges, motor_bodies)`` from the cache,
            or ``None`` when nothing is cached.  Never synthetic data.
        """
        result = self.load_cache(self._tag(neuron_type))
        if result is not None:
            self.last_source = "cache"
        return result

    def load_cached_full(
        self,
    ) -> Tuple[List[int], List[Tuple[int, int, float]]] | None:
        """Return the cached full connectome, never touching the network.

        Returns:
            tuple | None: ``(ids, edges)`` from the cache, or ``None``
            when nothing is cached.  Never synthetic data.
        """
        result = self.load_full_cache(self._tag())
        if result is not None:
            self.last_source = "cache"
        return result

    def load_or_fetch_full(
        self,
        min_weight: int = 50,
        edge_cap: int = 3_000_000,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]]:
        """Load the cached full connectome, fetching it live when absent.

        Real connectome data only: a failed fetch raises
        :class:`ConnectomeUnavailableError`.

        Args:
            min_weight (int): Minimum synaptic weight to include an edge.
            edge_cap (int): Maximum number of edges to keep after sorting.

        Returns:
            tuple[list[int], list[tuple[int, int, float]], list[int]]:
                ``(ids, edges, [])`` with an empty motor bodies
                list.

        Raises:
            ConnectomeUnavailableError: When the connectome is not
                cached and the live neuPrint fetch fails.
        """
        tag = self._tag()
        cached = self.load_full_cache(tag)
        if cached is not None:
            self.last_source = "cache"
            ids, edges = cached
            return ids, edges, []
        try:
            ids, edges, _ = self.fetch_full_connectome(min_weight, edge_cap)
        except Exception as exc:  # noqa: BLE001
            raise ConnectomeUnavailableError(
                f"could not load the real full connectome "
                f"(dataset {self.dataset!r}): {exc}\n"
                "flynet requires real connectome data and will not "
                "silently substitute synthetic data. Fix: set "
                "NEUPRINT_APPLICATION_CREDENTIALS (see .env) and install "
                "the 'neuprint' extra, or request synthetic test data "
                "explicitly via --offline (CLI) or "
                "ConnectomeLoader.synthetic_brain()."
            ) from exc
        self.save_full_cache(ids, edges, tag)
        self.last_source = "neuprint"
        return ids, edges, []

    # ------------------------------------------------------------------
    # synthetic brain
    # ------------------------------------------------------------------

    @staticmethod
    def synthetic_brain(
        n_neurons: int = _SYNTH_N,
        max_edges: int = 40,
        seed: int = 0,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]]:
        """Build a synthetic test brain without network access.

        Creates a small directed graph with sensor neurons, hub neurons, and
        motor neurons connected by strong structured edges and random noise
        edges.

        Args:
            n_neurons (int): Total number of neurons in the synthetic brain.
            max_edges (int): Maximum number of edges to include.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple[list[int], list[tuple[int, int, float]], list[int]]:
                ``(ids, edges, motor_bodies)`` where *ids* is the
                sorted list of neuron body IDs, *edges* holds
                ``(pre_body, post_body, weight)`` triples, and *motor_bodies*
                contains the two motor neuron IDs.
        """
        rng = np.random.default_rng(seed)
        ids = [_SYNTH_ID_BASE + i for i in range(n_neurons)]
        s0, s1 = ids[0], ids[1]
        hubs = ids[2 : 2 + _SYNTH_HUB_COUNT]
        m0, m1 = ids[-2], ids[-1]
        rows: list = []
        cols: list = []
        wts: list = []
        strong = [
            (s0, m0, 4.0),
            (s1, m1, 4.0),
            (s0, hubs[0], 2.5),
            (s1, hubs[2], 2.5),
            (hubs[0], m0, 1.8),
            (hubs[2], m1, 1.8),
            (hubs[0], hubs[1], 1.2),
            (hubs[1], hubs[0], 1.2),
            (m0, s0, 3.0),
            (m1, s1, 3.0),
        ]
        for a, b, w in strong:
            rows.append(a)
            cols.append(b)
            wts.append(w)
        edges = list(zip(rows, cols, wts))
        while len(edges) < max_edges:
            rows += [
                int(rng.integers(2, n_neurons)) + _SYNTH_ID_BASE
                for _ in range(_SYNTH_EDGES_PER_BATCH)
            ]
            cols += [
                int(rng.integers(2, n_neurons)) + _SYNTH_ID_BASE
                for _ in range(_SYNTH_EDGES_PER_BATCH)
            ]
            wts += [
                float(rng.random()) * _SYNTH_NOISE_WEIGHT
                for _ in range(_SYNTH_EDGES_PER_BATCH)
            ]
            edges = list(zip(rows, cols, wts))
        return ids, edges[:max_edges], [m0, m1]
