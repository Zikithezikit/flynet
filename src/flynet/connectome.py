"""Load Drosophila brain connectome data from neuPrint.

Provides :class:`ConnectomeLoader` which wraps neuPrint queries with local
CSV/JSON caching, graceful degradation to a synthetic brain when the server
is unreachable, and a chunked full-connectome fetch for the ~176k-neuron
male CNS dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".flynet", "cache")
_SYNTH_N = 41
_SYNTH_ID_BASE = 1000
_SYNTH_HUB_COUNT = 6
_SYNTH_NOISE_WEIGHT = 0.4
_SYNTH_EDGES_PER_BATCH = 8
_FULL_CHUNK = 20000
_FULL_MIN_WEIGHT = 50


class ConnectomeLoader:
    """Load and cache Drosophila connectome data from neuPrint."""

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
    ):
        self.server = server
        self.dataset = dataset
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._client = None

    # ------------------------------------------------------------------
    # neuPrint client (lazy, cached)
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return a cached neuPrint ``Client`` or ``None``."""
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
        """Filesystem-safe suffix derived from the dataset name."""
        safe = self.dataset.replace(":", "_").replace("/", "_")
        if neuron_type:
            return f".{safe}.{neuron_type}"
        return f".{safe}"

    def _edge_csv(self, prefix: str, tag: str) -> str:
        return os.path.join(self.cache_dir, f"{prefix}_edges{tag}.csv")

    def _meta_json(self, prefix: str, tag: str) -> str:
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

        Returns ``(ids, edges, motor_bodies)`` where *ids* is the sorted list
        of neuron body IDs, *edges* holds ``(pre_body, post_body, weight)``
        triples, and *motor_bodies* contains candidate motor neuron IDs of the
        requested cell type.
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

        Returns ``(ids, edges, [])`` with an empty *motor_bodies* list.
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
        """Persist fetched data to *cache_dir*."""
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
        """Persist a full-connectome fetch to *cache_dir*."""
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

        Returns ``(ids, edges, motor_bodies)`` or ``None`` when the cache is
        incomplete.
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

        Returns ``(ids, edges)`` or ``None`` when the cache is incomplete.
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
    # convenience wrappers (try cache → neuPrint → synthetic)
    # ------------------------------------------------------------------

    def load_or_fetch_mini(
        self,
        neuron_type: str,
        max_edges: int = 40,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]]:
        """Try cache first, then neuPrint, then synthetic fallback."""
        tag = self._tag(neuron_type)
        cached = self.load_cache(tag)
        if cached is not None:
            return cached
        try:
            ids, edges, motors = self.fetch_mini_brain(neuron_type, max_edges)
        except Exception:  # noqa: BLE001
            return self.synthetic_brain(max_edges=max_edges)
        self.save_cache(ids, edges, motors, tag)
        return ids, edges, motors

    def load_or_fetch_full(
        self,
        min_weight: int = 50,
        edge_cap: int = 3_000_000,
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[int]]:
        """Try cache first, then neuPrint."""
        tag = self._tag()
        cached = self.load_full_cache(tag)
        if cached is not None:
            ids, edges = cached
            return ids, edges, []
        ids, edges, _ = self.fetch_full_connectome(min_weight, edge_cap)
        self.save_full_cache(ids, edges, tag)
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

        Returns ``(ids, edges, motor_bodies)`` with the same structure as the
        real data: sensor neurons (``ids[0]``, ``ids[1]``), hub neurons
        (``ids[2:8]``), and motor neurons (``ids[-2:]``).  Strong edges connect
        sensors → hubs → motors and motors → sensors (recurrent); random noise
        edges fill up to *max_edges*.
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
        edges: list = list(zip(rows, cols, wts))
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
