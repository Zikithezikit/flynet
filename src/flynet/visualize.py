"""Visualization module for brain circuits, trajectories, and training metrics.

Uses matplotlib for all plots.  Optionally uses networkx for graph layout
in :func:`plot_brain_circuit` and Pillow for animated GIF creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import scipy.sparse
    from matplotlib.figure import Figure

    from flynet.learning import TrainingLogger
    from flynet.network import SpikingNetwork
    from flynet.synapses import SynapseList

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[assignment]

__all__ = [
    "plot_brain_circuit",
    "plot_trajectory",
    "plot_learning_curve",
    "plot_plasticity_heatmap",
    "plot_spike_raster",
    "plot_weight_matrix",
    "create_gif",
]


def plot_brain_circuit(
    network: SpikingNetwork,
    ax: matplotlib.axes.Axes | None = None,
    show_sensors: bool = True,
    show_motors: bool = True,
    max_edges: int = 50,
    title: str = "Brain Circuit",
) -> matplotlib.axes.Axes:
    """Plot the brain circuit as a directed graph.

    Sensors are red, motors are blue, other neurons are light blue.
    Edge thickness is proportional to weight.

    Args:
        network (SpikingNetwork): SpikingNetwork instance.
        ax (matplotlib.axes.Axes | None): Matplotlib axes.  Creates a new figure if ``None``.
        show_sensors (bool): Whether to highlight sensor neurons.
        show_motors (bool): Whether to highlight motor neurons.
        max_edges (int): Maximum number of edges to draw.
        title (str): Plot title.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if nx is not None:
        G = nx.DiGraph()
        sensor_set = set(network.sensors) if show_sensors else set()
        motor_set = set(network.motors) if show_motors else set()

        for nid in network.ids:
            G.add_node(nid)

        coo = network.W.tocoo()
        edge_list = list(zip(coo.row, coo.col, coo.data))
        edge_list.sort(key=lambda e: abs(e[2]), reverse=True)
        edge_list = edge_list[:max_edges]

        for r, c, w in edge_list:
            pre = network.ids[r]
            post = network.ids[c]
            G.add_edge(pre, post, weight=abs(w))

        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G.nodes)))

        node_colors = []
        for nid in G.nodes:
            if nid in sensor_set:
                node_colors.append("tomato")
            elif nid in motor_set:
                node_colors.append("dodgerblue")
            else:
                node_colors.append("lightblue")

        edge_weights = [G[u][v]["weight"] for u, v in G.edges]
        max_w = max(edge_weights) if edge_weights else 1.0
        edge_widths = [1.0 + 3.0 * w / max_w for w in edge_weights]

        nx.draw_networkx(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=200,
            with_labels=False,
            edge_color="gray",
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=12,
        )

        legend_items = []
        if show_sensors:
            legend_items.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
                           markersize=10, label="Sensors")
            )
        if show_motors:
            legend_items.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue",
                           markersize=10, label="Motors")
            )
        legend_items.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue",
                       markersize=10, label="Other")
        )
        ax.legend(handles=legend_items, loc="upper left")
    else:
        sensor_set = set(network.sensors) if show_sensors else set()
        motor_set = set(network.motors) if show_motors else set()
        n = len(network.ids)

        if nx is None:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos = {network.ids[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
        else:
            pos = {}

        coo = network.W.tocoo()
        edge_list = list(zip(coo.row, coo.col, coo.data))
        edge_list.sort(key=lambda e: abs(e[2]), reverse=True)
        edge_list = edge_list[:max_edges]

        for r, c, w in edge_list:
            pre = network.ids[r]
            post = network.ids[c]
            x0, y0 = pos[pre]
            x1, y1 = pos[post]
            width = 0.5 + 2.0 * abs(w)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=width, color="gray", alpha=0.6),
            )

        for nid in network.ids:
            x, y = pos[nid]
            if nid in sensor_set:
                color = "tomato"
            elif nid in motor_set:
                color = "dodgerblue"
            else:
                color = "lightblue"
            ax.plot(x, y, "o", color=color, markersize=10, zorder=5)

        legend_items = []
        if show_sensors:
            legend_items.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
                           markersize=10, label="Sensors")
            )
        if show_motors:
            legend_items.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue",
                           markersize=10, label="Motors")
            )
        legend_items.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue",
                       markersize=10, label="Other")
        )
        ax.legend(handles=legend_items, loc="upper left")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_trajectory(
    path: list[tuple[float, float]],
    food_pos: tuple[float, float],
    start_pos: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Fly Search Path",
) -> matplotlib.axes.Axes:
    """Plot the fly trajectory in the 2D arena.

    Args:
        path (list[tuple[float, float]]): List of ``(x, y)`` positions.
        food_pos (tuple[float, float]): ``(x, y)`` of the food source.
        start_pos (tuple[float, float] | None): ``(x, y)`` of the starting position.
        ax (matplotlib.axes.Axes | None): Matplotlib axes.
        title (str): Plot title.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    ax.plot(xs, ys, "b-", linewidth=1.0, alpha=0.7, label="Path")
    ax.plot(xs, ys, "b.", markersize=2, alpha=0.3)

    if start_pos is not None:
        ax.plot(start_pos[0], start_pos[1], "bo", markersize=10, label="Start", zorder=5)

    ax.plot(food_pos[0], food_pos[1], "o", color="gold", markersize=15,
            markeredgecolor="black", label="Food", zorder=5)

    if path:
        ax.plot(path[-1][0], path[-1][1], "r*", markersize=12, label="End", zorder=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    return ax


def plot_learning_curve(
    logger: TrainingLogger,
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Learning Curve",
    window: int = 10,
) -> matplotlib.axes.Axes:
    """Plot average steps per episode.

    Args:
        logger (TrainingLogger): TrainingLogger instance.
        ax (matplotlib.axes.Axes | None): Matplotlib axes.
        title (str): Plot title.
        window (int): Smoothing window size.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    episodes, steps = logger.get_learning_curve()
    if not episodes:
        ax.set_title(title)
        return ax

    ep_arr = np.array(episodes, dtype=np.float64)
    st_arr = np.array(steps, dtype=np.float64)

    ax.plot(ep_arr, st_arr, "b-", alpha=0.3, label="Raw")

    if len(st_arr) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(st_arr, kernel, mode="valid")
        sm_x = ep_arr[window - 1:]
        ax.plot(sm_x, smoothed, "b-", linewidth=2, label=f"Smoothed ({window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_plasticity_heatmap(
    W_before: scipy.sparse.spmatrix | np.ndarray,
    W_after: scipy.sparse.spmatrix | np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Weight Changes",
    max_display: int = 100,
) -> matplotlib.axes.Axes:
    """Plot heatmap of weight changes after training.

    Args:
        W_before (scipy.sparse.spmatrix | np.ndarray): Sparse matrix (before training).
        W_after (scipy.sparse.spmatrix | np.ndarray): Sparse matrix (after training).
        ax (matplotlib.axes.Axes | None): Matplotlib axes.
        title (str): Plot title.
        max_display (int): Maximum matrix dimension to display.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    dense_before = W_before.toarray() if hasattr(W_before, "toarray") else np.asarray(W_before)
    dense_after = W_after.toarray() if hasattr(W_after, "toarray") else np.asarray(W_after)

    delta = dense_after - dense_before
    n = min(delta.shape[0], max_display)
    m = min(delta.shape[1], max_display)
    delta = delta[:n, :m]

    vmax = np.abs(delta).max()
    if vmax == 0:
        vmax = 1.0

    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, label="Weight Change")
    ax.set_title(title)
    ax.set_xlabel("Pre-synaptic")
    ax.set_ylabel("Post-synaptic")
    return ax


def plot_spike_raster(
    spike_train: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Spike Raster",
) -> matplotlib.axes.Axes:
    """Plot spike raster (neuron index vs time).

    Args:
        spike_train (np.ndarray): Binary array ``(n_neurons, t_steps + 1)``.
        ax (matplotlib.axes.Axes | None): Matplotlib axes.
        title (str): Plot title.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    n_neurons, n_timesteps = spike_train.shape
    for t in range(n_timesteps):
        fired = np.nonzero(spike_train[:, t])[0]
        if fired.size > 0:
            ax.scatter(
                np.full_like(fired, t, dtype=np.float64),
                fired.astype(np.float64),
                s=1.0,
                c="black",
                marker="|",
            )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(0, n_timesteps)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    return ax


def plot_weight_matrix(
    synapses: SynapseList | scipy.sparse.spmatrix | np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Weight Matrix",
    max_display: int = 100,
) -> matplotlib.axes.Axes:
    """Plot the weight matrix as a heatmap.

    Args:
        synapses (SynapseList | scipy.sparse.spmatrix | np.ndarray): SynapseList or 2D array.
        ax (matplotlib.axes.Axes | None): Matplotlib axes.
        title (str): Plot title.
        max_display (int): Maximum matrix dimension to display.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if hasattr(synapses, "_weights"):
        W = synapses._weights
    elif hasattr(synapses, "toarray"):
        W = synapses.toarray()
    else:
        W = np.asarray(synapses)

    n = min(W.shape[0], max_display)
    m = min(W.shape[1], max_display)
    W = W[:n, :m]

    im = ax.imshow(W, cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_title(title)
    ax.set_xlabel("Pre-synaptic")
    ax.set_ylabel("Post-synaptic")
    return ax


def create_gif(
    frames: list[Figure | np.ndarray],
    path: str,
    fps: int = 20,
) -> None:
    """Create an animated GIF from a list of matplotlib figures or numpy arrays.

    Args:
        frames (list[Figure | np.ndarray]): List of Figure objects or 2D/3D numpy arrays (images).
        path (str): Output file path (should end in ``.gif``).
        fps (int): Frames per second.

    Raises:
        ImportError: If Pillow is not installed.
    """
    if PILImage is None:
        raise ImportError("Pillow is required for GIF creation: pip install Pillow")

    import io

    images: list[PILImage.Image] = []
    for frame in frames:
        if hasattr(frame, "savefig"):
            buf = io.BytesIO()
            frame.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img: Any = PILImage.open(buf)
        else:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                img = PILImage.fromarray(arr, mode="L")
            elif arr.ndim == 3 and arr.shape[2] == 4:
                img = PILImage.fromarray(arr, mode="RGBA")
            elif arr.ndim == 3 and arr.shape[2] == 3:
                img = PILImage.fromarray(arr, mode="RGB")
            else:
                img = PILImage.fromarray(arr)
        images.append(img.convert("P"))

    if not images:
        raise ValueError("No frames to save")

    duration_ms = int(1000 / fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
