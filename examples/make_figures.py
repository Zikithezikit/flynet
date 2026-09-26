#!/usr/bin/env python3
"""Regenerate the figures in ``docs/images/``.

The repository's documentation used to ship images with no code to
produce them, so the numbers in the README drifted away from what the
code actually does.  This script is that missing generator: it writes
every figure referenced by the README and prints the measured values so
the captions can be checked against them.

Run it after changing the connectome loader, the learning rules or the
examples::

    .venv/bin/python examples/make_figures.py
    .venv/bin/python examples/make_figures.py --only gui --outdir /tmp

It needs the real connectome, so it requires either a warm
``~/.flynet/cache`` or ``NEUPRINT_APPLICATION_CREDENTIALS``; it never
falls back to synthetic data.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flynet.connectome import ConnectomeLoader  # noqa: E402
from flynet.learning import RewardHebbian, TrainingLogger  # noqa: E402
from flynet.network import SpikingNetwork  # noqa: E402
from flynet.visualize import (  # noqa: E402
    plot_brain_circuit,
    plot_learning_curve,
    plot_trajectory,
)

_HERE = Path(__file__).resolve().parent
MINI_TYPE = "DNge104"
NAV_SEED = 7           # seed for the before/after navigation comparison
NAV_TRAIN_EPISODES = 5  # episodes between the before/after pair
CURVE_EPISODES = 30    # longer run so the learning curve has a trend
NAV_EVAL_SEED = 21
GUI_SEED = 11
GUI_TRAIN_EPISODES = 120
EVAL_SEED0 = 90000     # held-out hunt seeds for the before/after check
EVAL_EPISODES = 24


def _load(name: str):
    """Import a sibling example module by file path.

    ``examples/`` is not a package, so its modules are loaded explicitly
    instead of being imported by name.

    Args:
        name (str): File name of the module inside ``examples/``.

    Returns:
        module: The imported module.
    """
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"flynet_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_brain(loader: ConnectomeLoader, mini: bool) -> SpikingNetwork:
    """Load the mini or full connectome as a :class:`SpikingNetwork`.

    Args:
        loader (ConnectomeLoader): Loader to fetch through.
        mini (bool): Use the DNge104 mini brain instead of the whole
            connectome.

    Returns:
        SpikingNetwork: The assembled brain.
    """
    if mini:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=MINI_TYPE, max_edges=40
        )
    else:
        ids, edges, motors = loader.load_or_fetch_full(min_weight=50)
        motors = []
    return SpikingNetwork(ids, edges, motor_ids=motors if motors else None)


# ---------------------------------------------------------------------------
# Figure 1: brain circuit
# ---------------------------------------------------------------------------
def figure_brain_circuit(loader: ConnectomeLoader, outdir: Path) -> str:
    """Draw the wiring around the DNge104 descending neuron.

    Args:
        loader (ConnectomeLoader): Loader to fetch the mini brain through.
        outdir (Path): Directory to write the PNG into.

    Returns:
        str: One-line summary of what was measured.
    """
    net = _load_brain(loader, mini=True)
    fig = plt.figure(figsize=(9, 9))
    plot_brain_circuit(net, ax=fig.add_subplot(111),
                       title=f"Brain Circuit ({MINI_TYPE}, real connectome)")
    path = outdir / "real_brain_circuit.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return (f"{len(net.ids)} neurons, {net.W.nnz} synapses, "
            f"sensors={net.sensors}, motors={net.motors}")


# ---------------------------------------------------------------------------
# Figures 2 and 3: navigation before/after, and its learning curve
# ---------------------------------------------------------------------------
def _train_mini(net: SpikingNetwork, episodes: int) -> TrainingLogger:
    """Run Hebbian training on the mini brain and log each episode.

    Args:
        net (SpikingNetwork): Brain to train in place.
        episodes (int): Number of training episodes.

    Returns:
        TrainingLogger: Per-episode reward and step history.
    """
    nav = _load("navigation_task")
    hebbian = RewardHebbian(eta=0.6)
    logger = TrainingLogger()
    for ep in range(1, episodes + 1):
        rng = np.random.default_rng(NAV_SEED + ep)
        steps, _path, events, found = nav.play_game(net, rng)
        logger.log_episode(1.0 if found else 0.0, steps or nav.MAX_STEPS)
        if found and events:
            hebbian.update(net, events)
    return logger


def _steps_label(steps: int | None, found: bool, path_len: int) -> str:
    """Describe a hunt in a few words for a figure title.

    Args:
        steps (int | None): Steps reported by the episode, ``None`` when
            the food was never reached.
        found (bool): Whether the food was reached.
        path_len (int): Number of positions in the recorded path.

    Returns:
        str: A short label such as ``"97 steps"`` or ``"gave up (300
        steps)"``.
    """
    if found:
        return f"{steps} steps"
    return f"gave up ({path_len - 1} steps)"


def _frame_axes(ax, arena: float, legend: bool = True) -> None:
    """Give a navigation panel a fixed frame so panels stay comparable.

    ``plot_trajectory`` autoscales to the path, which makes a
    before/after pair hard to read; this pins both panels to the same
    square arena and moves the legend off the food.

    Args:
        ax (matplotlib.axes.Axes): Panel to adjust.
        arena (float): Side length of the square arena.
        legend (bool): Whether to reposition the legend.

    Returns:
        None: Adjusts the axes in place.
    """
    ax.set_xlim(0, arena)
    ax.set_ylim(0, arena)
    ax.set_aspect("equal")
    if legend and ax.get_legend() is not None:
        ax.get_legend().loc = "lower right"


def figure_navigation(loader: ConnectomeLoader, outdir: Path) -> str:
    """Draw the mini-brain navigation path before and after training.

    Both panels replay the *same* episode seed so the difference is due
    to the learned weights and nothing else.

    Args:
        loader (ConnectomeLoader): Loader to fetch the mini brain through.
        outdir (Path): Directory to write the PNG into.

    Returns:
        str: One-line summary of the measured before/after step counts.
    """
    nav = _load("navigation_task")
    before_net = _load_brain(loader, mini=True)

    steps_before, path_before, _, found_before = nav.play_game(
        before_net, np.random.default_rng(NAV_EVAL_SEED)
    )

    after_net = before_net.copy()
    logger = _train_mini(after_net, NAV_TRAIN_EPISODES)
    steps_after, path_after, _, found_after = nav.play_game(
        after_net, np.random.default_rng(NAV_EVAL_SEED)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, path, steps, found, label in (
        (axes[0], path_before, steps_before, found_before, "Before learning"),
        (axes[1], path_after, steps_after, found_after,
         f"After {NAV_TRAIN_EPISODES} episodes of Hebbian training"),
    ):
        plot_trajectory(
            [(p[0], p[1]) for p in path], nav.FOOD, tuple(nav.START), ax=ax,
            title=f"{label} - {_steps_label(steps, found, len(path))}",
        )
        _frame_axes(ax, nav.ARENA)
    fig.suptitle("Fly Navigation (real connectome, DNge104 mini brain)")
    fig.tight_layout()
    path = outdir / "real_trajectory.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    steps = logger.get_learning_curve()[1]
    return (f"before={_steps_label(steps_before, found_before, len(path_before))} "
            f"after={_steps_label(steps_after, found_after, len(path_after))} "
            f"(training steps: {steps})")


def figure_learning_curve(loader: ConnectomeLoader, outdir: Path) -> str:
    """Draw the mini-brain navigation learning curve.

    Args:
        loader (ConnectomeLoader): Loader to fetch the mini brain through.
        outdir (Path): Directory to write the PNG into.

    Returns:
        str: One-line summary of the curve's first and last episodes.
    """
    net = _load_brain(loader, mini=True)
    logger = _train_mini(net, CURVE_EPISODES)

    fig = plt.figure(figsize=(10, 5))
    plot_learning_curve(
        logger, ax=fig.add_subplot(111),
        title=f"Navigation learning curve ({MINI_TYPE}, "
              f"{CURVE_EPISODES} episodes)",
    )
    path = outdir / "real_learning_curve.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    steps = logger.get_learning_curve()[1]
    return f"steps per episode: {steps}"


# ---------------------------------------------------------------------------
# Figures 4 and 5: the interactive foraging GUI
# ---------------------------------------------------------------------------
def _gui_context(loader: ConnectomeLoader):
    """Load the full connectome and a foraging app on top of it.

    Args:
        loader (ConnectomeLoader): Loader to fetch the connectome through.

    Returns:
        tuple: ``(app, forager, rule)`` for driving and training.
    """
    gui = _load("fly_foraging_gui")
    net = _load_brain(loader, mini=False)
    forager = gui.FlyForager(
        net, rng=np.random.default_rng(GUI_SEED),
        settle_iters=gui.SETTLE_ITERS_FULL,
    )
    forager.invert_steer = forager.auto_steer_sign() == -1
    stats = gui.TrainingStats()
    rule = gui.MotorRewardHebbian(eta=0.05)
    app = gui.ForagingApp(
        forager, stats, rule,
        banner="[connectome] data source: real male CNS "
               f"({loader.last_source})",
        description=f"full connectome from {loader.dataset}",
    )
    return app, forager, rule


def figure_gui_frame(loader: ConnectomeLoader, outdir: Path) -> str:
    """Capture one frame of the running GUI on the full connectome.

    The frame is taken while the fly is actively tracking the food with
    one of its senses, which is the state worth showing in the docs.

    Args:
        loader (ConnectomeLoader): Loader to fetch the connectome through.
        outdir (Path): Directory to write the PNG into.

    Returns:
        str: One-line summary of the captured state.
    """
    gui = _load("fly_foraging_gui")
    app, forager, _rule = _gui_context(loader)
    app.add_controls()
    app.set_speed(16)

    # Wait for enough hunts that the HUD and the learning curve have
    # something in them, then catch the fly while it is tracking.
    for _ in range(3000):
        app.tick()
        if app.stats.episodes < 20:
            continue
        mode = forager.episode.mode
        if mode in ("sight", "smell") and 12.0 < forager.distance < 45.0:
            break
    # Settle transient UI (the "FOUND FOOD!" banner counts down per draw)
    # without advancing the simulation, so the still frame is the steady
    # state a reader would see.
    for _ in range(gui.FLASH_FRAMES + 1):
        app.draw()
    app.fig.canvas.draw()
    path = outdir / "gui_foraging.png"
    app.fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(app.fig)
    return (f"episode {app.stats.episodes}, "
            f"success {app.stats.successes}/{app.stats.episodes}, "
            f"frame mode={forager.episode.mode}")


def _evaluate_hunts(gui, net, Minv, invert, seeds) -> tuple[int, float]:
    """Score a brain on a fixed set of episode seeds.

    Uses the same held-out seeds for every brain so that before/after
    numbers are comparable; a single episode is far too noisy to judge
    learning on the full connectome.

    Args:
        gui (module): The foraging GUI example module.
        net (SpikingNetwork): Brain to score.
        Minv (np.ndarray): Frozen steering calibration to reuse.
        invert (bool): Whether the steering direction is flipped.
        seeds (range): Episode seeds to score.

    Returns:
        tuple: ``(successes, mean_steps)`` over the seeds.
    """
    runner = gui.FlyForager(
        net, rng=np.random.default_rng(EVAL_SEED0), Minv=Minv,
        invert_steer=invert,
    )
    hits, steps = 0, []
    for seed in seeds:
        runner.rng = np.random.default_rng(seed)
        runner.reset()
        while (not runner.episode.found
               and runner.episode.step < gui.MAX_STEPS):
            runner.step()
        if runner.episode.found:
            hits += 1
            steps.append(runner.episode.step)
    return hits, float(np.mean(steps)) if steps else float("nan")


def figure_gui_learning(loader: ConnectomeLoader, outdir: Path) -> str:
    """Show that the GUI's fly learns, on training and on held-out hunts.

    The left panel is the live training signal the window shows; the right
    panel is the honest check, scoring the untrained and the trained
    connectome on the *same* 24 held-out episode seeds.

    Args:
        loader (ConnectomeLoader): Loader to fetch the connectome through.
        outdir (Path): Directory to write the PNG into.

    Returns:
        str: One-line summary of the measured success rates.
    """
    gui = _load("fly_foraging_gui")
    _app, forager, rule = _gui_context(loader)
    stats = gui.TrainingStats()
    seeds = range(EVAL_SEED0, EVAL_SEED0 + EVAL_EPISODES)

    hits_before, steps_before = _evaluate_hunts(
        gui, forager.net, forager.Minv, forager.invert_steer, seeds
    )
    # Public API only: snapshot the connectome's own weights, then train.
    untrained = forager.net.copy()

    for ep in range(GUI_TRAIN_EPISODES):
        forager.rng = np.random.default_rng(GUI_SEED * 1000 + ep)
        forager.reset()
        while (not forager.episode.found
               and forager.episode.step < gui.MAX_STEPS):
            forager.step()
        stats.record(forager.episode.found, forager.episode.step)
        gui.apply_learning(
            forager.net, rule,
            forager.learning_batch(found=forager.episode.found),
        )

    hits_after, steps_after = _evaluate_hunts(
        gui, forager.net, forager.Minv, forager.invert_steer, seeds
    )
    del untrained

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # -- left: the live training signal --------------------------------
    episodes, steps = stats.logger.get_learning_curve()
    axes[0].plot(episodes, steps, "-", color="steelblue", alpha=0.3,
                 label="steps per hunt")
    sm_x, sm_y = stats.smoothed_steps()
    axes[0].plot(sm_x, sm_y, "-", color="crimson", linewidth=2,
                 label="smoothed")
    rate_x, rate_y = stats.success_series()
    axes[0].set_xlabel("hunt")
    axes[0].set_ylabel("steps to food")
    axes[0].set_ylim(0, float(gui.MAX_STEPS))
    axes[0].set_title(f"Training the fly ({GUI_TRAIN_EPISODES} hunts)")
    axes[0].legend(loc="upper right", fontsize=8)

    rate_ax = axes[0].twinx()
    rate_ax.plot(rate_x, rate_y, "-", color="seagreen", linewidth=1.5)
    rate_ax.set_ylabel("rolling success rate", color="seagreen", fontsize=9)
    rate_ax.tick_params(labelsize=8, colors="seagreen")
    rate_ax.set_ylim(-0.05, 1.05)

    # -- right: the held-out comparison --------------------------------
    labels = ["untrained\nconnectome", f"after {GUI_TRAIN_EPISODES}\nhunts"]
    rates = [100.0 * hits_before / EVAL_EPISODES,
             100.0 * hits_after / EVAL_EPISODES]
    means = [steps_before, steps_after]
    bars = axes[1].bar(labels, rates, color=["#9fb8cd", "#3f7f9f"], width=0.5)
    axes[1].set_ylabel(f"success rate on {EVAL_EPISODES} held-out hunts (%)")
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Held-out check (same 24 start conditions)")
    for bar, rate, mean in zip(bars, rates, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{rate:.0f}%\n{mean:.0f} steps avg",
            ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle("Foraging on the full real connectome: live learning")
    fig.tight_layout()
    path = outdir / "gui_foraging_learning.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return (f"held-out success {hits_before}/{EVAL_EPISODES} -> "
            f"{hits_after}/{EVAL_EPISODES}, "
            f"avg steps {steps_before:.0f} -> {steps_after:.0f}")


FIGURES = {
    "brain_circuit": figure_brain_circuit,
    "trajectory": figure_navigation,
    "curve": figure_learning_curve,
    "gui": figure_gui_frame,
    "gui_learning": figure_gui_learning,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser for the figure generator.
    """
    p = argparse.ArgumentParser(
        description="Regenerate the figures used by the README."
    )
    p.add_argument(
        "--outdir", default=str(_HERE.parent / "docs" / "images"),
        help="where to write the PNGs (default: %(default)s)",
    )
    p.add_argument(
        "--dataset", default="male-cns:v0.9",
        help="neuPrint dataset (default: %(default)s)",
    )
    p.add_argument(
        "--only", default="",
        help="comma-separated subset of: " + ", ".join(FIGURES),
    )
    p.add_argument(
        "--list", action="store_true", help="list the figure names and exit"
    )
    return p


def main() -> None:
    """Regenerate the requested figures.

    Raises:
        SystemExit: If the real connectome cannot be loaded.
    """
    args = build_parser().parse_args()
    if args.list:
        for name in FIGURES:
            print(name)
        return

    wanted = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only else list(FIGURES)
    )
    unknown = [s for s in wanted if s not in FIGURES]
    if unknown:
        raise SystemExit(
            f"unknown figure(s): {', '.join(unknown)}; "
            f"choose from {', '.join(FIGURES)}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    loader = ConnectomeLoader(dataset=args.dataset)

    for name in wanted:
        print(f"[figures] {name} ...", flush=True)
        summary = FIGURES[name](loader, outdir)
        print(f"[figures] {name}: {summary}", flush=True)
    print(f"[figures] wrote {len(wanted)} figure(s) to {outdir}")


if __name__ == "__main__":
    main()
