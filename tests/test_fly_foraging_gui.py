"""Tests for the interactive foraging GUI example.

The example lives in ``examples/`` and is not importable as a package, so
it is loaded by path.  Only display-free pieces are exercised: the sensing
geometry, the step/reward bookkeeping and the plasticity rule.  A short
headless run of ``main()`` covers the wiring end to end.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from flynet.connectome import ConnectomeLoader
from flynet.learning import MotorRewardHebbian, RewardHebbian
from flynet.network import SpikingNetwork

_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "fly_foraging_gui.py"
)


def _load_example():
    """Import ``examples/fly_foraging_gui.py`` by path.

    Returns:
        module: The imported example module.
    """
    spec = importlib.util.spec_from_file_location("fly_foraging_gui", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fly_foraging_gui"] = module
    spec.loader.exec_module(module)
    return module


fg = _load_example()


@pytest.fixture()
def small_net():
    """Return the 41-neuron synthetic brain as a SpikingNetwork."""
    ids, edges, motors = ConnectomeLoader.synthetic_brain(max_edges=40)
    return SpikingNetwork(ids, edges, motor_ids=motors)


# ---------------------------------------------------------------------------
# sensing geometry
# ---------------------------------------------------------------------------
def test_wrap_angle_stays_in_range():
    assert fg.wrap_angle(3.0 * np.pi) == pytest.approx(-np.pi)
    assert fg.wrap_angle(0.5) == pytest.approx(0.5)
    assert abs(fg.wrap_angle(7.0)) <= np.pi


def test_smell_decays_with_distance():
    food = (50.0, 50.0)
    near = fg.smell_at(np.array([50.0, 55.0]), food)
    far = fg.smell_at(np.array([50.0, 70.0]), food)
    assert near > far > 0.0
    assert fg.smell_at(np.array([50.0, 50.0]), food) == pytest.approx(2.0)


def test_visual_field_is_symmetric_when_food_is_ahead():
    # Facing +x, food directly ahead: both eyes see the same contrast.
    right, left = fg.visual_readings(np.array([20.0, 50.0]), 0.0, (50.0, 50.0))
    assert right == pytest.approx(left)
    assert right > 0.0


def test_visual_field_favours_the_eye_the_food_is_in():
    # Facing +x in a y-up frame, the fly's left is +y, so food at +y must
    # brighten the left eye and food at -y the right one.
    pos = np.array([20.0, 50.0])
    right, left = fg.visual_readings(pos, 0.0, (50.0, 60.0))
    assert left > right > 0.0
    right, left = fg.visual_readings(pos, 0.0, (50.0, 40.0))
    assert right > left > 0.0


def test_visual_field_ignores_food_behind_and_beyond_range():
    pos = np.array([20.0, 50.0])
    # Facing -x, so food at larger x is behind the fly.
    assert fg.visual_readings(pos, np.pi, (30.0, 50.0)) == (0.0, 0.0)
    # Food ahead but past the visual range.
    far = (20.0 + fg.SIGHT_RANGE + 10.0, 50.0)
    assert fg.visual_readings(pos, 0.0, far) == (0.0, 0.0)


def test_antennae_favour_the_side_the_plume_is_on():
    pos = np.array([20.0, 50.0])
    # Facing +x, its left antenna points to +y.
    right, left = fg.antenna_readings(pos, 0.0, (50.0, 60.0))
    assert left > right
    right, left = fg.antenna_readings(pos, 0.0, (50.0, 40.0))
    assert right > left


def test_topk_activity_is_sparse_and_sized():
    trace = [np.zeros(1000), np.arange(1000, dtype=float)]
    idx, vals = fg.topk_activity(trace, k=10)
    assert idx.size == 10
    assert vals.size == 10
    # The mean of the last two iterations is kept, and the largest
    # magnitudes survive the truncation.
    assert np.abs(vals).max() == pytest.approx(999.0 / 2.0)
    empty_i, empty_v = fg.topk_activity([], k=10)
    assert empty_i.size == 0 and empty_v.size == 0


# ---------------------------------------------------------------------------
# forager
# ---------------------------------------------------------------------------
def test_forager_runs_and_records_signed_rewards(small_net):
    forager = fg.FlyForager(small_net, rng=np.random.default_rng(0))
    for _ in range(40):
        forager.step()
    assert forager.episode.step == 40
    assert forager.step_count == 40
    for event in forager.episode.events:
        assert len(event) == 4
        _, _, activity, reward = event
        assert isinstance(activity, tuple)
        assert np.isfinite(reward)


def test_forager_stays_inside_the_arena(small_net):
    forager = fg.FlyForager(small_net, rng=np.random.default_rng(1))
    for _ in range(120):
        forager.step()
    assert np.all(forager.episode.pos >= 0.0)
    assert np.all(forager.episode.pos <= forager.arena)


def test_search_mode_actually_travels(small_net):
    """Search must cover ground, not spin on the spot."""
    forager = fg.FlyForager(small_net, rng=np.random.default_rng(2))
    forager.reset(start=np.array([10.0, 10.0]))
    while forager.episode.mode != "search" and forager.episode.step < 20:
        forager.step()
    forager.episode.events.clear()
    start = forager.episode.pos.copy()
    for _ in range(25):
        forager.step()
    assert float(np.linalg.norm(forager.episode.pos - start)) > 10.0


def test_learning_batch_bounds_length_and_adds_terminal_reward(small_net):
    forager = fg.FlyForager(small_net, rng=np.random.default_rng(3))
    for _ in range(4):
        forager.reset(start=np.array([10.0, 10.0]))
        while not forager.episode.events and forager.episode.step < fg.MAX_STEPS:
            forager.step()
        if forager.episode.events:
            break
    assert forager.episode.events, "no guided step produced a learning event"

    batch = forager.learning_batch(limit=5, found=True)
    assert 1 <= len(batch) <= 5
    plain = forager.learning_batch(limit=5)
    assert batch[-1][3] == pytest.approx(plain[-1][3] + fg.SUCCESS_BONUS)
    failed = forager.learning_batch(limit=5, found=False)
    assert failed[-1][3] == pytest.approx(
        plain[-1][3] + fg.FAILURE_PENALTY
    )
    assert forager.learning_batch(limit=0) == []


# ---------------------------------------------------------------------------
# learning rules
# ---------------------------------------------------------------------------
def test_motor_reward_hebbian_moves_the_motor_projection(small_net):
    rows, cols = small_net.rows, small_net.cols
    mask = np.zeros(cols.shape[0], dtype=bool)
    for body in small_net.motors[:2]:
        mask |= cols == small_net._ix[body]
    assert mask.any()

    before = small_net.W.data.copy()
    events = [(2.0, 0.5, (np.arange(20), np.linspace(0.2, 1.0, 20)), 1.0)]
    MotorRewardHebbian(eta=0.5).update(small_net, events)
    after = small_net.W.data
    assert not np.allclose(before[mask], after[mask])
    # Nothing outside the motor projection may move.
    assert np.allclose(before[~mask], after[~mask])


def test_motor_reward_hebbian_handles_all_negative_rewards(small_net):
    """A purely negative batch must depress, not be skipped."""
    mask = np.zeros(small_net.cols.shape[0], dtype=bool)
    for body in small_net.motors[:2]:
        mask |= small_net.cols == small_net._ix[body]
    before = small_net.W.data[mask].copy()
    # Realistic events: non-negative activity, negative reward.
    events = [(0.5, 2.0, (np.arange(20), np.linspace(0.2, 1.0, 20)), -1.0)]
    MotorRewardHebbian(eta=0.5).update(small_net, events)
    after = small_net.W.data[mask]
    assert not np.allclose(before, after)
    # Every delta is non-positive, so the total drive must shrink.
    assert after.sum() < before.sum()


def test_reward_hebbian_still_accepts_three_element_events(small_net):
    """The pre-existing 3-tuple contract must keep working."""
    before = float(small_net.W.data.mean())
    events = [(2.0, 0.5, (np.arange(20), np.linspace(0.2, 1.0, 20)))]
    RewardHebbian(eta=0.1).update(small_net, events)
    assert float(small_net.W.data.mean()) != before


# ---------------------------------------------------------------------------
# stats and app
# ---------------------------------------------------------------------------
def test_training_stats_rolling_window():
    stats = fg.TrainingStats(window=3)
    assert stats.success_rate() == 0.0
    assert stats.avg_steps() == 0.0
    for found, steps in ((True, 10), (False, 300), (True, 20)):
        stats.record(found, steps)
    assert stats.episodes == 3
    assert stats.successes == 2
    assert stats.best_steps == 10
    assert stats.success_rate() == pytest.approx(2 / 3)
    stats.record(True, 5)
    # The window drops the oldest entry, so the perfect run sticks.
    assert stats.best_steps == 5
    assert len(stats.recent) == 3
    xs, ys = stats.success_series()
    assert xs == [1, 2, 3, 4]
    assert ys[-1] == pytest.approx(2 / 3)


def test_app_clamps_speed_and_advances(small_net):
    forager = fg.FlyForager(small_net, rng=np.random.default_rng(4))
    app = fg.ForagingApp(
        forager, fg.TrainingStats(), MotorRewardHebbian(eta=0.25), "b", "b",
        speed=999,
    )
    assert app.speed == fg.MAX_SPEED
    app.set_speed(0)
    assert app.speed == fg.MIN_SPEED

    app.set_speed(3)
    before = forager.step_count
    app.tick()
    assert forager.step_count == before + 3

    app.paused = True
    held = forager.step_count
    app.tick()
    assert forager.step_count == held

    app.on_key(type("E", (), {"key": " "})())
    assert app.paused is False
    app.on_key(type("E", (), {"key": "q"})())
    assert app.running is False


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_headless_main_runs_on_the_synthetic_brain(tmp_path, capsys):
    out = tmp_path / "frame.png"
    argv = sys.argv
    sys.argv = [
        "fly_foraging_gui", "--synthetic", "--smoke", "--smoke-frames", "60",
        "--speed", "8", "--save", str(out),
    ]
    try:
        fg.main()
    finally:
        sys.argv = argv
    captured = capsys.readouterr().out
    assert "SYNTHETIC" in captured
    assert "steps=" in captured
    assert out.exists() and out.stat().st_size > 0


def test_offline_without_cache_exits_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        ConnectomeLoader, "cache_dir", str(tmp_path), raising=False
    )
    monkeypatch.setattr(
        ConnectomeLoader, "load_cached_full", lambda self: None
    )
    argv = sys.argv
    sys.argv = ["fly_foraging_gui", "--offline", "--smoke", "--smoke-frames", "1"]
    try:
        with pytest.raises(SystemExit) as exc:
            fg.main()
    finally:
        sys.argv = argv
    assert "no cached real brain" in str(exc.value)
