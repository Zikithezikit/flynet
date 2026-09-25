"""Regression tests for the real-brain load contract.

flynet must never silently substitute synthetic data for the real brain:
fetch failures raise ``ConnectomeUnavailableError``, explicit synthetic
data stays available only via ``ConnectomeLoader.synthetic_brain()``
(CLI: ``--offline``), and the loader reports provenance via ``last_source``.
All tests here are hermetic (isolated ``cache_dir``, monkeypatched fetch).
"""

import pytest

from flynet.connectome import ConnectomeLoader, ConnectomeUnavailableError


def test_mini_fetch_failure_raises_not_synthetic(tmp_path, monkeypatch):
    loader = ConnectomeLoader(cache_dir=str(tmp_path))

    def boom(*_args, **_kwargs):
        raise RuntimeError("neuprint client unavailable (no token)")

    monkeypatch.setattr(loader, "fetch_mini_brain", boom)
    with pytest.raises(ConnectomeUnavailableError) as exc:
        loader.load_or_fetch_mini("DNge104")
    msg = str(exc.value)
    assert "no token" in msg
    assert "synthetic" in msg.lower()
    assert "--offline" in msg


def test_full_fetch_failure_raises_not_synthetic(tmp_path, monkeypatch):
    loader = ConnectomeLoader(cache_dir=str(tmp_path))

    def boom(*_args, **_kwargs):
        raise RuntimeError("neuprint client unavailable (no token)")

    monkeypatch.setattr(loader, "fetch_full_connectome", boom)
    with pytest.raises(ConnectomeUnavailableError) as exc:
        loader.load_or_fetch_full()
    assert "no token" in str(exc.value)


def test_cache_hit_returns_cache_without_fetching(tmp_path, monkeypatch):
    loader = ConnectomeLoader(cache_dir=str(tmp_path))
    ids = [1, 2, 3]
    edges = [(1, 2, 5.0), (2, 3, 7.0)]
    motors = [3]
    loader.save_cache(ids, edges, motors, loader._tag("DNge104"))

    def must_not_fetch(*_args, **_kwargs):
        raise AssertionError("fetch attempted despite warm cache")

    monkeypatch.setattr(loader, "fetch_mini_brain", must_not_fetch)
    got = loader.load_or_fetch_mini("DNge104")
    assert got == (ids, edges, motors)
    assert loader.last_source == "cache"


def test_successful_fetch_saves_cache_and_reports_neuprint(tmp_path, monkeypatch):
    loader = ConnectomeLoader(cache_dir=str(tmp_path))
    ids = [10, 20]
    edges = [(10, 20, 6.0)]
    motors = [20]
    monkeypatch.setattr(
        loader, "fetch_mini_brain", lambda *a, **k: (ids, edges, motors))
    got = loader.load_or_fetch_mini("DNge104")
    assert got == (ids, edges, motors)
    assert loader.last_source == "neuprint"
    assert loader.load_cache(loader._tag("DNge104")) == (ids, edges, motors)


def test_load_cached_helpers_return_none_on_cold_cache(tmp_path):
    loader = ConnectomeLoader(cache_dir=str(tmp_path))
    assert loader.load_cached_mini("DNge104") is None
    assert loader.load_cached_full() is None


def test_explicit_synthetic_brain_still_available():
    ids, edges, motors = ConnectomeLoader.synthetic_brain(max_edges=40)
    assert len(ids) == 41
    assert ids[0] == 1000
    assert motors == [1039, 1040]
    assert 10 <= len(edges) <= 40
    assert all(pre in set(ids) and post in set(ids) for pre, post, _ in edges)


def test_cli_missing_connectome_exits_cleanly(monkeypatch, capsys):
    from flynet import cli
    from flynet import connectome

    def boom(*_args, **_kwargs):
        raise ConnectomeUnavailableError(
            "could not load the real neuPrint brain for 'DNge104': test")

    monkeypatch.setattr(
        connectome.ConnectomeLoader, "load_or_fetch_mini", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["simulate"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "test" in captured.err


def test_cli_offline_reports_synthetic(capsys):
    from flynet import cli

    cli.main(["simulate", "--offline", "--settle", "3"])
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out
    assert "Network: 41 neurons" in out


def test_cli_real_path_reports_provenance(capsys, tmp_path):
    from flynet import cli, connectome

    seed_ids = [1000 + i for i in range(41)]
    seed_edges = [(seed_ids[i], seed_ids[i + 1], 3.0) for i in range(40)]
    seed_motors = [1039, 1040]
    real_loader_cls = connectome.ConnectomeLoader

    def seeded_factory(*args, **kwargs):
        kwargs["cache_dir"] = str(tmp_path)
        loader = real_loader_cls(*args, **kwargs)
        loader.save_cache(seed_ids, seed_edges, seed_motors,
                          loader._tag("DNge104"))
        return loader

    saved = connectome.ConnectomeLoader
    connectome.ConnectomeLoader = seeded_factory
    try:
        cli.main(["simulate", "--settle", "3"])
    finally:
        connectome.ConnectomeLoader = saved
    out = capsys.readouterr().out
    assert "data source: cache" in out
    assert "SYNTHETIC" not in out
