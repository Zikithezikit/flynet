"""End-to-end check against the real neuPrint hemibrain/CNS data.

Skipped unless NEUPRINT_APPLICATION_CREDENTIALS is set, so offline CI stays
hermetic. Run with ``make test-live``.
"""

import os

import numpy as np
import pytest

from flynet import ConnectomeLoader, SpikingNetwork

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS"),
    reason="needs a neuPrint token",
)


@pytest.fixture(scope="module")
def real_net(tmp_path_factory):
    # Fresh cache dir so this really hits the server rather than a stale file.
    loader = ConnectomeLoader(cache_dir=str(tmp_path_factory.mktemp("cache")))
    ids, edges, motors = loader.load_or_fetch_mini(neuron_type="DNge104")
    assert loader.last_source == "neuprint"
    return SpikingNetwork(ids, edges, motor_ids=motors)


def test_real_brain_loads(real_net):
    assert len(real_net.ids) > 10
    assert len(real_net.motors) == 2


def test_real_brain_settles_to_finite_motor_output(real_net):
    real_net.settle(z_right=2.0, z_left=0.5)
    drv, _ = real_net.turn(rho_right=2.0, rho_left=0.5)
    assert np.all(np.isfinite(drv))
