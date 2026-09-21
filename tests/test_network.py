"""Tests for flynet.network module."""

import numpy as np
import pytest
from flynet.network import SpikingNetwork


def _make_simple_network():
    """4-neuron network: 2 sensors -> 1 hidden -> 1 motor."""
    ids = [10, 11, 12, 13]
    edges = [
        (10, 12, 3.0),  # sensor 0 -> hidden
        (11, 12, 2.0),  # sensor 1 -> hidden
        (12, 13, 4.0),  # hidden -> motor
        (13, 10, 1.0),  # motor -> sensor 0 (recurrent)
    ]
    return SpikingNetwork(ids, edges, sensor_ids=[10, 11], motor_ids=[13])


class TestSpikingNetwork:
    def test_init(self):
        net = _make_simple_network()
        assert len(net.ids) == 4
        assert net.sensors == [10, 11]
        assert net.motors == [13]

    def test_weight_matrix(self):
        net = _make_simple_network()
        assert net.W.nnz > 0
        assert net.W.shape == (4, 4)

    def test_get_set_weight(self):
        net = _make_simple_network()
        w = net.get_weight(10, 12)
        assert w > 0
        net.set_weight(10, 12, 99.0)
        assert net.get_weight(10, 12) == 99.0

    def test_settle(self):
        net = _make_simple_network()
        trace = net.settle(z_right=1.0, z_left=0.5, iterations=5)
        assert len(trace) == 5
        assert trace[0].shape == (4,)

    def test_calibrate(self):
        net = _make_simple_network()
        M, Minv = net.calibrate()
        assert M.shape == (2, 2)
        assert Minv.shape == (2, 2)

    def test_turn(self):
        net = _make_simple_network()
        drv, trace = net.turn(rho_right=2.0, rho_left=1.0)
        assert isinstance(drv, float)
        assert len(trace) > 0

    def test_presynaptic(self):
        net = _make_simple_network()
        pres = net.presynaptic(12)
        assert len(pres) == 2  # 10 and 11 project into 12

    def test_postsynaptic(self):
        net = _make_simple_network()
        posts = net.postsynaptic(12)
        assert len(posts) >= 1  # 12 projects to 13

    def test_copy(self):
        net = _make_simple_network()
        net2 = net.copy()
        net2.set_weight(10, 12, 999.0)
        assert net.get_weight(10, 12) != 999.0

    def test_auto_detect_motors(self):
        ids = [100, 101, 102]
        edges = [(100, 102, 5.0), (101, 102, 3.0)]
        net = SpikingNetwork(ids, edges)
        assert 102 in net.motors

    def test_auto_detect_sensors(self):
        ids = [100, 101, 102]
        edges = [(100, 102, 5.0), (101, 102, 3.0)]
        net = SpikingNetwork(ids, edges)
        assert len(net.sensors) >= 1
        assert 100 in net.sensors or 101 in net.sensors
