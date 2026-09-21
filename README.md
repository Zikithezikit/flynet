# flynet

Spiking neural network library based on the *Drosophila melanogaster* brain connectome.

Build, simulate, and train spiking neural networks using real fruit-fly connectome wiring
from [neuPrint](https://neuprint.janelia.org) / [FlyWire](https://flywire.ai), or define
your own brain structures programmatically.

```python
from flynet import SpikingNetwork, ConnectomeLoader

# Load the real fly brain
loader = ConnectomeLoader()
ids, edges, motors = loader.load_or_fetch_mini(neuron_type="DNge104")
net = SpikingNetwork(ids, edges, motor_ids=motors)

# Settle under sensory input and read motor output
trace = net.settle(z_right=2.0, z_left=0.5)
drv, _ = net.turn(rho_right=2.0, rho_left=0.5)
```

## Install

```bash
pip install -e ".[dev]"          # core only (numpy, scipy, matplotlib)
pip install -e ".[neuprint]"     # + neuPrint connectome access
pip install -e ".[all]"          # everything including dev tools
```

Requires Python >= 3.10.

## Features

| Module | What it does |
|---|---|
| `neurons` | Leaky Integrate-and-Fire and simple threshold neurons with refractory periods |
| `synapses` | Sparse (scipy CSR) and dense weight matrices with edge-list construction |
| `encoding` | Rate-coded and Poisson spike train generation |
| `receptive_fields` | Gaussian and Manhattan-distance spatial filtering kernels |
| `stdp` | Spike-Timing-Dependent Plasticity with reward-modulated eligibility traces |
| `inhibition` | Winner-takes-all and soft lateral inhibition, dynamic thresholds |
| `network` | `SpikingNetwork` -- settling dynamics, calibration, sensor-to-motor steering |
| `connectome` | `ConnectomeLoader` -- neuPrint fetch, disk caching, offline/synthetic fallback |
| `learning` | `RewardHebbian` (three-factor) and `STDPTrainer` training pipelines |
| `visualize` | Brain circuits, trajectories, learning curves, spike rasters, weight heatmaps |
| `cli` | `flynet simulate`, `flynet train`, `flynet list-datasets` |

## Quick Start

### 1. Load a real connectome

Get a free API token from https://neuprint.janelia.org (Account -> Token), then:

```bash
export NEUPRINT_APPLICATION_CREDENTIALS="your-token-here"
```

```python
from flynet.connectome import ConnectomeLoader
from flynet.network import SpikingNetwork

loader = ConnectomeLoader(dataset="male-cns:v1.0")

# Small brain slice (~41 neurons around a cell type)
ids, edges, motors = loader.load_or_fetch_mini(neuron_type="DNge104")
net = SpikingNetwork(ids, edges, motor_ids=motors)

# Or the whole fly brain (~176k neurons, stored sparse)
ids, edges, motors = loader.load_or_fetch_full(min_weight=50)
net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
```

All connectome data is cached to `~/.flynet/cache/` for offline use.

### 2. Simulate the brain

```python
# Inject sensory input and settle to a fixed point
trace = net.settle(z_right=2.0, z_left=0.5, iterations=12)
# trace is a list of 12 activity vectors, one per iteration

# Calibrate the sensor-to-motor mapping
M, Minv = net.calibrate()

# Compute a steering command from smell readings
drv, trace = net.turn(rho_right=2.0, rho_left=1.0)
# drv > 0 means turn right, < 0 means turn left

# Query the wiring
net.presynaptic(12781)   # who talks to this neuron?
net.postsynaptic(12781)  # who does this neuron talk to?
net.get_weight(25185, 12781)  # synapse strength
```

### 3. Train with reward-gated Hebbian plasticity

```python
from flynet.learning import RewardHebbian

hebbian = RewardHebbian(eta=0.6, reward_bonus=6.0)

# After a successful navigation episode, collect events and update:
events = [(rho_right, rho_left, activity), ...]
hebbian.update(net, events)
```

### 4. Train with STDP

```python
from flynet import SpikingNeuron, SynapseList, STDPRule, LateralInhibition

synapses = SynapseList(n_post=10, n_inputs=784)
stdp = STDPRule()
inhibition = LateralInhibition(mode="wta")

# In your training loop:
neurons = [SpikingNeuron(threshold=5.0) for _ in range(10)]
# ... simulate and apply STDP updates on spike pairs
```

## CLI

```bash
# List available connectome datasets
flynet list-datasets

# Simulate with the real brain (needs token)
NEUPRINT_APPLICATION_CREDENTIALS=... flynet simulate --plot

# Simulate offline with synthetic brain
flynet simulate --offline --plot

# Train the navigation task
NEUPRINT_APPLICATION_CREDENTIALS=... flynet train --episodes 10

# Train offline
flynet train --offline --episodes 5
```

## Examples

The `examples/` directory contains complete scripts:

| Script | Description |
|---|---|
| `basic_network.py` | Create a small network, settle, calibrate, STDP demo |
| `connectome_simulation.py` | Load real connectome and query wiring |
| `navigation_task.py` | Fly navigates to food using real brain wiring + Hebbian learning |
| `classification.py` | Unsupervised STDP pattern learning |

```bash
cd flynet
.venv/bin/python examples/basic_network.py
.venv/bin/python examples/navigation_task.py --episodes 5
```

## Architecture

```
flynet/
  neurons.py           LIFNeuron, SpikingNeuron, NeuronModel ABC
  synapses.py          SynapseMatrix (CSR sparse), SynapseList (dense)
  encoding.py          rate_encode, poisson_encode, SpikeTrain
  receptive_fields.py  ReceptiveField, manhattan_rf, apply_rf
  stdp.py              STDPRule, RewardModulatedSTDP
  inhibition.py        LateralInhibition, ThresholdManager
  network.py           SpikingNetwork (settle, calibrate, turn)
  connectome.py        ConnectomeLoader (neuPrint, caching, synthetic)
  learning.py          RewardHebbian, STDPTrainer, TrainingLogger
  visualize.py         plot_brain_circuit, plot_trajectory, etc.
  cli.py               flynet CLI entry point
```

**Total: ~2,600 lines** of Python across 12 modules, with 40 tests.

## Tested With

| Connectome | Neurons | Edges | Result |
|---|---|---|---|
| DNge104 mini brain | 41 | 40 | 100% navigation success, learning reduces path 77 -> 43 steps |
| Full male CNS | 66,729 | 228,220 | Settles in <0.1s, 1,236 neurons activate, steering works |

## License

MIT
