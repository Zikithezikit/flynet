"""flynet - Spiking neural network library based on the Drosophila brain connectome.

Build, simulate, and train spiking neural networks using real fruit-fly
connectome wiring from neuPrint/FlyWire, or define your own brain structures
programmatically.
"""

__version__ = "0.1.0"

from flynet.neurons import LIFNeuron, NeuronModel, SpikingNeuron
from flynet.synapses import SynapseMatrix, SynapseList
from flynet.encoding import rate_encode, SpikeTrain
from flynet.receptive_fields import ReceptiveField
from flynet.stdp import STDPRule
from flynet.network import SpikingNetwork
from flynet.connectome import ConnectomeLoader
from flynet.learning import RewardHebbian, STDPTrainer, TrainingLogger
from flynet.inhibition import LateralInhibition
from flynet.visualize import (
    plot_brain_circuit,
    plot_trajectory,
    plot_learning_curve,
    plot_plasticity_heatmap,
    plot_spike_raster,
    plot_weight_matrix,
    create_gif,
)

__all__ = [
    "LIFNeuron",
    "NeuronModel",
    "SpikingNeuron",
    "SynapseMatrix",
    "SynapseList",
    "rate_encode",
    "SpikeTrain",
    "ReceptiveField",
    "STDPRule",
    "SpikingNetwork",
    "ConnectomeLoader",
    "RewardHebbian",
    "STDPTrainer",
    "TrainingLogger",
    "LateralInhibition",
    "plot_brain_circuit",
    "plot_trajectory",
    "plot_learning_curve",
    "plot_plasticity_heatmap",
    "plot_spike_raster",
    "plot_weight_matrix",
    "create_gif",
]
