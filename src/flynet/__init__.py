"""flynet - Spiking neural network library based on the Drosophila brain connectome.

Build, simulate, and train spiking neural networks using real fruit-fly
connectome wiring fetched from neuPrint, or define your own brain structures
programmatically.
"""

__version__ = "0.1.0"

from flynet.neurons import LIFNeuron, NeuronModel, SpikingNeuron
from flynet.synapses import SynapseMatrix, SynapseList
from flynet.encoding import rate_encode, SpikeTrain
from flynet.receptive_fields import ReceptiveField
from flynet.stdp import STDPRule
from flynet.network import SpikingNetwork
from flynet.connectome import ConnectomeLoader, ConnectomeUnavailableError
from flynet.learning import (
    MotorRewardHebbian,
    RewardHebbian,
    STDPTrainer,
    TrainingLogger,
)
from flynet.inhibition import LateralInhibition

# Gradient training (optional – requires jax)
try:
    from flynet.jax_network import JaxNetwork
    from flynet.jax_trainer import GradientTrainer, ExpertTeacher
except ImportError:
    pass
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
    "ConnectomeUnavailableError",
    "MotorRewardHebbian",
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
