"""Compatibility import for the isolated PyTorch project."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / "projects/dualtopo/dualtopo")]


from .detector import DualTopo
from .heads import QueryWorldModel
from .bev_constructor import BEVWorldModel
from .decoder import DualPathwayDecoder
from .topology import DirectedGraphPropagation

__all__ = [
    "DualTopo",
    "QueryWorldModel",
    "BEVWorldModel",
    "DualPathwayDecoder",
    "DirectedGraphPropagation",
]
