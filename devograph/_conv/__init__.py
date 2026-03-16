"""Geometric convolution layers for hypergraphs.

Re-exports from hgx for backward compatibility.
"""

from hgx._conv._hyperbolic import PoincareHypergraphConv as PoincareHypergraphConv
from hgx._conv._lorentz import LorentzHypergraphConv as LorentzHypergraphConv
from hgx._conv._product import (
    ProductHypergraphConv as ProductHypergraphConv,
    ProductManifold as ProductManifold,
    ProductManifoldConv as ProductManifoldConv,
    ProductManifoldMLP as ProductManifoldMLP,
    ProductSpaceConv as ProductSpaceConv,
    ProductSpaceEmbedding as ProductSpaceEmbedding,
)

try:
    from hgx._conv._se3 import SE3HypergraphConv as SE3HypergraphConv
except ImportError:
    pass
