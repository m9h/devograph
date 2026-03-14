"""Geometric convolution layers for hypergraphs."""

from devograph._conv._hyperbolic import PoincareHypergraphConv as PoincareHypergraphConv
from devograph._conv._lorentz import LorentzHypergraphConv as LorentzHypergraphConv
from devograph._conv._product import (
    ProductHypergraphConv as ProductHypergraphConv,
    ProductManifold as ProductManifold,
    ProductManifoldConv as ProductManifoldConv,
    ProductManifoldMLP as ProductManifoldMLP,
    ProductSpaceConv as ProductSpaceConv,
    ProductSpaceEmbedding as ProductSpaceEmbedding,
)

try:
    from devograph._conv._se3 import SE3HypergraphConv as SE3HypergraphConv
except ImportError:
    pass
