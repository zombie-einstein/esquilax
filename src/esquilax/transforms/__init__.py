"""
Function transformations representing agent interactions and updates

The transformation implement map-reduce patterns that represent
agents/entities observing other agents/entities, and functions
that update the state of agents/entities.
"""

from ._graph import edge_map as edge_map
from ._graph import graph_reduce as graph_reduce
from ._graph import highest_weight as highest_weight
from ._graph import random_edge as random_edge
from ._grid import grid as grid
from ._grid import grid_local as grid_local
from ._map import amap as amap
from ._space import nearest_neighbour as nearest_neighbour
from ._space import spatial as spatial
