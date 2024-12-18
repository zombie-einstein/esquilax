"""
Function transformations representing agent interactions and updates

The transformation implement map-reduce patterns that represent
agents/entities observing other agents/entities, and functions
that update the state of agents/entities.
"""
from ._graph import edge_map, graph_reduce, highest_weight, random_edge
from ._grid import grid, grid_local
from ._map import amap
from ._space import nearest_neighbour, spatial
