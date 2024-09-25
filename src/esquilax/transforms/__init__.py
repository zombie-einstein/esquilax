"""
Function transformations representing agent interactions and updates

The transformation implement map-reduce patterns that represent
agents/entities observing other agents/entities, and functions
that update the state of agents/entities.
"""
from ._graph import edge_map, graph_reduce, highest_weight, random_neighbour
from ._map import amap
from ._space import spatial
