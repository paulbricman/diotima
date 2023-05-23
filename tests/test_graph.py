from diotima.graph import *
from diotima.world import *
import pytest
import networkx as nx


@pytest.fixture
def universe():
    return seed(UniverseConfig())


def test_universe_to_graph(universe: Universe):
    graph = universe_to_graph(universe)
    assert len(list(graph.nodes)) == 0

    universe = run(universe, 2)
    graph = universe_to_graph(universe)
    assert len(list(graph.nodes)) == universe.universe_config.n_atoms * (universe.step + 1)
    assert len(list(graph.edges)) == universe.universe_config.n_atoms ** 2 * universe.step


def test_graph_to_embs(universe: Universe):
    emb_dim = 2
    universe = run(universe, 2)
    graph = universe_to_graph(universe)
    embs = graph_to_embs(graph, dimensions=emb_dim)
    
    assert embs.shape == (len(graph.nodes), emb_dim)
