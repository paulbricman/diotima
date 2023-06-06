from diotima.world.universe import Universe

import jax.numpy as jnp
from jax import Array

import networkx as nx
from node2vec import Node2Vec
from einops import rearrange


def universe_to_graph(universe: Universe) -> nx.Graph:
    graph = nx.Graph()
    if universe.jac_history is None:
        return graph

    edge_list = []
    jac_history = universe.jac_history.tolist()
    for timestep in range(universe.step):
        for to_idx in range(universe.universe_config.n_atoms):
            for from_idx in range(universe.universe_config.n_atoms):
                edge_list += [(
                    timestep * universe.universe_config.n_atoms + from_idx,
                    (timestep + 1) * universe.universe_config.n_atoms + to_idx,
                    jac_history[timestep][to_idx][from_idx]
                )]
    graph.add_weighted_edges_from(edge_list)
    return graph


def graph_to_embs(
        graph: nx.Graph,
        dimensions: int = 2,
        walk_length: int = 4,
        num_walks: int = 100,
        workers: int = 1,
        window: int = 10,
        min_count: int = 1,
        batch_words: int = 4,
        p: float = 1.0,
        q: float = 1.0
):
    node2vec = Node2Vec(
        graph,
        dimensions,
        walk_length,
        num_walks,
        workers,
        p,
        q)
    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=batch_words)
    return jnp.array(model.wv.get_normed_vectors())
