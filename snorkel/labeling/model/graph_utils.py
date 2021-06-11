from typing import Iterable, List, Tuple

import networkx as nx


def get_clique_tree(nodes: Iterable[int], edges: List[Tuple[int, int]]) -> nx.Graph:
    """
    Given a set of int nodes i and edges (i,j), returns a clique tree.

    Clique tree is an object G for which:
    - G.nodes[i]['members'] contains the set of original nodes in the ith
        maximal clique
    - G[i][j]['members'] contains the set of original nodes in the seperator
        set between maximal cliques i and j

    Note: This method is currently only implemented for chordal graphs; TODO:
    add a step to triangulate non-chordal graphs.

    Parameters
    ----------
    nodes
        A list of nodes indices
    edges
        A list of tuples, where each tuple has indices for connected nodes

    Returns
    -------
    networkx.Graph
        An object G representing clique tree
    """
    # Form the original graph G1
    G1 = nx.Graph()
    G1.add_nodes_from(nodes)
    G1.add_edges_from(edges)

    # Check if graph is chordal
    # TODO: Add step to triangulate graph if not
    if not nx.is_chordal(G1):
        raise NotImplementedError("Graph triangulation not implemented.")

    # Create maximal clique graph G2
    # Each node is a maximal clique C_i
    # Let w = |C_i \cap C_j|; C_i, C_j have an edge with weight w if w > 0
    G2 = nx.Graph()
    for i, c in enumerate(nx.chordal_graph_cliques(G1)):
        G2.add_node(i, members=c)
    for i in G2.nodes:
        for j in G2.nodes:
            S = G2.nodes[i]["members"].intersection(G2.nodes[j]["members"])
            w = len(S)
            if w > 0:
                G2.add_edge(i, j, weight=w, members=S)

    # Return a minimum spanning tree of G2
    return nx.minimum_spanning_tree(G2)
