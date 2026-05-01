import random
import networkx as nx

def generate_graph(n, p):

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                G.add_edge(i, j)

    return G