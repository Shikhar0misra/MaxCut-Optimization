import itertools
import random
import time

def compute_cut(G, partition):
    cut_value = 0
    for u, v in G.edges():
        if partition[u] != partition[v]:
            cut_value += 1
    return cut_value


def brute_force_maxcut(G):

    n = len(G.nodes())
    best_cut = 0
    best_partition = None

    start = time.time()

    for bits in itertools.product([0,1], repeat=n):

        cut = compute_cut(G, bits)

        if cut > best_cut:
            best_cut = cut
            best_partition = bits

    end = time.time()

    return best_cut, best_partition, end-start


def greedy_maxcut(G):

    n = len(G.nodes())

    partition = [random.randint(0,1) for _ in range(n)]

    improved = True

    start = time.time()

    while improved:

        improved = False

        for node in range(n):

            current_cut = compute_cut(G, partition)

            partition[node] = 1 - partition[node]

            new_cut = compute_cut(G, partition)

            if new_cut > current_cut:
                improved = True
            else:
                partition[node] = 1 - partition[node]

    end = time.time()

    best_cut = compute_cut(G, partition)

    return best_cut, partition, end-start